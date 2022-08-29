import torch
from tqdm import tqdm
from torch.optim import Optimizer
import torch.nn as nn
from DeepPose.tools.config import TrainerConfig, DataSetConfig
from DeepPose.tools.loss import Loss
from DeepPose.tools.visualize import Vis
import os
from torch.utils.data import DataLoader
import math


class WarmUpCosineAnnealOptimizer:
    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            max_epoch_for_train: int,
            base_lr: float = 1e-3,
            warm_up_end_epoch: int = 1,
    ):
        self.optimizer = optimizer
        self.set_lr(base_lr)

        self.warm_up_epoch = warm_up_end_epoch
        self.base_lr = base_lr
        self.tmp_lr = base_lr

        self.max_epoch_for_train = max_epoch_for_train

    def set_lr(self, lr):
        self.tmp_lr = lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def warm(
            self,
            now_epoch_ind,
            now_batch_ind,
            max_batch_ind
    ):
        if now_epoch_ind < self.warm_up_epoch:
            self.tmp_lr = self.base_lr * pow(
                (now_batch_ind + now_epoch_ind * max_batch_ind) * 1. / (self.warm_up_epoch * max_batch_ind), 4)
            self.set_lr(self.tmp_lr)
        else:
            T = (self.max_epoch_for_train - self.warm_up_epoch + 1) * max_batch_ind
            t = (now_epoch_ind - self.warm_up_epoch) * max_batch_ind + now_batch_ind

            lr = 1.0 / 2 * (1.0 + math.cos(t * math.pi / T)) * self.base_lr
            self.set_lr(lr)

    def zero_grad(self):
        self.optimizer.zero_grad()

    def step(self):
        self.optimizer.step()


class DeepPoseTrainer:
    def __init__(
            self,
            model: nn.Module,
            connections: list
    ):
        self.model = model
        self.device = next(model.parameters()).device
        self.connections = connections

    def train_detector_one_epoch(
            self,
            data_loader_train: DataLoader,
            loss_func: Loss,
            optimizer: WarmUpCosineAnnealOptimizer,
            now_epoch: int,
            desc: str = '',
    ):
        loss_dict_vec = {}
        max_batch_ind = len(data_loader_train)

        for batch_id, (images, keypoints) in enumerate(tqdm(data_loader_train,
                                                            desc=desc,
                                                            position=0)):
            # optimizer.warm(
            #     now_epoch,
            #     batch_id,
            #     max_batch_ind
            # )

            self.model.train()
            images = images.to(self.device)
            targets = keypoints.to(self.device)

            output = self.model(images)
            loss_res = loss_func(output, targets)

            if not isinstance(loss_res, dict):
                print('You have not use our provided loss func, please overwrite method train_detector_one_epoch')
                pass
            else:
                loss = loss_res['total_loss']
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                for key, val in loss_res.items():
                    if key not in loss_dict_vec.keys():
                        loss_dict_vec[key] = []
                    loss_dict_vec[key].append(val.item())

        loss_dict = {}
        for key, val in loss_dict_vec.items():
            loss_dict[key] = sum(val) / len(val) if len(val) != 0 else 0.0
        return loss_dict

    def eval(self,
             data_loader,
             saved_dir: str):

        os.makedirs(saved_dir, exist_ok=True)
        self.model.eval()

        for batch_index, (x, y) in enumerate(tqdm(data_loader, desc='testing for batch')):
            x = x.to(self.device)
            y = y.to(self.device)
            out = self.model(x)

            for index in range(x.shape[0]):
                saved_abs_path = os.path.join(saved_dir, '{}_{}_right.png'.format(batch_index, index))
                Vis.plot_key_points(x[index], y[index],
                                    connections=self.connections,
                                    saved_abs_path=saved_abs_path)
                saved_abs_path = os.path.join(saved_dir, '{}_{}_predict.png'.format(batch_index, index))
                Vis.plot_key_points(x[index], out[index],
                                    connections=self.connections,
                                    saved_abs_path=saved_abs_path)

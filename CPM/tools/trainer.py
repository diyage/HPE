import torch
from tqdm import tqdm
from torch.optim import Optimizer
import torch.nn as nn
from CPM.tools.config import TrainerConfig, DataSetConfig
from CPM.tools.visualize import Vis
import os


class CPMTrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer = None,
                 opt_trainer: TrainerConfig = None,
                 opt_data_set: DataSetConfig = None):
        self.model = model

        if opt_trainer is None:
            self.opt_trainer = TrainerConfig()
        else:
            self.opt_trainer = opt_trainer

        if opt_data_set is None:
            self.opt_data_set = DataSetConfig()
        else:
            self.opt_data_set = opt_data_set

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.opt_trainer.lr)
        else:
            self.optimizer = optimizer

    def __train_one_epoch(self,
                          data_loader):
        self.model.train()

        for _, info in enumerate(data_loader):
            image = info['image']  # type: torch.Tensor
            gt_map = info['gt_map']  # type: torch.Tensor
            center_map = info['center_map']  # type: torch.Tensor

            x = image.to(self.opt_trainer.device)
            y = gt_map.to(self.opt_trainer.device)
            c = center_map.to(self.opt_trainer.device)

            out = self.model(x, c)
            loss_vec = []
            for i in range(out.shape[1]):
                loss_vec.append(nn.MSELoss()(out[:, i], y))
            loss = sum(loss_vec)/len(loss_vec)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self,
             data_loader,
             saved_dir: str):

        os.makedirs(saved_dir, exist_ok=True)
        self.model.eval()

        for batch_index, info in enumerate(data_loader):
            image = info['image']  # type: torch.Tensor
            gt_map = info['gt_map']  # type: torch.Tensor
            center_map = info['center_map']  # type: torch.Tensor

            x = image.to(self.opt_trainer.device)
            y = gt_map.to(self.opt_trainer.device)
            c = center_map.to(self.opt_trainer.device)

            out = self.model(x, c)
            out = out[:, -1]

            assert out.shape == y.shape

            for index in range(x.shape[0]):
                saved_abs_path = os.path.join(saved_dir, '{}_{}_right.png'.format(batch_index, index))
                Vis.plot_key_point_using_heat_map(x[index], y[index],
                                                  connections=self.opt_data_set.connections,
                                                  saved_abs_path=saved_abs_path)

                saved_abs_path = os.path.join(saved_dir, '{}_{}_predict.png'.format(batch_index, index))
                Vis.plot_key_point_using_heat_map(x[index], out[index],
                                                  connections=self.opt_data_set.connections,
                                                  saved_abs_path=saved_abs_path)

    def train(self,
              data_loader_train,
              data_loader_test,):
        for epoch in tqdm(range(self.opt_trainer.MAX_EPOCH), desc='training for epoch'):
            self.__train_one_epoch(data_loader_train)
            saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
            self.eval(data_loader_test, saved_dir)

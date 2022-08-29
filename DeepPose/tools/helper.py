from DeepPose.tools.config import Config
from DeepPose.tools.model_define import DeepPose
from DeepPose.tools.trainer import DeepPoseTrainer, WarmUpCosineAnnealOptimizer
from DeepPose.tools.loss import Loss
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch
import torch.nn as nn
import os


class Helper:
    def __init__(
            self,
            model: DeepPose,
            config: Config,
            restore_epoch: int = -1
    ):
        self.model = model
        self.config = config
        self.restore_epoch = restore_epoch

        if restore_epoch != -1:
            self.restore(restore_epoch)

        self.trainer = DeepPoseTrainer(
            model,
            config.data_config.connections
        )

    def restore(
            self,
            epoch: int
    ):
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
        saved_file_name = '{}/{}.pth'.format(saved_dir, epoch)
        self.model.load_state_dict(
            torch.load(saved_file_name)
        )

    def save(
            self,
            epoch: int
    ):
        # save model
        self.model.eval()
        saved_dir = self.config.ABS_PATH + os.getcwd() + '/model_pth_detector/'
        os.makedirs(saved_dir, exist_ok=True)
        torch.save(self.model.state_dict(), '{}/{}.pth'.format(saved_dir, epoch))

    def go(
            self,
            data_loader_train: DataLoader,
            data_loader_test: DataLoader,
    ):

        loss_func = Loss(rate=0.5)

        sgd_optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.config.trainer_config.lr,
            momentum=0.9,
            weight_decay=5e-4
        )

        adam_optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.config.trainer_config.lr,
        )
        warm_optimizer = WarmUpCosineAnnealOptimizer(
            adam_optimizer,
            self.config.trainer_config.MAX_EPOCH,
            base_lr=self.config.trainer_config.lr,
            warm_up_end_epoch=self.config.trainer_config.WARM_UP_END_EPOCH
        )

        for epoch in tqdm(range(self.restore_epoch + 1, self.config.trainer_config.MAX_EPOCH),
                          desc='training detector',
                          position=0):

            loss_dict = self.trainer.train_detector_one_epoch(
                data_loader_train,
                loss_func,
                warm_optimizer,
                now_epoch=epoch,
                desc='[train for detector epoch: {}/{}]'.format(epoch,
                                                                self.config.trainer_config.MAX_EPOCH - 1)
            )

            print_info = '\n\nepoch: {} [ now lr:{:.8f} ] , loss info-->\n'.format(
                epoch,
                warm_optimizer.tmp_lr
            )
            for key, val in loss_dict.items():
                print_info += '{:^30}:{:^15.6f}.\n'.format(key, val)
            tqdm.write(print_info)

            if epoch % self.config.eval_frequency == 0:
                # save model
                self.save(epoch)

                # show predict
                with torch.no_grad():
                    saved_dir = self.config.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
                    self.trainer.eval(data_loader_test, saved_dir)

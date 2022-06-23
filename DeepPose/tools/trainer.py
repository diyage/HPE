import torch
from tqdm import tqdm
from torch.optim import Optimizer
import torch.nn as nn
from DeepPose.tools.config import TrainerConfig
from DeepPose.tools.visualize import Vis
import os


class DeepPoseTrainer:
    def __init__(self,
                 model: nn.Module,
                 optimizer: Optimizer = None,
                 opt: TrainerConfig = None):
        self.model = model
        if opt is None:
            self.opt = TrainerConfig()
        else:
            self.opt = opt

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.opt.lr)
        else:
            self.optimizer = optimizer

    def __train_one_epoch(self,
                          data_loader):
        self.model.train()
        for _, (x, y) in enumerate(tqdm(data_loader, desc='training for batch')):
            x = x.to(self.opt.device)
            y = y.to(self.opt.device)
            out = self.model(x)
            loss = nn.MSELoss()(out, y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self,
             data_loader,
             saved_dir: str):

        os.makedirs(saved_dir, exist_ok=True)
        self.model.eval()

        for batch_index, (x, y) in enumerate(tqdm(data_loader, desc='testing for batch')):
            x = x.to(self.opt.device)
            y = y.to(self.opt.device)
            out = self.model(x)

            for index in range(x.shape[0]):
                saved_abs_path = os.path.join(saved_dir, '{}_{}_right.png'.format(batch_index, index))
                Vis.plot_key_points(x[index], y[index], saved_abs_path)
                saved_abs_path = os.path.join(saved_dir, '{}_{}_predict.png'.format(batch_index, index))
                Vis.plot_key_points(x[index], out[index], saved_abs_path)

    def train(self,
              data_loader_train,
              data_loader_test,):
        for epoch in tqdm(range(self.opt.MAX_EPOCH), desc='training for epoch'):
            self.__train_one_epoch(data_loader_train)
            saved_dir = self.opt.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
            self.eval(data_loader_test, saved_dir)




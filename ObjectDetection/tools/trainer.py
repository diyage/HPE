import torch
import torch.nn as nn
from ObjectDetection.tools.model_define import YoLoV1
from torch.optim.optimizer import Optimizer
from ObjectDetection.tools.config_define import *
from ObjectDetection.tools.others import YOLOV1Tools
from ObjectDetection.tools.loss_define import YoLoV1Loss, Loss
from ObjectDetection.tools.predictor import YoLoV1Predictor
import os
from tqdm import tqdm


class YOLOV1Trainer:
    def __init__(self,
                 model: YoLoV1,
                 yolo_tools: YOLOV1Tools,
                 yolo_predictor: YoLoV1Predictor,
                 optimizer: Optimizer = None,
                 data_opt: DataSetConfig = None,
                 train_opt: TrainConfig = None,
                 ):
        self.model = model
        self.yolo_tools = yolo_tools
        self.yolo_predictor = yolo_predictor

        self.yolo_loss_fn = YoLoV1Loss()
        self.right_yolo_loss = Loss()

        if train_opt is None:
            self.opt_trainer = TrainConfig()
        else:
            self.opt_trainer = train_opt

        if data_opt is None:
            self.opt_data_set = DataSetConfig()
        else:
            self.opt_data_set = data_opt

        if optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.opt_trainer.lr)
        else:
            self.optimizer = optimizer

    def __train_one_epoch(self,
                          data_loader):

        for batch_id, (images, labels) in enumerate(data_loader):
            self.model.train()
            images = images.cuda()
            ouput = self.model(images)

            pre_targets = self.yolo_tools.make_targets(labels)
            pre_targets = pre_targets.cuda()

            # loss, position_loss, conf_loss, class_loss = yolo_loss_fn(ouput, pre_targets)  # type: torch.Tensor
            # optimizer.zero_grad()
            #
            # loss.backward()
            # optimizer.step()
            #
            # print('*'*100)
            # # print(net.state_dict()['module.detector.2.bias'])
            # print("epoch: {:>4}, batch: {:>4}, loss: {:>10.9f}".format(epoch_id, batch_id, loss.item()))
            # print("  >> pos_loss: {:>10.6f}, conf_loss: {:>10.6f}, cls_loss: {:>10.6f}".format(
            #         position_loss.item(), conf_loss.item(), class_loss.item()))
            # print('*' * 100)

            loss = self.right_yolo_loss(ouput, pre_targets)  # type: torch.Tensor
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def eval(self,
             data_loader,
             saved_dir: str):

        os.makedirs(saved_dir, exist_ok=True)

        for batch_id, (images, labels) in enumerate(data_loader):
            self.model.eval()

            images = images.cuda()
            ouput = self.model(images)

            pre_targets = self.yolo_tools.make_targets(labels)
            pre_targets = pre_targets.cuda()

            for index in range(images.shape[0]):
                self.yolo_predictor.visualize(images[index],
                                              ouput[index],
                                              saved_path='{}/{}_{}_predict.png'.format(saved_dir, batch_id, index))
                self.yolo_predictor.visualize(images[index],
                                              pre_targets[index],
                                              saved_path='{}/{}_{}_gt.png'.format(saved_dir, batch_id, index))

    def train(self,
              data_loader_train,
              data_loader_test,):
        for epoch in tqdm(range(self.opt_trainer.max_epoch), desc='training for epoch'):
            self.__train_one_epoch(data_loader_train)
            saved_dir = self.opt_trainer.ABS_PATH + os.getcwd() + '/eval_images/{}/'.format(epoch)
            if epoch % 10 == 0:
                # eval image
                self.eval(data_loader_test, saved_dir)

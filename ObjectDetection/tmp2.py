import torch.nn as nn
import torchvision.transforms.transforms as transforms
from torch.utils.data import DataLoader
from ObjectDetection.tools.dataset_define import VOC2012DataSet
from ObjectDetection.tools.config_define import *
from ObjectDetection.tools.loss_define import YOLOV1Tools
from ObjectDetection.tools.model_define import YoLoV1
from ObjectDetection.tools.trainer import YOLOV1Trainer
from ObjectDetection.tools.predictor import YoLoV1Predictor


train_opt = TrainConfig()
data_opt = DataSetConfig()
yolo_v1_tools = YOLOV1Tools(data_opt.kinds_name, data_opt.grid_number, data_opt.image_size)

max_epoch = train_opt.max_epoch
batch_size = train_opt.batch_size

transform = transforms.Compose([
        transforms.ToTensor(),
    ])

train_d = VOC2012DataSet(root=data_opt.root_path,
                         train=True,
                         image_size=data_opt.image_size,
                         transform=transform)

train_l = DataLoader(train_d, batch_size=batch_size, collate_fn=VOC2012DataSet.collate_fn, shuffle=True)


test_d = VOC2012DataSet(root=data_opt.root_path,
                        train=False,
                        image_size=data_opt.image_size,
                        transform=transform)

test_l = DataLoader(test_d, batch_size=batch_size, collate_fn=VOC2012DataSet.collate_fn, shuffle=False)


net = YoLoV1()
net = nn.DataParallel(net)
net.cuda()

yolo_predictor = YoLoV1Predictor(net,
                                 conf_th=train_opt.conf_th,
                                 prob_th=train_opt.prob_th,
                                 iou_th=train_opt.iou_th,
                                 image_size=data_opt.image_size,
                                 grid_number=data_opt.grid_number,
                                 kinds_name=data_opt.kinds_name)
yolo_trainer = YOLOV1Trainer(net,
                             yolo_v1_tools,
                             yolo_predictor)
yolo_trainer.train(train_l, test_l)

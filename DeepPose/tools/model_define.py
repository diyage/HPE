import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F


class DeepPose(nn.Module):
    def __init__(self,
                 nJoints,
                 modelName='resnet50'
                 ):
        super(DeepPose, self).__init__()
        self.nJoints = nJoints
        self.block = 'BottleNeck' if (int(modelName[6:]) > 34) else 'BasicBlock'
        self.resnet = getattr(torchvision.models, modelName)(pretrained=True)
        self.resnet.fc = nn.Linear(512 * (4 if self.block == 'BottleNeck' else 1), self.nJoints * 2)  # type:nn.Linear
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.resnet(x)
        out = self.sigmoid(out)  # type: torch.Tensor
        return out.view(-1, self.nJoints, 2)


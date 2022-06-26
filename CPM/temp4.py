from CPM.tools.model_define import CPMNet
from CPM.tools.config import DataSetConfig
import torch
import torch.nn as nn
from CPM.tools.dataset_define import LspDataSet

from torch.utils.data import DataLoader


import numpy as np
import matplotlib.pyplot as plt
from CPM.tools.visualize import Vis
from DeepPose.tools.cv2_ import CV2
from CPM.tools.dataset_define import guassian_kernel

m = CPMNet(data_set_config=DataSetConfig()).to('cuda:1')
d = LspDataSet(root=DataSetConfig.root_path, use_visibility=True, train=True)
dl = DataLoader(d, batch_size=DataSetConfig.BATCH_SIZE, shuffle=False)

d_2 = LspDataSet(root=DataSetConfig.root_path, use_visibility=True, train=False)
dl_2 = DataLoader(d_2, batch_size=DataSetConfig.BATCH_SIZE, shuffle=False)

from CPM.tools.evaluation import OKS

oks = OKS(m, dl, 368, 368)
print(oks.sigma)
res = oks.compute(dl_2)
print(res)
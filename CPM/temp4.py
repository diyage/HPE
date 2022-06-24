from CPM.tools.model_define import CPMNet
from CPM.tools.config import DataSetConfig
import torch
import torch.nn as nn
from CPM.tools.dataset_define import LspDataSet

from torch.utils.data import DataLoader
d = LspDataSet(root='E:\PyCharm\DataSet\lsp')
dl = DataLoader(d, batch_size=2, shuffle=False)
import numpy as np
import matplotlib.pyplot as plt
from CPM.tools.visualize import Vis

for _, info in enumerate(dl):
    image = info['image']  # type: torch.Tensor
    gt_map = info['gt_map']  # type: torch.Tensor
    center_map = info['center_map']  # type: torch.Tensor
    index = 1

    Vis.plot_key_point_using_heat_map(image[index], gt_map[index], connections=DataSetConfig.connections)

    break

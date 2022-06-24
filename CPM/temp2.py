from CPM.tools.model_define import CPMNet
from CPM.tools.config import DataSetConfig
import torch
import torch.nn as nn
from CPM.tools.dataset_define import LspDataSet

import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

data_set_config = DataSetConfig()
data_set_config.sigma = 0.03

d = LspDataSet(root='E:\PyCharm\DataSet\lsp', data_set_opt=data_set_config)
dl = DataLoader(d, batch_size=2, shuffle=False)


for _, info in enumerate(dl):
    image = info['image']  # type: torch.Tensor
    gt_map = info['gt_map']  # type: torch.Tensor
    center_map = info['center_map']  # type: torch.Tensor
    index = 1

    im = image[index].numpy()
    im = np.transpose(im, axes=(1, 2, 0))
    plt.imshow(im)
    plt.show()

    cen_im = center_map[index].numpy()
    plt.imshow(cen_im, cmap='gray')
    plt.show()

    ans = 0
    for i in range(14):
        key_p = gt_map[index][i]  # type: torch.Tensor
        ans = key_p + ans
    plt.imshow(ans, cmap='gray')
    plt.show()

    break

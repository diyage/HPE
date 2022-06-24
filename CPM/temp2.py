from CPM.tools.model_define import CPMNet
from CPM.tools.config import DataSetConfig
import torch
import torch.nn as nn
from CPM.tools.dataset_define import LspDataSet

# data_set_config = DataSetConfig()
#
# m = CPMNet(data_set_config)
#
# x = torch.rand(size=(2, 3, 368, 368))
# center_map = torch.rand(size=(2, 368, 368))
# out = m(x, center_map)
# print(out.shape)

from torch.utils.data import DataLoader
d = LspDataSet(root='E:\PyCharm\DataSet\lsp')
dl = DataLoader(d, batch_size=2, shuffle=False)
import numpy as np
import matplotlib.pyplot as plt


def get_matrix_max_pos(matrix: torch.Tensor)->tuple:
    max_index = matrix.argmax()
    tmp = torch.zeros(*matrix.shape).view(-1)
    tmp[max_index] = 1.0
    tmp = tmp.view(*matrix.shape)
    # get r/c
    index_r = tmp.sum(dim=1).argmax()  # r
    index_c = tmp.sum(dim=0).argmax()  # c

    return int(index_c.item()), int(index_r.item())


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

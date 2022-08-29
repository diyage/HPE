import torch.nn as nn
import torch
from typing import List


class Loss(nn.Module):
    def __init__(
            self,
            rate: float = 0.5
    ):
        super().__init__()
        self.rate = rate
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def __compute(
            self,
            predict: torch.Tensor,
            target: torch.Tensor
    ):

        a = self.l1(predict, target)
        b = self.mse(predict, target)
        return self.rate * a + (1.0 - self.rate) * b

    def forward(
            self,
            out: torch.Tensor,
            target: torch.Tensor
    ):
        res = {}
        for i in range(out.shape[1]):
            predict = out[:, i]
            res['stage_{}'.format(i)] = self.__compute(
                predict,
                target
            )
        total = 0
        for _, val in res.items():
            total += val
        res['total_loss'] = total
        return res

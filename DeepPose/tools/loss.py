import torch.nn as nn
import torch


class Loss(nn.Module):
    def __init__(
            self,
            rate: float = 0.5
    ):
        super().__init__()
        self.rate = rate
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()

    def forward(
            self,
            predict: torch.Tensor,
            target: torch.Tensor
    ):
        a = self.l1(predict, target)
        b = self.mse(predict, target)
        return {
            'total_loss': self.rate * a + (1.0 - self.rate) * b
        }

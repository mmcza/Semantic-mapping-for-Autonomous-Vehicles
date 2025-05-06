import torch
from torch import nn

from monai import losses

class FocalLoss(nn.Module):
    def __init__(self, include_background=True, gamma=2.0, weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self._focal_loss = losses.FocalLoss(
            include_background=include_background,
            gamma=gamma,
            weight=torch.tensor(weights),
        )

    def forward(self, y_pred, y):
        return self._focal_loss(y_pred, y)
import torch
from torch import nn

from monai import losses

class DiceLoss(nn.Module):
    def __init__(self, include_background=True):
        super().__init__()
        self._dice_loss = losses.DiceLoss(
            include_background=include_background,
            sigmoid=True,
            smooth_nr=1e-6,
            smooth_dr=1e-6,
            batch=True,
            reduction="mean"
        )

    def forward(self, y_pred, y):
        return self._dice_loss(y_pred, y)
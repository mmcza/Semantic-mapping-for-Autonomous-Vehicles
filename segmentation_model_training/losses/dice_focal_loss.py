import torch
from torch import nn

from monai import losses

class DiceFocalLoss(nn.Module):
    def __init__(self, include_background=True, gamma=2.0, weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        self._dice_focal_loss = losses.DiceFocalLoss(
            include_background=include_background,
            gamma=gamma,
            weight=torch.tensor(weights),
            sigmoid=True,
            smooth_nr=1e-6,
            smooth_dr=1e-6,
            batch=True,
            reduction="mean"
        )

    def forward(self, y_pred, y):
        return self._dice_focal_loss(y_pred, y)

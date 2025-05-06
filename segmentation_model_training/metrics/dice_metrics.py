import torch
from torchmetrics import Metric
from monai import metrics

class DiceMetric(Metric):
    def __init__(self, include_background=True, num_classes=4, original_size=(720, 1280)):
        super().__init__()
        self._dice_metric = metrics.DiceMetric(
            include_background=include_background,
            reduction="mean",
        )
        self.num_classes = num_classes
        self.original_size = original_size  # (H, W) of the original images before padding
        self.add_state("dice_value", default=torch.tensor(0.0), dist_reduce_fx="mean")
        
    def update(self, preds, target):      
        batch_size = preds.shape[0]
        h, w = preds.shape[2], preds.shape[3]

        # Create padding mask to ignore the padded area in the metric calculation
        padding_mask = torch.ones((batch_size, h, w), dtype=torch.bool, device=preds.device)
        padding_offset = h - self.original_size[0]
        padding_mask[:, :padding_offset, :self.original_size[1]] = False
        
        # Handle different input formats
        if len(preds.shape) == 4 and preds.shape[1] > 1:  # [B, C, H, W] logits
            preds_argmax = torch.argmax(preds, dim=1)  # [B, H, W]
            preds_onehot = torch.nn.functional.one_hot(preds_argmax, num_classes=self.num_classes)  # [B, H, W, C]
            preds_onehot = preds_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        else:
            print(f"Unexpected prediction shape: {preds.shape}")
            return
            
        # Handle target format
        if len(target.shape) == 4 and target.shape[1] > 1:  # [B, C, H, W] already one-hot
            target_onehot = target.float()
        elif len(target.shape) == 3:  # [B, H, W] class indices
            target_onehot = torch.nn.functional.one_hot(target.long(), num_classes=self.num_classes)  # [B, H, W, C]
            target_onehot = target_onehot.permute(0, 3, 1, 2).float()  # [B, C, H, W]
        else:
            print(f"Unexpected target shape: {target.shape}")
            return
        
        for c in range(self.num_classes):
            preds_onehot[:, c, ...] = preds_onehot[:, c, ...] * padding_mask
            target_onehot[:, c, ...] = target_onehot[:, c, ...] * padding_mask
        
        self._dice_metric(y_pred=preds_onehot, y=target_onehot)
        
        # Get the result
        dice = self._dice_metric.aggregate().item()
        self.dice_value = torch.tensor(dice)
        
    def compute(self):
        return self.dice_value
        
    def reset(self):
        super().reset()
        self._dice_metric.reset()
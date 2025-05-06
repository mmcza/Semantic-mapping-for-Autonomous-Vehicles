import torch
import torch.nn as nn

class LossWrapper(nn.Module):
    def __init__(self, loss_fn, original_size=(720, 1280)):
        super().__init__()
        self.loss_fn = loss_fn
        self.original_size = original_size
        
    def forward(self, y_pred, y):
        batch_size = y_pred.shape[0]
        h, w = y_pred.shape[2], y_pred.shape[3]
        
        # Add padding mask to ignore the padded area in the loss calculation
        padding_mask = torch.ones((batch_size, 1, h, w), 
                                  device=y_pred.device,
                                  dtype=y_pred.dtype)
        
        padding_offset = h - self.original_size[0]
        padding_mask[:, :, :padding_offset, :self.original_size[1]] = 0.0
        
        masked_y_pred = y_pred.clone()
        for c in range(y_pred.shape[1]):
            masked_y_pred[:, c:c+1, ...] = masked_y_pred[:, c:c+1, ...] * padding_mask
        
        masked_y = y
        if y.shape[1] > 1:
            masked_y = y.clone()
            for c in range(y.shape[1]):
                masked_y[:, c:c+1, ...] = masked_y[:, c:c+1, ...] * padding_mask
        
        # Calculate loss with masked inputs
        return self.loss_fn(masked_y_pred, masked_y)
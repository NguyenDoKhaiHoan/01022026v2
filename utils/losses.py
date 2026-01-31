# utils/losses.py

import torch
import torch.nn as nn


# ==================================================
# 1. Charbonnier Loss (Smooth L1 dạng robust)
# ==================================================
class CharbonnierLoss(nn.Module):
    """
    Charbonnier Loss:
        L(x, y) = mean( sqrt( (x-y)^2 + eps^2 ) )

    Thường dùng thay SmoothL1 để regression ổn định hơn.
    """

    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        diff = pred - target
        loss = torch.mean(torch.sqrt(diff * diff + self.eps * self.eps))
        return loss

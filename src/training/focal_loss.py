"""
focal_loss.py
-------------
Focal Loss để chống Class Imbalance trong BC Training.
Phạt mạnh hơn khi model tự tin nhưng predict sai nhãn hiếm (Buy/Sell).

FL(pt) = -alpha_t × (1 - pt)^gamma × log(pt)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class FocalLoss(nn.Module):
    def __init__(
        self,
        gamma:         float               = 2.0,
        class_weights: Optional[torch.Tensor] = None,
        reduction:     str                 = "mean",
    ):
        super().__init__()
        self.gamma         = gamma
        self.class_weights = class_weights
        self.reduction     = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        logits  : (batch, n_classes) — raw logits (chưa softmax)
        targets : (batch,) — class indices [0, n_classes)
        """
        ce_loss = F.cross_entropy(logits, targets,
                                  weight=self.class_weights,
                                  reduction="none")
        pt      = torch.exp(-ce_loss)           # xác suất của class đúng
        fl      = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return fl.mean()
        elif self.reduction == "sum":
            return fl.sum()
        return fl
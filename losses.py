"""
Loss Functions for Semantic Segmentation
=========================================
Includes:
  - Cross-Entropy Loss (standard + weighted)
  - Dice Loss
  - Focal Loss
  - Tversky Loss
  - Combined (Dice + Focal) — best for class-imbalanced aerial imagery
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Dice Loss
# ---------------------------------------------------------------------------

class DiceLoss(nn.Module):
    """
    Soft Dice Loss for multi-class segmentation.

    Args:
        smooth         : Smoothing constant to avoid division by zero.
        ignore_index   : Class index to ignore (e.g., void/unlabeled).
        per_image      : Compute loss per image in batch then average.
    """

    def __init__(self, smooth: float = 1.0, ignore_index: int = -1):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits  : (B, C, H, W) raw model outputs.
            targets : (B, H, W) ground-truth class indices.
        """
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)

        # One-hot encode targets  →  (B, C, H, W)
        targets_oh = F.one_hot(targets.clamp(0), num_classes).permute(0, 3, 1, 2).float()

        # Mask out ignored class
        if self.ignore_index >= 0:
            valid = (targets != self.ignore_index).unsqueeze(1).float()
            probs = probs * valid
            targets_oh = targets_oh * valid

        # Flatten spatial dims
        probs_flat   = probs.contiguous().view(probs.size(0), num_classes, -1)
        targets_flat = targets_oh.contiguous().view(targets_oh.size(0), num_classes, -1)

        intersection = (probs_flat * targets_flat).sum(-1)
        union        = probs_flat.sum(-1) + targets_flat.sum(-1)

        dice_per_class = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice_per_class.mean()


# ---------------------------------------------------------------------------
# Focal Loss
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017) — down-weights easy examples.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        gamma        : Focusing parameter (2.0 recommended).
        alpha        : Tensor of per-class weights or None.
        ignore_index : Class index to ignore.
        reduction    : 'mean' | 'sum' | 'none'.
    """

    def __init__(
        self,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        ignore_index: int = -100,
        reduction: str = "mean",
    ):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha.to(logits.device) if self.alpha is not None else None,
            ignore_index=self.ignore_index,
            reduction="none",
        )
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        return focal_loss


# ---------------------------------------------------------------------------
# Tversky Loss  (generalisation of Dice; useful for imbalanced datasets)
# ---------------------------------------------------------------------------

class TverskyLoss(nn.Module):
    """
    Tversky Loss:  TI = TP / (TP + alpha*FP + beta*FN)
    alpha=beta=0.5 → Dice Loss
    alpha=0.3, beta=0.7 → penalises false negatives more (recall-focused)
    """

    def __init__(self, alpha: float = 0.3, beta: float = 0.7, smooth: float = 1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        probs = F.softmax(logits, dim=1)
        targets_oh = F.one_hot(targets.clamp(0), num_classes).permute(0, 3, 1, 2).float()

        probs_flat   = probs.contiguous().view(probs.size(0), num_classes, -1)
        targets_flat = targets_oh.contiguous().view(targets_oh.size(0), num_classes, -1)

        tp = (probs_flat * targets_flat).sum(-1)
        fp = (probs_flat * (1 - targets_flat)).sum(-1)
        fn = ((1 - probs_flat) * targets_flat).sum(-1)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)
        return 1.0 - tversky.mean()


# ---------------------------------------------------------------------------
# Combined Loss  (Dice + Focal — recommended for aerial segmentation)
# ---------------------------------------------------------------------------

class CombinedLoss(nn.Module):
    """
    Weighted sum of Dice Loss and Focal Loss.
    Dice ensures spatial overlap; Focal handles class imbalance.

    Args:
        dice_weight  : Weight for the Dice component.
        focal_weight : Weight for the Focal component.
        gamma        : Focal loss focusing parameter.
        alpha        : Per-class weights for Focal loss.
        smooth       : Dice smoothing constant.
        ignore_index : Class index to ignore.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        focal_weight: float = 0.5,
        gamma: float = 2.0,
        alpha: torch.Tensor = None,
        smooth: float = 1.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.dice_weight  = dice_weight
        self.focal_weight = focal_weight
        self.dice_loss  = DiceLoss(smooth=smooth)
        self.focal_loss = FocalLoss(gamma=gamma, alpha=alpha, ignore_index=ignore_index)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        dice  = self.dice_loss(logits, targets)
        focal = self.focal_loss(logits, targets)
        return self.dice_weight * dice + self.focal_weight * focal


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def build_criterion(config: dict, class_weights: torch.Tensor = None) -> nn.Module:
    """
    Build a loss function from config.

    Config keys:
        loss_type : 'ce' | 'dice' | 'focal' | 'tversky' | 'combined'
        class_weights : bool — whether to use computed class weights
        focal_gamma   : float
        tversky_alpha / tversky_beta : float
        dice_weight / focal_weight   : float (for combined)
        ignore_index  : int
    """
    loss_type    = config.get("loss_type", "combined").lower()
    ignore_index = config.get("ignore_index", -100)
    alpha = class_weights if config.get("class_weights", False) else None

    if loss_type == "ce":
        return nn.CrossEntropyLoss(
            weight=alpha,
            ignore_index=ignore_index,
        )
    elif loss_type == "dice":
        return DiceLoss(smooth=1.0)
    elif loss_type == "focal":
        return FocalLoss(
            gamma=config.get("focal_gamma", 2.0),
            alpha=alpha,
            ignore_index=ignore_index,
        )
    elif loss_type == "tversky":
        return TverskyLoss(
            alpha=config.get("tversky_alpha", 0.3),
            beta=config.get("tversky_beta", 0.7),
        )
    elif loss_type == "combined":
        return CombinedLoss(
            dice_weight=config.get("dice_weight", 0.5),
            focal_weight=config.get("focal_weight", 0.5),
            gamma=config.get("focal_gamma", 2.0),
            alpha=alpha,
            ignore_index=ignore_index,
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    B, C, H, W = 2, 6, 512, 512
    logits  = torch.randn(B, C, H, W)
    targets = torch.randint(0, C, (B, H, W))

    for name, fn in [
        ("DiceLoss",    DiceLoss()),
        ("FocalLoss",   FocalLoss()),
        ("TverskyLoss", TverskyLoss()),
        ("CombinedLoss",CombinedLoss()),
    ]:
        loss = fn(logits, targets)
        print(f"{name:15s}: {loss.item():.4f}")

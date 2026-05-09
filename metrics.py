"""
Segmentation Evaluation Metrics
================================
Pixel Accuracy, Mean IoU (Intersection-over-Union), Dice Coefficient,
Precision, Recall, and F1 per class.
"""

import torch
import numpy as np
from typing import Dict, List


class SegmentationMetrics:
    """
    Accumulates predictions over an entire epoch and computes metrics.

    Args:
        num_classes  : Number of segmentation classes.
        ignore_index : Class index to exclude from metrics (e.g. void).
        device       : Computation device for the confusion matrix.
    """

    def __init__(self, num_classes: int, ignore_index: int = -1, device: str = "cpu"):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.device       = device
        self.reset()

    def reset(self):
        self.confusion_matrix = torch.zeros(
            self.num_classes, self.num_classes, dtype=torch.long, device=self.device
        )

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            preds   : (B, H, W) predicted class indices.
            targets : (B, H, W) ground-truth class indices.
        """
        preds   = preds.to(self.device).reshape(-1)
        targets = targets.to(self.device).reshape(-1)

        # Mask ignored pixels
        if self.ignore_index >= 0:
            valid = targets != self.ignore_index
            preds   = preds[valid]
            targets = targets[valid]

        # Update confusion matrix
        mask = (targets >= 0) & (targets < self.num_classes)
        idx  = self.num_classes * targets[mask] + preds[mask]
        self.confusion_matrix += torch.bincount(idx, minlength=self.num_classes ** 2).reshape(
            self.num_classes, self.num_classes
        )

    def compute(self) -> Dict:
        cm = self.confusion_matrix.float()

        tp = torch.diag(cm)
        fp = cm.sum(dim=0) - tp
        fn = cm.sum(dim=1) - tp

        # Per-class IoU
        iou = tp / (tp + fp + fn + 1e-10)

        # Per-class Dice / F1
        dice = 2 * tp / (2 * tp + fp + fn + 1e-10)

        # Per-class Precision & Recall
        precision = tp / (tp + fp + 1e-10)
        recall    = tp / (tp + fn + 1e-10)

        # Pixel accuracy
        pixel_acc = tp.sum() / (cm.sum() + 1e-10)

        # Mean metrics (exclude classes with no ground-truth pixels)
        valid_classes = (cm.sum(dim=1) > 0)
        mean_iou   = iou[valid_classes].mean()
        mean_dice  = dice[valid_classes].mean()

        return {
            "mean_iou":       mean_iou.item(),
            "mean_dice":      mean_dice.item(),
            "pixel_accuracy": pixel_acc.item(),
            "per_class_iou":  iou.cpu().tolist(),
            "per_class_dice": dice.cpu().tolist(),
            "precision":      precision.cpu().tolist(),
            "recall":         recall.cpu().tolist(),
        }

    def pretty_print(self, class_names: List[str] = None):
        results = self.compute()
        print(f"\n{'='*60}")
        print(f"  Segmentation Metrics")
        print(f"{'='*60}")
        print(f"  Pixel Accuracy : {results['pixel_accuracy']:.4f}")
        print(f"  Mean IoU       : {results['mean_iou']:.4f}")
        print(f"  Mean Dice      : {results['mean_dice']:.4f}")
        print(f"\n  Per-Class IoU:")
        for i, (iou, dice, prec, rec) in enumerate(zip(
            results["per_class_iou"],
            results["per_class_dice"],
            results["precision"],
            results["recall"],
        )):
            name = class_names[i] if class_names else f"Class {i}"
            print(f"    [{i}] {name:<25s}  IoU={iou:.4f}  Dice={dice:.4f}  P={prec:.4f}  R={rec:.4f}")
        print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# Standalone metric functions (for quick use outside the class)
# ---------------------------------------------------------------------------

def compute_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute per-class IoU for a single batch."""
    ious = []
    for cls in range(num_classes):
        pred_cls   = pred == cls
        target_cls = target == cls
        intersection = (pred_cls & target_cls).sum().float()
        union        = (pred_cls | target_cls).sum().float()
        ious.append((intersection / (union + 1e-10)).item())
    return torch.tensor(ious)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Compute per-class Dice coefficient for a single batch."""
    dices = []
    for cls in range(num_classes):
        pred_cls   = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        dices.append((2 * intersection / (pred_cls.sum() + target_cls.sum() + 1e-10)).item())
    return torch.tensor(dices)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    NUM_CLASSES = 6
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    for _ in range(5):
        pred   = torch.randint(0, NUM_CLASSES, (4, 512, 512))
        target = torch.randint(0, NUM_CLASSES, (4, 512, 512))
        metrics.update(pred, target)

    metrics.pretty_print(
        class_names=["Impervious", "Building", "Low veg", "Tree", "Car", "Background"]
    )

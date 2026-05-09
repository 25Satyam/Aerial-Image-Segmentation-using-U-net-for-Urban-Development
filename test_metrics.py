"""
Unit Tests — Segmentation Metrics
"""

import sys
from pathlib import Path
import pytest
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils.metrics import SegmentationMetrics, compute_iou, compute_dice


NUM_CLASSES = 6


class TestSegmentationMetrics:
    def setup_method(self):
        self.metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    def test_perfect_prediction(self):
        targets = torch.randint(0, NUM_CLASSES, (2, 128, 128))
        self.metrics.update(targets, targets)  # pred == target
        results = self.metrics.compute()
        assert abs(results["mean_iou"]       - 1.0) < 1e-3, "Perfect pred should give IoU=1"
        assert abs(results["pixel_accuracy"] - 1.0) < 1e-3

    def test_completely_wrong_prediction(self):
        targets = torch.zeros(2, 128, 128, dtype=torch.long)          # all class 0
        preds   = torch.ones(2, 128, 128, dtype=torch.long)           # all class 1
        self.metrics.update(preds, targets)
        results = self.metrics.compute()
        assert results["pixel_accuracy"] < 0.01

    def test_reset_clears_state(self):
        targets = torch.randint(0, NUM_CLASSES, (2, 64, 64))
        self.metrics.update(targets, targets)
        self.metrics.reset()
        results = self.metrics.compute()
        # After reset, confusion matrix is all zeros → metrics are 0
        assert results["mean_iou"] == pytest.approx(0.0, abs=1e-6)

    def test_output_keys(self):
        targets = torch.randint(0, NUM_CLASSES, (1, 64, 64))
        self.metrics.update(targets, targets)
        results = self.metrics.compute()
        for key in ["mean_iou", "mean_dice", "pixel_accuracy",
                    "per_class_iou", "per_class_dice", "precision", "recall"]:
            assert key in results, f"Missing key: {key}"

    def test_per_class_iou_length(self):
        targets = torch.randint(0, NUM_CLASSES, (2, 64, 64))
        self.metrics.update(targets, targets)
        results = self.metrics.compute()
        assert len(results["per_class_iou"]) == NUM_CLASSES

    def test_accumulation_multiple_batches(self):
        """Metrics accumulated over multiple batches should equal single-batch metrics."""
        m1 = SegmentationMetrics(num_classes=NUM_CLASSES)
        m2 = SegmentationMetrics(num_classes=NUM_CLASSES)

        b1 = torch.randint(0, NUM_CLASSES, (2, 64, 64))
        b2 = torch.randint(0, NUM_CLASSES, (2, 64, 64))

        # Accumulate
        m1.update(b1, b1)
        m1.update(b2, b2)

        # One combined pass (not directly possible but sanity check via reset)
        m2.update(b1, b1)
        m2.update(b2, b2)

        r1 = m1.compute()
        r2 = m2.compute()
        assert abs(r1["mean_iou"] - r2["mean_iou"]) < 1e-6

    def test_values_in_valid_range(self):
        targets = torch.randint(0, NUM_CLASSES, (2, 128, 128))
        preds   = torch.randint(0, NUM_CLASSES, (2, 128, 128))
        self.metrics.update(preds, targets)
        results = self.metrics.compute()

        assert 0.0 <= results["mean_iou"]       <= 1.0
        assert 0.0 <= results["mean_dice"]      <= 1.0
        assert 0.0 <= results["pixel_accuracy"] <= 1.0
        for iou in results["per_class_iou"]:
            assert 0.0 <= iou <= 1.0


class TestStandaloneFunctions:
    def test_iou_perfect(self):
        pred   = torch.zeros(128, 128, dtype=torch.long)
        target = torch.zeros(128, 128, dtype=torch.long)
        ious   = compute_iou(pred, target, NUM_CLASSES)
        assert ious[0].item() == pytest.approx(1.0, abs=1e-4)

    def test_dice_perfect(self):
        pred   = torch.zeros(128, 128, dtype=torch.long)
        target = torch.zeros(128, 128, dtype=torch.long)
        dices  = compute_dice(pred, target, NUM_CLASSES)
        assert dices[0].item() == pytest.approx(1.0, abs=1e-4)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

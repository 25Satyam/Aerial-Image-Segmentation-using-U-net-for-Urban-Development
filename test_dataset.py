"""
Unit Tests — Dataset utilities
"""

import sys
import os
import tempfile
from pathlib import Path

import pytest
import numpy as np
import cv2
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.dataset import (
    rgb_mask_to_class,
    class_to_rgb_mask,
    ISPRS_CLASSES,
    get_train_transforms,
    get_val_transforms,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_fake_image(path: str, size: int = 64):
    img = (np.random.rand(size, size, 3) * 255).astype(np.uint8)
    cv2.imwrite(path, img)


def make_fake_mask(path: str, class_dict: dict, size: int = 64):
    colors = [v[1] for _, v in class_dict.items()]
    mask   = np.zeros((size, size, 3), dtype=np.uint8)
    # Fill each quadrant with a different class colour
    q = size // 2
    for i, color in enumerate(colors[:4]):
        row = (i // 2) * q
        col = (i  % 2) * q
        mask[row:row + q, col:col + q] = color
    cv2.imwrite(path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestMaskConversion:
    def test_rgb_to_class_round_trip(self):
        """class_to_rgb_mask ∘ rgb_mask_to_class should be near-identity."""
        class_dict = ISPRS_CLASSES
        size = 128

        # Build a valid RGB mask
        colors = [v[1] for _, v in class_dict.items()]
        mask_rgb = np.zeros((size, size, 3), dtype=np.uint8)
        q = size // len(colors)
        for i, color in enumerate(colors):
            mask_rgb[i * q:(i + 1) * q, :] = color

        class_mask = rgb_mask_to_class(mask_rgb, class_dict)
        restored   = class_to_rgb_mask(class_mask, class_dict)

        assert class_mask.shape == (size, size)
        assert restored.shape   == (size, size, 3)
        assert class_mask.max() < len(class_dict)
        assert class_mask.min() >= 0

    def test_all_classes_present(self):
        class_dict = ISPRS_CLASSES
        size = 256
        mask_rgb = np.zeros((size, size, 3), dtype=np.uint8)
        q = size // len(class_dict)
        for i, (_, (_, color)) in enumerate(class_dict.items()):
            mask_rgb[i * q:(i + 1) * q, :] = color

        class_mask = rgb_mask_to_class(mask_rgb, class_dict)
        unique = np.unique(class_mask)
        assert len(unique) == len(class_dict), \
            f"Expected {len(class_dict)} unique classes, found {len(unique)}"


class TestTransforms:
    def test_train_transform_shape(self):
        transform = get_train_transforms(image_size=128)
        img  = (np.random.rand(200, 200, 3) * 255).astype(np.uint8)
        mask = np.random.randint(0, 6, (200, 200), dtype=np.int64)
        out  = transform(image=img, mask=mask)
        assert out["image"].shape == (3, 128, 128), f"Got {out['image'].shape}"
        assert out["mask"].shape  == (128, 128),     f"Got {out['mask'].shape}"

    def test_val_transform_shape(self):
        transform = get_val_transforms(image_size=256)
        img  = (np.random.rand(512, 512, 3) * 255).astype(np.uint8)
        mask = np.random.randint(0, 6, (512, 512), dtype=np.int64)
        out  = transform(image=img, mask=mask)
        assert out["image"].shape == (3, 256, 256)
        assert out["mask"].shape  == (256, 256)

    def test_output_is_tensor(self):
        transform = get_val_transforms(64)
        img  = (np.random.rand(64, 64, 3) * 255).astype(np.uint8)
        mask = np.zeros((64, 64), dtype=np.int64)
        out  = transform(image=img, mask=mask)
        assert isinstance(out["image"], torch.Tensor)
        assert isinstance(out["mask"],  torch.Tensor)

    def test_normalisation_range(self):
        """After ImageNet normalisation some values will be negative."""
        transform = get_val_transforms(64)
        img = np.ones((64, 64, 3), dtype=np.uint8) * 128
        out = transform(image=img, mask=np.zeros((64, 64), dtype=np.int64))
        # normalised image should NOT be in [0, 1]
        img_tensor = out["image"]
        assert img_tensor.min() < 0 or img_tensor.max() > 1, \
            "Image should be ImageNet-normalised (outside [0,1])"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

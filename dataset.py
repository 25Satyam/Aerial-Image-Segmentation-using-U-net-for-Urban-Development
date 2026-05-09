"""
Dataset & DataLoader utilities for Aerial Image Segmentation
=============================================================

Supports:
  - ISPRS Potsdam / Vaihingen  (6 classes)
  - INRIA Aerial Image Dataset (2 classes: building / background)
  - DeepGlobe Land Cover        (7 classes)
  - Generic folder structure:   images/ + masks/

Usage:
    python src/dataset.py --prepare --input data/raw --output data/processed --tile-size 512
"""

import os
import json
import argparse
import random
from pathlib import Path
from typing import Optional, Tuple, List, Callable

import numpy as np
import cv2
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2


# ---------------------------------------------------------------------------
# Class definitions
# ---------------------------------------------------------------------------

# ISPRS Potsdam 6-class palette
ISPRS_CLASSES = {
    0: ("Impervious surfaces", (255, 255, 255)),
    1: ("Building",           (0,   0, 255)),
    2: ("Low vegetation",     (0,   255, 255)),
    3: ("Tree",               (0,   255, 0)),
    4: ("Car",                (255, 255, 0)),
    5: ("Background/Clutter", (255, 0,   0)),
}

DEEPGLOBE_CLASSES = {
    0: ("Urban land",    (0,   255, 255)),
    1: ("Agriculture",   (255, 255, 0)),
    2: ("Rangeland",     (255, 0,   255)),
    3: ("Forest land",   (0,   255, 0)),
    4: ("Water",         (0,   0,   255)),
    5: ("Barren land",   (255, 255, 255)),
    6: ("Unknown",       (0,   0,   0)),
}

INRIA_CLASSES = {
    0: ("Background", (0,   0,   0)),
    1: ("Building",   (255, 255, 255)),
}

DATASET_CONFIGS = {
    "isprs":     {"classes": ISPRS_CLASSES,    "num_classes": 6},
    "deepglobe": {"classes": DEEPGLOBE_CLASSES, "num_classes": 7},
    "inria":     {"classes": INRIA_CLASSES,     "num_classes": 2},
}


# ---------------------------------------------------------------------------
# RGB mask → class index conversion
# ---------------------------------------------------------------------------

def rgb_mask_to_class(mask_rgb: np.ndarray, class_dict: dict) -> np.ndarray:
    """Convert an RGB segmentation mask to a 2-D class-index array."""
    h, w = mask_rgb.shape[:2]
    class_mask = np.zeros((h, w), dtype=np.int64)
    for idx, (_, color) in class_dict.items():
        match = np.all(mask_rgb == np.array(color, dtype=np.uint8), axis=-1)
        class_mask[match] = idx
    return class_mask


def class_to_rgb_mask(class_mask: np.ndarray, class_dict: dict) -> np.ndarray:
    """Convert a 2-D class-index array back to an RGB mask for visualisation."""
    h, w = class_mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, (_, color) in class_dict.items():
        rgb[class_mask == idx] = color
    return rgb


# ---------------------------------------------------------------------------
# Augmentation pipelines
# ---------------------------------------------------------------------------

def get_train_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.5, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=6, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20),
            A.CLAHE(clip_limit=4.0),
        ], p=0.5),
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.1),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_test_transforms(image_size: int = 512) -> A.Compose:
    return get_val_transforms(image_size)


# ---------------------------------------------------------------------------
# Dataset Class
# ---------------------------------------------------------------------------

class AerialSegmentationDataset(Dataset):
    """
    Generic aerial image segmentation dataset.

    Directory structure expected:
        root/
          images/  *.jpg | *.png | *.tif
          masks/   *.png               (same filename as image)

    Args:
        images_dir   : Path to images folder.
        masks_dir    : Path to masks folder.
        dataset_type : One of 'isprs', 'deepglobe', 'inria'.
        transform    : Albumentations transform pipeline.
        cache_data   : Whether to cache all images in RAM (speeds up training).
    """

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        dataset_type: str = "isprs",
        transform: Optional[Callable] = None,
        cache_data: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.dataset_type = dataset_type
        self.transform = transform
        self.cache_data = cache_data

        cfg = DATASET_CONFIGS[dataset_type]
        self.class_dict = cfg["classes"]
        self.num_classes = cfg["num_classes"]

        # Collect valid (image, mask) pairs
        valid_exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        self.image_paths = sorted([
            p for p in self.images_dir.iterdir()
            if p.suffix.lower() in valid_exts
        ])

        # Verify masks exist
        self.mask_paths = []
        for img_path in self.image_paths:
            mask_path = self.masks_dir / (img_path.stem + ".png")
            if not mask_path.exists():
                # Try same extension
                mask_path = self.masks_dir / img_path.name
            if mask_path.exists():
                self.mask_paths.append(mask_path)
            else:
                print(f"[WARNING] No mask found for {img_path.name}, skipping.")

        assert len(self.image_paths) == len(self.mask_paths), \
            "Mismatch between number of images and masks."

        # Optionally cache
        self._image_cache = {}
        self._mask_cache = {}
        if cache_data:
            print(f"Caching {len(self.image_paths)} samples...")
            for i in range(len(self.image_paths)):
                self._image_cache[i] = self._load_image(self.image_paths[i])
                self._mask_cache[i] = self._load_mask(self.mask_paths[i])
            print("Done caching.")

    def _load_image(self, path: Path) -> np.ndarray:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _load_mask(self, path: Path) -> np.ndarray:
        mask_rgb = cv2.imread(str(path), cv2.IMREAD_COLOR)
        mask_rgb = cv2.cvtColor(mask_rgb, cv2.COLOR_BGR2RGB)
        return rgb_mask_to_class(mask_rgb, self.class_dict)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        if self.cache_data and idx in self._image_cache:
            image = self._image_cache[idx]
            mask = self._mask_cache[idx]
        else:
            image = self._load_image(self.image_paths[idx])
            mask = self._load_mask(self.mask_paths[idx])

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0
            mask = torch.from_numpy(mask).long()

        return image, mask

    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse-frequency class weights for weighted loss."""
        counts = torch.zeros(self.num_classes)
        for idx in range(len(self)):
            _, mask = self[idx]
            for c in range(self.num_classes):
                counts[c] += (mask == c).sum().item()
        total = counts.sum()
        weights = total / (self.num_classes * counts.clamp(min=1))
        return weights / weights.sum() * self.num_classes


# ---------------------------------------------------------------------------
# Tile generation (preprocessing large aerial images)
# ---------------------------------------------------------------------------

def tile_image_mask(
    image_path: str,
    mask_path: str,
    output_images_dir: str,
    output_masks_dir: str,
    tile_size: int = 512,
    overlap: int = 64,
    min_valid_ratio: float = 0.1,
) -> int:
    """
    Slice a large aerial image and its corresponding mask into smaller tiles.

    Args:
        image_path        : Path to source image.
        mask_path         : Path to source mask.
        output_images_dir : Where to save image tiles.
        output_masks_dir  : Where to save mask tiles.
        tile_size         : Tile dimensions (square).
        overlap           : Overlap between adjacent tiles in pixels.
        min_valid_ratio   : Discard tiles where valid (non-black) pixels < this ratio.

    Returns:
        Number of tiles saved.
    """
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_masks_dir, exist_ok=True)

    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)

    assert image is not None, f"Cannot read image: {image_path}"
    assert mask is not None, f"Cannot read mask:  {mask_path}"
    assert image.shape[:2] == mask.shape[:2], "Image and mask size mismatch."

    h, w = image.shape[:2]
    stem = Path(image_path).stem
    stride = tile_size - overlap
    count = 0

    for y in range(0, h - tile_size + 1, stride):
        for x in range(0, w - tile_size + 1, stride):
            img_tile  = image[y:y + tile_size, x:x + tile_size]
            mask_tile = mask[y:y + tile_size, x:x + tile_size]

            # Skip tiles that are mostly background / no-data
            valid_ratio = np.mean(img_tile > 0)
            if valid_ratio < min_valid_ratio:
                continue

            fname = f"{stem}_y{y:05d}_x{x:05d}.png"
            cv2.imwrite(os.path.join(output_images_dir, fname), img_tile)
            cv2.imwrite(os.path.join(output_masks_dir, fname), mask_tile)
            count += 1

    return count


def prepare_dataset(
    input_dir: str,
    output_dir: str,
    tile_size: int = 512,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    seed: int = 42,
):
    """
    Full preprocessing pipeline:
      1. Tile all images and masks.
      2. Split into train / val / test sets.
      3. Save a split manifest (JSON).
    """
    random.seed(seed)

    images_in = Path(input_dir) / "images"
    masks_in  = Path(input_dir) / "masks"

    images_out = Path(output_dir) / "images"
    masks_out  = Path(output_dir) / "masks"

    # Step 1: Tile
    image_files = sorted(images_in.glob("*.*"))
    total_tiles = 0
    for img_path in image_files:
        mask_path = masks_in / (img_path.stem + ".png")
        if not mask_path.exists():
            mask_path = masks_in / img_path.name
        if not mask_path.exists():
            print(f"[SKIP] No mask for {img_path.name}")
            continue

        n = tile_image_mask(
            str(img_path), str(mask_path),
            str(images_out), str(masks_out),
            tile_size=tile_size,
        )
        total_tiles += n
        print(f"  {img_path.name}: {n} tiles")

    print(f"\nTotal tiles: {total_tiles}")

    # Step 2: Split
    all_tiles = [p.name for p in sorted(images_out.glob("*.png"))]
    random.shuffle(all_tiles)

    n_test  = int(len(all_tiles) * test_ratio)
    n_val   = int(len(all_tiles) * val_ratio)
    n_train = len(all_tiles) - n_val - n_test

    splits = {
        "train": all_tiles[:n_train],
        "val":   all_tiles[n_train:n_train + n_val],
        "test":  all_tiles[n_train + n_val:],
    }

    manifest_path = Path(output_dir) / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(splits, f, indent=2)

    for split, names in splits.items():
        print(f"  {split}: {len(names)} tiles")

    print(f"\nManifest saved to {manifest_path}")


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------

def build_dataloaders(
    images_dir: str,
    masks_dir: str,
    dataset_type: str = "isprs",
    image_size: int = 512,
    batch_size: int = 8,
    num_workers: int = 4,
    val_ratio: float = 0.15,
    test_ratio: float = 0.10,
    cache_data: bool = False,
    seed: int = 42,
    manifest_path: Optional[str] = None,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train / val / test DataLoaders.

    If manifest_path is provided, use pre-defined file lists.
    Otherwise, randomly split the full dataset.
    """
    if manifest_path and Path(manifest_path).exists():
        with open(manifest_path) as f:
            splits = json.load(f)

        def subset_dataset(split_name, transform):
            fnames = splits[split_name]
            # Build a dataset then subset by indices
            full = AerialSegmentationDataset(
                images_dir, masks_dir, dataset_type, transform=transform, cache_data=False
            )
            name_to_idx = {full.image_paths[i].name: i for i in range(len(full))}
            indices = [name_to_idx[fn] for fn in fnames if fn in name_to_idx]
            return torch.utils.data.Subset(full, indices)

        train_ds = subset_dataset("train", get_train_transforms(image_size))
        val_ds   = subset_dataset("val",   get_val_transforms(image_size))
        test_ds  = subset_dataset("test",  get_test_transforms(image_size))
    else:
        full_ds = AerialSegmentationDataset(
            images_dir, masks_dir, dataset_type,
            transform=None, cache_data=cache_data
        )
        n = len(full_ds)
        n_test  = int(n * test_ratio)
        n_val   = int(n * val_ratio)
        n_train = n - n_val - n_test

        generator = torch.Generator().manual_seed(seed)
        train_ds, val_ds, test_ds = random_split(
            full_ds, [n_train, n_val, n_test], generator=generator
        )
        # Attach transforms per split
        train_ds.dataset.transform = get_train_transforms(image_size)
        val_ds.dataset.transform   = get_val_transforms(image_size)
        test_ds.dataset.transform  = get_test_transforms(image_size)

    loader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  **loader_kwargs)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# CLI for preprocessing
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare aerial image dataset")
    parser.add_argument("--prepare", action="store_true", help="Run preprocessing pipeline")
    parser.add_argument("--input",  type=str, default="data/raw",       help="Raw data directory")
    parser.add_argument("--output", type=str, default="data/processed", help="Output directory")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--val-ratio",  type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.prepare:
        print(f"Preparing dataset: {args.input} → {args.output}")
        prepare_dataset(
            input_dir=args.input,
            output_dir=args.output,
            tile_size=args.tile_size,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )
    else:
        parser.print_help()

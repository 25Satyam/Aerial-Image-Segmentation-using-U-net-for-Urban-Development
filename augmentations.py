"""
Augmentation Pipelines
========================
All transforms use Albumentations for consistency and speed.
Masks are automatically transformed in sync with images.
"""

import albumentations as A
from albumentations.pytorch import ToTensorV2


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_light_train_transforms(image_size: int = 512) -> A.Compose:
    """Minimal augmentation — for small or well-annotated datasets."""
    return A.Compose([
        A.RandomCrop(height=image_size, width=image_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_heavy_train_transforms(image_size: int = 512) -> A.Compose:
    """Aggressive augmentation — for limited data / better generalisation."""
    return A.Compose([
        A.RandomResizedCrop(height=image_size, width=image_size, scale=(0.4, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=0.4),
        A.OneOf([
            A.ElasticTransform(alpha=80, sigma=8, p=0.5),
            A.GridDistortion(num_steps=5, distort_limit=0.3, p=0.5),
            A.OpticalDistortion(distort_limit=0.5, shift_limit=0.3, p=0.5),
        ], p=0.35),
        A.OneOf([
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
            A.HueSaturationValue(hue_shift_limit=15, sat_shift_limit=40, val_shift_limit=30),
            A.CLAHE(clip_limit=6.0, tile_grid_size=(8, 8)),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        ], p=0.6),
        A.OneOf([
            A.GaussNoise(var_limit=(20.0, 80.0)),
            A.ISONoise(),
            A.MultiplicativeNoise(),
        ], p=0.2),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 7)),
            A.MedianBlur(blur_limit=5),
            A.MotionBlur(blur_limit=7),
        ], p=0.1),
        A.CoarseDropout(max_holes=8, max_height=image_size // 16,
                        max_width=image_size // 16, fill_value=0, p=0.2),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_tta_transforms(image_size: int = 512):
    """
    Test-Time Augmentation transforms.
    Returns a list of transforms; run inference with each and average.
    """
    return [
        get_val_transforms(image_size),                                 # Original
        A.Compose([A.HorizontalFlip(p=1.0), *get_val_transforms(image_size).transforms]),
        A.Compose([A.VerticalFlip(p=1.0),   *get_val_transforms(image_size).transforms]),
        A.Compose([A.Rotate(limit=(90, 90), p=1.0), *get_val_transforms(image_size).transforms]),
    ]

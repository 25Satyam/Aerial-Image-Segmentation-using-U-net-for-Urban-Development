"""
Inference Script — Predict segmentation masks on new aerial images.

Usage:
    # Single image
    python src/predict.py --image data/samples/aerial_001.jpg \\
                          --weights results/checkpoints/run_xxx/best_model.pth \\
                          --config  configs/config.yaml \\
                          --output  results/predictions/

    # Batch inference
    python src/predict.py --input-dir data/samples/ \\
                          --weights results/checkpoints/run_xxx/best_model.pth \\
                          --config  configs/config.yaml \\
                          --output  results/predictions/
"""

import sys
import argparse
import os
from pathlib import Path

import yaml
import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model
from src.dataset import DATASET_CONFIGS, class_to_rgb_mask


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def get_inference_transform(image_size: int = 512) -> A.Compose:
    return A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def load_model(config: dict, weights_path: str, device: torch.device) -> torch.nn.Module:
    model = build_model(config).to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def predict_single(
    image_path: str,
    model: torch.nn.Module,
    transform: A.Compose,
    device: torch.device,
    original_size: bool = True,
) -> np.ndarray:
    """
    Run inference on a single image.

    Returns:
        pred_mask : (H, W) np.uint8 array of class indices.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = image.shape[:2]

    augmented = transform(image=image)
    tensor    = augmented["image"].unsqueeze(0).to(device)   # (1, 3, H, W)

    with torch.no_grad(), torch.cuda.amp.autocast():
        logits = model(tensor)                                # (1, C, H, W)
        pred   = logits.argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    if original_size:
        pred = cv2.resize(pred, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)

    return pred


def overlay_prediction(
    image_path: str,
    pred_mask: np.ndarray,
    class_dict: dict,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Blend the RGB segmentation mask with the original image.

    Args:
        alpha : Transparency of the mask overlay (0=transparent, 1=opaque).

    Returns:
        overlay : (H, W, 3) np.uint8 blended image.
    """
    image    = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image    = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask_rgb = class_to_rgb_mask(pred_mask, class_dict)
    mask_rgb = cv2.resize(mask_rgb, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)
    overlay  = (alpha * mask_rgb + (1 - alpha) * image).astype(np.uint8)
    return overlay


def save_outputs(
    image_path: str,
    pred_mask: np.ndarray,
    class_dict: dict,
    output_dir: str,
    save_overlay: bool = True,
):
    os.makedirs(output_dir, exist_ok=True)
    stem = Path(image_path).stem

    # Save raw class-index mask (single-channel PNG)
    mask_path = Path(output_dir) / f"{stem}_pred_mask.png"
    cv2.imwrite(str(mask_path), pred_mask)

    # Save RGB coloured mask
    rgb_mask = class_to_rgb_mask(pred_mask, class_dict)
    rgb_path = Path(output_dir) / f"{stem}_pred_rgb.png"
    cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb_mask, cv2.COLOR_RGB2BGR))

    # Save overlay
    if save_overlay:
        overlay = overlay_prediction(image_path, pred_mask, class_dict)
        ov_path = Path(output_dir) / f"{stem}_overlay.png"
        cv2.imwrite(str(ov_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    return str(mask_path)


# ---------------------------------------------------------------------------
# Sliding window inference for large images
# ---------------------------------------------------------------------------

def predict_sliding_window(
    image_path: str,
    model: torch.nn.Module,
    device: torch.device,
    tile_size: int = 512,
    overlap: int = 64,
    num_classes: int = 6,
) -> np.ndarray:
    """
    Inference on large image via overlapping tiles with soft voting.
    Avoids border artefacts at tile boundaries.
    """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w  = image.shape[:2]

    score_map = np.zeros((num_classes, h, w), dtype=np.float32)
    count_map = np.zeros((h, w),              dtype=np.float32)

    norm_fn = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])

    stride = tile_size - overlap

    for y in range(0, h, stride):
        for x in range(0, w, stride):
            y1, x1 = y, x
            y2, x2 = min(y + tile_size, h), min(x + tile_size, w)

            tile = image[y1:y2, x1:x2]
            # Pad if smaller than tile_size
            tile_h, tile_w = tile.shape[:2]
            if tile_h < tile_size or tile_w < tile_size:
                tile = cv2.copyMakeBorder(
                    tile, 0, tile_size - tile_h, 0, tile_size - tile_w,
                    cv2.BORDER_REFLECT
                )

            tensor = norm_fn(image=tile)["image"].unsqueeze(0).to(device)

            with torch.no_grad(), torch.cuda.amp.autocast():
                logits = model(tensor)                         # (1, C, ts, ts)
                probs  = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()

            score_map[:, y1:y2, x1:x2] += probs[:, :y2 - y1, :x2 - x1]
            count_map[y1:y2, x1:x2]    += 1

    count_map = np.maximum(count_map, 1)
    score_map /= count_map[np.newaxis, :, :]
    pred_mask  = score_map.argmax(axis=0).astype(np.uint8)
    return pred_mask


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Aerial segmentation inference")
    parser.add_argument("--config",      type=str, required=True)
    parser.add_argument("--weights",     type=str, required=True)
    parser.add_argument("--output",      type=str, default="results/predictions/")
    parser.add_argument("--image",       type=str, default=None, help="Single image path")
    parser.add_argument("--input-dir",   type=str, default=None, help="Directory of images")
    parser.add_argument("--image-size",  type=int, default=512)
    parser.add_argument("--sliding-window", action="store_true",
                        help="Use sliding-window inference for large images")
    parser.add_argument("--no-overlay",  action="store_true", help="Skip saving overlay images")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model        = load_model(cfg, args.weights, device)
    transform    = get_inference_transform(args.image_size)
    dataset_type = cfg.get("dataset_type", "isprs")
    class_dict   = DATASET_CONFIGS[dataset_type]["classes"]
    num_classes  = cfg.get("num_classes", 6)

    print(f"Model loaded. Device: {device}")

    # Collect image paths
    if args.image:
        image_paths = [args.image]
    elif args.input_dir:
        valid_exts  = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        image_paths = [
            str(p) for p in sorted(Path(args.input_dir).iterdir())
            if p.suffix.lower() in valid_exts
        ]
    else:
        raise ValueError("Provide --image or --input-dir")

    print(f"Processing {len(image_paths)} image(s)...")

    for img_path in tqdm(image_paths):
        if args.sliding_window:
            pred = predict_sliding_window(img_path, model, device, num_classes=num_classes)
        else:
            pred = predict_single(img_path, model, transform, device)

        save_outputs(
            img_path, pred, class_dict, args.output,
            save_overlay=not args.no_overlay
        )

    print(f"\nPredictions saved to: {args.output}")

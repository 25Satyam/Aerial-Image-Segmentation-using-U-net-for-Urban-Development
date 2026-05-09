"""
Evaluation Script
==================
Runs a trained U-Net on the test split and reports segmentation metrics.

Usage:
    python src/evaluate.py --config configs/config.yaml \\
                           --weights results/checkpoints/run_xxx/best_model.pth \\
                           --split test
"""

import sys
import argparse
from pathlib import Path

import yaml
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model
from src.dataset import build_dataloaders, DATASET_CONFIGS
from src.losses import build_criterion
from src.utils.metrics import SegmentationMetrics


def evaluate(config: dict, weights_path: str, split: str = "test"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # DataLoaders
    train_loader, val_loader, test_loader = build_dataloaders(
        images_dir   = config["images_dir"],
        masks_dir    = config["masks_dir"],
        dataset_type = config.get("dataset_type", "isprs"),
        image_size   = config.get("image_size", 512),
        batch_size   = config.get("batch_size", 8),
        num_workers  = config.get("num_workers", 4),
        manifest_path= config.get("manifest_path", None),
    )
    loader_map = {"train": train_loader, "val": val_loader, "test": test_loader}
    loader = loader_map[split]

    # Model
    model = build_model(config).to(device)
    ckpt  = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded weights from {weights_path}")

    # Criterion
    criterion = build_criterion(config).to(device)

    # Metrics
    num_classes = config.get("num_classes", 6)
    dataset_type = config.get("dataset_type", "isprs")
    class_names = [v[0] for _, v in DATASET_CONFIGS[dataset_type]["classes"].items()]
    metrics_fn = SegmentationMetrics(num_classes=num_classes, device=device)

    total_loss = 0.0

    with torch.no_grad():
        for images, masks in tqdm(loader, desc=f"Evaluating [{split}]"):
            images = images.to(device, non_blocking=True)
            masks  = masks.to(device, non_blocking=True)

            with torch.cuda.amp.autocast():
                logits = model(images)
                loss   = criterion(logits, masks)

            preds = logits.argmax(dim=1)
            metrics_fn.update(preds, masks)
            total_loss += loss.item()

    avg_loss = total_loss / len(loader)
    print(f"\nAverage Loss ({split}): {avg_loss:.4f}")
    metrics_fn.pretty_print(class_names)

    return metrics_fn.compute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",  type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--split",   type=str, default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    evaluate(cfg, args.weights, args.split)

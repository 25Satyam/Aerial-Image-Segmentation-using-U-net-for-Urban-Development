"""
Training Script — Aerial Image Segmentation with U-Net
=======================================================

Usage:
    python src/train.py --config configs/config.yaml
    python src/train.py --config configs/config.yaml --resume results/checkpoints/best_model.pth
"""

import os
import sys
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models.unet import build_model
from src.dataset import build_dataloaders
from src.losses import build_criterion
from src.utils.metrics import SegmentationMetrics


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

def setup_logger(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    log_file = Path(log_dir) / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)

    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger


# ---------------------------------------------------------------------------
# Training / Validation step
# ---------------------------------------------------------------------------

def train_epoch(model, loader, criterion, optimizer, scheduler, device, scaler, logger, epoch):
    model.train()
    total_loss = 0.0
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"[Train] Epoch {epoch}", leave=False, dynamic_ncols=True)

    for batch_idx, (images, masks) in enumerate(pbar):
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            logits = model(images)
            loss   = criterion(logits, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if isinstance(scheduler, OneCycleLR):
            scheduler.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{optimizer.param_groups[0]['lr']:.2e}")

    return total_loss / num_batches


@torch.no_grad()
def validate_epoch(model, loader, criterion, metrics_fn, device, epoch):
    model.eval()
    total_loss = 0.0
    metrics_fn.reset()
    num_batches = len(loader)

    pbar = tqdm(loader, desc=f"[Val]   Epoch {epoch}", leave=False, dynamic_ncols=True)

    for images, masks in pbar:
        images = images.to(device, non_blocking=True)
        masks  = masks.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            logits = model(images)
            loss   = criterion(logits, masks)

        preds = logits.argmax(dim=1)
        metrics_fn.update(preds, masks)

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / num_batches
    results  = metrics_fn.compute()
    return avg_loss, results


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(state: dict, path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path: str, model, optimizer=None, scheduler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    if scheduler and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
    return ckpt.get("epoch", 0), ckpt.get("best_miou", 0.0)


# ---------------------------------------------------------------------------
# Optimizer / Scheduler factory
# ---------------------------------------------------------------------------

def build_optimizer(config: dict, model_params):
    opt_type = config.get("optimizer", "adamw").lower()
    lr       = config.get("lr", 1e-4)
    wd       = config.get("weight_decay", 1e-4)

    if opt_type == "adam":
        return Adam(model_params, lr=lr, weight_decay=wd)
    elif opt_type == "adamw":
        return AdamW(model_params, lr=lr, weight_decay=wd)
    elif opt_type == "sgd":
        return SGD(model_params, lr=lr, weight_decay=wd, momentum=0.9, nesterov=True)
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")


def build_scheduler(config: dict, optimizer, num_train_steps: int = None):
    sched_type = config.get("scheduler", "cosine").lower()
    epochs     = config.get("epochs", 100)

    if sched_type == "cosine":
        return CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    elif sched_type == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5, verbose=True)
    elif sched_type == "onecycle":
        assert num_train_steps, "Need num_train_steps for OneCycleLR"
        return OneCycleLR(optimizer, max_lr=config.get("lr", 1e-4) * 10,
                          total_steps=num_train_steps * epochs)
    elif sched_type == "cosine_restart":
        return CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
    else:
        raise ValueError(f"Unknown scheduler: {sched_type}")


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(config: dict, resume_path: str = None):
    # ---- Setup ----
    run_name = config.get("run_name", f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    ckpt_dir = Path(config.get("checkpoint_dir", "results/checkpoints")) / run_name
    log_dir  = Path(config.get("log_dir", "results/logs")) / run_name
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir,  exist_ok=True)

    logger  = setup_logger(str(log_dir))
    writer  = SummaryWriter(log_dir=str(log_dir))

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    logger.info(f"Run: {run_name}")

    # Save config snapshot
    with open(ckpt_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # ---- Data ----
    train_loader, val_loader, _ = build_dataloaders(
        images_dir   = config["images_dir"],
        masks_dir    = config["masks_dir"],
        dataset_type = config.get("dataset_type", "isprs"),
        image_size   = config.get("image_size", 512),
        batch_size   = config.get("batch_size", 8),
        num_workers  = config.get("num_workers", 4),
        val_ratio    = config.get("val_ratio", 0.15),
        test_ratio   = config.get("test_ratio", 0.10),
        cache_data   = config.get("cache_data", False),
        manifest_path= config.get("manifest_path", None),
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ---- Model ----
    model = build_model(config).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {config.get('architecture', 'unet')} | Trainable params: {params:,}")

    # ---- Loss ----
    criterion = build_criterion(config).to(device)

    # ---- Optimizer & Scheduler ----
    optimizer = build_optimizer(config, model.parameters())
    scheduler = build_scheduler(config, optimizer, num_train_steps=len(train_loader))

    # ---- Mixed Precision ----
    use_amp = config.get("mixed_precision", True) and torch.cuda.is_available()
    scaler  = torch.cuda.amp.GradScaler() if use_amp else None
    logger.info(f"Mixed precision: {use_amp}")

    # ---- Metrics ----
    num_classes = config.get("num_classes", 6)
    metrics_fn  = SegmentationMetrics(num_classes=num_classes, device=device)

    # ---- Resume ----
    start_epoch = 0
    best_miou   = 0.0
    if resume_path and Path(resume_path).exists():
        start_epoch, best_miou = load_checkpoint(resume_path, model, optimizer, scheduler)
        logger.info(f"Resumed from {resume_path} at epoch {start_epoch}, best_miou={best_miou:.4f}")

    # ---- Training loop ----
    epochs       = config.get("epochs", 100)
    patience     = config.get("patience", 20)
    no_improve   = 0

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(start_epoch + 1, epochs + 1):
        t0 = time.time()

        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, scheduler,
            device, scaler, logger, epoch
        )
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, metrics_fn, device, epoch
        )

        miou        = val_metrics["mean_iou"]
        dice        = val_metrics["mean_dice"]
        pixel_acc   = val_metrics["pixel_accuracy"]

        # Scheduler step (non-OneCycleLR)
        if isinstance(scheduler, ReduceLROnPlateau):
            scheduler.step(miou)
        elif not isinstance(scheduler, OneCycleLR):
            scheduler.step()

        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:04d}/{epochs} | "
            f"TrainLoss={train_loss:.4f} | ValLoss={val_loss:.4f} | "
            f"mIoU={miou:.4f} | Dice={dice:.4f} | PixAcc={pixel_acc:.4f} | "
            f"LR={optimizer.param_groups[0]['lr']:.2e} | {elapsed:.1f}s"
        )

        # TensorBoard
        writer.add_scalars("Loss",         {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("Metrics/Val",  {"mIoU": miou, "Dice": dice, "PixelAcc": pixel_acc}, epoch)
        writer.add_scalar ("LR",           optimizer.param_groups[0]["lr"], epoch)

        # Per-class IoU
        for cls_idx, iou in enumerate(val_metrics["per_class_iou"]):
            writer.add_scalar(f"IoU/class_{cls_idx}", iou, epoch)

        # Checkpoint
        state = {
            "epoch":           epoch,
            "model_state":     model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else {},
            "best_miou":       best_miou,
            "config":          config,
        }

        save_checkpoint(state, str(ckpt_dir / "last_model.pth"))

        if miou > best_miou:
            best_miou = miou
            no_improve = 0
            save_checkpoint(state, str(ckpt_dir / "best_model.pth"))
            logger.info(f"  ✓ New best mIoU: {best_miou:.4f} — checkpoint saved.")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping triggered after {patience} epochs without improvement.")
                break

    logger.info(f"Training complete. Best mIoU: {best_miou:.4f}")
    logger.info(f"Best checkpoint: {ckpt_dir / 'best_model.pth'}")
    writer.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train U-Net for aerial segmentation")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    train(cfg, resume_path=args.resume)

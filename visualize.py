"""
Visualization Utilities
========================
Functions for plotting segmentation results, training curves,
and confusion matrices.
"""

import os
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
import cv2


# ---------------------------------------------------------------------------
# Color palettes
# ---------------------------------------------------------------------------

ISPRS_PALETTE = {
    0: ("Impervious surfaces", np.array([255, 255, 255]) / 255),
    1: ("Building",            np.array([0,   0,   255]) / 255),
    2: ("Low vegetation",      np.array([0,   255, 255]) / 255),
    3: ("Tree",                np.array([0,   255,   0]) / 255),
    4: ("Car",                 np.array([255, 255,   0]) / 255),
    5: ("Background",          np.array([255,   0,   0]) / 255),
}


def denormalize(tensor: torch.Tensor, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)) -> np.ndarray:
    """Convert a normalised image tensor to a displayable numpy array."""
    t = tensor.clone().cpu().float()
    for c, (m, s) in enumerate(zip(mean, std)):
        t[c] = t[c] * s + m
    return t.permute(1, 2, 0).numpy().clip(0, 1)


def mask_to_rgb(mask: np.ndarray, palette: dict) -> np.ndarray:
    """Convert a class-index mask to an RGB image using the palette."""
    h, w   = mask.shape
    rgb    = np.zeros((h, w, 3), dtype=np.float32)
    for cls_idx, (_, color) in palette.items():
        rgb[mask == cls_idx] = color
    return rgb


# ---------------------------------------------------------------------------
# Sample visualisation
# ---------------------------------------------------------------------------

def plot_sample(
    image: torch.Tensor,
    mask: torch.Tensor,
    pred: Optional[torch.Tensor] = None,
    palette: dict = ISPRS_PALETTE,
    title: str = "",
    save_path: Optional[str] = None,
):
    """
    Plot image | ground-truth mask | predicted mask in a single row.

    Args:
        image : (3, H, W) normalised tensor.
        mask  : (H, W) ground-truth class-index tensor.
        pred  : (H, W) predicted class-index tensor (optional).
        palette: class → (name, RGB) dict.
    """
    img_np  = denormalize(image)
    mask_np = mask.cpu().numpy()

    n_cols = 3 if pred is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 5))

    axes[0].imshow(img_np)
    axes[0].set_title("Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(mask_to_rgb(mask_np, palette))
    axes[1].set_title("Ground Truth", fontsize=12)
    axes[1].axis("off")

    if pred is not None:
        pred_np = pred.cpu().numpy()
        axes[2].imshow(mask_to_rgb(pred_np, palette))
        axes[2].set_title("Prediction", fontsize=12)
        axes[2].axis("off")

    # Legend
    patches = [
        mpatches.Patch(color=color, label=name)
        for _, (name, color) in palette.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.02))

    if title:
        fig.suptitle(title, fontsize=14, fontweight="bold")

    plt.tight_layout()
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Batch visualisation
# ---------------------------------------------------------------------------

def plot_batch(
    images: torch.Tensor,
    masks: torch.Tensor,
    preds: Optional[torch.Tensor] = None,
    palette: dict = ISPRS_PALETTE,
    n_samples: int = 4,
    save_path: Optional[str] = None,
):
    """Plot up to n_samples examples from a batch."""
    n = min(n_samples, images.size(0))
    n_cols = 3 if preds is not None else 2
    fig, axes = plt.subplots(n, n_cols, figsize=(6 * n_cols, 5 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    col_titles = ["Image", "Ground Truth"] + (["Prediction"] if preds is not None else [])
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=13, fontweight="bold")

    for row in range(n):
        axes[row, 0].imshow(denormalize(images[row]))
        axes[row, 0].axis("off")

        axes[row, 1].imshow(mask_to_rgb(masks[row].cpu().numpy(), palette))
        axes[row, 1].axis("off")

        if preds is not None:
            axes[row, 2].imshow(mask_to_rgb(preds[row].cpu().numpy(), palette))
            axes[row, 2].axis("off")

    patches = [
        mpatches.Patch(color=color, label=name)
        for _, (name, color) in palette.items()
    ]
    fig.legend(handles=patches, loc="lower center", ncol=len(patches),
               fontsize=9, frameon=True, bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout()
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    miou_scores: List[float],
    save_path: Optional[str] = None,
):
    """Plot training / validation loss and mIoU curves."""
    epochs = range(1, len(train_losses) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, train_losses, label="Train Loss", color="#E74C3C", linewidth=2)
    ax1.plot(epochs, val_losses,   label="Val Loss",   color="#3498DB", linewidth=2)
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss"); ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, miou_scores, label="Val mIoU", color="#2ECC71", linewidth=2)
    best_epoch = np.argmax(miou_scores) + 1
    best_miou  = max(miou_scores)
    ax2.axvline(x=best_epoch, color="orange", linestyle="--", alpha=0.8, label=f"Best epoch {best_epoch}")
    ax2.scatter([best_epoch], [best_miou], color="orange", s=80, zorder=5)
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("mIoU")
    ax2.set_title(f"Validation mIoU (best={best_miou:.4f})"); ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    normalize: bool = True,
    save_path: Optional[str] = None,
):
    """Plot a confusion matrix heatmap."""
    if normalize:
        cm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-10)
        fmt = ".2f"; vmax = 1.0
    else:
        fmt = "d"; vmax = None

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, vmin=0, vmax=vmax, ax=ax,
    )
    ax.set_ylabel("True Label",      fontsize=12)
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_title("Confusion Matrix (normalised)" if normalize else "Confusion Matrix", fontsize=14)
    plt.xticks(rotation=30, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()


# ---------------------------------------------------------------------------
# Per-class IoU bar chart
# ---------------------------------------------------------------------------

def plot_class_iou(
    iou_scores: List[float],
    class_names: List[str],
    palette: dict = ISPRS_PALETTE,
    save_path: Optional[str] = None,
):
    colors = [palette.get(i, (None, "#95a5a6"))[1] for i in range(len(class_names))]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(class_names, iou_scores, color=colors, edgecolor="black", linewidth=0.5)

    for bar, score in zip(bars, iou_scores):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01, f"{score:.3f}",
                ha="center", va="bottom", fontsize=9)

    mean_iou = np.mean(iou_scores)
    ax.axhline(mean_iou, color="red", linestyle="--", linewidth=1.5,
               label=f"mIoU = {mean_iou:.4f}")

    ax.set_ylim(0, 1.1)
    ax.set_ylabel("IoU Score", fontsize=12)
    ax.set_title("Per-class IoU", fontsize=14)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()

    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.show()
    plt.close()

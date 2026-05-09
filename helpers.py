"""
General Utility Functions
"""

import os
import random
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42):
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False
    os.environ["PYTHONHASHSEED"] = str(seed)


def load_config(path: str) -> Dict:
    """Load YAML config file."""
    with open(path) as f:
        return yaml.safe_load(f)


def save_config(config: Dict, path: str):
    """Save config dict to YAML."""
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def count_parameters(model: torch.nn.Module) -> Dict[str, int]:
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def model_size_mb(model: torch.nn.Module) -> float:
    """Estimate model size in megabytes."""
    param_bytes = sum(p.nelement() * p.element_size() for p in model.parameters())
    buf_bytes   = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_bytes + buf_bytes) / (1024 ** 2)


def get_device(prefer_gpu: bool = True) -> torch.device:
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


def save_json(data: Any, path: str):
    os.makedirs(Path(path).parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: str) -> Any:
    with open(path) as f:
        return json.load(f)


def format_metrics(metrics: Dict) -> str:
    """Format metrics dict as a readable string."""
    lines = []
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<20s}: {v:.4f}")
        elif isinstance(v, list):
            vals = ", ".join(f"{x:.4f}" for x in v)
            lines.append(f"  {k:<20s}: [{vals}]")
    return "\n".join(lines)


class EarlyStopping:
    """Stop training when a monitored metric stops improving."""

    def __init__(self, patience: int = 15, mode: str = "max", min_delta: float = 1e-4):
        self.patience  = patience
        self.mode      = mode
        self.min_delta = min_delta
        self.counter   = 0
        self.best      = None
        self.stop      = False

    def __call__(self, value: float) -> bool:
        if self.best is None:
            self.best = value
            return False

        improved = (value > self.best + self.min_delta) if self.mode == "max" \
                   else (value < self.best - self.min_delta)

        if improved:
            self.best    = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True

        return self.stop

"""Utility functions for reproducibility, config loading, and metrics."""

import random

import numpy as np
import torch
import yaml


def set_seed(seed: int, deterministic: bool = True):
    """Set random seeds for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)
    else:
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)


def resolve_device(requested: str | None = "auto") -> str:
    """Resolve requested device string into an available runtime device."""
    if requested is None:
        requested = "auto"

    requested_device = str(requested).lower()
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"

    if requested_device.startswith("cuda"):
        return "cuda" if torch.cuda.is_available() else "cpu"

    return "cpu"


def count_trainable_parameters(model) -> int:
    """Count trainable parameters for a model."""
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def exact_solution(x: torch.Tensor, t: torch.Tensor, alpha: float) -> torch.Tensor:
    """Exact solution: u(x,t) = t^alpha * sin(pi*x)"""
    t_safe = torch.where(t > 0, t, torch.ones_like(t) * 1e-15)
    result = (t_safe ** alpha) * torch.sin(np.pi * x)
    result = torch.where(t > 0, result, torch.zeros_like(result))
    return result


def compute_errors(model, x, t, alpha, device="cpu"):
    """Compute L2 relative and L-infinity errors."""
    model.eval()
    with torch.no_grad():
        u_pred = model(x, t).squeeze()
        u_exact = exact_solution(x, t, alpha)
        
        mask = t > 1e-10
        u_pred_masked = u_pred[mask]
        u_exact_masked = u_exact[mask]
        
        diff = u_pred_masked - u_exact_masked
        l2_error = torch.norm(diff) / torch.norm(u_exact_masked)
        linf_error = torch.max(torch.abs(diff))
        
    return l2_error.item(), linf_error.item()


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_device(config: dict) -> str:
    """Get compute device."""
    requested = config.get("device", "auto")
    return resolve_device(requested)

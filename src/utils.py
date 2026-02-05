"""Utility Functions"""

import torch
import numpy as np
import yaml
from pathlib import Path


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


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
    requested = config.get('device', 'cuda')
    if requested == 'cuda' and torch.cuda.is_available():
        return 'cuda'
    return 'cpu'

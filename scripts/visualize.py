#!/usr/bin/env python3
"""Visualization script for trained PINN checkpoints."""

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PINN


def exact_solution(x, t, alpha, problem_type):
    """Return exact solution for supported problems."""
    if problem_type == "integro_differential":
        return (t ** alpha) * torch.cos(np.pi * x)
    return (t ** alpha) * torch.sin(np.pi * x)


def resolve_default_checkpoint():
    """Pick the most relevant checkpoint path for the current branch."""
    candidates = [
        Path("outputs/checkpoints_integro_diff/final_model.pt"),
        Path("outputs/checkpoints/final_model.pt"),
    ]
    for path in candidates:
        if path.exists():
            return path
    return candidates[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate solution visualizations from a checkpoint")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path")
    parser.add_argument("--config", type=str, default="configs/integro_differential.yaml", help="Config path")
    parser.add_argument("--output-dir", type=str, default="outputs/figures", help="Output directory")
    parser.add_argument("--n-grid", type=int, default=100, help="Grid size per dimension")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Device")
    return parser.parse_args()


def main():
    args = parse_args()

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else resolve_default_checkpoint()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")

    problem = config.get("problem", {})
    problem_type = problem.get("type", "fractional")
    alpha = problem.get("alpha", 0.5)
    x_min = problem.get("x_min", 0.0)
    x_max = problem.get("x_max", 1.0)
    t_max = problem.get("t_max", 1.0)

    net_cfg = config.get("network", config.get("model", {}))
    model = PINN(
        input_dim=net_cfg.get("input_dim", 2),
        output_dim=net_cfg.get("output_dim", 1),
        hidden_layers=net_cfg.get("hidden_layers", [64, 64, 64, 64]),
        activation=net_cfg.get("activation", "tanh"),
        device=device,
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Using device: {device}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")

    N = args.n_grid
    x = torch.linspace(x_min, x_max, N, dtype=torch.float64, device=device)
    t = torch.linspace(0.0, t_max, N, dtype=torch.float64, device=device)
    X, T = torch.meshgrid(x, t, indexing="ij")

    with torch.no_grad():
        u_pred = model(X.flatten(), T.flatten()).reshape(N, N).cpu().numpy()

    u_exact = exact_solution(X, T, alpha, problem_type).cpu().numpy()
    error = np.abs(u_pred - u_exact)

    x_np = x.cpu().numpy()
    t_np = t.cpu().numpy()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    im0 = axes[0].pcolormesh(x_np, t_np, u_pred.T, cmap="viridis", shading="auto")
    axes[0].set_title("Predicted")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(x_np, t_np, u_exact.T, cmap="viridis", shading="auto")
    axes[1].set_title("Exact")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(x_np, t_np, error.T, cmap="hot", shading="auto")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_dir / "solution_comparison.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    t_slices = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for i, t_val in enumerate(t_slices):
        ax = axes[i // 3, i % 3]
        t_idx = min(int(t_val * (N - 1)), N - 1)
        ax.plot(x_np, u_pred[:, t_idx], "b-", linewidth=2, label="Predicted")
        ax.plot(x_np, u_exact[:, t_idx], "r--", linewidth=2, label="Exact")
        ax.set_xlabel("x")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"t = {t_np[t_idx]:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Solution Slices at Fixed t", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "slices_fixed_t.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    x_slices = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    for i, x_val in enumerate(x_slices):
        ax = axes[i // 3, i % 3]
        x_idx = min(int(x_val * (N - 1)), N - 1)
        ax.plot(t_np, u_pred[x_idx, :], "b-", linewidth=2, label="Predicted")
        ax.plot(t_np, u_exact[x_idx, :], "r--", linewidth=2, label="Exact")
        ax.set_xlabel("t")
        ax.set_ylabel("u(x, t)")
        ax.set_title(f"x = {x_np[x_idx]:.3f}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.suptitle("Solution Slices at Fixed x", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "slices_fixed_x.png", dpi=150)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for t_val in [0.25, 0.5, 0.75, 1.0]:
        t_idx = min(int(t_val * (N - 1)), N - 1)
        axes[0].plot(x_np, error[:, t_idx], linewidth=1.5, label=f"t={t_np[t_idx]:.3f}")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("|Error|")
    axes[0].set_title("Error vs x at Fixed t")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    for x_val in [0.25, 0.5, 0.75]:
        x_idx = min(int(x_val * (N - 1)), N - 1)
        axes[1].plot(t_np, error[x_idx, :], linewidth=1.5, label=f"x={x_np[x_idx]:.3f}")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("|Error|")
    axes[1].set_title("Error vs t at Fixed x")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "error_slices.png", dpi=150)
    plt.close()

    print(f"Saved visualizations to: {output_dir}")


if __name__ == "__main__":
    main()

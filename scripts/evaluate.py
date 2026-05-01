#!/usr/bin/env python3
"""Evaluation script for trained PINN checkpoints."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_factory import build_model, model_name_from_config
from src.physics_integro import exact_solution as integro_exact_solution


def exact_solution(x, t, alpha, problem_type):
    """Return the exact solution associated with the configured problem."""
    if problem_type == "integro_differential":
        return integro_exact_solution(x, t, alpha, {"family": "cosine", "spatial_frequency": 1.0})
    return (t ** alpha) * torch.sin(np.pi * x)


def compute_error_metrics(u_pred, u_exact, t_mesh):
    """Compute relative L2 and Linf errors while excluding t=0."""
    mask = t_mesh > 1e-10
    pred = u_pred[mask]
    exact = u_exact[mask]
    diff = pred - exact

    l2_rel = torch.norm(diff) / torch.norm(exact)
    linf = torch.max(torch.abs(diff))
    mean_abs = torch.mean(torch.abs(diff))

    return {
        "l2_relative": l2_rel.item(),
        "linf": linf.item(),
        "mean_absolute": mean_abs.item(),
    }


def plot_solution_comparison(x_mesh, t_mesh, u_pred, u_exact, output_path):
    """Plot exact/predicted/error fields."""
    error = torch.abs(u_pred - u_exact)

    X = x_mesh.cpu().numpy()
    T = t_mesh.cpu().numpy()
    pred = u_pred.cpu().numpy()
    exact = u_exact.cpu().numpy()
    err = error.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    im0 = axes[0].pcolormesh(X, T, exact, cmap="viridis", shading="auto")
    axes[0].set_title("Exact")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("t")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].pcolormesh(X, T, pred, cmap="viridis", shading="auto")
    axes[1].set_title("Predicted")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("t")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].pcolormesh(X, T, err, cmap="hot", shading="auto")
    axes[2].set_title(f"Absolute Error (max={err.max():.3e})")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("t")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_training_history(checkpoint, output_path):
    """Plot available training history from checkpoint formats used in this repo."""
    history = checkpoint.get("history", {})

    if history.get("epochs") and history.get("l2") and history.get("linf"):
        epochs = history["epochs"]
        l2_vals = history["l2"]
        linf_vals = history["linf"]
        loss_vals = history.get("loss", [])

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))

        if loss_vals:
            axes[0].semilogy(epochs, loss_vals, "b-")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("Epoch")
        axes[0].grid(True, alpha=0.3)

        axes[1].semilogy(epochs, l2_vals, "g-")
        axes[1].set_title("L2 Relative Error")
        axes[1].set_xlabel("Epoch")
        axes[1].grid(True, alpha=0.3)

        axes[2].semilogy(epochs, linf_vals, "r-")
        axes[2].set_title("Linf Error")
        axes[2].set_xlabel("Epoch")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_path, dpi=150)
        plt.close()
        return

    loss_history = checkpoint.get("loss_history")
    error_history = checkpoint.get("error_history")
    eval_epochs = checkpoint.get("eval_epochs", [])

    if not loss_history or not error_history or not eval_epochs:
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    axes[0].semilogy(loss_history.get("total", []), "b-")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Step")
    axes[0].grid(True, alpha=0.3)

    axes[1].semilogy(eval_epochs, error_history.get("l2", []), "g-")
    axes[1].set_title("L2 Relative Error")
    axes[1].set_xlabel("Epoch")
    axes[1].grid(True, alpha=0.3)

    axes[2].semilogy(eval_epochs, error_history.get("linf", []), "r-")
    axes[2].set_title("Linf Error")
    axes[2].set_xlabel("Epoch")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a trained PINN checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint file")
    parser.add_argument("--config", type=str, default="configs/integro_differential.yaml", help="Path to YAML config")
    parser.add_argument("--output-dir", type=str, default="outputs/eval", help="Output directory")
    parser.add_argument("--n-test", type=int, default=100, help="Grid resolution per dimension")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"], help="Evaluation device")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
        if device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")

    print(f"Using device: {device}")

    problem = config.get("problem", {})
    problem_type = problem.get("type", "fractional")
    alpha = problem.get("alpha", 0.5)
    solution_cfg = problem.get("solution", {})
    x_min = problem.get("x_min", 0.0)
    x_max = problem.get("x_max", 1.0)
    t_max = problem.get("t_max", 1.0)

    net_cfg = config.get("network", config.get("model", {}))
    model = build_model(net_cfg, device=device, problem_config=problem)
    print(f"Model type: {model_name_from_config(net_cfg)}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    epoch = checkpoint.get("epoch", "unknown")
    print(f"Loaded checkpoint epoch: {epoch}")

    x = torch.linspace(x_min, x_max, args.n_test, dtype=torch.float64, device=device)
    t = torch.linspace(0.0, t_max, args.n_test, dtype=torch.float64, device=device)
    x_mesh, t_mesh = torch.meshgrid(x, t, indexing="ij")

    with torch.no_grad():
        u_pred = model(x_mesh.flatten(), t_mesh.flatten()).reshape(x_mesh.shape)

    if problem_type == "integro_differential":
        u_exact = integro_exact_solution(x_mesh, t_mesh, alpha, solution_cfg)
    else:
        u_exact = exact_solution(x_mesh, t_mesh, alpha, problem_type)
    metrics = compute_error_metrics(u_pred, u_exact, t_mesh)

    print("\nError metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.6e}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    plot_solution_comparison(
        x_mesh,
        t_mesh,
        u_pred,
        u_exact,
        output_dir / "solution_comparison.png",
    )
    plot_training_history(checkpoint, output_dir / "training_history.png")

    with open(output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSaved outputs to: {output_dir}")


if __name__ == "__main__":
    main()

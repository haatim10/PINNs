#!/usr/bin/env python3
"""Evaluation script for trained PINN model."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PINN
from src.dataset import create_test_grid
from src.utils import (
    exact_solution_example,
    compute_errors,
    plot_comparison,
    plot_training_history,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate trained PINN")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/figures",
        help="Directory to save figures",
    )
    parser.add_argument(
        "--n_test",
        type=int,
        default=100,
        help="Number of test points per dimension",
    )
    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Problem parameters
    problem_cfg = config.get("problem", {})
    alpha = problem_cfg.get("alpha", 0.5)
    T = problem_cfg.get("T", 1.0)
    x_min = problem_cfg.get("x_min", 0.0)
    x_max = problem_cfg.get("x_max", 1.0)

    # Create model
    model_cfg = config.get("model", {})
    model = PINN(
        input_dim=model_cfg.get("input_dim", 2),
        output_dim=model_cfg.get("output_dim", 1),
        hidden_layers=model_cfg.get("hidden_layers", [64, 64, 64]),
        activation=model_cfg.get("activation", "tanh"),
    ).to(device)

    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")

    # Create test grid
    t_grid, x_grid, T_mesh, X_mesh = create_test_grid(
        t_range=(0.0, T),
        x_range=(x_min, x_max),
        N_t=args.n_test,
        N_x=args.n_test,
        device=device,
    )

    # Predict
    with torch.no_grad():
        u_pred = model(t_grid, x_grid)
        u_pred = u_pred.reshape(args.n_test, args.n_test)

    # Compute exact solution (if available)
    u_exact = exact_solution_example(T_mesh, X_mesh, alpha)

    # Compute errors
    errors = compute_errors(u_pred, u_exact)
    print("\nError Metrics:")
    for name, value in errors.items():
        print(f"  {name}: {value:.6e}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot comparison
    plot_comparison(
        T_mesh,
        X_mesh,
        u_pred,
        u_exact,
        save_path=output_dir / "solution_comparison.png",
    )

    # Plot training history if available
    history_path = Path(args.checkpoint).parent.parent / "logs" / "training_history.json"
    if history_path.exists():
        import json
        with open(history_path, "r") as f:
            history = json.load(f)
        plot_training_history(history, save_path=output_dir / "training_history.png")

    print(f"\nFigures saved to {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Main training script for PI-fMI-Fractional-PDE."""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PINN
from src.loss import PINNLoss
from src.dataset import CollocationDataset
from src.trainer import Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def main():
    parser = argparse.ArgumentParser(description="Train PINN for Fractional PDE")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Set device
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set random seed
    seed = config.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)

    # Problem parameters
    problem_cfg = config.get("problem", {})
    alpha = problem_cfg.get("alpha", 0.5)
    T = problem_cfg.get("T", 1.0)
    x_min = problem_cfg.get("x_min", 0.0)
    x_max = problem_cfg.get("x_max", 1.0)

    # Define initial and boundary conditions
    def ic_fn(x):
        return torch.sin(np.pi * x)

    def bc_fn(t, x):
        return torch.zeros_like(t)

    # Create dataset
    coll_cfg = config.get("collocation", {})
    dataset = CollocationDataset(
        t_range=(0.0, T),
        x_range=(x_min, x_max),
        N_pde=coll_cfg.get("N_pde", 5000),
        N_ic=coll_cfg.get("N_ic", 200),
        N_bc=coll_cfg.get("N_bc", 200),
        ic_fn=ic_fn,
        bc_fn=bc_fn,
        device=device,
        seed=seed,
    )

    # Create model
    model_cfg = config.get("model", {})
    model = PINN(
        input_dim=model_cfg.get("input_dim", 2),
        output_dim=model_cfg.get("output_dim", 1),
        hidden_layers=model_cfg.get("hidden_layers", [64, 64, 64]),
        activation=model_cfg.get("activation", "tanh"),
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Create loss function
    loss_cfg = config.get("loss", {})
    loss_fn = PINNLoss(
        lambda_pde=loss_cfg.get("lambda_pde", 1.0),
        lambda_ic=loss_cfg.get("lambda_ic", 10.0),
        lambda_bc=loss_cfg.get("lambda_bc", 10.0),
        alpha=alpha,
    )

    # Create trainer
    output_dir = config.get("logging", {}).get("output_dir", "outputs")
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        dataset=dataset,
        config=config,
        output_dir=output_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    train_cfg = config.get("training", {})
    log_cfg = config.get("logging", {})

    trainer.train(
        epochs=train_cfg.get("epochs", 10000),
        log_every=log_cfg.get("log_every", 100),
        save_every=log_cfg.get("save_every", 1000),
    )

    print("Training complete!")


if __name__ == "__main__":
    main()

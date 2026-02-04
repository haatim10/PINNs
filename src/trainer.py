"""Training loop with logging."""

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau
from typing import Dict, Optional
from pathlib import Path
import json
from tqdm import tqdm

from .model import PINN
from .loss import PINNLoss
from .dataset import CollocationDataset


class Trainer:
    """PINN Trainer with logging and checkpointing."""

    def __init__(
        self,
        model: PINN,
        loss_fn: PINNLoss,
        dataset: CollocationDataset,
        config: Dict,
        output_dir: str = "outputs",
    ):
        """
        Initialize trainer.

        Args:
            model: PINN model
            loss_fn: Loss function
            dataset: Collocation dataset
            config: Training configuration
            output_dir: Directory for outputs
        """
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.config = config

        # Setup output directories
        self.output_dir = Path(output_dir)
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Setup optimizer
        train_cfg = config.get("training", {})
        self.optimizer = Adam(
            model.parameters(),
            lr=train_cfg.get("learning_rate", 1e-3),
            weight_decay=train_cfg.get("weight_decay", 0.0),
        )

        # Setup scheduler
        self.scheduler = self._setup_scheduler(train_cfg)

        # Training state
        self.epoch = 0
        self.best_loss = float("inf")
        self.history = {"total": [], "pde": [], "ic": [], "bc": [], "lr": []}

    def _setup_scheduler(self, train_cfg: Dict):
        """Setup learning rate scheduler."""
        sched_cfg = train_cfg.get("scheduler", {})
        sched_type = sched_cfg.get("type", "StepLR")

        if sched_type == "StepLR":
            return StepLR(
                self.optimizer,
                step_size=sched_cfg.get("step_size", 2000),
                gamma=sched_cfg.get("gamma", 0.5),
            )
        elif sched_type == "CosineAnnealing":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=train_cfg.get("epochs", 10000),
            )
        elif sched_type == "ReduceLROnPlateau":
            return ReduceLROnPlateau(
                self.optimizer,
                patience=sched_cfg.get("patience", 500),
                factor=sched_cfg.get("gamma", 0.5),
            )
        else:
            return None

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        # Get batch data
        data = self.dataset.get_batch()

        # Zero gradients
        self.optimizer.zero_grad()

        # Compute loss
        losses = self.loss_fn(self.model, data)

        # Backward pass
        losses["total"].backward()

        # Gradient clipping (optional)
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        # Update weights
        self.optimizer.step()

        # Update scheduler
        if self.scheduler is not None:
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(losses["total"])
            else:
                self.scheduler.step()

        return {k: v.item() for k, v in losses.items()}

    def train(
        self,
        epochs: int,
        log_every: int = 100,
        save_every: int = 1000,
        resample_every: Optional[int] = None,
    ):
        """
        Full training loop.

        Args:
            epochs: Number of epochs
            log_every: Logging frequency
            save_every: Checkpoint save frequency
            resample_every: Resample PDE points every N epochs (None to disable)
        """
        pbar = tqdm(range(epochs), desc="Training")

        for epoch in pbar:
            self.epoch = epoch

            # Resample PDE points periodically
            if resample_every is not None and epoch > 0 and epoch % resample_every == 0:
                self.dataset.resample_pde_points()

            # Train one epoch
            losses = self.train_epoch()

            # Record history
            current_lr = self.optimizer.param_groups[0]["lr"]
            self.history["total"].append(losses["total"])
            self.history["pde"].append(losses["pde"])
            self.history["ic"].append(losses["ic"])
            self.history["bc"].append(losses["bc"])
            self.history["lr"].append(current_lr)

            # Update progress bar
            pbar.set_postfix(
                loss=f"{losses['total']:.2e}",
                pde=f"{losses['pde']:.2e}",
                ic=f"{losses['ic']:.2e}",
                bc=f"{losses['bc']:.2e}",
            )

            # Logging
            if epoch % log_every == 0:
                self._log_metrics(epoch, losses)

            # Save checkpoint
            if epoch % save_every == 0 and epoch > 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pt")

            # Save best model
            if losses["total"] < self.best_loss:
                self.best_loss = losses["total"]
                self.save_checkpoint("best_model.pt")

        # Save final model
        self.save_checkpoint("final_model.pt")
        self._save_history()

    def _log_metrics(self, epoch: int, losses: Dict[str, float]):
        """Log metrics to console."""
        lr = self.optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}: Loss={losses['total']:.4e}, "
            f"PDE={losses['pde']:.4e}, IC={losses['ic']:.4e}, "
            f"BC={losses['bc']:.4e}, LR={lr:.2e}"
        )

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            "epoch": self.epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_loss": self.best_loss,
            "config": self.config,
        }
        if self.scheduler is not None:
            checkpoint["scheduler_state_dict"] = self.scheduler.state_dict()

        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filepath: str):
        """Load model checkpoint."""
        checkpoint = torch.load(filepath)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epoch = checkpoint["epoch"]
        self.best_loss = checkpoint["best_loss"]
        if self.scheduler is not None and "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    def _save_history(self):
        """Save training history to JSON."""
        with open(self.log_dir / "training_history.json", "w") as f:
            json.dump(self.history, f, indent=2)

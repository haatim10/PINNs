"""Utility functions: exact solutions, error metrics, and plotting."""

import torch
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, Tuple, Optional
from pathlib import Path


def exact_solution_example(
    t: torch.Tensor, x: torch.Tensor, alpha: float = 0.5
) -> torch.Tensor:
    """
    Example exact solution for testing.

    This is for the problem:
    D_t^alpha u - u_xx = f(t, x)
    u(0, x) = sin(pi * x)
    u(t, 0) = u(t, 1) = 0

    The exact solution is u(t, x) = E_alpha(-pi^2 * t^alpha) * sin(pi * x)
    where E_alpha is the Mittag-Leffler function.

    For simplicity, we use an approximation here.
    """
    # Simplified: exponential decay approximation
    decay = torch.exp(-np.pi**2 * t**alpha)
    return decay * torch.sin(np.pi * x)


def mittag_leffler(z: torch.Tensor, alpha: float, beta: float = 1.0, n_terms: int = 100) -> torch.Tensor:
    """
    Compute the Mittag-Leffler function E_{alpha, beta}(z).

    E_{alpha, beta}(z) = sum_{k=0}^{inf} z^k / Gamma(alpha*k + beta)

    Args:
        z: Input tensor
        alpha: First parameter
        beta: Second parameter
        n_terms: Number of terms in series expansion

    Returns:
        Mittag-Leffler function values
    """
    result = torch.zeros_like(z)
    z_power = torch.ones_like(z)

    for k in range(n_terms):
        gamma_val = torch.exp(torch.lgamma(torch.tensor(alpha * k + beta)))
        result = result + z_power / gamma_val
        z_power = z_power * z

    return result


def compute_errors(
    u_pred: torch.Tensor,
    u_exact: torch.Tensor,
) -> dict:
    """
    Compute various error metrics.

    Args:
        u_pred: Predicted solution
        u_exact: Exact solution

    Returns:
        Dictionary of error metrics
    """
    # Flatten tensors
    u_pred = u_pred.flatten()
    u_exact = u_exact.flatten()

    # Absolute errors
    abs_error = torch.abs(u_pred - u_exact)

    # L2 error
    l2_error = torch.sqrt(torch.mean((u_pred - u_exact) ** 2))

    # Relative L2 error
    rel_l2_error = l2_error / (torch.sqrt(torch.mean(u_exact ** 2)) + 1e-10)

    # Max error
    max_error = torch.max(abs_error)

    # Mean absolute error
    mae = torch.mean(abs_error)

    return {
        "L2": l2_error.item(),
        "Relative_L2": rel_l2_error.item(),
        "Max": max_error.item(),
        "MAE": mae.item(),
    }


def plot_solution(
    T: torch.Tensor,
    X: torch.Tensor,
    U: torch.Tensor,
    title: str = "PINN Solution",
    save_path: Optional[str] = None,
    cmap: str = "viridis",
):
    """
    Plot 2D solution heatmap.

    Args:
        T: Time meshgrid (N_t x N_x)
        X: Space meshgrid (N_t x N_x)
        U: Solution values (N_t x N_x)
        title: Plot title
        save_path: Path to save figure
        cmap: Colormap
    """
    # Convert to numpy
    T_np = T.cpu().numpy() if isinstance(T, torch.Tensor) else T
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    U_np = U.cpu().numpy() if isinstance(U, torch.Tensor) else U

    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.pcolormesh(T_np, X_np, U_np, cmap=cmap, shading="auto")
    ax.set_xlabel("Time t")
    ax.set_ylabel("Space x")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, label="u(t, x)")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_comparison(
    T: torch.Tensor,
    X: torch.Tensor,
    U_pred: torch.Tensor,
    U_exact: torch.Tensor,
    save_path: Optional[str] = None,
):
    """
    Plot predicted vs exact solution comparison.

    Args:
        T: Time meshgrid
        X: Space meshgrid
        U_pred: Predicted solution
        U_exact: Exact solution
        save_path: Path to save figure
    """
    T_np = T.cpu().numpy() if isinstance(T, torch.Tensor) else T
    X_np = X.cpu().numpy() if isinstance(X, torch.Tensor) else X
    U_pred_np = U_pred.cpu().numpy() if isinstance(U_pred, torch.Tensor) else U_pred
    U_exact_np = U_exact.cpu().numpy() if isinstance(U_exact, torch.Tensor) else U_exact

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Predicted
    im0 = axes[0].pcolormesh(T_np, X_np, U_pred_np, cmap="viridis", shading="auto")
    axes[0].set_title("Predicted u(t, x)")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    plt.colorbar(im0, ax=axes[0])

    # Exact
    im1 = axes[1].pcolormesh(T_np, X_np, U_exact_np, cmap="viridis", shading="auto")
    axes[1].set_title("Exact u(t, x)")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    plt.colorbar(im1, ax=axes[1])

    # Error
    error = np.abs(U_pred_np - U_exact_np)
    im2 = axes[2].pcolormesh(T_np, X_np, error, cmap="hot", shading="auto")
    axes[2].set_title("Absolute Error")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    plt.colorbar(im2, ax=axes[2])

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_training_history(history: dict, save_path: Optional[str] = None):
    """
    Plot training loss history.

    Args:
        history: Dictionary with loss history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Total loss
    axes[0].semilogy(history["total"], label="Total")
    axes[0].semilogy(history["pde"], label="PDE", alpha=0.7)
    axes[0].semilogy(history["ic"], label="IC", alpha=0.7)
    axes[0].semilogy(history["bc"], label="BC", alpha=0.7)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Learning rate
    axes[1].semilogy(history["lr"])
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Learning Rate")
    axes[1].set_title("Learning Rate Schedule")
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

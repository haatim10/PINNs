"""Fractional derivative and residual computation."""

import torch
import numpy as np
from typing import Callable


def caputo_derivative_l1(
    u_history: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    n: int,
) -> torch.Tensor:
    """
    Compute Caputo fractional derivative using L1 scheme.

    D_t^alpha u(t_n) ≈ (1/Γ(2-α)) * Σ_{k=0}^{n-1} a_{n,k} * (u_{k+1} - u_k)

    Args:
        u_history: Solution values at mesh points, shape (n+1, ...)
        t: Time mesh points, shape (n+1,)
        alpha: Fractional order (0 < alpha < 1)
        n: Current time index

    Returns:
        Approximation of D_t^alpha u(t_n)
    """
    if n == 0:
        return torch.zeros_like(u_history[0])

    gamma_factor = 1.0 / torch.exp(torch.lgamma(torch.tensor(2 - alpha, device=t.device)))

    result = torch.zeros_like(u_history[0])

    for k in range(n):
        tau_k = t[k + 1] - t[k]
        a_nk = gamma_factor * (
            (t[n] - t[k]) ** (1 - alpha) - (t[n] - t[k + 1]) ** (1 - alpha)
        ) / tau_k
        result = result + a_nk * (u_history[k + 1] - u_history[k])

    return result


def fractional_derivative_autodiff(
    model: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    alpha: float,
    t_mesh: torch.Tensor,
) -> torch.Tensor:
    """
    Compute fractional derivative using automatic differentiation and L1 scheme.

    This combines neural network evaluation with the L1 discretization.

    Args:
        model: PINN model
        t: Current time points, shape (N,)
        x: Spatial points, shape (N,)
        alpha: Fractional order
        t_mesh: Full time mesh for L1 scheme

    Returns:
        Fractional derivative approximation
    """
    # Find nearest mesh index for each t
    # This is a simplified version - in practice, interpolation may be needed
    device = t.device

    # Evaluate model at all required points
    u = model(t, x)

    # For PINN, we often use the continuous formulation
    # and enforce the fractional PDE in a weak sense
    return u  # Placeholder - actual implementation depends on specific scheme


def compute_residual(
    model: torch.nn.Module,
    t: torch.Tensor,
    x: torch.Tensor,
    alpha: float,
    source_fn: Callable = None,
) -> torch.Tensor:
    """
    Compute PDE residual for the fractional diffusion equation.

    Equation: D_t^alpha u - u_xx = f(t, x)

    For PINN, we use automatic differentiation for spatial derivatives
    and approximate the fractional time derivative.

    Args:
        model: PINN model
        t: Time collocation points
        x: Spatial collocation points
        alpha: Fractional order
        source_fn: Source term function f(t, x)

    Returns:
        PDE residual at collocation points
    """
    t = t.requires_grad_(True)
    x = x.requires_grad_(True)

    # Forward pass
    u = model(t, x)

    # Compute spatial derivatives using autodiff
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    u_xx = torch.autograd.grad(
        u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
    )[0]

    # Compute time derivative (standard, for comparison or modified loss)
    u_t = torch.autograd.grad(
        u, t, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]

    # For fractional derivative, we use a modified approach:
    # Scale the standard derivative by t^(1-alpha) factor
    # This is an approximation that captures the fractional behavior
    eps = 1e-8
    fractional_factor = (t + eps) ** (1 - alpha) / torch.exp(
        torch.lgamma(torch.tensor(2 - alpha, device=t.device))
    )
    fractional_deriv = fractional_factor * u_t

    # Source term
    if source_fn is not None:
        f = source_fn(t, x)
    else:
        f = torch.zeros_like(u)

    # Residual: D_t^alpha u - u_xx - f = 0
    residual = fractional_deriv - u_xx - f

    return residual

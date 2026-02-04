"""Collocation points and data generation."""

import torch
import numpy as np
from typing import Dict, Callable, Optional, Tuple


class CollocationDataset:
    """Generate collocation points for PINN training."""

    def __init__(
        self,
        t_range: Tuple[float, float] = (0.0, 1.0),
        x_range: Tuple[float, float] = (0.0, 1.0),
        N_pde: int = 5000,
        N_ic: int = 200,
        N_bc: int = 200,
        ic_fn: Optional[Callable] = None,
        bc_fn: Optional[Callable] = None,
        device: str = "cpu",
        seed: int = 42,
    ):
        """
        Initialize collocation dataset.

        Args:
            t_range: Time domain (t_min, t_max)
            x_range: Spatial domain (x_min, x_max)
            N_pde: Number of interior collocation points
            N_ic: Number of initial condition points
            N_bc: Number of boundary condition points (per boundary)
            ic_fn: Initial condition function u(0, x)
            bc_fn: Boundary condition function u(t, x_boundary)
            device: Computation device
            seed: Random seed
        """
        self.t_range = t_range
        self.x_range = x_range
        self.N_pde = N_pde
        self.N_ic = N_ic
        self.N_bc = N_bc
        self.device = device

        # Set default IC/BC if not provided
        self.ic_fn = ic_fn if ic_fn is not None else lambda x: torch.sin(np.pi * x)
        self.bc_fn = bc_fn if bc_fn is not None else lambda t, x: torch.zeros_like(t)

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Generate points
        self._generate_points()

    def _generate_points(self):
        """Generate all collocation points."""
        t_min, t_max = self.t_range
        x_min, x_max = self.x_range

        # Interior PDE points (Latin Hypercube or random sampling)
        self.t_pde = torch.rand(self.N_pde, 1, device=self.device) * (t_max - t_min) + t_min
        self.x_pde = torch.rand(self.N_pde, 1, device=self.device) * (x_max - x_min) + x_min

        # Avoid t=0 for interior points (singularity)
        self.t_pde = torch.clamp(self.t_pde, min=1e-6)

        # Initial condition points (t=0)
        self.x_ic = torch.rand(self.N_ic, 1, device=self.device) * (x_max - x_min) + x_min
        self.u_ic = self.ic_fn(self.x_ic)

        # Boundary condition points
        self.t_bc = torch.rand(self.N_bc, 1, device=self.device) * (t_max - t_min) + t_min
        self.x_bc_left = torch.full((self.N_bc, 1), x_min, device=self.device)
        self.x_bc_right = torch.full((self.N_bc, 1), x_max, device=self.device)
        self.u_bc_left = self.bc_fn(self.t_bc, self.x_bc_left)
        self.u_bc_right = self.bc_fn(self.t_bc, self.x_bc_right)

    def get_batch(self) -> Dict[str, torch.Tensor]:
        """Get all collocation data as a dictionary."""
        return {
            "t_pde": self.t_pde,
            "x_pde": self.x_pde,
            "x_ic": self.x_ic,
            "u_ic": self.u_ic,
            "t_bc": self.t_bc,
            "x_bc_left": self.x_bc_left,
            "x_bc_right": self.x_bc_right,
            "u_bc_left": self.u_bc_left,
            "u_bc_right": self.u_bc_right,
        }

    def resample_pde_points(self):
        """Resample interior PDE points (for adaptive training)."""
        t_min, t_max = self.t_range
        x_min, x_max = self.x_range

        self.t_pde = torch.rand(self.N_pde, 1, device=self.device) * (t_max - t_min) + t_min
        self.x_pde = torch.rand(self.N_pde, 1, device=self.device) * (x_max - x_min) + x_min
        self.t_pde = torch.clamp(self.t_pde, min=1e-6)


def create_test_grid(
    t_range: Tuple[float, float],
    x_range: Tuple[float, float],
    N_t: int = 100,
    N_x: int = 100,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a regular grid for testing/visualization.

    Args:
        t_range: Time domain
        x_range: Spatial domain
        N_t: Number of time points
        N_x: Number of spatial points
        device: Computation device

    Returns:
        t_grid, x_grid: Flattened grid points for model evaluation
        T, X: Meshgrid arrays for plotting
    """
    t = torch.linspace(t_range[0], t_range[1], N_t, device=device)
    x = torch.linspace(x_range[0], x_range[1], N_x, device=device)

    T, X = torch.meshgrid(t, x, indexing="ij")

    t_grid = T.flatten().unsqueeze(-1)
    x_grid = X.flatten().unsqueeze(-1)

    return t_grid, x_grid, T, X

"""Combined loss function for PINN training."""

import torch
import torch.nn as nn
from typing import Dict, Callable, Optional


class PINNLoss(nn.Module):
    """Combined loss for Physics-Informed Neural Network."""

    def __init__(
        self,
        lambda_pde: float = 1.0,
        lambda_ic: float = 10.0,
        lambda_bc: float = 10.0,
        alpha: float = 0.5,
        source_fn: Optional[Callable] = None,
    ):
        """
        Initialize PINN loss.

        Args:
            lambda_pde: Weight for PDE residual loss
            lambda_ic: Weight for initial condition loss
            lambda_bc: Weight for boundary condition loss
            alpha: Fractional order
            source_fn: Source term function
        """
        super().__init__()
        self.lambda_pde = lambda_pde
        self.lambda_ic = lambda_ic
        self.lambda_bc = lambda_bc
        self.alpha = alpha
        self.source_fn = source_fn

    def pde_loss(
        self, model: nn.Module, t_pde: torch.Tensor, x_pde: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute PDE residual loss.

        Args:
            model: PINN model
            t_pde: Time collocation points
            x_pde: Spatial collocation points

        Returns:
            Mean squared PDE residual
        """
        t_pde = t_pde.requires_grad_(True)
        x_pde = x_pde.requires_grad_(True)

        u = model(t_pde, x_pde)

        # Spatial derivatives
        u_x = torch.autograd.grad(
            u, x_pde, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]
        u_xx = torch.autograd.grad(
            u_x, x_pde, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        # Time derivative
        u_t = torch.autograd.grad(
            u, t_pde, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        # Fractional derivative approximation
        eps = 1e-8
        gamma_val = torch.exp(torch.lgamma(torch.tensor(2 - self.alpha, device=t_pde.device)))
        fractional_factor = (t_pde + eps) ** (1 - self.alpha) / gamma_val
        D_alpha_u = fractional_factor * u_t

        # Source term
        if self.source_fn is not None:
            f = self.source_fn(t_pde, x_pde)
        else:
            f = torch.zeros_like(u)

        # PDE: D_t^alpha u - u_xx = f
        residual = D_alpha_u - u_xx - f

        return torch.mean(residual ** 2)

    def ic_loss(
        self,
        model: nn.Module,
        x_ic: torch.Tensor,
        u_ic: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute initial condition loss.

        Args:
            model: PINN model
            x_ic: Spatial points for IC
            u_ic: True initial values

        Returns:
            Mean squared IC error
        """
        t_ic = torch.zeros_like(x_ic)
        u_pred = model(t_ic, x_ic)
        return torch.mean((u_pred - u_ic) ** 2)

    def bc_loss(
        self,
        model: nn.Module,
        t_bc: torch.Tensor,
        x_bc_left: torch.Tensor,
        x_bc_right: torch.Tensor,
        u_bc_left: torch.Tensor,
        u_bc_right: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute boundary condition loss (Dirichlet).

        Args:
            model: PINN model
            t_bc: Time points for BC
            x_bc_left: Left boundary x values
            x_bc_right: Right boundary x values
            u_bc_left: Left boundary true values
            u_bc_right: Right boundary true values

        Returns:
            Mean squared BC error
        """
        u_pred_left = model(t_bc, x_bc_left)
        u_pred_right = model(t_bc, x_bc_right)

        loss_left = torch.mean((u_pred_left - u_bc_left) ** 2)
        loss_right = torch.mean((u_pred_right - u_bc_right) ** 2)

        return loss_left + loss_right

    def forward(
        self,
        model: nn.Module,
        data: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """
        Compute total loss.

        Args:
            model: PINN model
            data: Dictionary containing:
                - t_pde, x_pde: PDE collocation points
                - x_ic, u_ic: Initial condition data
                - t_bc, x_bc_left, x_bc_right, u_bc_left, u_bc_right: BC data

        Returns:
            Dictionary with individual and total losses
        """
        # PDE loss
        loss_pde = self.pde_loss(model, data["t_pde"], data["x_pde"])

        # Initial condition loss
        loss_ic = self.ic_loss(model, data["x_ic"], data["u_ic"])

        # Boundary condition loss
        loss_bc = self.bc_loss(
            model,
            data["t_bc"],
            data["x_bc_left"],
            data["x_bc_right"],
            data["u_bc_left"],
            data["u_bc_right"],
        )

        # Total weighted loss
        total_loss = (
            self.lambda_pde * loss_pde
            + self.lambda_ic * loss_ic
            + self.lambda_bc * loss_bc
        )

        return {
            "total": total_loss,
            "pde": loss_pde,
            "ic": loss_ic,
            "bc": loss_bc,
        }

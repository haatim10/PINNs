"""Loss Function Module"""

import torch
from dataclasses import dataclass


@dataclass
class LossComponents:
    total: torch.Tensor
    pde: torch.Tensor
    bc: torch.Tensor
    ic: torch.Tensor


class PIFMILoss:
    """
    Physics-Informed Loss Function.
    
    L_total = w_pde * L_PDE + w_bc * L_BC + w_ic * L_IC
    """
    
    def __init__(self, model, mesh, l1_coeffs, alpha: float, 
                 weights: dict = None, device: str = "cpu"):
        from .physics import PDEResidual
        
        self.model = model
        self.mesh = mesh
        self.l1_coeffs = l1_coeffs
        self.alpha = alpha
        self.device = device
        self.weights = weights or {'pde': 1.0, 'bc': 10.0, 'ic': 10.0}
        self.pde_residual = PDEResidual(model, mesh, l1_coeffs, alpha, device)
        
    def compute_pde_loss(self, x, t, n_indices):
        """L_PDE = mean(residual^2)"""
        residual = self.pde_residual.compute(x, t, n_indices)
        return torch.mean(residual ** 2)
    
    def compute_bc_loss(self, x, t, u_target):
        """L_BC = mean((u_pred - u_target)^2)"""
        u_pred = self.model(x, t).squeeze()
        return torch.mean((u_pred - u_target) ** 2)
    
    def compute_ic_loss(self, x, t, u_target):
        """L_IC = mean((u_pred - u_target)^2)"""
        u_pred = self.model(x, t).squeeze()
        return torch.mean((u_pred - u_target) ** 2)
    
    def compute(self, data):
        """Compute total loss."""
        loss_pde = self.compute_pde_loss(data.x_coll, data.t_coll, data.n_coll)
        loss_bc = self.compute_bc_loss(data.x_bc, data.t_bc, data.u_bc)
        loss_ic = self.compute_ic_loss(data.x_ic, data.t_ic, data.u_ic)
        
        loss_total = (
            self.weights['pde'] * loss_pde +
            self.weights['bc'] * loss_bc +
            self.weights['ic'] * loss_ic
        )
        
        return LossComponents(total=loss_total, pde=loss_pde, bc=loss_bc, ic=loss_ic)
    
    def __call__(self, data):
        return self.compute(data)

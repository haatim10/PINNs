"""
Physics Module: Full L1 Scheme for Caputo Fractional Derivative

PDE: D_t^alpha u - u_xx = f(x,t)
f(x,t) = [Gamma(alpha+1) + pi^2 * t^alpha] * sin(pi*x)
Exact solution: u(x,t) = t^alpha * sin(pi*x)
"""

import torch
import numpy as np
from scipy.special import gamma


class PDEResidual:
    """
    Computes PDE residual using full L1 scheme from the paper.
    
    L1 scheme: D_t^alpha u(t_n) = d_{n,1}*u^n - d_{n,n}*u^0 - sum_{k=1}^{n-1}(d_{n,k}-d_{n,k+1})*u^{n-k}
    """
    
    def __init__(self, model, mesh, l1_coeffs, alpha: float, device: str = "cpu"):
        self.model = model
        self.mesh = mesh
        self.l1_coeffs = l1_coeffs
        self.alpha = alpha
        self.device = device
        self.gamma_alpha_plus_1 = gamma(alpha + 1)
        
    def source_term(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """f(x,t) = [Gamma(alpha+1) + pi^2 * t^alpha] * sin(pi*x)"""
        return (self.gamma_alpha_plus_1 + np.pi**2 * t**self.alpha) * torch.sin(np.pi * x)
    
    def compute_u_and_u_xx(self, x: torch.Tensor, t: torch.Tensor):
        """Compute u and u_xx using automatic differentiation."""
        x = x.clone().requires_grad_(True)
        t = t.clone().requires_grad_(True)
        
        u = self.model(x, t)
        
        u_x = torch.autograd.grad(
            outputs=u, inputs=x,
            grad_outputs=torch.ones_like(u),
            create_graph=True, retain_graph=True
        )[0]
        
        u_xx = torch.autograd.grad(
            outputs=u_x, inputs=x,
            grad_outputs=torch.ones_like(u_x),
            create_graph=True, retain_graph=True
        )[0]
        
        return u, u_xx
    
    def compute_fractional_derivative_l1(self, x: torch.Tensor, n_indices: torch.Tensor, u_current: torch.Tensor):
        """
        Compute L1 approximation of Caputo derivative using full history.
        
        L1 formula: D_t^alpha u(t_n) = d_{n,1}*u^n - d_{n,n}*u^0 - sum_{k=1}^{n-1}(d_{n,k}-d_{n,k+1})*u^{n-k}
        """
        batch_size = x.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        
        unique_n = torch.unique(n_indices)
        t_nodes = self.mesh.get_nodes()
        
        for n in unique_n:
            n_val = n.item()
            if n_val == 0:
                continue
                
            mask = (n_indices == n_val)
            x_n = x[mask]
            u_n = u_current[mask].squeeze()
            num_points = x_n.shape[0]
            
            if num_points == 0:
                continue
                
            coeffs = self.l1_coeffs.get_coefficients_for_n(n_val)
            
            # d_{n,1} * u^n (current value - has gradient)
            frac_deriv = coeffs[1] * u_n
            
            # - d_{n,n} * u^0 (initial value)
            t_0 = t_nodes[0].expand(num_points)
            with torch.no_grad():
                u_0 = self.model(x_n, t_0).squeeze()
            frac_deriv = frac_deriv - coeffs[n_val] * u_0
            
            # - sum_{k=1}^{n-1} (d_{n,k} - d_{n,k+1}) * u^{n-k}
            for k in range(1, n_val):
                idx = n_val - k
                t_idx = t_nodes[idx].expand(num_points)
                with torch.no_grad():
                    u_idx = self.model(x_n, t_idx).squeeze()
                diff_coeff = coeffs[k] - coeffs[k + 1]
                frac_deriv = frac_deriv - diff_coeff * u_idx
            
            result[mask] = frac_deriv
            
        return result
    
    def compute(self, x: torch.Tensor, t: torch.Tensor, n_indices: torch.Tensor):
        """Compute full PDE residual: D_t^alpha u - u_xx - f = 0"""
        u, u_xx = self.compute_u_and_u_xx(x, t)
        frac_deriv = self.compute_fractional_derivative_l1(x, n_indices, u)
        f = self.source_term(x.detach(), t.detach())
        
        residual = frac_deriv - u_xx.squeeze() - f
        return residual

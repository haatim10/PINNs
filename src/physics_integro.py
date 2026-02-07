"""
Physics Module for Time-Fractional Integro-Differential Equation

PDE: D_t^α u - (x²+1) u_xx + ∫₀ᵗ sin(x)(t-s)^{-β} u(x,s) ds = f(x,t)

Domain: x ∈ [0,1], t ∈ (0,1]
BC: u(0,t) = t^α, u(1,t) = -t^α  
IC: u(x,0) = 0
Exact solution: u(x,t) = t^α cos(πx)

Source term derivation:
- D_t^α[t^α cos(πx)] = Γ(α+1) cos(πx)
- (x²+1) u_xx = -(x²+1) π² t^α cos(πx)
- ∫₀ᵗ sin(x)(t-s)^{-β} s^α cos(πx) ds = sin(x)cos(πx) t^{α+1-β} Γ(α+1)Γ(1-β)/Γ(α+2-β)

f(x,t) = cos(πx)[Γ(α+1) + (x²+1)π²t^α + sin(x) t^{α+1-β} Γ(α+1)Γ(1-β)/Γ(α+2-β)]
"""

import torch
import numpy as np
from scipy.special import gamma


class IntegroDifferentialResidual:
    """
    Computes PDE residual for time-fractional integro-differential equation.
    
    Uses:
    - L1 scheme for Caputo fractional derivative
    - Automatic differentiation for spatial derivatives
    - Composite quadrature for weakly singular integral
    """
    
    def __init__(self, model, mesh, l1_coeffs, alpha: float, beta: float, 
                 n_quad: int = 20, device: str = "cpu"):
        self.model = model
        self.mesh = mesh
        self.l1_coeffs = l1_coeffs
        self.alpha = alpha
        self.beta = beta
        self.n_quad = n_quad
        self.device = device
        
        # Precompute gamma function values
        self.gamma_alpha_plus_1 = gamma(alpha + 1)
        self.gamma_1_minus_beta = gamma(1 - beta)
        self.gamma_alpha_plus_2_minus_beta = gamma(alpha + 2 - beta)
        
        # Coefficient for integral term in source
        self.integral_coeff = (self.gamma_alpha_plus_1 * self.gamma_1_minus_beta / 
                               self.gamma_alpha_plus_2_minus_beta)
        
    def exact_solution(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """u(x,t) = t^α cos(πx)"""
        return (t ** self.alpha) * torch.cos(np.pi * x)
    
    def source_term(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        f(x,t) = cos(πx)[Γ(α+1) + (x²+1)π²t^α + sin(x) t^{α+1-β} Γ(α+1)Γ(1-β)/Γ(α+2-β)]
        """
        cos_pi_x = torch.cos(np.pi * x)
        sin_x = torch.sin(x)
        
        # Three terms
        term1 = self.gamma_alpha_plus_1
        term2 = (x**2 + 1) * (np.pi**2) * (t ** self.alpha)
        term3 = sin_x * (t ** (self.alpha + 1 - self.beta)) * self.integral_coeff
        
        return cos_pi_x * (term1 + term2 + term3)
    
    def compute_u_and_derivatives(self, x: torch.Tensor, t: torch.Tensor):
        """Compute u, u_x, and u_xx using automatic differentiation."""
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
    
    def compute_fractional_derivative_l1(self, x: torch.Tensor, n_indices: torch.Tensor, 
                                          u_current: torch.Tensor):
        """
        Compute L1 approximation of Caputo derivative using full history.
        
        L1 formula: D_t^α u(t_n) = d_{n,1}*u^n - d_{n,n}*u^0 - Σ_{k=1}^{n-1}(d_{n,k}-d_{n,k+1})*u^{n-k}
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
            
            # - d_{n,n} * u^0 (initial value = 0 for this problem)
            # Since u(x,0) = 0, this term vanishes
            
            # - Σ_{k=1}^{n-1} (d_{n,k} - d_{n,k+1}) * u^{n-k}
            for k in range(1, n_val):
                idx = n_val - k
                t_idx = t_nodes[idx].expand(num_points)
                with torch.no_grad():
                    u_idx = self.model(x_n, t_idx).squeeze()
                diff_coeff = coeffs[k] - coeffs[k + 1]
                frac_deriv = frac_deriv - diff_coeff * u_idx
            
            result[mask] = frac_deriv
            
        return result
    
    def compute_integral_term(self, x: torch.Tensor, t: torch.Tensor, n_indices: torch.Tensor):
        """
        Compute weakly singular integral: ∫₀ᵗ sin(x)(t-s)^{-β} u(x,s) ds
        
        Uses composite trapezoidal rule with graded mesh points.
        For weakly singular integrals, we use the available mesh points.
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
            t_n = t_nodes[n_val]
            num_points = x_n.shape[0]
            
            if num_points == 0:
                continue
            
            sin_x = torch.sin(x_n)
            
            # Compute integral using trapezoidal rule on graded mesh
            integral_sum = torch.zeros(num_points, dtype=torch.float64, device=self.device)
            
            for j in range(n_val):
                t_j = t_nodes[j]
                t_j_plus_1 = t_nodes[j + 1]
                h_j = t_j_plus_1 - t_j
                
                # Midpoint of interval for kernel evaluation
                t_mid = (t_j + t_j_plus_1) / 2
                
                # Kernel at midpoint: (t_n - t_mid)^{-β}
                kernel = (t_n - t_mid) ** (-self.beta)
                
                # u at midpoint (use linear interpolation or just evaluate at midpoint)
                with torch.no_grad():
                    t_mid_expanded = t_mid.expand(num_points)
                    u_mid = self.model(x_n, t_mid_expanded).squeeze()
                
                integral_sum = integral_sum + kernel * u_mid * h_j
            
            result[mask] = sin_x * integral_sum
            
        return result
    
    def compute_integral_term_quadrature(self, x: torch.Tensor, t: torch.Tensor, n_indices: torch.Tensor):
        """
        Alternative: Gauss-Jacobi quadrature for weakly singular integral.
        
        Transform: let s = t*τ, then ∫₀ᵗ (t-s)^{-β} u(x,s) ds = t^{1-β} ∫₀¹ (1-τ)^{-β} u(x,tτ) dτ
        
        Use Gauss-Jacobi quadrature with weight (1-τ)^{-β}.
        """
        batch_size = x.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        
        # Get Gauss-Jacobi nodes and weights for weight function (1-τ)^{-β} on [0,1]
        # For simplicity, use Gauss-Legendre and absorb singularity
        from numpy.polynomial.legendre import leggauss
        nodes, weights = leggauss(self.n_quad)
        
        # Transform from [-1,1] to [0,1]
        nodes = (nodes + 1) / 2
        weights = weights / 2
        
        unique_n = torch.unique(n_indices)
        t_nodes_mesh = self.mesh.get_nodes()
        
        for n in unique_n:
            n_val = n.item()
            if n_val == 0:
                continue
                
            mask = (n_indices == n_val)
            x_n = x[mask]
            t_n = t_nodes_mesh[n_val].item()
            num_points = x_n.shape[0]
            
            if num_points == 0:
                continue
            
            sin_x = torch.sin(x_n)
            
            # Compute integral using quadrature
            integral_sum = torch.zeros(num_points, dtype=torch.float64, device=self.device)
            
            for i, (tau, w) in enumerate(zip(nodes, weights)):
                s = t_n * tau
                
                # Kernel: (t-s)^{-β} = (t_n - t_n*τ)^{-β} = t_n^{-β} * (1-τ)^{-β}
                # Full integrand factor: t_n * (t_n - s)^{-β} = t_n^{1-β} * (1-τ)^{-β}
                kernel = (t_n ** (1 - self.beta)) * ((1 - tau) ** (-self.beta))
                
                # Evaluate u(x, s) = u(x, t_n * τ)
                s_tensor = torch.tensor(s, dtype=torch.float64, device=self.device).expand(num_points)
                with torch.no_grad():
                    u_s = self.model(x_n, s_tensor).squeeze()
                
                integral_sum = integral_sum + w * kernel * u_s
            
            result[mask] = sin_x * integral_sum
            
        return result
    
    def compute(self, x: torch.Tensor, t: torch.Tensor, n_indices: torch.Tensor):
        """
        Compute full PDE residual:
        D_t^α u - (x²+1) u_xx + ∫₀ᵗ sin(x)(t-s)^{-β} u(x,s) ds - f = 0
        """
        # Get u and u_xx
        u, u_xx = self.compute_u_and_derivatives(x, t)
        
        # Fractional derivative
        frac_deriv = self.compute_fractional_derivative_l1(x, n_indices, u)
        
        # Variable coefficient diffusion term: (x²+1) u_xx
        diffusion = (x**2 + 1) * u_xx.squeeze()
        
        # Integral term (use mesh-based quadrature)
        integral_term = self.compute_integral_term(x, t, n_indices)
        
        # Source term
        f = self.source_term(x.detach(), t.detach())
        
        # PDE residual: D_t^α u - (x²+1) u_xx + integral - f = 0
        residual = frac_deriv - diffusion + integral_term - f
        
        return residual


class BoundaryConditions:
    """
    Boundary conditions for the integro-differential problem.
    
    u(0,t) = t^α
    u(1,t) = -t^α
    """
    
    def __init__(self, model, alpha: float, device: str = "cpu"):
        self.model = model
        self.alpha = alpha
        self.device = device
        
    def left_bc(self, t: torch.Tensor) -> torch.Tensor:
        """u(0,t) = t^α"""
        return t ** self.alpha
    
    def right_bc(self, t: torch.Tensor) -> torch.Tensor:
        """u(1,t) = -t^α"""
        return -(t ** self.alpha)
    
    def compute_bc_loss(self, t_left: torch.Tensor, t_right: torch.Tensor):
        """Compute boundary condition residuals."""
        x_left = torch.zeros_like(t_left)
        x_right = torch.ones_like(t_right)
        
        u_pred_left = self.model(x_left, t_left).squeeze()
        u_pred_right = self.model(x_right, t_right).squeeze()
        
        u_exact_left = self.left_bc(t_left)
        u_exact_right = self.right_bc(t_right)
        
        bc_residual_left = u_pred_left - u_exact_left
        bc_residual_right = u_pred_right - u_exact_right
        
        return bc_residual_left, bc_residual_right


class InitialCondition:
    """
    Initial condition: u(x,0) = 0
    """
    
    def __init__(self, model, device: str = "cpu"):
        self.model = model
        self.device = device
        
    def compute_ic_loss(self, x: torch.Tensor):
        """Compute initial condition residual: u(x,0) = 0"""
        t_zero = torch.zeros_like(x)
        u_pred = self.model(x, t_zero).squeeze()
        
        # IC is u(x,0) = 0
        return u_pred  # Should be zero

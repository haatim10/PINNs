"""
Physics Module for Time-Fractional Integro-Differential Equation

PDE: D_t^α u - (x²+1) u_xx + ∫₀ᵗ sin(x)(t-s)^{-β} u(x,s) ds = f(x,t)

Domain: x ∈ [0,1], t ∈ (0,1]
BC: u(x,t) = exact_solution(x,t) on x = 0,1
IC: u(x,0) = 0

Default exact solution family:
u(x,t) = A t^p cos(kπx + φ)

The frequency k and time power p are configurable so the same code path can
represent the standard branch problem and a harder oscillatory variant.
"""

import torch
import numpy as np
from scipy.special import gamma


def resolve_solution_config(solution_cfg: dict | None, alpha: float) -> dict:
    """Return normalized exact-solution parameters."""
    cfg = dict(solution_cfg or {})
    cfg.setdefault("family", "cosine")
    cfg.setdefault("spatial_frequency", 1.0)
    cfg.setdefault("time_power", alpha)
    cfg.setdefault("amplitude", 1.0)
    cfg.setdefault("phase", 0.0)
    return cfg


def exact_solution(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    solution_cfg: dict | None = None,
) -> torch.Tensor:
    """Evaluate the configured exact solution."""
    cfg = resolve_solution_config(solution_cfg, alpha)
    family = str(cfg.get("family", "cosine")).lower()
    frequency = float(cfg.get("spatial_frequency", 1.0))
    time_power = float(cfg.get("time_power", alpha))
    amplitude = float(cfg.get("amplitude", 1.0))
    phase = float(cfg.get("phase", 0.0))

    argument = frequency * np.pi * x + phase
    time_factor = amplitude * (t ** time_power)

    if family == "cosine":
        return time_factor * torch.cos(argument)
    if family == "sine":
        return time_factor * torch.sin(argument)

    raise ValueError(f"Unsupported solution family '{family}'")


def source_term(
    x: torch.Tensor,
    t: torch.Tensor,
    alpha: float,
    beta: float,
    solution_cfg: dict | None = None,
) -> torch.Tensor:
    """Compute the forcing term consistent with the configured exact solution."""
    cfg = resolve_solution_config(solution_cfg, alpha)
    family = str(cfg.get("family", "cosine")).lower()
    frequency = float(cfg.get("spatial_frequency", 1.0))
    time_power = float(cfg.get("time_power", alpha))
    amplitude = float(cfg.get("amplitude", 1.0))
    phase = float(cfg.get("phase", 0.0))

    argument = frequency * np.pi * x + phase
    trig = torch.cos(argument) if family == "cosine" else torch.sin(argument)

    gamma_time = gamma(time_power + 1)
    gamma_time_fractional = gamma(time_power + 1 - alpha)
    gamma_kernel = gamma(1 - beta)
    gamma_kernel_ratio = gamma(time_power + 2 - beta)

    derivative_term = amplitude * (gamma_time / gamma_time_fractional) * (t ** (time_power - alpha))
    diffusion_term = amplitude * (x**2 + 1) * (frequency * np.pi) ** 2 * (t ** time_power)
    integral_term = (
        amplitude
        * torch.sin(x)
        * (t ** (time_power + 1 - beta))
        * (gamma_time * gamma_kernel / gamma_kernel_ratio)
    )

    return trig * (derivative_term + diffusion_term + integral_term)


class IntegroDifferentialResidual:
    """
    Computes PDE residual for time-fractional integro-differential equation.
    
    Uses:
    - L1 scheme for Caputo fractional derivative
    - Automatic differentiation for spatial derivatives
    - Composite quadrature for weakly singular integral
    """
    
    def __init__(self, model, mesh, l1_coeffs, alpha: float, beta: float,
                 solution_cfg: dict | None = None,
                 n_quad: int = 20, device: str = "cpu"):
        self.model = model
        self.mesh = mesh
        self.l1_coeffs = l1_coeffs
        self.alpha = alpha
        self.beta = beta
        self.solution_cfg = resolve_solution_config(solution_cfg, alpha)
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
        """Evaluate the configured exact solution."""
        return exact_solution(x, t, self.alpha, self.solution_cfg)
    
    def source_term(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Return the forcing term matching the configured exact solution."""
        return source_term(x, t, self.alpha, self.beta, self.solution_cfg)
    
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
        
        Uses KERNEL-AWARE PRODUCT INTEGRATION on the graded mesh.
        
        Idea: On each sub-interval [t_j, t_{j+1}], approximate u(x,s) linearly
        and integrate the singular kernel (t_n - s)^{-β} analytically:
        
            ∫_{t_j}^{t_{j+1}} (t_n - s)^{-β} u(x,s) ds
              ≈ u(x, t_j) · w_j^L  +  u(x, t_{j+1}) · w_j^R
        
        where the weights are computed by integrating the kernel against
        the piecewise-linear basis functions exactly:
        
            w_j^L = (1/h_j) ∫_{t_j}^{t_{j+1}} (t_n - s)^{-β} (t_{j+1} - s) ds
            w_j^R = (1/h_j) ∫_{t_j}^{t_{j+1}} (t_n - s)^{-β} (s - t_j) ds
        
        With substitution τ = t_n - s, letting a = t_n - t_j, b = t_n - t_{j+1}:
        
            w_j^L = (1/h_j) [ (a^{2-β} - b^{2-β})/(2-β) - b·(a^{1-β} - b^{1-β})/(1-β) ]
            w_j^R = (1/h_j) [ a·(a^{1-β} - b^{1-β})/(1-β) - (a^{2-β} - b^{2-β})/(2-β) ]
        
        Advantages over midpoint rule:
        - Singular kernel handled analytically (no numerical error from singularity)
        - Only the smooth part u(x,s) is approximated
        - Exact for piecewise-linear u (second-order in smooth regions)
        """
        batch_size = x.shape[0]
        result = torch.zeros(batch_size, dtype=torch.float64, device=self.device)
        
        unique_n = torch.unique(n_indices)
        t_nodes = self.mesh.get_nodes()
        beta = self.beta
        one_minus_beta = 1.0 - beta
        two_minus_beta = 2.0 - beta
        
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
            
            # Precompute u at all mesh nodes t_0, t_1, ..., t_n in one batched call
            # Stack all (x, t) pairs: for each of the num_points x-values,
            # we need u evaluated at t_0, t_1, ..., t_n
            with torch.no_grad():
                u_at_nodes = []
                for j_idx in range(n_val + 1):
                    t_j_expanded = t_nodes[j_idx].expand(num_points)
                    u_j = self.model(x_n, t_j_expanded).squeeze()
                    u_at_nodes.append(u_j)
                # u_at_nodes[j] has shape (num_points,) for j = 0, ..., n_val
            
            # Compute integral using product integration weights
            integral_sum = torch.zeros(num_points, dtype=torch.float64, device=self.device)
            
            for j in range(n_val):
                # a = t_n - t_j,  b = t_n - t_{j+1},  h_j = t_{j+1} - t_j = a - b
                a = (t_n - t_nodes[j]).item()
                b = (t_n - t_nodes[j + 1]).item()
                h_j = a - b  # = t_{j+1} - t_j
                
                if h_j < 1e-30:
                    continue
                
                # Compute analytically integrated kernel moments
                a_1 = a ** one_minus_beta  # a^{1-β}
                a_2 = a ** two_minus_beta  # a^{2-β}
                
                if b > 1e-30:
                    b_1 = b ** one_minus_beta  # b^{1-β}
                    b_2 = b ** two_minus_beta  # b^{2-β}
                else:
                    # b = 0 (last interval, j = n-1): b^{1-β} = 0, b^{2-β} = 0
                    b_1 = 0.0
                    b_2 = 0.0
                
                # Product integration weights
                # w_j^L = (1/h_j) [ (a^{2-β} - b^{2-β})/(2-β) - b·(a^{1-β} - b^{1-β})/(1-β) ]
                # w_j^R = (1/h_j) [ a·(a^{1-β} - b^{1-β})/(1-β) - (a^{2-β} - b^{2-β})/(2-β) ]
                moment_1 = (a_1 - b_1) / one_minus_beta   # ∫ τ^{-β} dτ
                moment_2 = (a_2 - b_2) / two_minus_beta   # ∫ τ^{1-β} dτ
                
                b_val = (t_n - t_nodes[j + 1]).item()
                a_val = (t_n - t_nodes[j]).item()
                
                w_left = (moment_2 - b_val * moment_1) / h_j
                w_right = (a_val * moment_1 - moment_2) / h_j
                
                # Accumulate: w_L * u(x, t_j) + w_R * u(x, t_{j+1})
                integral_sum = integral_sum + w_left * u_at_nodes[j] + w_right * u_at_nodes[j + 1]
            
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
    
    def evaluate_integral_convergence(self, x_test: torch.Tensor, t_test: torch.Tensor, 
                                      n_test: torch.Tensor, exact_integral: torch.Tensor) -> dict:
        """
        Evaluate convergence of product integration approximation.
        
        Computes integral approximation error and convergence metrics.
        
        Args:
            x_test: Test spatial points
            t_test: Test temporal points
            n_test: Test time indices
            exact_integral: Exact integral values (for comparison)
            
        Returns:
            dict with L2, Linf errors and convergence info
        """
        with torch.no_grad():
            integral_approx = self.compute_integral_term(x_test, t_test, n_test)
            
            error = (integral_approx - exact_integral).abs()
            l2_error = torch.sqrt((error ** 2).mean())
            linf_error = error.max()
            
            return {
                'integral_l2_error': l2_error.item(),
                'integral_linf_error': linf_error.item(),
                'integral_approx_mean': integral_approx.mean().item(),
                'integral_exact_mean': exact_integral.mean().item(),
            }


class IntegralConvergenceMonitor:
    """
    Monitor convergence of product integration approximation over training.
    
    Tracks how the integral approximation improves as the neural network improves.
    """
    
    def __init__(self, residual_computer: IntegroDifferentialResidual, 
                 mesh, alpha: float, beta: float, device: str = "cpu"):
        self.residual = residual_computer
        self.mesh = mesh
        self.alpha = alpha
        self.beta = beta
        self.device = device
        self.history = {
            'epoch': [],
            'integral_l2_error': [],
            'integral_linf_error': [],
        }
    
    def compute_reference_integral(self, x: torch.Tensor, t: torch.Tensor, 
                                   n_indices: torch.Tensor) -> torch.Tensor:
        """
        Approximate the integral using refined product integration 
        (finer mesh as reference).
        
        Uses the alternative quadrature-based routine as a reference estimate.
        """
        with torch.no_grad():
            return self.residual.compute_integral_term_quadrature(x, t, n_indices)
    
    def log_convergence(self, epoch: int, x_test: torch.Tensor, t_test: torch.Tensor,
                       n_test: torch.Tensor, log_file: str = None):
        """Log integral convergence metrics to file."""
        reference = self.compute_reference_integral(x_test, t_test, n_test)
        metrics = self.residual.evaluate_integral_convergence(
            x_test, t_test, n_test, 
            exact_integral=reference
        )
        
        self.history['epoch'].append(epoch)
        self.history['integral_l2_error'].append(metrics['integral_l2_error'])
        self.history['integral_linf_error'].append(metrics['integral_linf_error'])
        
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Epoch {epoch}: Integral L2={metrics['integral_l2_error']:.6e}, "
                       f"Linf={metrics['integral_linf_error']:.6e}\n")


class BoundaryConditions:
    """
    Boundary conditions for the integro-differential problem.
    
    u(0,t) = t^α
    u(1,t) = -t^α
    """
    
    def __init__(self, model, alpha: float, solution_cfg: dict | None = None, device: str = "cpu"):
        self.model = model
        self.alpha = alpha
        self.solution_cfg = resolve_solution_config(solution_cfg, alpha)
        self.device = device
        
    def left_bc(self, t: torch.Tensor) -> torch.Tensor:
        """Exact boundary value at x = 0."""
        x = torch.zeros_like(t)
        return exact_solution(x, t, self.alpha, self.solution_cfg)
    
    def right_bc(self, t: torch.Tensor) -> torch.Tensor:
        """Exact boundary value at x = 1."""
        x = torch.ones_like(t)
        return exact_solution(x, t, self.alpha, self.solution_cfg)
    
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

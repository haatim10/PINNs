"""Graded Mesh and L1 Discretization Coefficients"""

import torch
import numpy as np
from scipy.special import gamma


class GradedMesh:
    def __init__(self, N: int, beta: float, t_max: float = 1.0, device: str = "cpu"):
        self.N = N
        self.beta = beta
        self.t_max = t_max
        self.device = device
        
        n = torch.arange(N + 1, dtype=torch.float64, device=device)
        self.t_nodes = t_max * (n / N) ** beta
        self.tau = self.t_nodes[1:] - self.t_nodes[:-1]
        
    def get_nodes(self):
        return self.t_nodes


class L1Coefficients:
    def __init__(self, mesh: GradedMesh, alpha: float, device: str = "cpu"):
        self.mesh = mesh
        self.alpha = alpha
        self.device = device
        self.N = mesh.N
        self.gamma_2_minus_alpha = gamma(2 - alpha)
        self._coefficients = {}
        self._precompute_coefficients()
        
    def _precompute_coefficients(self):
        t_nodes = self.mesh.t_nodes.cpu().numpy()
        tau = self.mesh.tau.cpu().numpy()
        alpha = self.alpha
        gamma_val = self.gamma_2_minus_alpha
        
        for n in range(1, self.N + 1):
            t_n = t_nodes[n]
            coeffs = np.zeros(n + 1)
            
            for k in range(1, n + 1):
                t_n_minus_k = t_nodes[n - k]
                t_n_minus_k_plus_1 = t_nodes[n - k + 1]
                tau_n_minus_k_plus_1 = tau[n - k]
                
                numerator = (t_n - t_n_minus_k) ** (1 - alpha) - (t_n - t_n_minus_k_plus_1) ** (1 - alpha)
                denominator = gamma_val * tau_n_minus_k_plus_1
                coeffs[k] = numerator / denominator
                
            self._coefficients[n] = torch.tensor(coeffs, dtype=torch.float64, device=self.device)
    
    def get_coefficients_for_n(self, n: int):
        return self._coefficients[n]

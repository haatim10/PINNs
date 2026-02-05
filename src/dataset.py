"""Dataset Module: Training Data Generation"""

import torch
from dataclasses import dataclass


@dataclass
class TrainingData:
    x_coll: torch.Tensor
    t_coll: torch.Tensor
    n_coll: torch.Tensor
    x_bc: torch.Tensor
    t_bc: torch.Tensor
    u_bc: torch.Tensor
    x_ic: torch.Tensor
    t_ic: torch.Tensor
    u_ic: torch.Tensor


class CollocationDataset:
    """
    Generates training data:
    - Collocation points (interior) for PDE residual
    - Boundary points for BC loss
    - Initial points for IC loss
    """
    
    def __init__(self, mesh, N_x: int, N_collocation: int, N_boundary: int, 
                 N_initial: int, x_min: float = 0.0, x_max: float = 1.0,
                 device: str = "cpu", seed: int = None):
        
        self.mesh = mesh
        self.N_x = N_x
        self.N_t = mesh.N
        self.N_collocation = N_collocation
        self.N_boundary = N_boundary
        self.N_initial = N_initial
        self.x_min = x_min
        self.x_max = x_max
        self.device = device
        
        if seed is not None:
            torch.manual_seed(seed)
            
        self.x_grid = torch.linspace(x_min, x_max, N_x, dtype=torch.float64, device=device)
        self.t_grid = mesh.get_nodes()
        
        # Create interior grid (exclude boundaries and t=0)
        x_interior = self.x_grid[1:-1]
        t_interior = self.t_grid[1:]
        
        X, T = torch.meshgrid(x_interior, t_interior, indexing='ij')
        n_values = torch.arange(1, self.N_t + 1, device=device)
        N_grid = n_values.unsqueeze(0).expand(len(x_interior), -1)
        
        self.x_interior_flat = X.flatten()
        self.t_interior_flat = T.flatten()
        self.n_interior_flat = N_grid.flatten()
        self.total_interior_points = len(self.x_interior_flat)
        
    def sample_collocation_points(self):
        """Randomly sample collocation points from interior grid."""
        indices = torch.randperm(self.total_interior_points, device=self.device)[:self.N_collocation]
        return (self.x_interior_flat[indices], 
                self.t_interior_flat[indices], 
                self.n_interior_flat[indices])
    
    def get_boundary_points(self):
        """Get boundary points at x=0 and x=1."""
        t_indices = torch.randint(1, self.N_t + 1, (self.N_boundary,), device=self.device)
        t = self.t_grid[t_indices]
        
        x_left = torch.zeros(self.N_boundary, dtype=torch.float64, device=self.device)
        x_right = torch.ones(self.N_boundary, dtype=torch.float64, device=self.device)
        
        x = torch.cat([x_left, x_right])
        t = torch.cat([t, t])
        u = torch.zeros_like(x)  # u(0,t) = u(1,t) = 0
        
        return x, t, u
    
    def get_initial_points(self):
        """Get initial condition points at t=0."""
        x_indices = torch.randint(0, self.N_x, (self.N_initial,), device=self.device)
        x = self.x_grid[x_indices]
        t = torch.zeros(self.N_initial, dtype=torch.float64, device=self.device)
        u = torch.zeros_like(x)  # u(x,0) = 0
        return x, t, u
    
    def get_training_data(self, resample_collocation: bool = True):
        """Get all training data."""
        x_coll, t_coll, n_coll = self.sample_collocation_points()
        x_bc, t_bc, u_bc = self.get_boundary_points()
        x_ic, t_ic, u_ic = self.get_initial_points()
        
        return TrainingData(
            x_coll=x_coll, t_coll=t_coll, n_coll=n_coll,
            x_bc=x_bc, t_bc=t_bc, u_bc=u_bc,
            x_ic=x_ic, t_ic=t_ic, u_ic=u_ic
        )

"""Graded mesh generation and L1 scheme coefficients."""

import numpy as np
import torch


class GradedMesh:
    """Graded mesh for fractional PDEs with weak singularity at t=0."""

    def __init__(self, T: float, N: int, r: float = 1.0):
        """
        Initialize graded mesh.

        Args:
            T: Final time
            N: Number of time steps
            r: Grading parameter (r=1 gives uniform mesh, r>1 gives graded mesh)
        """
        self.T = T
        self.N = N
        self.r = r
        self.t = self._generate_mesh()
        self.tau = np.diff(self.t)  # Time step sizes

    def _generate_mesh(self) -> np.ndarray:
        """Generate graded mesh points."""
        j = np.arange(self.N + 1)
        t = self.T * (j / self.N) ** self.r
        return t

    def to_tensor(self, device: str = "cpu") -> torch.Tensor:
        """Convert mesh to PyTorch tensor."""
        return torch.tensor(self.t, dtype=torch.float32, device=device)


def compute_l1_coefficients(alpha: float, mesh: GradedMesh) -> np.ndarray:
    """
    Compute L1 scheme coefficients for Caputo fractional derivative.

    The L1 scheme approximates the Caputo derivative as:
    D_t^alpha u(t_n) ≈ sum_{k=0}^{n-1} a_{n,k} (u(t_{k+1}) - u(t_k))

    Args:
        alpha: Fractional order (0 < alpha < 1)
        mesh: Graded mesh object

    Returns:
        Coefficient matrix a[n, k] for n=1,...,N and k=0,...,n-1
    """
    N = mesh.N
    t = mesh.t
    gamma_factor = 1.0 / np.math.gamma(2 - alpha)

    # Initialize coefficient matrix (upper triangular structure)
    a = np.zeros((N + 1, N + 1))

    for n in range(1, N + 1):
        for k in range(n):
            # L1 coefficient formula
            a[n, k] = gamma_factor * (
                (t[n] - t[k]) ** (1 - alpha) - (t[n] - t[k + 1]) ** (1 - alpha)
            ) / (t[k + 1] - t[k])

    return a


def compute_l1_coefficients_torch(
    alpha: float, t: torch.Tensor, device: str = "cpu"
) -> torch.Tensor:
    """
    Compute L1 coefficients in PyTorch for autodiff compatibility.

    Args:
        alpha: Fractional order
        t: Time mesh tensor of shape (N+1,)
        device: Computation device

    Returns:
        Coefficient tensor
    """
    N = len(t) - 1
    gamma_factor = 1.0 / torch.exp(torch.lgamma(torch.tensor(2 - alpha)))

    a = torch.zeros((N + 1, N + 1), device=device)

    for n in range(1, N + 1):
        for k in range(n):
            tau_k = t[k + 1] - t[k]
            a[n, k] = gamma_factor * (
                (t[n] - t[k]) ** (1 - alpha) - (t[n] - t[k + 1]) ** (1 - alpha)
            ) / tau_k

    return a

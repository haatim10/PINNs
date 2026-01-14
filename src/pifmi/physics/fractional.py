import torch
import math

def generate_graded_mesh(N, beta=2.0, T=1.0):
    """Generates temporal nodes clustered near t=0 (t_n = (n/N)^beta)."""
    n = torch.linspace(0, N, N + 1)
    t = (n / N) ** beta * T
    return t

def compute_l1_coefficients(t, alpha):
    """
    Computes d_{n,k} weights for L1 discretization on graded mesh.
    Matches paper Eq. 3.6: d_{n,k} = [(t_n - t_{k-1})^{1-a} - (t_n - t_k)^{1-a}] / (t_k - t_{k-1})
    """
    N = len(t) - 1
    gamma_const = math.gamma(2 - alpha)
    all_coeffs = [] 

    for n in range(1, N + 1):
        # We need coefficients for k = 1 to n
        n_weights = torch.zeros(n)
        for k in range(1, n + 1):
            dt_k = t[k] - t[k-1]
            # Standard L1 weight formula for non-uniform grids
            term = ((t[n] - t[k-1])**(1-alpha) - (t[n] - t[k])**(1-alpha)) / (gamma_const * dt_k)
            n_weights[k-1] = term
        all_coeffs.append(n_weights)
    return all_coeffs
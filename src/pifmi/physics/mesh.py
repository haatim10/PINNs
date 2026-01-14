import torch

def generate_graded_mesh(N, beta=2.0, T=1.0):
    """
    Generates a non-uniform graded mesh for the temporal domain.
    t_n = (n/N)^beta [cite: 340]
    """
    n = torch.linspace(0, N, N + 1)
    t = (n / N) ** beta * T
    return t
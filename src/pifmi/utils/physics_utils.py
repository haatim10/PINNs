import torch
import math

def get_source_f(x, t, alpha=0.5):
    """f(x,t) based on the prof's note: (Gamma(alpha+1) + pi^2) * sin(pi*x)"""
    g_val = math.gamma(alpha + 1)
    pi = math.pi
    return (g_val + pi**2) * torch.sin(pi * x)

def exact_sol(x, t, alpha=0.5):
    """Target solution: u(x,t) = t^alpha * sin(pi*x)"""
    return (t**alpha) * torch.sin(math.pi * x)
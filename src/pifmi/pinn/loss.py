import torch
import math

def compute_total_loss(model, x_colloc, t_nodes, coeffs, alpha, f_func):
    """
    Computes PI-fMI Loss: PDE Residual + Initial Condition + Boundary Conditions.
    """
    x = x_colloc.clone().detach().requires_grad_(True)
    
    # 1. PRECOMPUTE U at all temporal nodes to build history
    u_hist = [model(x, torch.full_like(x, tn)) for tn in t_nodes]
    
    pde_losses = []
    # Loop from n=1 to N (Caputo derivative is defined for t > 0)
    for n in range(1, len(t_nodes)):
        u_n = u_hist[n]
        
        # Spatial u_xx via Auto-Diff (AD)
        u_x = torch.autograd.grad(u_n, x, torch.ones_like(u_n), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
        
        # CORRECTED L1 Summation: sum_{k=1}^n d_{n,k} * (u_k - u_{k-1})
        d_weights = coeffs[n-1]
        caputo_dt = 0
        for k in range(1, n + 1):
            caputo_dt += d_weights[k-1] * (u_hist[k] - u_hist[k-1])
        
        # PDE Residual: D_t^alpha u - u_xx - f(x, t_n)
        f_n = f_func(x, t_nodes[n], alpha)
        res = caputo_dt - u_xx - f_n
        pde_losses.append(torch.mean(res**2))

    l_pde = torch.mean(torch.stack(pde_losses))
    
    # 2. BOUNDARY CONDITIONS (u=0 at x=0,1)
    # Sample more time points for BCs to ensure stability
    t_bc = torch.linspace(0, 1, 100).view(-1, 1).to(x.device)
    l_bc = torch.mean(model(torch.zeros_like(t_bc), t_bc)**2) + \
           torch.mean(model(torch.ones_like(t_bc), t_bc)**2)
    
    # 3. INITIAL CONDITION (u=0 at t=0)
    l_ic = torch.mean(u_hist[0]**2)
    
    return l_pde + l_bc + l_ic
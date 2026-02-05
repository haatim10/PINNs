#!/usr/bin/env python3
"""Visualization Script for PI-fMI Results"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.model import PINN


def main():
    # Load model
    checkpoint = torch.load('outputs/checkpoints/final_model.pt')
    alpha = checkpoint['config']['problem']['alpha']
    
    model = PINN(hidden_layers=[64,64,64,64], device='cuda')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Plot solution comparison
    N = 100
    x = torch.linspace(0, 1, N, dtype=torch.float64, device='cuda')
    t = torch.linspace(0, 1, N, dtype=torch.float64, device='cuda')
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    with torch.no_grad():
        u_pred = model(X.flatten(), T.flatten()).reshape(N, N).cpu().numpy()
    
    u_exact = (T.cpu().numpy() ** alpha) * np.sin(np.pi * X.cpu().numpy())
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    im0 = axes[0].pcolormesh(T.cpu(), X.cpu(), u_pred, cmap='viridis')
    axes[0].set_title('Predicted')
    axes[0].set_xlabel('t')
    axes[0].set_ylabel('x')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].pcolormesh(T.cpu(), X.cpu(), u_exact, cmap='viridis')
    axes[1].set_title('Exact')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('x')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].pcolormesh(T.cpu(), X.cpu(), np.abs(u_pred-u_exact), cmap='hot')
    axes[2].set_title('Absolute Error')
    axes[2].set_xlabel('t')
    axes[2].set_ylabel('x')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('outputs/solution_comparison.png', dpi=150)
    print('Plot saved to outputs/solution_comparison.png')
    plt.show()


if __name__ == "__main__":
    main()

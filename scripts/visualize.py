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
    
    # Generate grid
    N = 100
    x = torch.linspace(0, 1, N, dtype=torch.float64, device='cuda')
    t = torch.linspace(0, 1, N, dtype=torch.float64, device='cuda')
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    with torch.no_grad():
        u_pred = model(X.flatten(), T.flatten()).reshape(N, N).cpu().numpy()
    
    u_exact = (T.cpu().numpy() ** alpha) * np.sin(np.pi * X.cpu().numpy())
    x_np = x.cpu().numpy()
    t_np = t.cpu().numpy()
    
    # ========== Plot 1: Solution Comparison ==========
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
    
    # ========== Plot 2: Slices at Fixed t ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    t_slices = [0.1, 0.25, 0.5, 0.75, 0.9, 1.0]
    for i, t_val in enumerate(t_slices):
        ax = axes[i // 3, i % 3]
        t_idx = int(t_val * (N - 1))
        
        ax.plot(x_np, u_pred[:, t_idx], 'b-', linewidth=2, label='Predicted')
        ax.plot(x_np, u_exact[:, t_idx], 'r--', linewidth=2, label='Exact')
        ax.set_xlabel('x')
        ax.set_ylabel('u(x, t)')
        ax.set_title(f't = {t_val:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Solution Slices at Fixed t', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/slices_fixed_t.png', dpi=150)
    print('Plot saved to outputs/slices_fixed_t.png')
    
    # ========== Plot 3: Slices at Fixed x ==========
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    
    x_slices = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]
    for i, x_val in enumerate(x_slices):
        ax = axes[i // 3, i % 3]
        x_idx = int(x_val * (N - 1))
        
        ax.plot(t_np, u_pred[x_idx, :], 'b-', linewidth=2, label='Predicted')
        ax.plot(t_np, u_exact[x_idx, :], 'r--', linewidth=2, label='Exact')
        ax.set_xlabel('t')
        ax.set_ylabel('u(x, t)')
        ax.set_title(f'x = {x_val:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Solution Slices at Fixed x', fontsize=14)
    plt.tight_layout()
    plt.savefig('outputs/slices_fixed_x.png', dpi=150)
    print('Plot saved to outputs/slices_fixed_x.png')
    
    # ========== Plot 4: Error Slices ==========
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    error = np.abs(u_pred - u_exact)
    
    # Error along x at different t
    for t_val in [0.25, 0.5, 0.75, 1.0]:
        t_idx = int(t_val * (N - 1))
        axes[0].plot(x_np, error[:, t_idx], linewidth=1.5, label=f't={t_val}')
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('|Error|')
    axes[0].set_title('Error vs x at Fixed t')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Error along t at different x
    for x_val in [0.25, 0.5, 0.75]:
        x_idx = int(x_val * (N - 1))
        axes[1].plot(t_np, error[x_idx, :], linewidth=1.5, label=f'x={x_val}')
    axes[1].set_xlabel('t')
    axes[1].set_ylabel('|Error|')
    axes[1].set_title('Error vs t at Fixed x')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('outputs/error_slices.png', dpi=150)
    print('Plot saved to outputs/error_slices.png')
    
    plt.show()
    print('\nAll plots generated!')


if __name__ == "__main__":
    main()

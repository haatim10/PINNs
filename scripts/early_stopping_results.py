#!/usr/bin/env python
"""
Early Stopping Analysis Script
Finds the epoch with minimum error from training history
and loads the best model to generate results.
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PINN


def analyze_checkpoint_history(checkpoint_dir):
    """
    Analyze checkpoint files to find the best epoch.
    Returns the epoch with minimum L2 error.
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    # Try to find checkpoint with history
    best_models = {}
    
    for checkpoint_file in sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt')):
        try:
            checkpoint = torch.load(checkpoint_file, weights_only=False, map_location='cpu')
            if 'history' in checkpoint:
                history = checkpoint['history']
                if history['l2']:  # Has evaluation history
                    best_models[checkpoint_file.name] = {
                        'epoch': checkpoint['epoch'],
                        'file': checkpoint_file,
                        'final_l2': history['l2'][-1],
                        'final_linf': history['linf'][-1],
                        'min_l2': min(history['l2']),
                        'min_linf': min(history['linf']),
                        'best_l2_epoch': history['epochs'][history['l2'].index(min(history['l2']))],
                        'best_linf_epoch': history['epochs'][history['linf'].index(min(history['linf']))],
                    }
        except Exception as e:
            print(f"Could not load {checkpoint_file}: {e}")
            pass
    
    return best_models


def load_best_checkpoint(checkpoint_dir, metric='l2'):
    """
    Load checkpoint with best minimum error.
    metric: 'l2' or 'linf'
    """
    checkpoint_dir = Path(checkpoint_dir)
    best_models = analyze_checkpoint_history(checkpoint_dir)
    
    if not best_models:
        print("No checkpoints with history found!")
        return None, None
    
    # Find best across all checkpoints
    metric_key = f'min_{metric}'
    best_ckpt_name = min(best_models.keys(), key=lambda x: best_models[x][metric_key])
    best_model_info = best_models[best_ckpt_name]
    
    print("\n" + "="*70)
    print(f"EARLY STOPPING RESULTS - Best {metric.upper()} Error")
    print("="*70)
    print(f"\nCheckpoint analyzed: {best_ckpt_name}")
    print(f"Checkpoint epoch: {best_model_info['epoch']}")
    print(f"\nMinimum L2 Error: {best_model_info['min_l2']:.6e} at epoch {best_model_info['best_l2_epoch']}")
    print(f"Minimum Linf Error: {best_model_info['min_linf']:.6e} at epoch {best_model_info['best_linf_epoch']}")
    print(f"\nFinal L2 Error (epoch {best_model_info['epoch']}): {best_model_info['final_l2']:.6e}")
    print(f"Final Linf Error (epoch {best_model_info['epoch']}): {best_model_info['final_linf']:.6e}")
    
    if metric == 'l2':
        improvement = (best_model_info['final_l2'] - best_model_info['min_l2']) / best_model_info['min_l2'] * 100
        print(f"\nL2 Error increased by {improvement:.2f}% after best epoch")
    else:
        improvement = (best_model_info['final_linf'] - best_model_info['min_linf']) / best_model_info['min_linf'] * 100
        print(f"Linf Error increased by {improvement:.2f}% after best epoch")
    
    # Load the final checkpoint (has all the history)
    checkpoint = torch.load(best_model_info['file'], weights_only=False)
    
    print("\n" + "="*70)
    
    return checkpoint, best_model_info


def compute_errors(model, alpha, device, N_x=100, N_t=100):
    """Compute relative L2 and Linf errors against exact solution."""
    model.eval()
    x = torch.linspace(0, 1, N_x, dtype=torch.float64, device=device)
    t = torch.linspace(0.01, 1, N_t, dtype=torch.float64, device=device)  # Avoid t=0
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    with torch.no_grad():
        u_pred = model(X.flatten(), T.flatten()).reshape(X.shape)
        u_exact = (T ** alpha) * torch.cos(np.pi * X)
        
        error = torch.abs(u_pred - u_exact)
        l2_error = (torch.norm(error) / torch.norm(u_exact)).item()
        linf_error = torch.max(error).item()
    
    return l2_error, linf_error, u_pred, u_exact, error


def plot_error_history(checkpoint, results_dir):
    """Plot training history highlighting best epochs."""
    history = checkpoint.get('history', {})
    
    if not history.get('epochs'):
        print("No history available in checkpoint")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = history['epochs']
    l2_errors = history['l2']
    linf_errors = history['linf']
    
    # Find best epochs
    best_l2_idx = l2_errors.index(min(l2_errors))
    best_linf_idx = linf_errors.index(min(linf_errors))
    
    best_l2_epoch = epochs[best_l2_idx]
    best_linf_epoch = epochs[best_linf_idx]
    
    # L2 Error
    ax1.semilogy(epochs, l2_errors, 'b-o', linewidth=2, markersize=6, label='L2 Error')
    ax1.semilogy([best_l2_epoch], [l2_errors[best_l2_idx]], 'r*', markersize=20, 
                  label=f'Best L2 @ epoch {best_l2_epoch}')
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('L2 Error', fontsize=12)
    ax1.set_title('L2 Error vs Epoch', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Linf Error
    ax2.semilogy(epochs, linf_errors, 'g-o', linewidth=2, markersize=6, label='Linf Error')
    ax2.semilogy([best_linf_epoch], [linf_errors[best_linf_idx]], 'r*', markersize=20,
                  label=f'Best Linf @ epoch {best_linf_epoch}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Linf Error', fontsize=12)
    ax2.set_title('Linf Error vs Epoch', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'error_history_best_epochs.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error history plot to {results_dir / 'error_history_best_epochs.png'}")


def generate_bestepoch_plots(model, alpha, results_dir, device, u_pred, u_exact, error):
    """Generate visualization plots for best epoch."""
    
    model.eval()
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    x = np.linspace(0, 1, 100)
    t = np.linspace(0.01, 1, 100)
    X, T = np.meshgrid(x, t)
    
    u_pred_np = u_pred.cpu().numpy()
    u_exact_np = u_exact.cpu().numpy()
    error_np = error.cpu().numpy()
    
    # 1. Solution Comparison (2D heatmaps)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    
    im0 = axes[0].pcolormesh(X, T, u_exact_np, cmap='viridis', shading='auto')
    axes[0].set_xlabel('x', fontsize=11)
    axes[0].set_ylabel('t', fontsize=11)
    axes[0].set_title('Exact Solution', fontsize=12, fontweight='bold')
    plt.colorbar(im0, ax=axes[0])
    
    im1 = axes[1].pcolormesh(X, T, u_pred_np, cmap='viridis', shading='auto')
    axes[1].set_xlabel('x', fontsize=11)
    axes[1].set_ylabel('t', fontsize=11)
    axes[1].set_title('PINN Prediction (Best Epoch)', fontsize=12, fontweight='bold')
    plt.colorbar(im1, ax=axes[1])
    
    im2 = axes[2].pcolormesh(X, T, error_np, cmap='hot', shading='auto')
    axes[2].set_xlabel('x', fontsize=11)
    axes[2].set_ylabel('t', fontsize=11)
    axes[2].set_title(f'Absolute Error (max={error_np.max():.4f})', fontsize=12, fontweight='bold')
    plt.colorbar(im2, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(results_dir / 'best_epoch_solution_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved solution comparison to {results_dir / 'best_epoch_solution_comparison.png'}")
    
    # 2. Slices at fixed t
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    t_vals = [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
    
    for i, t_val in enumerate(t_vals):
        ax = axes[i // 3, i % 3]
        t_idx = int(t_val * 99)
        ax.plot(x, u_exact_np[t_idx, :], 'b-', label='Exact', linewidth=2)
        ax.plot(x, u_pred_np[t_idx, :], 'r--', label='Predicted', linewidth=2)
        ax.fill_between(x, u_exact_np[t_idx, :], u_pred_np[t_idx, :], alpha=0.2, color='orange')
        ax.set_xlabel('x', fontsize=10)
        ax.set_ylabel('u(x,t)', fontsize=10)
        ax.set_title(f't = {t_val:.1f}', fontsize=11, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'best_epoch_slices_fixed_t.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved time slices to {results_dir / 'best_epoch_slices_fixed_t.png'}")
    
    # 3. Error distribution
    fig, ax = plt.subplots(figsize=(12, 5))
    error_flat = error_np.flatten()
    ax.hist(error_flat, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
    ax.axvline(error_flat.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {error_flat.mean():.2e}')
    ax.axvline(np.median(error_flat), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(error_flat):.2e}')
    ax.set_xlabel('Absolute Error', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Error Distribution (Best Epoch)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(results_dir / 'best_epoch_error_distribution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved error distribution to {results_dir / 'best_epoch_error_distribution.png'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Early Stopping Analysis')
    parser.add_argument('--config', type=str, default='configs/integro_differential.yaml',
                        help='Config file path')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints_integro_diff',
                        help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='outputs/integro_diff_results',
                        help='Results directory')
    parser.add_argument('--metric', type=str, default='l2', choices=['l2', 'linf'],
                        help='Metric to optimize for early stopping')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    alpha = config['problem']['alpha']
    
    # Find and load best checkpoint
    checkpoint, best_info = load_best_checkpoint(args.checkpoint_dir, metric=args.metric)
    
    if checkpoint is None:
        print("Error: Could not load checkpoint")
        return
    
    # Load model
    print(f"\nLoading model from epoch {checkpoint['epoch']}...")
    net_cfg = config['network']
    model = PINN(
        input_dim=net_cfg['input_dim'],
        output_dim=net_cfg['output_dim'],
        hidden_layers=net_cfg['hidden_layers'],
        activation=net_cfg['activation'],
        device=device
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print("\n" + "="*70)
    print("Computing Best Epoch Results")
    print("="*70)
    
    # Compute errors for best epoch model
    l2_err, linf_err, u_pred, u_exact, error = compute_errors(model, alpha, device)
    
    print(f"\nBest Epoch ({checkpoint['epoch']}) - High-Resolution Evaluation (100×100 grid):")
    print(f"  L2 Relative Error: {l2_err*100:.4f}%")
    print(f"  Linf Error: {linf_err:.6e}")
    print(f"  Mean Error: {error.mean().item():.6e}")
    print(f"  Median Error: {torch.median(error).item():.6e}")
    print(f"  Std Dev: {error.std().item():.6e}")
    
    # Generate results directory
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    print("\n" + "="*70)
    print("Generating visualizations...")
    print("="*70)
    
    plot_error_history(checkpoint, results_dir)
    generate_bestepoch_plots(model, alpha, results_dir, device, u_pred, u_exact, error)
    
    print("\n" + "="*70)
    print(f"✓ All results saved to {results_dir}")
    print("="*70)
    
    return model, checkpoint, best_info


if __name__ == '__main__':
    main()

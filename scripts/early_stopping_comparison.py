#!/usr/bin/env python
"""
Early Stopping Comparison
Compare models at different epochs (best L2 vs best Linf)
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PINN


def compute_errors(model, alpha, device, N_x=100, N_t=100):
    """Compute relative L2 and Linf errors against exact solution."""
    model.eval()
    x = torch.linspace(0, 1, N_x, dtype=torch.float64, device=device)
    t = torch.linspace(0.01, 1, N_t, dtype=torch.float64, device=device)
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    with torch.no_grad():
        u_pred = model(X.flatten(), T.flatten()).reshape(X.shape)
        u_exact = (T ** alpha) * torch.cos(np.pi * X)
        error = torch.abs(u_pred - u_exact)
        
        l2_error = (torch.norm(error) / torch.norm(u_exact)).item()
        linf_error = torch.max(error).item()
    
    return l2_error, linf_error, u_pred, u_exact, error


def load_and_eval_model(checkpoint_path, config, device):
    """Load model from checkpoint and evaluate."""
    checkpoint = torch.load(checkpoint_path, weights_only=False, map_location=device)
    
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
    
    alpha = config['problem']['alpha']
    l2_err, linf_err, u_pred, u_exact, error = compute_errors(model, alpha, device)
    
    return {
        'epoch': checkpoint['epoch'],
        'model': model,
        'l2_error': l2_err,
        'linf_error': linf_err,
        'u_pred': u_pred,
        'u_exact': u_exact,
        'error': error,
        'checkpoint': checkpoint
    }


def plot_comparison(results_dict, results_dir):
    """Create comprehensive comparison plot."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # 4-panel comparison
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)
    
    x_grid = np.linspace(0, 1, 100)
    t_grid = np.linspace(0.01, 1, 100)
    X, T = np.meshgrid(x_grid, t_grid)
    
    epoch_names = sorted(results_dict.keys(), key=lambda k: results_dict[k]['epoch'])
    
    for col_idx, epoch_name in enumerate(epoch_names):
        r = results_dict[epoch_name]
        u_pred = r['u_pred'].cpu().numpy()
        u_exact = r['u_exact'].cpu().numpy()
        error = r['error'].cpu().numpy()
        
        # Predicted solution
        ax = fig.add_subplot(gs[0, col_idx])
        im = ax.pcolormesh(X, T, u_pred, cmap='viridis', shading='auto')
        ax.set_title(f"Predicted (Epoch {r['epoch']})\nL2: {r['l2_error']*100:.4f}%, Linf: {r['linf_error']:.4e}", 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('t' if col_idx == 0 else '')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Error map
        ax = fig.add_subplot(gs[1, col_idx])
        im = ax.pcolormesh(X, T, error, cmap='hot', shading='auto', norm=plt.matplotlib.colors.LogNorm())
        ax.set_title(f"Error Map (log scale)\nMax: {error.max():.4e}, Mean: {error.mean():.4e}", 
                    fontsize=11, fontweight='bold')
        ax.set_ylabel('t' if col_idx == 0 else '')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        # Slice at t=0.5
        ax = fig.add_subplot(gs[2, col_idx])
        t_idx = 50  # t ≈ 0.5
        ax.plot(x_grid, u_exact[t_idx, :], 'b-', linewidth=2.5, label='Exact', zorder=3)
        ax.plot(x_grid, u_pred[t_idx, :], 'r--', linewidth=2.5, label='Predicted', zorder=2)
        ax.fill_between(x_grid, u_exact[t_idx, :], u_pred[t_idx, :], alpha=0.2, color='orange')
        ax.set_title(f"Slice at t ≈ 0.5", fontsize=11, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('u' if col_idx == 0 else '')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    
    # Overall title
    fig.suptitle('Early Stopping Comparison: Best L2 vs Best Linf Epochs', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(results_dir / 'early_stopping_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved comparison plot to {results_dir / 'early_stopping_comparison.png'}")


def create_summary_table(results_dict, results_dir):
    """Create summary table as text."""
    results_dir = Path(results_dir)
    
    summary_text = "\n" + "="*80 + "\n"
    summary_text += "EARLY STOPPING COMPARISON SUMMARY\n"
    summary_text += "="*80 + "\n\n"
    
    # Table header
    summary_text += f"{'Metric':<20} {'Epoch':<12} {'Best L2 Epoch':<20} {'Best Linf Epoch':<20}\n"
    summary_text += "-"*80 + "\n"
    
    results_sorted = sorted(results_dict.items(), key=lambda x: x[1]['epoch'])
    
    best_l2_epoch = results_sorted[0]
    best_linf_epoch = results_sorted[-1] if len(results_sorted) > 1 else results_sorted[0]
    
    # Find which is which
    l2_vals = {name: r['l2_error'] for name, r in results_dict.items()}
    linf_vals = {name: r['linf_error'] for name, r in results_dict.items()}
    
    best_l2_name = min(l2_vals.keys(), key=lambda k: l2_vals[k])
    best_linf_name = min(linf_vals.keys(), key=lambda k: linf_vals[k])
    
    r_l2 = results_dict[best_l2_name]
    r_linf = results_dict[best_linf_name]
    
    # L2 Error
    summary_text += f"{'L2 Error (%):':<20}"
    summary_text += f"{r_l2['l2_error']*100:<12.4f}"
    summary_text += f"({r_l2['epoch']} epochs)   "
    summary_text += f"{r_linf['l2_error']*100:<12.4f} ({r_linf['epoch']} epochs)\n"
    
    # Linf Error
    summary_text += f"{'Linf Error:':<20}"
    summary_text += f"{r_l2['linf_error']:<12.4e}"
    summary_text += f"({r_l2['epoch']} epochs)   "
    summary_text += f"{r_linf['linf_error']:<12.4e} ({r_linf['epoch']} epochs)\n"
    
    # Mean Error
    summary_text += f"{'Mean Error:':<20}"
    summary_text += f"{r_l2['error'].mean().item():<12.4e}"
    summary_text += f"({r_l2['epoch']} epochs)   "
    summary_text += f"{r_linf['error'].mean().item():<12.4e} ({r_linf['epoch']} epochs)\n"
    
    # Median Error
    summary_text += f"{'Median Error:':<20}"
    summary_text += f"{torch.median(r_l2['error']).item():<12.4e}"
    summary_text += f"({r_l2['epoch']} epochs)   "
    summary_text += f"{torch.median(r_linf['error']).item():<12.4e} ({r_linf['epoch']} epochs)\n"
    
    summary_text += "\n" + "="*80 + "\n"
    
    # Recommendations
    summary_text += "EARLY STOPPING RECOMMENDATIONS:\n"
    summary_text += "-"*80 + "\n"
    
    if best_l2_name == best_linf_name:
        summary_text += f"✓ Single Best Epoch: {r_l2['epoch']}\n"
        summary_text += f"  - Minimizes both L2 and Linf errors\n"
        summary_text += f"  - No overfitting detected\n"
    else:
        l2_diff = abs(r_linf['l2_error'] - r_l2['l2_error']) / r_l2['l2_error'] * 100
        linf_diff = abs(r_l2['linf_error'] - r_linf['linf_error']) / r_linf['linf_error'] * 100
        
        summary_text += f"► Best L2 Error: Epoch {r_l2['epoch']}\n"
        summary_text += f"  - L2: {r_l2['l2_error']*100:.4f}%\n"
        summary_text += f"  - Linf: {r_l2['linf_error']:.4e} ({linf_diff:.2f}% worse than best)\n\n"
        
        summary_text += f"► Best Linf Error: Epoch {r_linf['epoch']}\n"
        summary_text += f"  - Linf: {r_linf['linf_error']:.4e}\n"
        summary_text += f"  - L2: {r_linf['l2_error']*100:.4f}% ({l2_diff:.2f}% worse than best)\n\n"
        
        summary_text += f"► Trade-off assessment:\n"
        summary_text += f"  - L2 error increases by {l2_diff:.2f}% if using Linf-best epoch\n"
        summary_text += f"  - Linf error increases by {linf_diff:.2f}% if using L2-best epoch\n"
        
        if l2_diff < linf_diff:
            summary_text += f"  - Recommendation: Use Epoch {r_l2['epoch']} (better L2 trade-off)\n"
        else:
            summary_text += f"  - Recommendation: Use Epoch {r_linf['epoch']} (better Linf trade-off)\n"
    
    summary_text += "="*80 + "\n"
    
    # Print and save
    print(summary_text)
    
    with open(results_dir / 'early_stopping_summary.txt', 'w') as f:
        f.write(summary_text)
    
    print(f"\nSaved summary to {results_dir / 'early_stopping_summary.txt'}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Early Stopping Comparison')
    parser.add_argument('--config', type=str, default='configs/integro_differential.yaml',
                        help='Config file path')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints_integro_diff',
                        help='Checkpoint directory')
    parser.add_argument('--results-dir', type=str, default='outputs/integro_diff_results',
                        help='Results directory')
    parser.add_argument('--epochs', type=int, nargs='+', default=[11500, 20000],
                        help='Epochs to compare')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Comparing epochs: {args.epochs}\n")
    
    # Load and evaluate models
    results_dict = {}
    checkpoint_dir = Path(args.checkpoint_dir)
    
    print("="*80)
    print("Loading and evaluating models...")
    print("="*80 + "\n")
    
    for epoch in args.epochs:
        ckpt_path = checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        if not ckpt_path.exists():
            print(f"Warning: Checkpoint for epoch {epoch} not found: {ckpt_path}")
            continue
        
        print(f"Loading epoch {epoch}...", end=' ', flush=True)
        try:
            results = load_and_eval_model(ckpt_path, config, device)
            results_dict[f'epoch_{epoch}'] = results
            print(f"✓ (L2: {results['l2_error']*100:.4f}%, Linf: {results['linf_error']:.4e})")
        except Exception as e:
            print(f"✗ Error: {e}")
    
    if not results_dict:
        print("Error: No models loaded successfully!")
        return
    
    # Generate results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("Generating comparison plots...")
    print("="*80 + "\n")
    
    plot_comparison(results_dict, results_dir)
    create_summary_table(results_dict, results_dir)
    
    print("\n" + "="*80)
    print(f"✓ All results saved to {results_dir}")
    print("="*80)


if __name__ == '__main__':
    main()

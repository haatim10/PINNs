#!/usr/bin/env python
"""
Early Stopping Recommendation Engine
Analyzes all checkpoints and recommends optimal stopping point
"""

import torch
import numpy as np
import yaml
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PINN


def analyze_all_checkpoints(checkpoint_dir):
    """Extract error metrics from all checkpoints.
    
    Note: History values stored in checkpoints are RMSE (not relative L2).
    Epoch rankings are the same for both metrics since relative L2 = RMSE / ||u_exact||
    and ||u_exact|| is constant. Displayed values are RMSE; multiply by ~1.98 for
    approximate relative L2 percentage.
    """
    checkpoint_dir = Path(checkpoint_dir)
    all_data = {}
    
    for checkpoint_file in sorted(checkpoint_dir.glob('checkpoint_epoch_*.pt')):
        try:
            checkpoint = torch.load(checkpoint_file, weights_only=False, map_location='cpu')
            if 'history' in checkpoint:
                history = checkpoint['history']
                if history['l2']:
                    epoch = checkpoint['epoch']
                    all_data[epoch] = {
                        'l2': history['l2'][-1],
                        'linf': history['linf'][-1],
                        'loss': history['loss'][-1] if history['loss'] else None,
                        'checkpoint_path': checkpoint_file
                    }
        except:
            pass
    
    return all_data


def compute_recommendation_scores(all_data):
    """Compute different scoring metrics for early stopping recommendation."""
    
    if not all_data:
        return None
    
    epochs = sorted(all_data.keys())
    l2_vals = np.array([all_data[e]['l2'] for e in epochs])
    linf_vals = np.array([all_data[e]['linf'] for e in epochs])
    
    min_l2_idx = np.argmin(l2_vals)
    min_linf_idx = np.argmin(linf_vals)
    
    min_l2_epoch = epochs[min_l2_idx]
    min_linf_epoch = epochs[min_linf_idx]
    
    # Compute normalized metrics (0-1 scale)
    l2_norm = (l2_vals - l2_vals.min()) / (l2_vals.max() - l2_vals.min() + 1e-10)
    linf_norm = (linf_vals - linf_vals.min()) / (linf_vals.max() - linf_vals.min() + 1e-10)
    
    # Weighted combined score (equal weight L2 and Linf)
    combined_score = 0.5 * l2_norm + 0.5 * linf_norm
    best_combined_idx = np.argmin(combined_score)
    best_combined_epoch = epochs[best_combined_idx]
    
    # Find "sweet spot" - minimum error increase after improvement plateaus
    # Look for the point where improvements become marginal (within 1% of best)
    l2_threshold = all_data[min_l2_epoch]['l2'] * 1.01
    linf_threshold = all_data[min_linf_epoch]['linf'] * 1.01
    
    sweet_spot_epoch = min_l2_epoch
    for epoch in epochs:
        if all_data[epoch]['l2'] <= l2_threshold and all_data[epoch]['linf'] <= linf_threshold:
            if epoch < sweet_spot_epoch:
                sweet_spot_epoch = epoch
    
    return {
        'epochs': epochs,
        'l2_values': l2_vals,
        'linf_values': linf_vals,
        'min_l2_epoch': min_l2_epoch,
        'min_linf_epoch': min_linf_epoch,
        'min_l2_value': all_data[min_l2_epoch]['l2'],
        'min_linf_value': all_data[min_linf_epoch]['linf'],
        'best_combined_epoch': best_combined_epoch,
        'best_combined_score': combined_score[best_combined_idx],
        'sweet_spot_epoch': sweet_spot_epoch,
        'all_data': all_data
    }


def print_recommendations(analysis):
    """Print comprehensive early stopping recommendations."""
    
    print("\n" + "="*80)
    print("EARLY STOPPING ANALYSIS AND RECOMMENDATIONS")
    print("="*80 + "\n")
    
    data = analysis['all_data']
    
    print("KEY FINDINGS:")
    print("-"*80)
    print(f"Total checkpoints analyzed: {len(analysis['epochs'])}")
    print(f"Epoch range: {analysis['epochs'][0]} - {analysis['epochs'][-1]}\n")
    
    # Best L2
    l2_best_epoch = analysis['min_l2_epoch']
    l2_best_val = analysis['min_l2_value']
    print(f"► Minimum L2 Error: Epoch {l2_best_epoch}")
    print(f"   L2: {l2_best_val*100:.4f}% = {l2_best_val:.6e}")
    print(f"   Linf: {data[l2_best_epoch]['linf']:.6e}\n")
    
    # Best Linf
    linf_best_epoch = analysis['min_linf_epoch']
    linf_best_val = analysis['min_linf_value']
    print(f"► Minimum Linf Error: Epoch {linf_best_epoch}")
    print(f"   Linf: {linf_best_val:.6e}")
    print(f"   L2: {data[linf_best_epoch]['l2']*100:.4f}% = {data[linf_best_epoch]['l2']:.6e}\n")
    
    # Best combined
    combined_epoch = analysis['best_combined_epoch']
    print(f"► Best Combined (L2 + Linf): Epoch {combined_epoch}")
    print(f"   L2: {data[combined_epoch]['l2']*100:.4f}%")
    print(f"   Linf: {data[combined_epoch]['linf']:.6e}\n")
    
    # Sweet spot
    sweet_epoch = analysis['sweet_spot_epoch']
    print(f"► Early Enough to Stop (≤1% error increase): Epoch {sweet_epoch}")
    print(f"   L2: {data[sweet_epoch]['l2']*100:.4f}%")
    print(f"   Linf: {data[sweet_epoch]['linf']:.6e}\n")
    
    print("="*80)
    print("RECOMMENDATIONS BY USE CASE:")
    print("-"*80)
    
    print(f"\n1. Minimize Average Error (L2):")
    print(f"   ➜ Use Epoch {l2_best_epoch} (L2 = {l2_best_val*100:.4f}%)")
    print(f"     - Best for MSE/regression metrics")
    print(f"     - Linf is {(data[l2_best_epoch]['linf']/linf_best_val - 1)*100:.2f}% worse than minimum")
    
    print(f"\n2. Minimize Maximum Error (Linf):")
    print(f"   ➜ Use Epoch {linf_best_epoch} (Linf = {linf_best_val:.6e})")
    print(f"     - Best for worst-case error bounds")
    print(f"     - L2 is {(data[linf_best_epoch]['l2']/l2_best_val - 1)*100:.2f}% worse than minimum")
    
    print(f"\n3. Balanced Performance (Recommended):")
    print(f"   ➜ Use Epoch {combined_epoch}")
    print(f"     - L2: {data[combined_epoch]['l2']*100:.4f}% ({(data[combined_epoch]['l2']/l2_best_val - 1)*100:.2f}% vs best)")
    print(f"     - Linf: {data[combined_epoch]['linf']:.6e} ({(data[combined_epoch]['linf']/linf_best_val - 1)*100:.2f}% vs best)")
    
    print(f"\n4. Early Stopping (Save Time):")
    print(f"   ➜ Use Epoch {sweet_epoch}")
    print(f"     - Training time saved: ~{(1 - sweet_epoch/analysis['epochs'][-1])*100:.1f}%")
    print(f"     - L2: {data[sweet_epoch]['l2']*100:.4f}% ({(data[sweet_epoch]['l2']/l2_best_val - 1)*100:.2f}% vs best)")
    print(f"     - Linf: {data[sweet_epoch]['linf']:.6e} ({(data[sweet_epoch]['linf']/linf_best_val - 1)*100:.2f}% vs best)")
    
    print("\n" + "="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Early Stopping Recommendation Engine')
    parser.add_argument('--checkpoint-dir', type=str, default='outputs/checkpoints_integro_diff',
                        help='Checkpoint directory')
    
    args = parser.parse_args()
    
    print("Analyzing all checkpoints...")
    all_data = analyze_all_checkpoints(args.checkpoint_dir)
    
    if not all_data:
        print("No checkpoints with history found!")
        return
    
    analysis = compute_recommendation_scores(all_data)
    print_recommendations(analysis)


if __name__ == '__main__':
    main()

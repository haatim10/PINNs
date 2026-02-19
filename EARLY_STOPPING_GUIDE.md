# Early Stopping Analysis Guide

## Overview

We've implemented a comprehensive early stopping analysis toolkit to identify the optimal training epoch for your fractional integro-differential PINN. Instead of always training to the full 20,000 epochs, you can now use these tools to find where performance is truly optimal.

## Key Findings

### Results Summary

| Metric | Best L2 Epoch | Best Linf Epoch | Optimal Balance |
|--------|---------------|-----------------|-----------------|
| **Epoch** | 20,000 | 11,500 | **17,500** ⭐ |
| **L2 Error** | **1.52%** | 2.80% | 1.60% |
| **Linf Error** | 6.0594e-02 | **5.0698e-02** | 5.1197e-02 |
| **Training Time** | Full (32 hrs) | 57.5% complete | 87.5% complete |

### Interpretation

**L2 Error (Relative L2 Norm):**
- Continues improving until epoch 20,000
- Final value: 1.52% (excellent accuracy)
- Training through full epochs is worthwhile

**Linf Error (Maximum Error):**
- Best at epoch 11,500
- Slightly degrades by 19.5% at epoch 20,000
- But L2 improves by 83.5% in the same period

**Sweet Spot: Epoch 17,500**
- Balances both metrics
- L2 only 5.06% worse than absolute minimum
- Linf only 0.98% worse than absolute minimum  
- **Saves ~12.5% training time** while maintaining excellent performance

## Usage Guide

### 1. For New Training Runs - Implement Early Stopping

To avoid training for 32 hours every time, apply early stopping at epoch 17,500:

```python
# In your training loop, add:
if epoch >= 17500:
    print(f"Epoch {epoch}: Reached early stopping point")
    break
```

Or use this command to train with early stopping:

```bash
python scripts/train_integro_diff.py --config configs/integro_differential.yaml --early-stop-epoch 17500
```

### 2. Analyze Existing Training

Find the best epoch in your current training:

```bash
# Get detailed analysis
python scripts/early_stopping_recommendation.py --checkpoint-dir outputs/checkpoints_integro_diff

# Compare two specific epochs
python scripts/early_stopping_comparison.py --config configs/integro_differential.yaml \
  --checkpoint-dir outputs/checkpoints_integro_diff --epochs 11500 20000

# Get best epoch results
python scripts/early_stopping_results.py --config configs/integro_differential.yaml \
  --checkpoint-dir outputs/checkpoints_integro_diff --results-dir outputs/integro_diff_results
```

### 3. Load Best Model

```python
import torch
from src.model import PINN

# Load the best balanced epoch
checkpoint_path = 'outputs/checkpoints_integro_diff/checkpoint_epoch_17500.pt'
checkpoint = torch.load(checkpoint_path)

# Create model
model = PINN(input_dim=2, output_dim=1, hidden_layers=[64]*4, activation='tanh')
model.load_state_dict(checkpoint['model_state_dict'])

# You now have the best model loaded!
```

## Scripts Included

### `early_stopping_results.py`
**Purpose:** Find the best epoch by a single metric (L2 or Linf)

**Usage:**
```bash
python scripts/early_stopping_results.py \
  --config configs/integro_differential.yaml \
  --checkpoint-dir outputs/checkpoints_integro_diff \
  --metric l2  # or 'linf'
```

**Output:** 
- Detailed error metrics
- Visualization plots
- Error distribution histogram

### `early_stopping_comparison.py`
**Purpose:** Compare two specific epochs side-by-side

**Usage:**
```bash
python scripts/early_stopping_comparison.py \
  --config configs/integro_differential.yaml \
  --checkpoint-dir outputs/checkpoints_integro_diff \
  --epochs 11500 20000
```

**Output:**
- 3×4 comparison grid (solution, error, slices)
- Summary table
- Trade-off analysis

### `early_stopping_recommendation.py`
**Purpose:** Comprehensive analysis of ALL checkpoints

**Usage:**
```bash
python scripts/early_stopping_recommendation.py \
  --checkpoint-dir outputs/checkpoints_integro_diff
```

**Output:**
- 4 different recommendations:
  1. For minimizing M.S.E. (L2)
  2. For minimizing worst-case error (Linf)
  3. For balanced performance ⭐
  4. For early stopping (save time)

## Visualizations Generated

### Error History
- Plots showing L2 and Linf error vs epoch
- Marked minimum error points
- Helps visualize convergence behavior

### Solution Comparison
- Side-by-side of exact vs predicted solutions
- Error heatmaps
- Slice comparisons at fixed time values

### Error Distribution
- Histogram of absolute errors
- Mean and median markers
- Shows error variability

## Recommendation Matrix

Choose your stopping epoch based on your priorities:

| Priority | Recommendation | Details |
|----------|-----------------|---------|
| **Maximum Accuracy** | Epoch 20,000 | Best L2 (1.52%), sacrifice some Linf |
| **Minimize Max Error** | Epoch 11,500 | Best Linf (5.07e-02), but L2=2.80% |
| **⭐ Best Balance** | **Epoch 17,500** | **L2=1.60%, Linf=5.12e-02** |
| **Save Time** | Epoch 17,500 | 12.5% faster, near-optimal accuracy |
| **Already Trained** | Epoch 20,000 | Full convergence achieved |

## For Future Implementations

To add early stopping to your training script:

1. **Monitor validation metrics** every N epochs
2. **Track best epoch** for L2 and Linf separately
3. **Set patience threshold** (e.g., stop if no improvement for 2000 epochs)
4. **Save checkpoints** at regular intervals

Example:

```python
best_l2 = float('inf')
patience = 2000
no_improve_count = 0

for epoch in range(1, max_epochs + 1):
    # ... training ...
    
    if epoch % eval_interval == 0:
        l2_err = compute_error(model)
        
        if l2_err < best_l2:
            best_l2 = l2_err
            no_improve_count = 0
            torch.save(model.state_dict(), 'best_model.pt')
        else:
            no_improve_count += 1
        
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

## Conclusion

**Recommended Protocol for Future Training:**
1. Use **Epoch 17,500** as default stopping point
2. Saves ~12.5% training time
3. Achieves 95%+ of maximum possible accuracy
4. Both metrics remain near-optimal

This balances practical concerns (training time) with performance requirements (accuracy).

---

**Generated:** February 19, 2026  
**Branch:** `integro-differential-product-integration`  
**Repository:** https://github.com/haatim10/PINNs

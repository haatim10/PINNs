# PI-fMI-Fractional-PDE

Physics-Informed Neural Network for Fractional Partial Differential Equations using the L1 scheme on graded meshes.

## Problem

Solving the time-fractional diffusion equation:
```
D_t^α u - u_xx = f(x,t)
```
where:
- `f(x,t) = [Γ(α+1) + π² t^α] sin(πx)`
- Exact solution: `u(x,t) = t^α sin(πx)`
- Domain: `x ∈ [0,1]`, `t ∈ [0,1]`
- Boundary conditions: `u(0,t) = u(1,t) = 0`
- Initial condition: `u(x,0) = 0`

## Results

| Metric | Value |
|--------|-------|
| **L2 Relative Error** | 3.84% |
| **L∞ Error** | 6.10% |
| **Training Time** | 94.8 minutes |
| **Epochs** | 4000 |
| **α (fractional order)** | 0.5 |
| **β (mesh grading)** | 2.0 |

## Project Structure

```
PINNs/
├── configs/          # Hyperparameter configurations
├── src/              # Source code modules
├── scripts/          # Training and evaluation scripts
├── notebooks/        # Jupyter notebooks for analysis
├── outputs/          # Generated outputs (checkpoints, figures)
└── tests/            # Unit tests
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Training
```bash
python scripts/train.py --config configs/default.yaml
python scripts/train.py --epochs 4000
python scripts/train.py --resume  # Resume from checkpoint
```

### Visualization
```bash
python scripts/visualize.py
```

## License

MIT

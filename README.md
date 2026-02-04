# PI-fMI-Fractional-PDE

Physics-Informed Neural Network for Fractional Partial Differential Equations using the L1 scheme on graded meshes.

## Project Structure

```
PI-fMI-Fractional-PDE/
├── configs/          # Hyperparameter configurations
├── src/              # Source code modules
├── scripts/          # Training and evaluation scripts
├── notebooks/        # Jupyter notebooks for analysis
├── outputs/          # Generated outputs (checkpoints, logs, figures)
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
```

### Evaluation
```bash
python scripts/evaluate.py --checkpoint outputs/checkpoints/best_model.pt
```

## License

MIT

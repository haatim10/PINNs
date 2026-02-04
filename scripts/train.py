#!/usr/bin/env python3
"""PI-fMI Training Script"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import torch

from src.mesh import GradedMesh, L1Coefficients
from src.model import PINN
from src.dataset import CollocationDataset
from src.loss import PIFMILoss
from src.trainer import Trainer
from src.utils import load_config, set_seed, get_device


def parse_args():
    parser = argparse.ArgumentParser(description='Train PI-fMI model')
    parser.add_argument('--config', type=str, default='configs/default.yaml')
    parser.add_argument('--alpha', type=float, default=None)
    parser.add_argument('--beta', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--device', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    
    config = load_config(args.config)
    
    if args.alpha:
        config.setdefault('problem', {})['alpha'] = args.alpha
    if args.beta:
        config.setdefault('problem', {})['beta'] = args.beta
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.lr:
        config.setdefault('training', {})['learning_rate'] = args.lr
    if args.device:
        config['device'] = args.device
    
    set_seed(config.get('seed', 42))
    device = get_device(config)
    print(f"Using device: {device}")
    
    prob = config.get('problem', {})
    disc = config.get('discretization', {})
    
    alpha = prob.get('alpha', 0.5)
    beta = prob.get('beta', 2.0)
    N_x = disc.get('N_x', 100)
    N_t = disc.get('N_t', 100)
    N_coll = disc.get('N_collocation', 200)
    
    print(f"\nProblem: alpha={alpha}, beta={beta}")
    print(f"Grid: {N_x}x{N_t}, Collocation points: {N_coll}")
    
    mesh = GradedMesh(N_t, beta, t_max=1.0, device=device)
    l1_coeffs = L1Coefficients(mesh, alpha, device=device)
    
    net = config.get('network', {})
    model = PINN(
        hidden_layers=net.get('hidden_layers', [64, 64, 64, 64]),
        activation=net.get('activation', 'tanh'),
        device=device
    )
    print(f"Model parameters: {model.count_parameters()}")
    
    dataset = CollocationDataset(
        mesh=mesh, N_x=N_x, N_collocation=N_coll,
        N_boundary=disc.get('N_boundary', 100),
        N_initial=disc.get('N_initial', 100),
        device=device, seed=config.get('seed', 42)
    )
    
    train_cfg = config.get('training', {})
    loss_fn = PIFMILoss(
        model, mesh, l1_coeffs, alpha,
        weights=train_cfg.get('weights', {'pde': 1.0, 'bc': 10.0, 'ic': 10.0}),
        device=device
    )
    
    trainer = Trainer(model, loss_fn, dataset, config, device)
    results = trainer.train(verbose=True)
    
    print(f"\nFinal L2 Error: {results['final_l2_error']:.6e}")
    print(f"Final Linf Error: {results['final_linf_error']:.6e}")


if __name__ == "__main__":
    main()

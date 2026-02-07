#!/usr/bin/env python
"""
Training Script for Time-Fractional Integro-Differential Equation

PDE: D_t^α u - (x²+1) u_xx + ∫₀ᵗ sin(x)(t-s)^{-β} u(x,s) ds = f(x,t)

Domain: x ∈ [0,1], t ∈ (0,1]
BC: u(0,t) = t^α, u(1,t) = -t^α  
IC: u(x,0) = 0
Exact solution: u(x,t) = t^α cos(πx)
"""

import torch
import torch.optim as optim
import numpy as np
import yaml
import argparse
from pathlib import Path
from tqdm import tqdm
import time
import math
import csv

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PINN
from src.mesh import GradedMesh, L1Coefficients
from src.physics_integro import IntegroDifferentialResidual, BoundaryConditions, InitialCondition


class IntegroDiffDataset:
    """
    Dataset for integro-differential problem with non-homogeneous BCs.
    """
    
    def __init__(self, mesh, config, device="cpu"):
        self.mesh = mesh
        self.device = device
        
        disc = config['discretization']
        prob = config['problem']
        
        self.N_x = disc['N_x']
        self.N_t = disc['N_t']
        self.N_collocation = disc['N_collocation']
        self.N_boundary = disc['N_boundary']
        self.N_initial = disc['N_initial']
        self.alpha = prob['alpha']
        
        self.x_grid = torch.linspace(0, 1, self.N_x, dtype=torch.float64, device=device)
        self.t_grid = mesh.get_nodes()
        
        # Interior grid (exclude boundaries x=0,1 and t=0)
        x_interior = self.x_grid[1:-1]
        t_interior = self.t_grid[1:]
        
        X, T = torch.meshgrid(x_interior, t_interior, indexing='ij')
        n_values = torch.arange(1, self.N_t + 1, device=device)
        N_grid = n_values.unsqueeze(0).expand(len(x_interior), -1)
        
        self.x_interior = X.flatten()
        self.t_interior = T.flatten()
        self.n_interior = N_grid.flatten()
        self.total_interior = len(self.x_interior)
        
    def sample_collocation(self):
        """Sample collocation points from interior."""
        idx = torch.randperm(self.total_interior, device=self.device)[:self.N_collocation]
        return self.x_interior[idx], self.t_interior[idx], self.n_interior[idx]
    
    def get_boundary_points(self):
        """Get boundary points with non-homogeneous BCs."""
        t_idx = torch.randint(1, self.N_t + 1, (self.N_boundary,), device=self.device)
        t = self.t_grid[t_idx]
        
        # Left boundary: x=0, u(0,t) = t^α
        x_left = torch.zeros(self.N_boundary, dtype=torch.float64, device=self.device)
        u_left = t ** self.alpha
        
        # Right boundary: x=1, u(1,t) = -t^α
        x_right = torch.ones(self.N_boundary, dtype=torch.float64, device=self.device)
        u_right = -(t ** self.alpha)
        
        return (torch.cat([x_left, x_right]), 
                torch.cat([t, t]), 
                torch.cat([u_left, u_right]))
    
    def get_initial_points(self):
        """Get IC points: u(x,0) = 0."""
        x_idx = torch.randint(0, self.N_x, (self.N_initial,), device=self.device)
        x = self.x_grid[x_idx]
        t = torch.zeros(self.N_initial, dtype=torch.float64, device=self.device)
        u = torch.zeros(self.N_initial, dtype=torch.float64, device=self.device)
        return x, t, u
    
    def get_training_data(self):
        """Get all training data."""
        x_coll, t_coll, n_coll = self.sample_collocation()
        x_bc, t_bc, u_bc = self.get_boundary_points()
        x_ic, t_ic, u_ic = self.get_initial_points()
        
        return {
            'x_coll': x_coll, 't_coll': t_coll, 'n_coll': n_coll,
            'x_bc': x_bc, 't_bc': t_bc, 'u_bc': u_bc,
            'x_ic': x_ic, 't_ic': t_ic, 'u_ic': u_ic
        }


class IntegroDiffLoss:
    """Loss function for integro-differential equation."""
    
    def __init__(self, model, mesh, l1_coeffs, config, device="cpu"):
        prob = config['problem']
        weights = config['training']['weights']
        
        self.model = model
        self.device = device
        self.weights = weights
        
        self.pde_residual = IntegroDifferentialResidual(
            model, mesh, l1_coeffs,
            alpha=prob['alpha'],
            beta=prob['beta'],
            n_quad=config['discretization'].get('N_integral_quad', 20),
            device=device
        )
        
    def compute(self, data):
        """Compute total loss."""
        # PDE residual loss
        residual = self.pde_residual.compute(data['x_coll'], data['t_coll'], data['n_coll'])
        loss_pde = torch.mean(residual ** 2)
        
        # Boundary condition loss
        u_pred_bc = self.model(data['x_bc'], data['t_bc']).squeeze()
        loss_bc = torch.mean((u_pred_bc - data['u_bc']) ** 2)
        
        # Initial condition loss
        u_pred_ic = self.model(data['x_ic'], data['t_ic']).squeeze()
        loss_ic = torch.mean((u_pred_ic - data['u_ic']) ** 2)
        
        # Total loss
        loss_total = (self.weights['pde'] * loss_pde + 
                      self.weights['bc'] * loss_bc + 
                      self.weights['ic'] * loss_ic)
        
        return {
            'total': loss_total,
            'pde': loss_pde,
            'bc': loss_bc,
            'ic': loss_ic
        }


def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, min_lr):
    """Cosine annealing with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return max(min_lr / optimizer.defaults['lr'], 
                   0.5 * (1 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def compute_errors(model, alpha, device, N_x=100, N_t=100):
    """Compute L2 and Linf errors against exact solution."""
    model.eval()
    x = torch.linspace(0, 1, N_x, dtype=torch.float64, device=device)
    t = torch.linspace(0.01, 1, N_t, dtype=torch.float64, device=device)  # Avoid t=0
    X, T = torch.meshgrid(x, t, indexing='ij')
    
    with torch.no_grad():
        u_pred = model(X.flatten(), T.flatten()).reshape(X.shape)
        u_exact = (T ** alpha) * torch.cos(np.pi * X)
        
        error = torch.abs(u_pred - u_exact)
        l2_error = torch.sqrt(torch.mean(error ** 2)).item()
        linf_error = torch.max(error).item()
    
    model.train()
    return l2_error, linf_error


def train(config_path, resume=False):
    """Main training function."""
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    prob = config['problem']
    disc = config['discretization']
    train_cfg = config['training']
    paths = config.get('paths', {})
    
    alpha = prob['alpha']
    beta = prob['beta']
    N_x = disc['N_x']
    N_t = disc['N_t']
    
    print(f"Problem: alpha={alpha}, beta={beta}")
    print(f"Grid: {N_x}x{N_t}, Collocation points: {disc['N_collocation']}")
    
    # Create mesh and L1 coefficients
    mesh = GradedMesh(N=N_t, T=prob['t_max'], beta=2.0, device=device)
    l1_coeffs = L1Coefficients(mesh, alpha)
    
    # Create model
    net_cfg = config['network']
    model = PINN(
        input_dim=net_cfg['input_dim'],
        output_dim=net_cfg['output_dim'],
        hidden_layers=net_cfg['hidden_layers'],
        activation=net_cfg['activation'],
        device=device
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Create dataset and loss
    dataset = IntegroDiffDataset(mesh, config, device)
    loss_fn = IntegroDiffLoss(model, mesh, l1_coeffs, config, device)
    
    # Optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=train_cfg['learning_rate'])
    
    scheduler_cfg = train_cfg.get('scheduler', {})
    if scheduler_cfg.get('enabled', True) and scheduler_cfg.get('type') == 'cosine_warmup':
        warmup = scheduler_cfg.get('warmup_epochs', 500)
        min_lr = scheduler_cfg.get('min_lr', 1e-5)
        scheduler = get_cosine_warmup_scheduler(optimizer, warmup, train_cfg['epochs'], min_lr)
        print(f"Using Cosine Warmup scheduler: warmup={warmup} epochs, min_lr={min_lr}")
    else:
        scheduler = None
    
    # Setup checkpoints
    checkpoint_dir = Path(paths.get('checkpoint_dir', 'outputs/checkpoints_integro_diff'))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    results_dir = Path(paths.get('results_dir', 'outputs/integro_diff_results'))
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Logging
    log_cfg = config.get('logging', {})
    eval_interval = log_cfg.get('eval_interval', 500)
    checkpoint_interval = log_cfg.get('checkpoint_interval', 1000)
    track_points = log_cfg.get('track_points', False)
    points_file = log_cfg.get('points_file', 'outputs/integro_diff_points.csv')
    
    # Point tracking
    if track_points:
        with open(points_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'point_idx', 'x', 't', 'n'])
        print(f"Point tracking enabled: {points_file}")
    
    # Training history
    history = {'loss': [], 'l2': [], 'linf': [], 'epochs': []}
    start_epoch = 1
    
    # Resume from checkpoint
    if resume:
        ckpt_path = checkpoint_dir / 'latest_checkpoint.pt'
        if ckpt_path.exists():
            checkpoint = torch.load(ckpt_path, weights_only=False)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if scheduler and 'scheduler_state_dict' in checkpoint:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            history = checkpoint.get('history', history)
            print(f"Resumed from epoch {checkpoint['epoch']}")
    
    # Training loop
    print("=" * 60)
    print("Starting Training")
    print(f"Device: {device}")
    print(f"Epochs: {start_epoch} to {train_cfg['epochs']}")
    print(f"Checkpoints saved every {checkpoint_interval} epochs")
    print("=" * 60)
    
    start_time = time.time()
    pbar = tqdm(range(start_epoch, train_cfg['epochs'] + 1), desc="Training")
    
    for epoch in pbar:
        model.train()
        data = dataset.get_training_data()
        
        optimizer.zero_grad()
        losses = loss_fn.compute(data)
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if scheduler:
            scheduler.step()
        
        # Track points
        if track_points and (epoch == 1 or epoch % 1000 == 0):
            with open(points_file, 'a', newline='') as f:
                writer = csv.writer(f)
                for i in range(len(data['x_coll'])):
                    writer.writerow([epoch, i, 
                                     data['x_coll'][i].item(), 
                                     data['t_coll'][i].item(),
                                     data['n_coll'][i].item()])
        
        # Evaluation
        if epoch % eval_interval == 0 or epoch == 1:
            l2_err, linf_err = compute_errors(model, alpha, device)
            history['loss'].append(losses['total'].item())
            history['l2'].append(l2_err)
            history['linf'].append(linf_err)
            history['epochs'].append(epoch)
            
            lr = optimizer.param_groups[0]['lr']
            print(f"\nEpoch {epoch}: Loss={losses['total'].item():.4e}, "
                  f"L2={l2_err:.4e}, Linf={linf_err:.4e}, LR={lr:.2e}")
        
        # Checkpoint
        if epoch % checkpoint_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history,
                'config': config
            }, checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'history': history,
                'config': config
            }, checkpoint_dir / 'latest_checkpoint.pt')
            print(f">>> Checkpoint saved at epoch {epoch}")
        
        pbar.set_postfix({'Loss': f'{losses["total"].item():.2e}', 
                          'LR': f'{optimizer.param_groups[0]["lr"]:.2e}'})
    
    adam_time = (time.time() - start_time) / 60
    print("=" * 60)
    print(f"Adam Training Complete! Time: {adam_time:.1f} minutes")
    
    l2_err, linf_err = compute_errors(model, alpha, device)
    print(f"L2 Error after Adam: {l2_err:.6e}")
    print(f"Linf Error after Adam: {linf_err:.6e}")
    print("=" * 60)
    
    # L-BFGS fine-tuning
    lbfgs_cfg = train_cfg.get('lbfgs', {})
    if lbfgs_cfg.get('enabled', False):
        print("=" * 60)
        print("Starting L-BFGS Fine-tuning")
        print(f"Epochs: {lbfgs_cfg['epochs']}, LR: {lbfgs_cfg['lr']}")
        print("=" * 60)
        
        lbfgs_optimizer = optim.LBFGS(
            model.parameters(),
            lr=lbfgs_cfg['lr'],
            max_iter=lbfgs_cfg.get('max_iter', 20),
            history_size=lbfgs_cfg.get('history_size', 50),
            line_search_fn='strong_wolfe'
        )
        
        pbar_lbfgs = tqdm(range(1, lbfgs_cfg['epochs'] + 1), desc="L-BFGS")
        
        for epoch in pbar_lbfgs:
            data = dataset.get_training_data()
            
            def closure():
                lbfgs_optimizer.zero_grad()
                losses = loss_fn.compute(data)
                losses['total'].backward()
                return losses['total']
            
            loss = lbfgs_optimizer.step(closure)
            
            if epoch % 100 == 0:
                l2_err, linf_err = compute_errors(model, alpha, device)
                print(f"\nL-BFGS Epoch {epoch}: Loss={loss.item():.4e}, "
                      f"L2={l2_err:.4e}, Linf={linf_err:.4e}")
            
            pbar_lbfgs.set_postfix({'Loss': f'{loss.item():.2e}'})
        
        print("L-BFGS Fine-tuning Complete!")
    
    # Final evaluation and save
    l2_err, linf_err = compute_errors(model, alpha, device)
    print("=" * 60)
    print(f"Final L2 Error: {l2_err:.6e}")
    print(f"Final Linf Error: {linf_err:.6e}")
    print("=" * 60)
    
    torch.save({
        'epoch': train_cfg['epochs'],
        'model_state_dict': model.state_dict(),
        'history': history,
        'config': config,
        'final_l2': l2_err,
        'final_linf': linf_err
    }, checkpoint_dir / 'final_model.pt')
    
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/integro_differential.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    
    train(args.config, args.resume)

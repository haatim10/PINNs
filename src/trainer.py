"""Trainer Module with Checkpointing"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, LambdaLR
from tqdm import tqdm
import time
from pathlib import Path
import csv

from .utils import compute_errors


def get_cosine_warmup_scheduler(optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
    """
    Creates a learning rate scheduler with linear warmup followed by cosine annealing.
    """
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            return (epoch + 1) / warmup_epochs
        else:
            # Cosine annealing after warmup
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            return min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)).item())
    
    return LambdaLR(optimizer, lr_lambda)


class Trainer:
    """Training manager with checkpoint support."""
    
    def __init__(self, model, loss_fn, dataset, config, device="cpu"):
        self.model = model
        self.loss_fn = loss_fn
        self.dataset = dataset
        self.config = config
        self.device = device
        
        train_config = config.get('training', {})
        self.epochs = train_config.get('epochs', 10000)
        self.learning_rate = train_config.get('learning_rate', 0.001)
        
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        
        # Scheduler setup with warmup support
        scheduler_config = train_config.get('scheduler', {})
        if scheduler_config.get('enabled', True):
            scheduler_type = scheduler_config.get('type', 'step')
            if scheduler_type == 'cosine_warmup':
                warmup_epochs = scheduler_config.get('warmup_epochs', 500)
                min_lr = scheduler_config.get('min_lr', 0.00001)
                min_lr_ratio = min_lr / self.learning_rate
                self.scheduler = get_cosine_warmup_scheduler(
                    self.optimizer, warmup_epochs, self.epochs, min_lr_ratio
                )
                self.warmup_epochs = warmup_epochs
                print(f"Using Cosine Warmup scheduler: warmup={warmup_epochs} epochs, min_lr={min_lr}")
            else:
                self.scheduler = StepLR(
                    self.optimizer,
                    step_size=scheduler_config.get('step_size', 2000),
                    gamma=scheduler_config.get('gamma', 0.5)
                )
                self.warmup_epochs = 0
        else:
            self.scheduler = None
            self.warmup_epochs = 0
        
        log_config = config.get('logging', {})
        self.eval_interval = log_config.get('eval_interval', 500)
        self.checkpoint_interval = log_config.get('checkpoint_interval', 1000)
        
        # Point tracking setup
        self.track_points = log_config.get('track_points', False)
        self.points_file = log_config.get('points_file', 'outputs/sampled_points_log.csv')
        if self.track_points:
            Path(self.points_file).parent.mkdir(parents=True, exist_ok=True)
            # Initialize the points log file with header
            with open(self.points_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['epoch', 'point_type', 'index', 'x', 't', 'n_or_u_target'])
            print(f"Point tracking enabled: {self.points_file}")
        
        paths = config.get('paths', {})
        self.checkpoint_dir = Path(paths.get('checkpoint_dir', 'outputs/checkpoints'))
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.loss_history = {'total': [], 'pde': [], 'bc': [], 'ic': []}
        self.error_history = {'l2': [], 'linf': []}
        self.eval_epochs = []
        
        prob_config = config.get('problem', {})
        self.alpha = prob_config.get('alpha', 0.5)
        
        disc_config = config.get('discretization', {})
        self.N_x = disc_config.get('N_x', 100)
        self.N_t = disc_config.get('N_t', 100)
        
        self.start_epoch = 1
        
    def save_checkpoint(self, epoch):
        """Save training checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'loss_history': self.loss_history,
            'error_history': self.error_history,
            'eval_epochs': self.eval_epochs,
            'config': self.config,
        }
        torch.save(checkpoint, self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, self.checkpoint_dir / 'latest_checkpoint.pt')
        print(f"\n>>> Checkpoint saved at epoch {epoch}")
        
    def load_checkpoint(self, path):
        """Load training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.loss_history = checkpoint['loss_history']
        self.error_history = checkpoint['error_history']
        self.eval_epochs = checkpoint.get('eval_epochs', [])
        self.start_epoch = checkpoint['epoch'] + 1
        print(f">>> Resumed from epoch {checkpoint['epoch']}")
        
    def train_step(self, data):
        """Single training step."""
        self.model.train()
        self.optimizer.zero_grad()
        losses = self.loss_fn(data)
        losses.total.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        return losses
    
    def evaluate(self):
        """Evaluate model errors."""
        x = torch.linspace(0, 1, self.N_x, dtype=torch.float64, device=self.device)
        t = torch.linspace(0, 1, self.N_t + 1, dtype=torch.float64, device=self.device)
        X, T = torch.meshgrid(x, t, indexing='ij')
        return compute_errors(self.model, X.flatten(), T.flatten(), self.alpha, self.device)
    
    def log_training_points(self, epoch, data):
        """Log all training points for this epoch to CSV."""
        with open(self.points_file, 'a', newline='') as f:
            writer = csv.writer(f)
            
            # Log collocation points
            for i in range(len(data.x_coll)):
                writer.writerow([
                    epoch, 'collocation', i,
                    data.x_coll[i].item(),
                    data.t_coll[i].item(),
                    int(data.n_coll[i].item())
                ])
            
            # Log boundary points
            for i in range(len(data.x_bc)):
                writer.writerow([
                    epoch, 'boundary', i,
                    data.x_bc[i].item(),
                    data.t_bc[i].item(),
                    data.u_bc[i].item()
                ])
            
            # Log initial condition points
            for i in range(len(data.x_ic)):
                writer.writerow([
                    epoch, 'initial', i,
                    data.x_ic[i].item(),
                    data.t_ic[i].item(),
                    data.u_ic[i].item()
                ])
    
    def train(self, verbose=True, resume_path=None):
        """Main training loop."""
        if resume_path and Path(resume_path).exists():
            self.load_checkpoint(resume_path)
        
        print("=" * 60)
        print("Starting Training")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.start_epoch} to {self.epochs}")
        if self.warmup_epochs > 0:
            print(f"Warmup epochs: {self.warmup_epochs}")
        print(f"Checkpoints saved every {self.checkpoint_interval} epochs")
        if self.track_points:
            print(f"Point tracking: {self.points_file}")
        print("=" * 60)
        
        start_time = time.time()
        pbar = tqdm(range(self.start_epoch, self.epochs + 1), desc="Training")
        
        for epoch in pbar:
            data = self.dataset.get_training_data()
            
            # Log training points every 1000 epochs if tracking is enabled
            if self.track_points and (epoch % 1000 == 0 or epoch == 1):
                self.log_training_points(epoch, data)
            
            losses = self.train_step(data)
            
            self.loss_history['total'].append(losses.total.item())
            self.loss_history['pde'].append(losses.pde.item())
            self.loss_history['bc'].append(losses.bc.item())
            self.loss_history['ic'].append(losses.ic.item())
            
            if self.scheduler:
                self.scheduler.step()
            
            if epoch % self.eval_interval == 0 or epoch == 1:
                l2_error, linf_error = self.evaluate()
                self.error_history['l2'].append(l2_error)
                self.error_history['linf'].append(linf_error)
                self.eval_epochs.append(epoch)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                if verbose:
                    print(f"\nEpoch {epoch}: Loss={losses.total.item():.4e}, L2={l2_error:.4e}, Linf={linf_error:.4e}, LR={current_lr:.2e}")
            
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            pbar.set_postfix({'Loss': f'{losses.total.item():.2e}', 'LR': f'{current_lr:.2e}'})
        
        total_time = time.time() - start_time
        
        print("=" * 60)
        print(f"Adam Training Complete! Time: {total_time/60:.1f} minutes")
        print(f"L2 Error after Adam: {self.error_history['l2'][-1]:.6e}")
        print(f"Linf Error after Adam: {self.error_history['linf'][-1]:.6e}")
        print("=" * 60)
        
        # L-BFGS fine-tuning phase
        lbfgs_config = self.config.get('training', {}).get('lbfgs', {})
        if lbfgs_config.get('enabled', False):
            self.train_lbfgs(lbfgs_config, verbose)
        
        # Save final model
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'loss_history': self.loss_history,
            'error_history': self.error_history,
            'eval_epochs': self.eval_epochs,
        }, self.checkpoint_dir / 'final_model.pt')
        
        print("=" * 60)
        print(f"Final L2 Error: {self.error_history['l2'][-1]:.6e}")
        print(f"Final Linf Error: {self.error_history['linf'][-1]:.6e}")
        print("=" * 60)
        
        return {
            'final_l2_error': self.error_history['l2'][-1],
            'final_linf_error': self.error_history['linf'][-1],
            'loss_history': self.loss_history,
            'error_history': self.error_history,
        }
    
    def train_lbfgs(self, lbfgs_config, verbose=True):
        """L-BFGS fine-tuning phase after Adam."""
        lbfgs_epochs = lbfgs_config.get('epochs', 500)
        lbfgs_lr = lbfgs_config.get('lr', 1.0)
        max_iter = lbfgs_config.get('max_iter', 20)
        history_size = lbfgs_config.get('history_size', 50)
        
        print("\n" + "=" * 60)
        print("Starting L-BFGS Fine-tuning")
        print(f"Epochs: {lbfgs_epochs}, LR: {lbfgs_lr}, Max iter: {max_iter}")
        print("=" * 60)
        
        lbfgs_optimizer = optim.LBFGS(
            self.model.parameters(),
            lr=lbfgs_lr,
            max_iter=max_iter,
            history_size=history_size,
            line_search_fn='strong_wolfe'
        )
        
        # Use fixed data for L-BFGS (more stable)
        data = self.dataset.get_training_data()
        
        pbar = tqdm(range(lbfgs_epochs), desc="L-BFGS")
        
        for epoch in pbar:
            def closure():
                lbfgs_optimizer.zero_grad()
                losses = self.loss_fn(data)
                losses.total.backward()
                return losses.total
            
            loss = lbfgs_optimizer.step(closure)
            
            self.loss_history['total'].append(loss.item())
            
            if (epoch + 1) % 100 == 0:
                l2_error, linf_error = self.evaluate()
                self.error_history['l2'].append(l2_error)
                self.error_history['linf'].append(linf_error)
                self.eval_epochs.append(self.epochs + epoch + 1)
                
                if verbose:
                    print(f"\nL-BFGS Epoch {epoch+1}: Loss={loss.item():.4e}, L2={l2_error:.4e}, Linf={linf_error:.4e}")
            
            pbar.set_postfix({'Loss': f'{loss.item():.2e}'})
        
        print(f"\nL-BFGS Fine-tuning Complete!")

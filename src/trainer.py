"""Trainer Module with Checkpointing"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm
import time
from pathlib import Path

from .utils import compute_errors


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
        
        scheduler_config = train_config.get('scheduler', {})
        if scheduler_config.get('enabled', True):
            self.scheduler = StepLR(
                self.optimizer,
                step_size=scheduler_config.get('step_size', 2000),
                gamma=scheduler_config.get('gamma', 0.5)
            )
        else:
            self.scheduler = None
        
        log_config = config.get('logging', {})
        self.eval_interval = log_config.get('eval_interval', 500)
        self.checkpoint_interval = log_config.get('checkpoint_interval', 1000)
        
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
    
    def train(self, verbose=True, resume_path=None):
        """Main training loop."""
        if resume_path and Path(resume_path).exists():
            self.load_checkpoint(resume_path)
        
        print("=" * 60)
        print("Starting Training")
        print(f"Device: {self.device}")
        print(f"Epochs: {self.start_epoch} to {self.epochs}")
        print(f"Checkpoints saved every {self.checkpoint_interval} epochs")
        print("=" * 60)
        
        start_time = time.time()
        pbar = tqdm(range(self.start_epoch, self.epochs + 1), desc="Training")
        
        for epoch in pbar:
            data = self.dataset.get_training_data()
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
                
                if verbose:
                    print(f"\nEpoch {epoch}: Loss={losses.total.item():.4e}, L2={l2_error:.4e}, Linf={linf_error:.4e}")
            
            if epoch % self.checkpoint_interval == 0:
                self.save_checkpoint(epoch)
            
            pbar.set_postfix({'Loss': f'{losses.total.item():.2e}'})
        
        total_time = time.time() - start_time
        
        print("=" * 60)
        print(f"Training Complete! Time: {total_time/60:.1f} minutes")
        print(f"Final L2 Error: {self.error_history['l2'][-1]:.6e}")
        print(f"Final Linf Error: {self.error_history['linf'][-1]:.6e}")
        print("=" * 60)
        
        # Save final model
        torch.save({
            'epoch': self.epochs,
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'loss_history': self.loss_history,
            'error_history': self.error_history,
            'eval_epochs': self.eval_epochs,
        }, self.checkpoint_dir / 'final_model.pt')
        
        return {
            'final_l2_error': self.error_history['l2'][-1],
            'final_linf_error': self.error_history['linf'][-1],
            'loss_history': self.loss_history,
            'error_history': self.error_history,
        }

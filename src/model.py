"""Physics-Informed Neural Network Architecture"""

import torch
import torch.nn as nn
from typing import List


class PINN(nn.Module):
    """
    Fully connected neural network for approximating PDE solutions.
    Input: (x, t) -> Output: u(x, t)
    """
    
    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: List[int] = [64, 64, 64, 64],
        activation: str = "tanh",
        device: str = "cpu"
    ):
        super(PINN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers
        self.device = device
        
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        self.activation = activations.get(activation.lower(), nn.Tanh())
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(self.activation)
            prev_dim = hidden_dim
            
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
        
        self._initialize_weights()
        self.to(device)
        self.double()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        inputs = torch.cat([x, t], dim=-1)
        return self.network(inputs)
    
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

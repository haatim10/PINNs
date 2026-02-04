"""Neural network architecture for PINN."""

import torch
import torch.nn as nn
from typing import List


class PINN(nn.Module):
    """Physics-Informed Neural Network for fractional PDEs."""

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: List[int] = [64, 64, 64],
        activation: str = "tanh",
    ):
        """
        Initialize PINN.

        Args:
            input_dim: Input dimension (typically 2 for (t, x))
            output_dim: Output dimension (typically 1 for u(t, x))
            hidden_layers: List of hidden layer sizes
            activation: Activation function name
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # Build network layers
        layers = []
        in_features = input_dim

        for hidden_size in hidden_layers:
            layers.append(nn.Linear(in_features, hidden_size))
            layers.append(self._get_activation(activation))
            in_features = hidden_size

        layers.append(nn.Linear(in_features, output_dim))

        self.network = nn.Sequential(*layers)

        # Initialize weights
        self._init_weights()

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "sigmoid": nn.Sigmoid(),
        }
        if name.lower() not in activations:
            raise ValueError(f"Unknown activation: {name}")
        return activations[name.lower()]

    def _init_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            t: Time coordinates, shape (N,) or (N, 1)
            x: Spatial coordinates, shape (N,) or (N, 1)

        Returns:
            Network output u(t, x), shape (N, 1)
        """
        # Ensure proper shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)

        # Concatenate inputs
        inputs = torch.cat([t, x], dim=-1)

        return self.network(inputs)

    def forward_with_grads(
        self, t: torch.Tensor, x: torch.Tensor
    ) -> tuple:
        """
        Forward pass with gradient computation.

        Returns:
            u: Network output
            u_t: Partial derivative w.r.t. t
            u_x: Partial derivative w.r.t. x
            u_xx: Second partial derivative w.r.t. x
        """
        t = t.requires_grad_(True)
        x = x.requires_grad_(True)

        u = self.forward(t, x)

        # Compute gradients
        u_t = torch.autograd.grad(
            u, t, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        u_x = torch.autograd.grad(
            u, x, grad_outputs=torch.ones_like(u), create_graph=True
        )[0]

        u_xx = torch.autograd.grad(
            u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True
        )[0]

        return u, u_t, u_x, u_xx

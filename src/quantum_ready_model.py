"""Quantum-ready hybrid PINN components.

This module does not execute a real quantum circuit. It provides a small,
configurable hybrid block that mimics quantum feature mixing while preserving
the same PINN interface: model(x, t) -> u(x, t).
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class QuantumInspiredBlock(nn.Module):
    """A lightweight quantum-inspired variational block.

    The block applies trainable phase shifts and periodic nonlinear mixing,
    followed by nearest-neighbor feature coupling to emulate entanglement-like
    interactions in a differentiable classical layer.
    """

    def __init__(self, n_qubits: int, n_layers: int, entanglement_strength: float = 0.15):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_layers = n_layers
        self.entanglement_strength = entanglement_strength

        self.theta = nn.Parameter(torch.zeros(n_layers, n_qubits))
        nn.init.normal_(self.theta, mean=0.0, std=0.1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        for layer in range(self.n_layers):
            phase = z + self.theta[layer]
            mixed = torch.sin(phase) + 0.5 * torch.cos(phase)
            if self.n_qubits > 1 and self.entanglement_strength != 0.0:
                mixed = mixed + self.entanglement_strength * torch.roll(mixed, shifts=1, dims=-1)
            z = mixed
        return z


class QuantumReadyPINN(nn.Module):
    """Hybrid quantum-ready PINN with the same API as the classical PINN."""

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        quantum: Optional[Dict] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or [64, 64, 64, 64]
        self.device = device

        qcfg = quantum or {}
        self.backend = qcfg.get("backend", "classical_emulator")
        self.n_qubits = int(qcfg.get("n_qubits", 6))
        self.n_layers = int(qcfg.get("n_layers", 2))
        self.feature_scale = float(qcfg.get("feature_scale", 1.0))
        self.entanglement_strength = float(qcfg.get("entanglement_strength", 0.15))
        self.use_residual = bool(qcfg.get("residual_connection", True))

        activations = {
            "tanh": nn.Tanh(),
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
        }
        self.activation = activations.get(str(activation).lower(), nn.Tanh())

        encoder_hidden = int(qcfg.get("encoder_hidden_dim", self.hidden_layers[0]))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, encoder_hidden),
            self.activation,
            nn.Linear(encoder_hidden, self.n_qubits),
        )

        self.variational = QuantumInspiredBlock(
            n_qubits=self.n_qubits,
            n_layers=self.n_layers,
            entanglement_strength=self.entanglement_strength,
        )

        self.skip = nn.Linear(input_dim, self.n_qubits) if self.use_residual else None
        self.readout = nn.Linear(self.n_qubits, output_dim)

        self._initialize_weights()
        self.to(device)
        self.double()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        inputs = torch.cat([x, t], dim=-1)
        z = self.feature_scale * self.encoder(inputs)
        z = self.variational(z)

        if self.skip is not None:
            z = z + torch.tanh(self.skip(inputs))

        return self.readout(z)

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

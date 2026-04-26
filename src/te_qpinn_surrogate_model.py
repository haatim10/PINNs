"""TE-QPINN-inspired quantum surrogate model.

This module implements a fully differentiable PyTorch surrogate inspired by
TE-QPINN ideas:
- trainable embedding FNN phi(input)
- input rescaling to configurable range (default [-0.95, 0.95])
- angle embedding theta_i = phi_i(input) * cycled_coordinate
- sinusoidal feature map with optional pairwise "entanglement-like" terms
- expectation-style readout
- optional residual classical branch
"""

from typing import Dict, List, Optional

import torch
import torch.nn as nn


def _activation_from_name(name: str) -> nn.Module:
    activations = {
        "tanh": nn.Tanh(),
        "relu": nn.ReLU(),
        "gelu": nn.GELU(),
        "silu": nn.SiLU(),
    }
    return activations.get(str(name).lower(), nn.Tanh())


def _coerce_bounds(bounds, input_dim: int, default_value: float) -> torch.Tensor:
    if bounds is None:
        values = [default_value] * input_dim
    elif isinstance(bounds, (int, float)):
        values = [float(bounds)] * input_dim
    else:
        values = list(bounds)
        if len(values) == 1:
            values = values * input_dim
        if len(values) != input_dim:
            raise ValueError(
                f"Expected {input_dim} bounds values, received {len(values)}"
            )
    return torch.tensor(values, dtype=torch.float64)


class InputRescaler(nn.Module):
    """Affine rescaler for coordinate inputs."""

    def __init__(
        self,
        input_dim: int,
        input_mins: torch.Tensor,
        input_maxs: torch.Tensor,
        target_min: float = -0.95,
        target_max: float = 0.95,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.target_min = float(target_min)
        self.target_max = float(target_max)
        self.register_buffer("input_mins", input_mins.reshape(1, input_dim))
        self.register_buffer("input_maxs", input_maxs.reshape(1, input_dim))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(self.input_maxs - self.input_mins, min=1e-12)
        normalized = (inputs - self.input_mins) / denom
        return normalized * (self.target_max - self.target_min) + self.target_min


class TrainableEmbeddingFNN(nn.Module):
    """Small MLP producing trainable embedding scales phi_i(input)."""

    def __init__(
        self,
        input_dim: int,
        n_qubits: int,
        hidden_layers: List[int],
        activation: str,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(_activation_from_name(activation))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, n_qubits))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class TEQPINNSurrogatePINN(nn.Module):
    """TE-QPINN-inspired surrogate preserving PINN forward contract."""

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        te_qpinn: Optional[Dict] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_layers = hidden_layers or [64, 64, 64, 64]
        self.device = device

        cfg = te_qpinn or {}
        self.n_qubits = int(cfg.get("n_qubits", 8))
        self.rescale_min = float(cfg.get("rescale_min", -0.95))
        self.rescale_max = float(cfg.get("rescale_max", 0.95))
        self.use_pairwise_entanglement = bool(cfg.get("use_pairwise_entanglement", True))
        self.use_residual = bool(cfg.get("residual_connection", True))
        self.residual_scale = float(cfg.get("residual_scale", 1.0))
        self.expectation_activation = str(cfg.get("expectation_activation", "tanh")).lower()

        embedding_hidden = cfg.get("embedding_hidden_layers", [32, 32])
        if isinstance(embedding_hidden, int):
            embedding_hidden = [embedding_hidden]
        embedding_hidden = [int(width) for width in embedding_hidden]

        embedding_activation = cfg.get("embedding_activation", activation)

        input_mins = _coerce_bounds(cfg.get("input_mins"), input_dim, 0.0)
        input_maxs = _coerce_bounds(cfg.get("input_maxs"), input_dim, 1.0)

        self.rescaler = InputRescaler(
            input_dim=input_dim,
            input_mins=input_mins,
            input_maxs=input_maxs,
            target_min=self.rescale_min,
            target_max=self.rescale_max,
        )
        self.embedding_fnn = TrainableEmbeddingFNN(
            input_dim=input_dim,
            n_qubits=self.n_qubits,
            hidden_layers=embedding_hidden,
            activation=embedding_activation,
        )

        feature_dim = 2 * self.n_qubits
        if self.use_pairwise_entanglement and self.n_qubits > 1:
            feature_dim += self.n_qubits - 1

        variational_hidden = int(cfg.get("variational_hidden_dim", max(16, feature_dim)))
        self.variational = nn.Sequential(
            nn.Linear(feature_dim, variational_hidden),
            _activation_from_name(cfg.get("variational_activation", "tanh")),
            nn.Linear(variational_hidden, self.n_qubits),
        )
        self.expectation_head = nn.Linear(self.n_qubits, self.n_qubits)
        self.readout = nn.Linear(self.n_qubits, output_dim)

        self.residual_branch = None
        if self.use_residual:
            residual_hidden = int(cfg.get("residual_hidden_dim", max(8, input_dim * 4)))
            self.residual_branch = nn.Sequential(
                nn.Linear(input_dim, residual_hidden),
                _activation_from_name(cfg.get("residual_activation", "tanh")),
                nn.Linear(residual_hidden, output_dim),
            )

        self._initialize_weights()
        self.to(device)
        self.double()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _prepare_inputs(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.cat([x, t], dim=-1)

    def _cycled_coordinates(self, scaled_inputs: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(self.n_qubits, device=scaled_inputs.device) % self.input_dim
        return scaled_inputs[:, indices]

    def _build_quantum_features(self, theta: torch.Tensor) -> torch.Tensor:
        parts = [torch.sin(theta), torch.cos(theta)]
        if self.use_pairwise_entanglement and self.n_qubits > 1:
            pairwise = torch.sin(theta[:, :-1]) * torch.cos(theta[:, 1:])
            parts.append(pairwise)
        return torch.cat(parts, dim=-1)

    def _apply_expectation_activation(self, values: torch.Tensor) -> torch.Tensor:
        if self.expectation_activation == "identity":
            return values
        if self.expectation_activation == "sin":
            return torch.sin(values)
        return torch.tanh(values)

    def compute_embedding(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return scaled inputs, phi, and theta for diagnostics/visualization."""
        inputs = self._prepare_inputs(x, t)
        scaled_inputs = self.rescaler(inputs)
        phi = self.embedding_fnn(scaled_inputs)
        cycled_coord = self._cycled_coordinates(scaled_inputs)
        theta = phi * cycled_coord
        return {
            "inputs_scaled": scaled_inputs,
            "phi": phi,
            "theta": theta,
        }

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedding = self.compute_embedding(x, t)
        theta = embedding["theta"]
        scaled_inputs = embedding["inputs_scaled"]

        q_features = self._build_quantum_features(theta)
        latent = self.variational(q_features)
        expectation = self._apply_expectation_activation(self.expectation_head(latent))
        output = self.readout(expectation)

        if self.residual_branch is not None:
            output = output + self.residual_scale * self.residual_branch(scaled_inputs)

        return output

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)

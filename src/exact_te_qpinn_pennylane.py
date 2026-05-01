"""Exact paper-aligned TE-QPINN hybrid model using PennyLane.

This module implements a hybrid classical-quantum PINN model:
- classical FNN embedding phi(x, t)
- angle embedding theta_i = phi_i * cycled_coordinate_i
- parameterized quantum circuit (RY embedding + RX/RY/RZ ansatz + CNOT entanglement)
- expectation-value readout

Notes:
- Requires PennyLane.
- Uses default.qubit simulator and Torch backprop interface.
- Batch execution is implemented sample-by-sample for reliability.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

try:  # Optional dependency: existing repo behavior must remain usable without PennyLane.
    import pennylane as qml

    _PENNYLANE_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - exercised in tests via ImportError path.
    qml = None
    _PENNYLANE_IMPORT_ERROR = exc


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
        target_min: float = -1.0,
        target_max: float = 1.0,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
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
        output_dim: int,
        hidden_layers: List[int],
        activation: str,
    ):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = int(input_dim)
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, int(hidden_dim)))
            layers.append(_activation_from_name(activation))
            prev_dim = int(hidden_dim)
        layers.append(nn.Linear(prev_dim, int(output_dim)))
        self.net = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.net(inputs)


class ExactTEQPINNPennyLane(nn.Module):
    """Paper-aligned TE-QPINN hybrid model with PennyLane PQC."""

    SUPPORTED_ENTANGLEMENT = {"chain", "ring"}
    SUPPORTED_READOUT = {"z_sum", "z_tensor"}

    def __init__(
        self,
        input_dim: int = 2,
        output_dim: int = 1,
        hidden_layers: Optional[List[int]] = None,
        activation: str = "tanh",
        te_qpinn_pennylane: Optional[Dict] = None,
        problem: Optional[Dict] = None,
        device: str = "cpu",
    ):
        super().__init__()
        if qml is None:
            raise ImportError(
                "PennyLane is required for model_type='te_qpinn_pennylane'. "
                "Install with `python3 -m pip install pennylane`."
            ) from _PENNYLANE_IMPORT_ERROR

        if str(device).lower().startswith("cuda"):
            raise ValueError(
                "ExactTEQPINNPennyLane currently supports CPU execution only "
                "(PennyLane default.qubit backend). Set device='cpu'."
            )

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = hidden_layers or [64, 64, 64, 64]
        self.problem = problem or {}
        self.device_name = "cpu"

        cfg = te_qpinn_pennylane or {}
        self.num_qubits = int(cfg.get("num_qubits", cfg.get("n_qubits", 4)))
        self.num_layers = int(cfg.get("num_layers", cfg.get("n_layers", 5)))
        self.entanglement = str(cfg.get("entanglement", "chain")).lower()
        self.readout_type = str(cfg.get("readout_type", "z_sum")).lower()
        if self.entanglement not in self.SUPPORTED_ENTANGLEMENT:
            raise ValueError(
                f"Unsupported entanglement '{self.entanglement}'. "
                "Use 'chain' or 'ring'."
            )
        if self.readout_type not in self.SUPPORTED_READOUT:
            raise ValueError(
                f"Unsupported readout_type '{self.readout_type}'. "
                "Use 'z_sum' or 'z_tensor'."
            )

        self.rescale_min = float(cfg.get("rescale_min", -1.0))
        self.rescale_max = float(cfg.get("rescale_max", 1.0))
        input_mins = _coerce_bounds(cfg.get("input_mins"), self.input_dim, 0.0)
        input_maxs = _coerce_bounds(cfg.get("input_maxs"), self.input_dim, 1.0)
        self.rescaler = InputRescaler(
            input_dim=self.input_dim,
            input_mins=input_mins,
            input_maxs=input_maxs,
            target_min=self.rescale_min,
            target_max=self.rescale_max,
        )

        embedding_hidden = cfg.get("embedding_hidden_layers", [10, 10])
        if isinstance(embedding_hidden, int):
            embedding_hidden = [int(embedding_hidden)]
        embedding_hidden = [int(width) for width in embedding_hidden]
        embedding_activation = str(cfg.get("embedding_activation", "tanh"))
        self.embedding_fnn = TrainableEmbeddingFNN(
            input_dim=self.input_dim,
            output_dim=self.num_qubits,
            hidden_layers=embedding_hidden,
            activation=embedding_activation,
        )

        # Variational circuit parameters: [num_layers, num_qubits, (RX, RY, RZ)].
        self.theta = nn.Parameter(
            torch.zeros(self.num_layers, self.num_qubits, 3, dtype=torch.float64)
        )
        nn.init.normal_(self.theta, mean=0.0, std=0.1)

        readout_input_dim = self.num_qubits if self.readout_type == "z_sum" else 1
        self.readout_linear = nn.Linear(readout_input_dim, self.output_dim)
        self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float64))
        self.output_bias = nn.Parameter(torch.tensor(0.0, dtype=torch.float64))

        self.quantum_device = qml.device("default.qubit", wires=self.num_qubits)
        self._qnode_z_vector, self._qnode_z_tensor = self._build_qnodes()

        self._initialize_weights()
        self.to(self.device_name)
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
        inputs = torch.cat([x, t], dim=-1)
        return inputs.to(device=self.device_name, dtype=torch.float64)

    def _cycled_coordinates(self, scaled_inputs: torch.Tensor) -> torch.Tensor:
        indices = torch.arange(self.num_qubits, device=scaled_inputs.device) % self.input_dim
        return scaled_inputs[:, indices]

    def _apply_variational_layers(self, weights: torch.Tensor):
        for layer in range(self.num_layers):
            for wire in range(self.num_qubits):
                qml.RX(weights[layer, wire, 0], wires=wire)
                qml.RY(weights[layer, wire, 1], wires=wire)
                qml.RZ(weights[layer, wire, 2], wires=wire)

            for wire in range(self.num_qubits - 1):
                qml.CNOT(wires=[wire, wire + 1])
            if self.entanglement == "ring" and self.num_qubits > 2:
                qml.CNOT(wires=[self.num_qubits - 1, 0])

    def _tensor_z_observable(self):
        observable = qml.PauliZ(0)
        for wire in range(1, self.num_qubits):
            observable = observable @ qml.PauliZ(wire)
        return observable

    def _build_qnodes(self):
        @qml.qnode(self.quantum_device, interface="torch", diff_method="backprop")
        def qnode_z_vector(angles: torch.Tensor, weights: torch.Tensor):
            for wire in range(self.num_qubits):
                qml.RY(angles[wire], wires=wire)
            self._apply_variational_layers(weights)
            return tuple(qml.expval(qml.PauliZ(wire)) for wire in range(self.num_qubits))

        @qml.qnode(self.quantum_device, interface="torch", diff_method="backprop")
        def qnode_z_tensor(angles: torch.Tensor, weights: torch.Tensor):
            for wire in range(self.num_qubits):
                qml.RY(angles[wire], wires=wire)
            self._apply_variational_layers(weights)
            return qml.expval(self._tensor_z_observable())

        return qnode_z_vector, qnode_z_tensor

    def compute_embedding(self, x: torch.Tensor, t: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Return scaled inputs, embedding features phi, and angles theta."""
        inputs = self._prepare_inputs(x, t)
        scaled_inputs = self.rescaler(inputs)
        phi = self.embedding_fnn(scaled_inputs)
        cycled_coord = self._cycled_coordinates(scaled_inputs)
        angles = phi * cycled_coord
        return {
            "inputs_scaled": scaled_inputs,
            "phi": phi,
            "theta": angles,
        }

    def _evaluate_quantum_batch(self, angles: torch.Tensor) -> torch.Tensor:
        # PennyLane QNodes are evaluated per sample for robust autograd.
        if self.readout_type == "z_sum":
            feature_rows = []
            for sample_angles in angles:
                sample_features = self._qnode_z_vector(sample_angles, self.theta)
                if isinstance(sample_features, (tuple, list)):
                    sample_features = torch.stack(list(sample_features))
                feature_rows.append(sample_features.to(dtype=torch.float64))
            return torch.stack(feature_rows, dim=0)

        scalar_rows = []
        for sample_angles in angles:
            sample_scalar = self._qnode_z_tensor(sample_angles, self.theta)
            if sample_scalar.dim() == 0:
                sample_scalar = sample_scalar.unsqueeze(0)
            scalar_rows.append(sample_scalar.to(dtype=torch.float64))
        return torch.stack(scalar_rows, dim=0).reshape(-1, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        embedding = self.compute_embedding(x, t)
        angles = embedding["theta"]
        quantum_features = self._evaluate_quantum_batch(angles)
        output = self.readout_linear(quantum_features)
        output = self.output_scale * output + self.output_bias
        return output

    def count_parameters(self) -> int:
        return sum(param.numel() for param in self.parameters() if param.requires_grad)


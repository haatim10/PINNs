"""Unit tests for model factory and quantum-ready model compatibility."""

import pytest
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model import PINN
from src.model_factory import build_model, model_name_from_config
from src.quantum_ready_model import QuantumReadyPINN


class TestModelFactory:
    """Factory should build expected model variants from config."""

    def test_build_classical_model(self):
        config = {
            "model_type": "classical",
            "input_dim": 2,
            "output_dim": 1,
            "hidden_layers": [16, 16],
            "activation": "tanh",
        }
        model = build_model(config, device="cpu")
        assert isinstance(model, PINN)

    def test_build_classical_model_with_memory_features(self):
        config = {
            "model_type": "classical",
            "input_dim": 2,
            "output_dim": 1,
            "hidden_layers": [16, 16],
            "activation": "tanh",
            "memory_features": "analytic",
            "memory_feature_set": "basic_fractional",
            "memory_feature_normalization": "scale",
            "memory_epsilon": 1e-8,
        }
        problem = {
            "alpha": 0.5,
            "beta": 0.5,
            "x_min": 0.0,
            "x_max": 1.0,
            "t_min": 0.0,
            "t_max": 1.0,
        }
        model = build_model(config, device="cpu", problem_config=problem)
        assert isinstance(model, PINN)
        first_linear = next(m for m in model.network if isinstance(m, torch.nn.Linear))
        assert first_linear.in_features == 8

    def test_build_quantum_ready_model(self):
        config = {
            "model_type": "quantum_ready",
            "input_dim": 2,
            "output_dim": 1,
            "hidden_layers": [16, 16],
            "activation": "tanh",
            "quantum": {
                "n_qubits": 4,
                "n_layers": 2,
                "entanglement_strength": 0.1,
            },
        }
        model = build_model(config, device="cpu")
        assert isinstance(model, QuantumReadyPINN)

    def test_alias_hybrid_quantum_maps_to_quantum_ready(self):
        config = {
            "model_type": "hybrid_quantum",
            "input_dim": 2,
            "output_dim": 1,
            "hidden_layers": [8, 8],
            "activation": "tanh",
        }
        model = build_model(config, device="cpu")
        assert isinstance(model, QuantumReadyPINN)

    def test_invalid_model_type_raises(self):
        with pytest.raises(ValueError, match="Unsupported model_type"):
            build_model({"model_type": "not_a_real_model"}, device="cpu")


class TestQuantumReadyModel:
    """Quantum-ready model must preserve baseline PINN forward contract."""

    def test_forward_shape_and_dtype(self):
        config = {
            "model_type": "quantum_ready",
            "input_dim": 2,
            "output_dim": 1,
            "hidden_layers": [16, 16],
            "activation": "tanh",
            "quantum": {
                "n_qubits": 6,
                "n_layers": 2,
                "residual_connection": True,
            },
        }
        model = build_model(config, device="cpu")

        x = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)
        t = torch.linspace(0.0, 1.0, 10, dtype=torch.float64)

        y = model(x, t)
        assert y.shape == (10, 1)
        assert y.dtype == torch.float64

    def test_count_parameters_positive(self):
        config = {
            "model_type": "quantum_ready",
            "input_dim": 2,
            "output_dim": 1,
            "hidden_layers": [12, 12],
            "activation": "tanh",
            "quantum": {
                "n_qubits": 4,
                "n_layers": 2,
            },
        }
        model = build_model(config, device="cpu")
        assert model.count_parameters() > 0


class TestModelNameNormalization:
    def test_default_name(self):
        assert model_name_from_config({}) == "classical"

    def test_alias_name_normalization(self):
        assert model_name_from_config({"model_type": "hybrid_quantum"}) == "quantum_ready"
        assert model_name_from_config({"model_type": "qready"}) == "quantum_ready"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

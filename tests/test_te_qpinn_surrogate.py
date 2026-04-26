"""Unit tests for the TE-QPINN-inspired surrogate model."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_factory import build_model, model_name_from_config
from src.te_qpinn_surrogate_model import TEQPINNSurrogatePINN


def _base_config(model_type: str = "te_qpinn_surrogate") -> dict:
    return {
        "model_type": model_type,
        "input_dim": 2,
        "output_dim": 1,
        "hidden_layers": [16, 16],
        "activation": "tanh",
        "te_qpinn": {
            "n_qubits": 6,
            "embedding_hidden_layers": [12, 12],
            "rescale_min": -0.95,
            "rescale_max": 0.95,
            "input_mins": [0.0, 0.0],
            "input_maxs": [1.0, 1.0],
            "use_pairwise_entanglement": True,
            "residual_connection": True,
            "residual_scale": 0.25,
            "expectation_activation": "tanh",
        },
    }


def test_model_factory_routes_te_qpinn_surrogate():
    model = build_model(_base_config("te_qpinn_surrogate"), device="cpu")
    assert isinstance(model, TEQPINNSurrogatePINN)

    model_alias = build_model(_base_config("te_qpinn"), device="cpu")
    assert isinstance(model_alias, TEQPINNSurrogatePINN)
    assert model_name_from_config({"model_type": "te_qpinn"}) == "te_qpinn_surrogate"


def test_forward_pass_shape_and_dtype():
    model = build_model(_base_config(), device="cpu")

    x = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 9, dtype=torch.float64)
    y = model(x, t)

    assert y.shape == (9, 1)
    assert y.dtype == torch.float64

    embedding = model.compute_embedding(x, t)
    assert embedding["phi"].shape == (9, model.n_qubits)
    assert embedding["theta"].shape == (9, model.n_qubits)


def test_parameter_count_positive():
    model = build_model(_base_config(), device="cpu")
    count = model.count_parameters()
    assert count > 0
    assert count == sum(p.numel() for p in model.parameters() if p.requires_grad)


def test_device_compatibility():
    cpu_model = build_model(_base_config(), device="cpu")
    x_cpu = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    t_cpu = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    y_cpu = cpu_model(x_cpu, t_cpu)
    assert y_cpu.device.type == "cpu"

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable in this environment")

    cuda_model = build_model(_base_config(), device="cuda")
    x_cuda = torch.linspace(0.0, 1.0, 5, dtype=torch.float64, device="cuda")
    t_cuda = torch.linspace(0.0, 1.0, 5, dtype=torch.float64, device="cuda")
    y_cuda = cuda_model(x_cuda, t_cuda)
    assert y_cuda.device.type == "cuda"
    assert y_cuda.shape == (5, 1)

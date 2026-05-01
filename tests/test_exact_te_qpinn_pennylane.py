"""Tests for exact PennyLane-based TE-QPINN model (Milestone A core)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_factory import build_model, model_name_from_config
from src.te_qpinn_surrogate_model import TEQPINNSurrogatePINN

try:
    import pennylane as _qml  # noqa: F401

    HAS_PENNYLANE = True
except Exception:
    HAS_PENNYLANE = False


def _problem_config() -> dict:
    return {
        "type": "integro_differential",
        "alpha": 0.5,
        "beta": 0.5,
        "x_min": 0.0,
        "x_max": 1.0,
        "t_min": 0.0,
        "t_max": 1.0,
    }


def _exact_config(model_type: str = "te_qpinn_pennylane") -> dict:
    return {
        "model_type": model_type,
        "input_dim": 2,
        "output_dim": 1,
        "hidden_layers": [16, 16],
        "activation": "tanh",
        "te_qpinn_pennylane": {
            "num_qubits": 4,
            "num_layers": 2,
            "entanglement": "chain",
            "readout_type": "z_sum",
            "embedding_hidden_layers": [10, 10],
            "embedding_activation": "tanh",
            "rescale_min": -1.0,
            "rescale_max": 1.0,
            "input_mins": [0.0, 0.0],
            "input_maxs": [1.0, 1.0],
        },
    }


def _surrogate_config() -> dict:
    return {
        "model_type": "te_qpinn_surrogate",
        "input_dim": 2,
        "output_dim": 1,
        "hidden_layers": [16, 16],
        "activation": "tanh",
        "te_qpinn": {
            "n_qubits": 6,
            "embedding_hidden_layers": [12, 12],
            "input_mins": [0.0, 0.0],
            "input_maxs": [1.0, 1.0],
            "residual_connection": True,
        },
    }


def test_factory_raises_clear_error_when_pennylane_missing():
    if HAS_PENNYLANE:
        pytest.skip("PennyLane is installed; missing-dependency behavior not applicable")

    with pytest.raises(ImportError, match="PennyLane is required"):
        build_model(
            _exact_config(),
            device="cpu",
            problem_config=_problem_config(),
        )


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
def test_model_instantiation_and_forward_shape():
    model = build_model(_exact_config(), device="cpu", problem_config=_problem_config())
    x = torch.tensor([0.1, 0.4, 0.8], dtype=torch.float64)
    t = torch.tensor([0.2, 0.5, 0.9], dtype=torch.float64)
    y = model(x, t)
    assert y.shape == (3, 1)
    assert y.dtype == torch.float64


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
def test_parameter_count_positive():
    model = build_model(_exact_config(), device="cpu", problem_config=_problem_config())
    count = model.count_parameters()
    assert count > 0
    assert count == sum(p.numel() for p in model.parameters() if p.requires_grad)


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
def test_gradients_flow_to_embedding_and_quantum_weights():
    model = build_model(_exact_config(), device="cpu", problem_config=_problem_config())
    x = torch.tensor([0.2, 0.7], dtype=torch.float64)
    t = torch.tensor([0.3, 0.6], dtype=torch.float64)
    loss = model(x, t).pow(2).mean()
    loss.backward()

    embedding_grads = [
        p.grad for name, p in model.named_parameters() if "embedding_fnn" in name and p.requires_grad
    ]
    assert embedding_grads, "Expected embedding_fnn parameters to exist"
    assert any(g is not None for g in embedding_grads), "Expected embedding_fnn gradients"
    for grad in embedding_grads:
        if grad is not None:
            assert torch.isfinite(grad).all()

    assert model.theta.grad is not None, "Expected quantum variational weights gradient"
    assert torch.isfinite(model.theta.grad).all()


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
def test_autograd_supports_u_x_and_u_xx():
    model = build_model(_exact_config(), device="cpu", problem_config=_problem_config())
    x = torch.tensor([0.25, 0.75], dtype=torch.float64, requires_grad=True)
    t = torch.tensor([0.15, 0.55], dtype=torch.float64, requires_grad=True)

    u = model(x, t)
    u_x = torch.autograd.grad(
        outputs=u,
        inputs=x,
        grad_outputs=torch.ones_like(u),
        create_graph=True,
        retain_graph=True,
    )[0]
    u_xx = torch.autograd.grad(
        outputs=u_x,
        inputs=x,
        grad_outputs=torch.ones_like(u_x),
        create_graph=True,
        retain_graph=True,
    )[0]

    assert u.shape == (2, 1)
    assert u_x.shape == (2,)
    assert u_xx.shape == (2,)
    for tensor in (u, u_x, u_xx):
        assert torch.isfinite(tensor).all()


@pytest.mark.skipif(not HAS_PENNYLANE, reason="PennyLane not installed")
def test_model_factory_aliases_and_name_normalization():
    aliases = [
        "te_qpinn_pennylane",
        "exact_te_qpinn_pennylane",
        "exact_pqc_te_qpinn",
    ]
    for alias in aliases:
        cfg = _exact_config(alias)
        model = build_model(cfg, device="cpu", problem_config=_problem_config())
        assert model_name_from_config(cfg) == "te_qpinn_pennylane"
        y = model(
            torch.tensor([0.3, 0.6], dtype=torch.float64),
            torch.tensor([0.2, 0.7], dtype=torch.float64),
        )
        assert y.shape == (2, 1)


def test_existing_surrogate_model_still_instantiates():
    model = build_model(_surrogate_config(), device="cpu", problem_config=_problem_config())
    assert isinstance(model, TEQPINNSurrogatePINN)
    y = model(
        torch.tensor([0.1, 0.6], dtype=torch.float64),
        torch.tensor([0.2, 0.8], dtype=torch.float64),
    )
    assert y.shape == (2, 1)


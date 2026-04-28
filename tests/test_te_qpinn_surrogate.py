"""Unit tests for the TE-QPINN-inspired surrogate model."""

import sys
from copy import deepcopy
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model_factory import build_model, model_name_from_config
from src.te_qpinn_surrogate_model import TEQPINNSurrogatePINN


def _base_config(model_type: str = "te_qpinn_surrogate", te_overrides: dict | None = None) -> dict:
    config = {
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
    if te_overrides:
        config["te_qpinn"].update(deepcopy(te_overrides))
    return config


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


def test_fixed_residual_mode_matches_legacy_formula():
    model = build_model(
        _base_config(
            te_overrides={
                "residual_blend_mode": "fixed",
                "residual_connection": True,
                "residual_scale": 0.25,
            }
        ),
        device="cpu",
    )

    assert model.residual_blend_mode == "fixed"
    assert model.residual_gate_logit is None

    x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    with torch.no_grad():
        embedding = model.compute_embedding(x, t)
        theta = embedding["theta"]
        scaled_inputs = embedding["inputs_scaled"]
        q_features = model._build_quantum_features(theta)
        latent = model.variational(q_features)
        expectation = model._apply_expectation_activation(model.expectation_head(latent))
        quantum_output = model.readout(expectation)
        residual_output = model.residual_branch(scaled_inputs)
        expected = quantum_output + model.residual_scale * residual_output
        actual = model(x, t)

    assert torch.allclose(actual, expected, atol=1e-10, rtol=1e-10)


def test_gated_residual_forward_and_gate_properties():
    model = build_model(
        _base_config(
            te_overrides={
                "residual_blend_mode": "gated",
                "residual_gate_init": 0.5,
                "residual_connection": True,
            }
        ),
        device="cpu",
    )
    assert model.residual_blend_mode == "gated"
    assert model.residual_gate_logit is not None
    assert model.residual_gate_logit.requires_grad

    gate_value = torch.sigmoid(model.residual_gate_logit.detach()).item()
    assert 0.0 < gate_value < 1.0

    x = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    y = model(x, t)
    assert y.shape == (6, 1)
    assert y.dtype == torch.float64


def test_gated_residual_backward_updates_gate_grad():
    model = build_model(
        _base_config(
            te_overrides={
                "residual_blend_mode": "gated",
                "residual_gate_init": 0.3,
                "residual_connection": True,
            }
        ),
        device="cpu",
    )
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    loss = model(x, t).pow(2).mean()
    loss.backward()

    assert model.residual_gate_logit.grad is not None
    assert torch.isfinite(model.residual_gate_logit.grad).all()


def test_factory_instantiates_gated_mode_and_param_delta_small():
    fixed_model = build_model(
        _base_config(
            te_overrides={
                "residual_blend_mode": "fixed",
                "residual_connection": True,
            }
        ),
        device="cpu",
    )
    gated_model = build_model(
        _base_config(
            te_overrides={
                "residual_blend_mode": "gated",
                "residual_gate_init": 0.5,
                "residual_connection": True,
            }
        ),
        device="cpu",
    )

    fixed_count = fixed_model.count_parameters()
    gated_count = gated_model.count_parameters()
    assert gated_model.residual_blend_mode == "gated"
    assert gated_count == fixed_count + 1


def test_gated_mode_requires_residual_connection():
    with pytest.raises(ValueError, match="requires residual_connection=true"):
        build_model(
            _base_config(
                te_overrides={
                    "residual_blend_mode": "gated",
                    "residual_connection": False,
                }
            ),
            device="cpu",
        )


def test_feature_norm_none_preserves_quantum_feature_construction():
    model = build_model(
        _base_config(
            te_overrides={
                "feature_norm": "none",
                "feature_norm_position": "post_entanglement",
                "use_pairwise_entanglement": True,
            }
        ),
        device="cpu",
    )

    x = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 6, dtype=torch.float64)
    theta = model.compute_embedding(x, t)["theta"]

    manual = torch.cat(
        [
            torch.sin(theta),
            torch.cos(theta),
            torch.sin(theta[:, :-1]) * torch.cos(theta[:, 1:]),
        ],
        dim=-1,
    )
    built = model._build_quantum_features(theta)
    assert torch.allclose(built, manual, atol=1e-12, rtol=1e-12)


def test_layernorm_post_quantum_forward_works():
    model = build_model(
        _base_config(
            te_overrides={
                "feature_norm": "layernorm",
                "feature_norm_position": "post_quantum",
            }
        ),
        device="cpu",
    )
    assert model.quantum_feature_norm is not None
    assert model.entangled_feature_norm is None

    x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    y = model(x, t)
    assert y.shape == (7, 1)
    assert y.dtype == torch.float64


def test_layernorm_post_entanglement_forward_works():
    model = build_model(
        _base_config(
            te_overrides={
                "feature_norm": "layernorm",
                "feature_norm_position": "post_entanglement",
            }
        ),
        device="cpu",
    )
    assert model.quantum_feature_norm is None
    assert model.entangled_feature_norm is not None

    x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 7, dtype=torch.float64)
    y = model(x, t)
    assert y.shape == (7, 1)
    assert y.dtype == torch.float64


def test_layernorm_backward_pass_works():
    model = build_model(
        _base_config(
            te_overrides={
                "feature_norm": "layernorm",
                "feature_norm_position": "post_entanglement",
            }
        ),
        device="cpu",
    )
    x = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 8, dtype=torch.float64)
    loss = model(x, t).pow(2).mean()
    loss.backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads
    for grad in grads:
        assert torch.isfinite(grad).all()


def test_invalid_feature_norm_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported feature_norm"):
        build_model(
            _base_config(
                te_overrides={
                    "feature_norm": "batchnorm",
                }
            ),
            device="cpu",
        )


def test_invalid_feature_norm_position_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported feature_norm_position"):
        build_model(
            _base_config(
                te_overrides={
                    "feature_norm": "layernorm",
                    "feature_norm_position": "pre_quantum",
                }
            ),
            device="cpu",
        )


def test_layernorm_parameter_count_delta_is_small_and_positive():
    fixed_model = build_model(
        _base_config(
            te_overrides={
                "feature_norm": "none",
            }
        ),
        device="cpu",
    )
    ln_model = build_model(
        _base_config(
            te_overrides={
                "feature_norm": "layernorm",
                "feature_norm_position": "post_entanglement",
            }
        ),
        device="cpu",
    )
    fixed_count = fixed_model.count_parameters()
    ln_count = ln_model.count_parameters()
    assert ln_count > fixed_count

    feature_dim = 2 * ln_model.n_qubits
    if ln_model.use_pairwise_entanglement and ln_model.n_qubits > 1:
        feature_dim += ln_model.n_qubits - 1
    expected_delta = 2 * feature_dim  # LayerNorm weight + bias
    assert ln_count == fixed_count + expected_delta

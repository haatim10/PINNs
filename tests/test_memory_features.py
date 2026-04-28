"""Tests for shared deterministic memory feature builder."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.memory_features import MemoryFeatureBuilder


def _builder(**overrides):
    kwargs = {
        "memory_features": "none",
        "memory_feature_set": "basic_fractional",
        "memory_feature_normalization": "scale",
        "memory_epsilon": 1e-8,
        "alpha": 0.5,
        "beta": 0.5,
        "x_min": 0.0,
        "x_max": 1.0,
        "t_min": 0.0,
        "t_max": 1.0,
    }
    kwargs.update(overrides)
    return MemoryFeatureBuilder(**kwargs)


def test_memory_features_none_returns_2d_input():
    builder = _builder(memory_features="none")
    x = torch.tensor([0.0, 0.3, 0.9], dtype=torch.float64)
    t = torch.tensor([0.1, 0.2, 0.4], dtype=torch.float64)
    features = builder(x, t)
    assert features.shape == (3, 2)
    assert torch.allclose(features[:, 0], x)
    assert torch.allclose(features[:, 1], t)


def test_memory_features_analytic_returns_8d_input():
    builder = _builder(memory_features="analytic")
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    t = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    features = builder(x, t)
    assert features.shape == (5, 8)


def test_memory_features_t0_safety_has_no_nan_or_inf():
    builder = _builder(memory_features="analytic")
    x = torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64)
    t = torch.tensor([0.0, 1e-12, 0.2], dtype=torch.float64)
    features = builder(x, t)
    assert torch.isfinite(features).all()


def test_memory_features_backward_pass_works():
    builder = _builder(memory_features="analytic")
    x = torch.linspace(0.0, 1.0, 7, dtype=torch.float64, requires_grad=True)
    t = torch.linspace(0.0, 1.0, 7, dtype=torch.float64, requires_grad=True)
    output = builder(x, t).pow(2).mean()
    output.backward()
    assert x.grad is not None
    assert t.grad is not None
    assert torch.isfinite(x.grad).all()
    assert torch.isfinite(t.grad).all()


def test_invalid_memory_features_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported memory_features"):
        _builder(memory_features="history_tokens")


def test_invalid_memory_feature_set_raises_clear_error():
    with pytest.raises(ValueError, match="Unsupported memory_feature_set"):
        _builder(memory_features="analytic", memory_feature_set="unknown_set")

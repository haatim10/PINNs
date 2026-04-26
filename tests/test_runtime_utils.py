"""Tests for runtime utility helpers added in Phase 0."""

import numpy as np
import torch
import torch.nn as nn

from src.utils import count_trainable_parameters, resolve_device, set_seed


def test_set_seed_reproducible_for_torch_and_numpy():
    set_seed(1234, deterministic=True)
    torch_a = torch.randn(5, dtype=torch.float64)
    np_a = np.random.rand(5)

    set_seed(1234, deterministic=True)
    torch_b = torch.randn(5, dtype=torch.float64)
    np_b = np.random.rand(5)

    assert torch.allclose(torch_a, torch_b)
    assert np.allclose(np_a, np_b)


def test_resolve_device_behaves_safely():
    auto_device = resolve_device("auto")
    assert auto_device in {"cpu", "cuda"}

    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
    else:
        assert resolve_device("cuda") == "cpu"

    assert resolve_device("cpu") == "cpu"
    assert resolve_device(None) in {"cpu", "cuda"}


def test_count_trainable_parameters():
    model = nn.Sequential(
        nn.Linear(2, 3, bias=True),
        nn.Tanh(),
        nn.Linear(3, 1, bias=False),
    )
    # First linear: 2*3 weights + 3 bias = 9
    # Second linear: 3*1 weights = 3
    assert count_trainable_parameters(model) == 12

"""Unit tests for the configurable integro-differential exact solution."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics_integro import BoundaryConditions, exact_solution, source_term


def test_exact_solution_respects_frequency_and_time_power():
    x = torch.tensor([0.0, 1.0], dtype=torch.float64)
    t = torch.tensor([0.5, 0.5], dtype=torch.float64)
    cfg = {
        "family": "cosine",
        "spatial_frequency": 3.0,
        "time_power": 0.35,
        "amplitude": 1.2,
        "phase": 0.0,
    }

    u = exact_solution(x, t, alpha=0.5, solution_cfg=cfg)
    expected_left = 1.2 * (0.5 ** 0.35)
    expected_right = -expected_left

    assert torch.isclose(u[0], torch.tensor(expected_left, dtype=torch.float64))
    assert torch.isclose(u[1], torch.tensor(expected_right, dtype=torch.float64))


def test_source_term_returns_finite_tensor_for_oscillatory_case():
    x = torch.linspace(0.0, 1.0, 5, dtype=torch.float64)
    t = torch.full_like(x, 0.25)
    cfg = {
        "family": "cosine",
        "spatial_frequency": 3.0,
        "time_power": 0.35,
        "amplitude": 1.0,
        "phase": 0.0,
    }

    f = source_term(x, t, alpha=0.5, beta=0.5, solution_cfg=cfg)

    assert f.shape == x.shape
    assert torch.isfinite(f).all()


def test_boundary_conditions_follow_exact_solution():
    class DummyModel:
        def __call__(self, x, t):
            return torch.zeros_like(x).unsqueeze(-1)

    cfg = {
        "family": "cosine",
        "spatial_frequency": 3.0,
        "time_power": 0.35,
        "amplitude": 1.0,
        "phase": 0.0,
    }
    bc = BoundaryConditions(DummyModel(), alpha=0.5, solution_cfg=cfg)
    t = torch.tensor([0.2, 0.4], dtype=torch.float64)

    left = bc.left_bc(t)
    right = bc.right_bc(t)

    expected = t ** 0.35
    assert torch.allclose(left, expected)
    assert torch.allclose(right, -expected)

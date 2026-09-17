"""Regression tests for the audit fixes.

Covers:
  1. The reference quadrature is genuinely Gauss-Jacobi and is more accurate than
     the product-integration rule it validates.
  2. The CSI feature builder does not receive the generator's true Doppler
     frequency (the estimator works from training data only).
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from scipy.special import gamma as gamma_fn

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh import GradedMesh, L1Coefficients
from src.physics_integro import IntegroDifferentialResidual, IntegralConvergenceMonitor
from src.applications.wireless_channel import (
    WirelessChannelConfig,
    build_channel_features,
    estimate_doppler_hz,
    fit_linear_baseline,
    generate_wireless_channel,
)

ALPHA = BETA = 0.5


class _PowerLawModel(torch.nn.Module):
    """u(x, s) = s**alpha, so the Volterra integral has a closed form."""

    def forward(self, x, t):
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return torch.clamp(t, min=0.0).pow(ALPHA)


def _exact_integral(t: float) -> float:
    """int_0^t (t-s)^{-beta} s^alpha ds via the Beta function."""
    return (
        t ** (ALPHA + 1.0 - BETA)
        * gamma_fn(ALPHA + 1.0)
        * gamma_fn(1.0 - BETA)
        / gamma_fn(ALPHA + 2.0 - BETA)
    )


def _residual(n_quad: int, n_t: int = 100):
    mesh = GradedMesh(n_t, 2.0)
    coeffs = L1Coefficients(mesh, ALPHA)
    return IntegroDifferentialResidual(
        _PowerLawModel(), mesh, coeffs, ALPHA, BETA, n_quad=n_quad, device="cpu"
    ), mesh


def _quadrature_error(n_quad: int, n_t: int = 100) -> float:
    residual, mesh = _residual(n_quad, n_t)
    n = n_t
    value = residual.compute_integral_term_quadrature(
        torch.tensor([0.3], dtype=torch.float64),
        mesh.get_nodes()[n].reshape(1),
        torch.tensor([n]),
    ).item()
    return abs(value - np.sin(0.3) * _exact_integral(mesh.get_nodes()[n].item()))


class TestGaussJacobiReference:
    def test_reference_is_accurate_at_default_order(self):
        """A 15-point rule should already be far below the old 1.7e-2 plateau."""
        assert _quadrature_error(15) < 1e-4

    def test_reference_converges_with_quadrature_order(self):
        """Gauss-Legendre-on-a-singularity stagnated; Gauss-Jacobi must converge."""
        errors = [_quadrature_error(nq) for nq in (10, 20, 40, 80)]
        for coarse, fine in zip(errors, errors[1:]):
            assert fine < coarse
        assert errors[-1] < errors[0] / 100.0

    def test_monitor_reference_beats_product_integration(self):
        """The convergence reference must be better than the rule under test."""
        residual, mesh = _residual(n_quad=15, n_t=200)
        monitor = IntegralConvergenceMonitor(residual, mesh, ALPHA, BETA, device="cpu")

        n = 200
        x = torch.tensor([0.3], dtype=torch.float64)
        t = mesh.get_nodes()[n].reshape(1)
        idx = torch.tensor([n])
        exact = np.sin(0.3) * _exact_integral(t.item())

        reference_err = abs(monitor.compute_reference_integral(x, t, idx).item() - exact)
        product_err = abs(residual.compute_integral_term(x, t, idx).item() - exact)
        assert reference_err < product_err

    def test_monitor_restores_working_quadrature_order(self):
        residual, mesh = _residual(n_quad=15)
        monitor = IntegralConvergenceMonitor(residual, mesh, ALPHA, BETA, device="cpu")
        monitor.compute_reference_integral(
            torch.tensor([0.3], dtype=torch.float64),
            mesh.get_nodes()[100].reshape(1),
            torch.tensor([100]),
        )
        assert residual.n_quad == 15


class TestDopplerLeakage:
    def test_estimator_uses_only_supplied_samples(self):
        """Estimate recovers a plausible frequency without being told the truth."""
        t = np.linspace(0.0, 1.0, 200)
        y = np.cos(2.0 * np.pi * 8.0 * t + 0.3) + 0.15 * t
        assert abs(estimate_doppler_hz(t, y) - 8.0) < 0.5

    def test_estimate_is_not_hardcoded_to_generator_default(self):
        """A channel at a different Doppler must produce a different estimate."""
        t = np.linspace(0.0, 1.0, 200)
        y5 = np.cos(2.0 * np.pi * 5.0 * t)
        y12 = np.cos(2.0 * np.pi * 12.0 * t)
        assert abs(estimate_doppler_hz(t, y5) - 5.0) < 0.5
        assert abs(estimate_doppler_hz(t, y12) - 12.0) < 0.5

    def test_wrong_doppler_degrades_domain_features(self):
        """Domain features must actually depend on the frequency they are given.

        This is the property that made the original result circular: with the true
        frequency the basis spans the signal exactly.
        """
        cfg = WirelessChannelConfig(noise_std=0.0)
        data = generate_wireless_channel(cfg)
        t = np.asarray(data["t"], dtype=float)
        y = np.asarray(data["h_clean"], dtype=float).reshape(len(t), -1)[:, 0]

        def linear_fit_error(freq: float) -> float:
            feats = build_channel_features(t, "domain", doppler_hz=freq)
            pred = fit_linear_baseline(feats, y, feats)
            return float(np.linalg.norm(y - pred) / np.linalg.norm(y))

        assert linear_fit_error(cfg.doppler_hz) < linear_fit_error(cfg.doppler_hz + 7.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

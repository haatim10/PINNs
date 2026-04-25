"""Unit tests for graded mesh generation and L1 coefficients."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh import GradedMesh, L1Coefficients


class TestGradedMesh:
    """Tests for GradedMesh class."""

    def test_uniform_mesh(self):
        """beta=1 should generate a uniform mesh."""
        mesh = GradedMesh(N=10, beta=1.0, t_max=1.0)
        expected = np.linspace(0.0, 1.0, 11)
        np.testing.assert_allclose(mesh.get_nodes().cpu().numpy(), expected, rtol=1e-10)

    def test_graded_mesh_monotonic(self):
        """Mesh nodes must be strictly increasing."""
        mesh = GradedMesh(N=20, beta=2.0, t_max=1.0)
        t = mesh.get_nodes().cpu().numpy()
        assert np.all(np.diff(t) > 0)

    def test_graded_mesh_endpoints(self):
        """Mesh must start at 0 and end at t_max."""
        t_max = 2.5
        mesh = GradedMesh(N=15, beta=1.5, t_max=t_max)
        t = mesh.get_nodes().cpu().numpy()
        assert t[0] == pytest.approx(0.0)
        assert t[-1] == pytest.approx(t_max)

    def test_graded_mesh_finer_near_zero(self):
        """For beta > 1, early steps should be smaller than late steps."""
        mesh = GradedMesh(N=10, beta=2.0, t_max=1.0)
        tau = mesh.tau.cpu().numpy()
        assert tau[0] < tau[-1]

    def test_mesh_length(self):
        """Check number of nodes and intervals."""
        N = 25
        mesh = GradedMesh(N=N, beta=1.5, t_max=1.0)
        assert len(mesh.get_nodes()) == N + 1
        assert len(mesh.tau) == N

    def test_nodes_dtype(self):
        """Mesh nodes are expected to be float64 for stable fractional arithmetic."""
        mesh = GradedMesh(N=10, beta=1.0, t_max=1.0)
        assert isinstance(mesh.get_nodes(), torch.Tensor)
        assert mesh.get_nodes().dtype == torch.float64


class TestL1Coefficients:
    """Tests for L1 discretization coefficients."""

    def test_l1_coefficients_shape_per_time_level(self):
        """Each time level n must expose n+1 coefficients (including index 0)."""
        mesh = GradedMesh(N=10, beta=1.0, t_max=1.0)
        l1 = L1Coefficients(alpha=0.5, mesh=mesh)

        for n in range(1, 11):
            coeffs = l1.get_coefficients_for_n(n)
            assert coeffs.shape == (n + 1,)

    def test_l1_coefficients_positive_for_k_ge_1(self):
        """All active L1 coefficients d_{n,k}, k>=1, should be positive."""
        mesh = GradedMesh(N=10, beta=1.0, t_max=1.0)
        l1 = L1Coefficients(alpha=0.5, mesh=mesh)

        for n in range(1, 11):
            coeffs = l1.get_coefficients_for_n(n).cpu().numpy()
            assert np.all(coeffs[1:] > 0)

    def test_l1_coefficients_zero_at_index_zero(self):
        """Index 0 is unused in this formulation and should remain zero."""
        mesh = GradedMesh(N=10, beta=1.0, t_max=1.0)
        l1 = L1Coefficients(alpha=0.5, mesh=mesh)

        for n in range(1, 11):
            coeffs = l1.get_coefficients_for_n(n).cpu().numpy()
            assert coeffs[0] == pytest.approx(0.0)

    def test_l1_different_alphas(self):
        """Changing alpha should change the coefficients."""
        mesh = GradedMesh(N=5, beta=1.0, t_max=1.0)
        l1_a = L1Coefficients(alpha=0.5, mesh=mesh)
        l1_b = L1Coefficients(alpha=0.8, mesh=mesh)

        coeffs_a = l1_a.get_coefficients_for_n(5).cpu().numpy()
        coeffs_b = l1_b.get_coefficients_for_n(5).cpu().numpy()
        assert not np.allclose(coeffs_a, coeffs_b)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

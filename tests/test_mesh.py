"""Unit tests for mesh generation and L1 coefficients."""

import pytest
import torch
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh import GradedMesh, compute_l1_coefficients


class TestGradedMesh:
    """Tests for GradedMesh class."""

    def test_uniform_mesh(self):
        """Test that r=1 gives uniform mesh."""
        mesh = GradedMesh(T=1.0, N=10, r=1.0)
        expected = np.linspace(0, 1, 11)
        np.testing.assert_allclose(mesh.t, expected, rtol=1e-10)

    def test_graded_mesh_monotonic(self):
        """Test that graded mesh is monotonically increasing."""
        mesh = GradedMesh(T=1.0, N=20, r=2.0)
        assert np.all(np.diff(mesh.t) > 0)

    def test_graded_mesh_endpoints(self):
        """Test that mesh starts at 0 and ends at T."""
        T = 2.5
        mesh = GradedMesh(T=T, N=15, r=1.5)
        assert mesh.t[0] == 0.0
        assert mesh.t[-1] == T

    def test_graded_mesh_finer_near_zero(self):
        """Test that graded mesh (r>1) has finer spacing near t=0."""
        mesh = GradedMesh(T=1.0, N=10, r=2.0)
        # First step should be smaller than last step
        assert mesh.tau[0] < mesh.tau[-1]

    def test_mesh_length(self):
        """Test correct number of mesh points."""
        N = 25
        mesh = GradedMesh(T=1.0, N=N, r=1.5)
        assert len(mesh.t) == N + 1
        assert len(mesh.tau) == N

    def test_to_tensor(self):
        """Test conversion to PyTorch tensor."""
        mesh = GradedMesh(T=1.0, N=10, r=1.0)
        t_tensor = mesh.to_tensor()
        assert isinstance(t_tensor, torch.Tensor)
        assert t_tensor.dtype == torch.float32


class TestL1Coefficients:
    """Tests for L1 scheme coefficients."""

    def test_l1_coefficients_shape(self):
        """Test coefficient matrix has correct shape."""
        mesh = GradedMesh(T=1.0, N=10, r=1.0)
        a = compute_l1_coefficients(alpha=0.5, mesh=mesh)
        assert a.shape == (11, 11)

    def test_l1_coefficients_positive(self):
        """Test that non-zero coefficients are positive."""
        mesh = GradedMesh(T=1.0, N=10, r=1.0)
        a = compute_l1_coefficients(alpha=0.5, mesh=mesh)
        # Check upper triangular part
        for n in range(1, 11):
            for k in range(n):
                assert a[n, k] > 0

    def test_l1_coefficients_zero_first_row(self):
        """Test that first row is all zeros."""
        mesh = GradedMesh(T=1.0, N=10, r=1.0)
        a = compute_l1_coefficients(alpha=0.5, mesh=mesh)
        np.testing.assert_array_equal(a[0, :], 0)

    def test_l1_different_alphas(self):
        """Test coefficients for different alpha values."""
        mesh = GradedMesh(T=1.0, N=5, r=1.0)

        a_05 = compute_l1_coefficients(alpha=0.5, mesh=mesh)
        a_08 = compute_l1_coefficients(alpha=0.8, mesh=mesh)

        # Coefficients should be different for different alphas
        assert not np.allclose(a_05, a_08)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

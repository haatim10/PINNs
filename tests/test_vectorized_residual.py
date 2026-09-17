"""The vectorized residual path must be algebraically identical to the loop.

compute() dispatches to compute_vectorized(); compute_reference() is the original
O(N^2) implementation. These tests pin the two together in values AND gradients,
under both history gradient modes, so the speedup can never silently change the
physics.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.mesh import GradedMesh, L1Coefficients
from src.model import PINN
from src.physics_integro import IntegroDifferentialResidual

ALPHA = BETA = 0.5


def _setup(n_t: int, mode: str, seed: int = 0):
    torch.manual_seed(seed)
    mesh = GradedMesh(n_t, 2.0)
    coeffs = L1Coefficients(mesh, ALPHA)
    model = PINN(hidden_layers=[32, 32, 32], device="cpu")
    residual = IntegroDifferentialResidual(
        model, mesh, coeffs, ALPHA, BETA, n_quad=15,
        device="cpu", history_gradient_mode=mode,
    )
    batch = 32
    x = torch.rand(batch, dtype=torch.float64)
    n_idx = torch.randint(1, n_t + 1, (batch,))
    t = mesh.get_nodes()[n_idx]
    return model, residual, x, t, n_idx


def _param_grads(model, residual_values):
    model.zero_grad()
    torch.mean(residual_values ** 2).backward()
    return torch.cat([p.grad.flatten().clone() for p in model.parameters()])


@pytest.mark.parametrize("mode", ["full", "detached_legacy"])
@pytest.mark.parametrize("n_t", [10, 25, 50])
def test_residual_values_match(mode, n_t):
    _, residual, x, t, n_idx = _setup(n_t, mode)
    ref = residual.compute_reference(x, t, n_idx)
    vec = residual.compute_vectorized(x, t, n_idx)
    assert torch.allclose(ref, vec, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("mode", ["full", "detached_legacy"])
def test_parameter_gradients_match(mode):
    model, residual, x, t, n_idx = _setup(50, mode)
    g_ref = _param_grads(model, residual.compute_reference(x, t, n_idx))
    g_vec = _param_grads(model, residual.compute_vectorized(x, t, n_idx))
    assert torch.allclose(g_ref, g_vec, atol=1e-10, rtol=0.0)
    # relative check, since absolute tolerance alone can hide a scaled gradient
    assert (g_ref - g_vec).norm() / g_ref.norm() < 1e-10


def test_compute_dispatches_to_vectorized_by_default():
    _, residual, x, t, n_idx = _setup(25, "full")
    assert residual.use_vectorized_residual is True
    default = residual.compute(x, t, n_idx)
    residual.use_vectorized_residual = False
    fallback = residual.compute(x, t, n_idx)
    assert torch.allclose(default, fallback, atol=1e-12, rtol=0.0)


def test_weight_matrices_are_lower_triangular():
    """Node t_n may only depend on history nodes t_j with j <= n."""
    _, residual, _, _, _ = _setup(25, "full")
    residual._ensure_weight_matrices()
    n_t = residual.mesh.N
    for n in range(1, n_t + 1):
        future_l1 = residual._w_l1[n, n + 1:]
        future_int = residual._w_int[n, n + 1:]
        if future_l1.numel():          # empty for the final row
            assert future_l1.abs().max() == 0.0
            assert future_int.abs().max() == 0.0
        # the j = n and j = 0 L1 slots are handled outside the matrix
        assert residual._w_l1[n, n] == 0.0
        assert residual._w_l1[n, 0] == 0.0


def test_history_matrix_shape_and_nodes():
    _, residual, x, _, _ = _setup(25, "full")
    history = residual._history_matrix(x)
    assert history.shape == (x.shape[0], residual.mesh.N + 1)
    # column j must equal a direct evaluation at mesh node t_j
    t_nodes = residual.mesh.get_nodes()
    for j in (0, 7, 25):
        direct = residual.model(x, t_nodes[j].expand(x.shape[0])).squeeze()
        assert torch.allclose(history[:, j], direct, atol=1e-12, rtol=0.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

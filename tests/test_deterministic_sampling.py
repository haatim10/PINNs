"""Tests for deterministic data sampling in the integro-differential dataset."""

import torch

from scripts.train_integro_diff import IntegroDiffDataset
from src.mesh import GradedMesh


def _build_config(data_seed: int, fixed_collocation: bool = False) -> dict:
    return {
        "seed": 42,
        "problem": {
            "alpha": 0.5,
            "x_min": 0.0,
            "x_max": 1.0,
            "solution": {"family": "cosine"},
        },
        "discretization": {
            "N_x": 16,
            "N_t": 10,
            "N_collocation": 8,
            "N_boundary": 4,
            "N_initial": 4,
        },
        "reproducibility": {
            "deterministic_sampling": True,
            "fixed_collocation": fixed_collocation,
            "data_seed": data_seed,
        },
    }


def _assert_batches_equal(batch_a: dict, batch_b: dict):
    keys = {"x_coll", "t_coll", "n_coll", "x_bc", "t_bc", "u_bc", "x_ic", "t_ic", "u_ic"}
    for key in keys:
        assert torch.equal(batch_a[key], batch_b[key]), f"Mismatch in field: {key}"


def test_same_data_seed_produces_same_sampling_sequence():
    mesh_1 = GradedMesh(N=10, beta=2.0, t_max=1.0, device="cpu")
    mesh_2 = GradedMesh(N=10, beta=2.0, t_max=1.0, device="cpu")
    config = _build_config(data_seed=777, fixed_collocation=False)

    ds_1 = IntegroDiffDataset(mesh_1, config, device="cpu")
    ds_2 = IntegroDiffDataset(mesh_2, config, device="cpu")

    batch_1_epoch_1 = ds_1.get_training_data()
    batch_2_epoch_1 = ds_2.get_training_data()
    _assert_batches_equal(batch_1_epoch_1, batch_2_epoch_1)

    batch_1_epoch_2 = ds_1.get_training_data()
    batch_2_epoch_2 = ds_2.get_training_data()
    _assert_batches_equal(batch_1_epoch_2, batch_2_epoch_2)


def test_different_data_seeds_change_collocation_sampling():
    mesh_a = GradedMesh(N=10, beta=2.0, t_max=1.0, device="cpu")
    mesh_b = GradedMesh(N=10, beta=2.0, t_max=1.0, device="cpu")

    ds_a = IntegroDiffDataset(mesh_a, _build_config(data_seed=101), device="cpu")
    ds_b = IntegroDiffDataset(mesh_b, _build_config(data_seed=202), device="cpu")

    batch_a = ds_a.get_training_data()
    batch_b = ds_b.get_training_data()
    assert not torch.equal(batch_a["n_coll"], batch_b["n_coll"])


def test_fixed_collocation_reuses_same_batch():
    mesh = GradedMesh(N=10, beta=2.0, t_max=1.0, device="cpu")
    ds = IntegroDiffDataset(mesh, _build_config(data_seed=909, fixed_collocation=True), device="cpu")

    first_batch = ds.get_training_data()
    second_batch = ds.get_training_data()
    _assert_batches_equal(first_batch, second_batch)

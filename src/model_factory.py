"""Model factory for classical and quantum-ready PINN variants."""

from typing import Dict, Optional

from .exact_te_qpinn_pennylane import ExactTEQPINNPennyLane
from .memory_features import MemoryFeatureBuilder
from .model import PINN
from .quantum_ready_model import QuantumReadyPINN
from .te_qpinn_surrogate_model import TEQPINNSurrogatePINN


def _build_memory_feature_builder(
    network_config: Dict,
    problem_config: Optional[Dict],
) -> MemoryFeatureBuilder:
    problem_cfg = problem_config or {}
    return MemoryFeatureBuilder(
        memory_features=network_config.get("memory_features", "none"),
        memory_feature_set=network_config.get("memory_feature_set", "basic_fractional"),
        memory_epsilon=network_config.get("memory_epsilon", 1e-8),
        memory_feature_normalization=network_config.get("memory_feature_normalization", "scale"),
        alpha=problem_cfg.get("alpha"),
        beta=problem_cfg.get("beta"),
        x_min=problem_cfg.get("x_min", 0.0),
        x_max=problem_cfg.get("x_max", 1.0),
        t_min=problem_cfg.get("t_min", 0.0),
        t_max=problem_cfg.get("t_max", 1.0),
    )


def build_model(network_config: Dict, device: str = "cpu", problem_config: Optional[Dict] = None):
    """Build a model from network configuration.

    Supported model_type values:
    - classical (default)
    - quantum_ready
    - hybrid_quantum (alias for quantum_ready)
    - te_qpinn_surrogate
    - te_qpinn (alias for te_qpinn_surrogate)
    - te_qpinn_pennylane
    - exact_te_qpinn_pennylane (alias for te_qpinn_pennylane)
    - exact_pqc_te_qpinn (alias for te_qpinn_pennylane)

    Memory features are configured through network keys:
    - memory_features: none | analytic
    - memory_feature_set: basic_fractional
    - memory_epsilon
    - memory_feature_normalization: none | scale
    """
    if network_config is None:
        network_config = {}

    model_type = str(network_config.get("model_type", "classical")).lower()
    memory_builder = _build_memory_feature_builder(network_config, problem_config)
    use_memory_builder = memory_builder.memory_features != "none"
    base_input_dim = int(network_config.get("input_dim", 2))
    effective_input_dim = int(memory_builder.output_dim) if use_memory_builder else base_input_dim

    common_kwargs = {
        "input_dim": base_input_dim,
        "output_dim": network_config.get("output_dim", 1),
        "hidden_layers": network_config.get("hidden_layers", [64, 64, 64, 64]),
        "activation": network_config.get("activation", "tanh"),
        "device": device,
    }

    if model_type in {"classical", "pinn"}:
        return PINN(
            input_dim=effective_input_dim,
            output_dim=common_kwargs["output_dim"],
            hidden_layers=common_kwargs["hidden_layers"],
            activation=common_kwargs["activation"],
            device=device,
            memory_feature_builder=memory_builder if use_memory_builder else None,
        )

    if model_type in {"quantum_ready", "hybrid_quantum", "qready"}:
        return QuantumReadyPINN(
            **common_kwargs,
            quantum=network_config.get("quantum", {}),
        )

    if model_type in {"te_qpinn_surrogate", "te_qpinn", "teqpinn", "te-qpinn"}:
        return TEQPINNSurrogatePINN(
            **common_kwargs,
            te_qpinn=network_config.get("te_qpinn", {}),
            problem=problem_config or {},
            memory_feature_builder=memory_builder if use_memory_builder else None,
        )

    if model_type in {
        "te_qpinn_pennylane",
        "exact_te_qpinn_pennylane",
        "exact_pqc_te_qpinn",
    }:
        return ExactTEQPINNPennyLane(
            **common_kwargs,
            te_qpinn_pennylane=network_config.get("te_qpinn_pennylane", {}),
            problem=problem_config or {},
        )

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Use one of: classical, quantum_ready, hybrid_quantum, te_qpinn_surrogate, "
        "te_qpinn, te_qpinn_pennylane, exact_te_qpinn_pennylane, exact_pqc_te_qpinn."
    )


def model_name_from_config(network_config: Dict) -> str:
    """Return normalized model type name for logging/outputs."""
    if network_config is None:
        return "classical"

    model_type = str(network_config.get("model_type", "classical")).lower()
    if model_type in {"hybrid_quantum", "qready"}:
        return "quantum_ready"
    if model_type in {"te_qpinn", "teqpinn", "te-qpinn"}:
        return "te_qpinn_surrogate"
    if model_type in {
        "te_qpinn_pennylane",
        "exact_te_qpinn_pennylane",
        "exact_pqc_te_qpinn",
    }:
        return "te_qpinn_pennylane"
    return model_type

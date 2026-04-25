"""Model factory for classical and quantum-ready PINN variants."""

from typing import Dict

from .model import PINN
from .quantum_ready_model import QuantumReadyPINN


def build_model(network_config: Dict, device: str = "cpu"):
    """Build a model from network configuration.

    Supported model_type values:
    - classical (default)
    - quantum_ready
    - hybrid_quantum (alias for quantum_ready)
    """
    if network_config is None:
        network_config = {}

    model_type = str(network_config.get("model_type", "classical")).lower()

    common_kwargs = {
        "input_dim": network_config.get("input_dim", 2),
        "output_dim": network_config.get("output_dim", 1),
        "hidden_layers": network_config.get("hidden_layers", [64, 64, 64, 64]),
        "activation": network_config.get("activation", "tanh"),
        "device": device,
    }

    if model_type in {"classical", "pinn"}:
        return PINN(**common_kwargs)

    if model_type in {"quantum_ready", "hybrid_quantum", "qready"}:
        return QuantumReadyPINN(
            **common_kwargs,
            quantum=network_config.get("quantum", {}),
        )

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        "Use one of: classical, quantum_ready, hybrid_quantum."
    )


def model_name_from_config(network_config: Dict) -> str:
    """Return normalized model type name for logging/outputs."""
    if network_config is None:
        return "classical"

    model_type = str(network_config.get("model_type", "classical")).lower()
    if model_type in {"hybrid_quantum", "qready"}:
        return "quantum_ready"
    return model_type

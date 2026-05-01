"""Application-focused demo modules."""

from .wireless_channel import (
    MLPTrainConfig,
    WirelessChannelConfig,
    build_channel_features,
    generate_wireless_channel,
    make_channel_split,
    run_ar1_baseline,
    run_linear_baseline,
    train_mlp_regressor,
)

__all__ = [
    "MLPTrainConfig",
    "WirelessChannelConfig",
    "build_channel_features",
    "generate_wireless_channel",
    "make_channel_split",
    "run_ar1_baseline",
    "run_linear_baseline",
    "train_mlp_regressor",
]


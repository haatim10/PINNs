"""Shared deterministic memory feature builder for PINN variants."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


class MemoryFeatureBuilder(nn.Module):
    """Build local or memory-aware analytic features from (x, t)."""

    SUPPORTED_MEMORY_FEATURES = {"none", "analytic"}
    SUPPORTED_FEATURE_SETS = {"basic_fractional"}
    SUPPORTED_NORMALIZATION = {"none", "scale"}

    def __init__(
        self,
        memory_features: str = "none",
        memory_feature_set: str = "basic_fractional",
        memory_epsilon: float = 1e-8,
        memory_feature_normalization: str = "scale",
        *,
        alpha: float | None = None,
        beta: float | None = None,
        x_min: float = 0.0,
        x_max: float = 1.0,
        t_min: float = 0.0,
        t_max: float = 1.0,
    ):
        super().__init__()
        self.memory_features = str(memory_features).lower()
        self.memory_feature_set = str(memory_feature_set).lower()
        self.memory_feature_normalization = str(memory_feature_normalization).lower()
        self.memory_epsilon = float(memory_epsilon)

        if self.memory_features not in self.SUPPORTED_MEMORY_FEATURES:
            raise ValueError(
                f"Unsupported memory_features '{memory_features}'. "
                "Use 'none' or 'analytic'."
            )
        if self.memory_feature_set not in self.SUPPORTED_FEATURE_SETS:
            raise ValueError(
                f"Unsupported memory_feature_set '{memory_feature_set}'. "
                "Use 'basic_fractional'."
            )
        if self.memory_feature_normalization not in self.SUPPORTED_NORMALIZATION:
            raise ValueError(
                f"Unsupported memory_feature_normalization '{memory_feature_normalization}'. "
                "Use 'none' or 'scale'."
            )
        if self.memory_epsilon <= 0.0:
            raise ValueError("memory_epsilon must be > 0.")

        self.alpha = None if alpha is None else float(alpha)
        self.beta = None if beta is None else float(beta)
        self.x_min = float(x_min)
        self.x_max = float(x_max)
        self.t_min = float(t_min)
        self.t_max = float(t_max)

        if self.memory_features == "analytic" and (self.alpha is None or self.beta is None):
            raise ValueError(
                "memory_features='analytic' requires problem alpha and beta."
            )

        scale_mins, scale_maxs = self._default_feature_bounds()
        self.register_buffer("scale_mins", scale_mins.reshape(1, -1))
        self.register_buffer("scale_maxs", scale_maxs.reshape(1, -1))

    @property
    def output_dim(self) -> int:
        if self.memory_features == "analytic":
            return 8
        return 2

    def default_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return deterministic per-feature bounds for current mode."""
        mins = self.scale_mins.reshape(-1).detach().clone()
        maxs = self.scale_maxs.reshape(-1).detach().clone()
        return mins, maxs

    def _default_feature_bounds(self) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.memory_features == "none":
            return (
                torch.tensor([self.x_min, self.t_min], dtype=torch.float64),
                torch.tensor([self.x_max, self.t_max], dtype=torch.float64),
            )

        # Analytic feature set bounds over x in [x_min, x_max], t in [t_min, t_max].
        # t powers/log terms are evaluated over t_safe to match forward path.
        t_min_safe = max(self.t_min, self.memory_epsilon)
        t_max_safe = max(self.t_max, self.memory_epsilon)
        alpha = float(self.alpha)
        beta = float(self.beta)

        t_alpha_min = min(t_min_safe**alpha, t_max_safe**alpha)
        t_alpha_max = max(t_min_safe**alpha, t_max_safe**alpha)

        one_minus_alpha = 1.0 - alpha
        t_one_minus_alpha_min = min(
            t_min_safe**one_minus_alpha, t_max_safe**one_minus_alpha
        )
        t_one_minus_alpha_max = max(
            t_min_safe**one_minus_alpha, t_max_safe**one_minus_alpha
        )

        one_minus_beta = 1.0 - beta
        t_one_minus_beta_min = min(t_min_safe**one_minus_beta, t_max_safe**one_minus_beta)
        t_one_minus_beta_max = max(t_min_safe**one_minus_beta, t_max_safe**one_minus_beta)

        log_min = min(torch.log1p(torch.tensor(t_min_safe, dtype=torch.float64)).item(),
                      torch.log1p(torch.tensor(t_max_safe, dtype=torch.float64)).item())
        log_max = max(torch.log1p(torch.tensor(t_min_safe, dtype=torch.float64)).item(),
                      torch.log1p(torch.tensor(t_max_safe, dtype=torch.float64)).item())

        xt_candidates = [
            self.x_min * self.t_min,
            self.x_min * self.t_max,
            self.x_max * self.t_min,
            self.x_max * self.t_max,
        ]
        xt_min = min(xt_candidates)
        xt_max = max(xt_candidates)

        xt_alpha_candidates = [
            self.x_min * t_alpha_min,
            self.x_min * t_alpha_max,
            self.x_max * t_alpha_min,
            self.x_max * t_alpha_max,
        ]
        xt_alpha_min = min(xt_alpha_candidates)
        xt_alpha_max = max(xt_alpha_candidates)

        mins = torch.tensor(
            [
                self.x_min,
                self.t_min,
                t_alpha_min,
                t_one_minus_alpha_min,
                t_one_minus_beta_min,
                log_min,
                xt_min,
                xt_alpha_min,
            ],
            dtype=torch.float64,
        )
        maxs = torch.tensor(
            [
                self.x_max,
                self.t_max,
                t_alpha_max,
                t_one_minus_alpha_max,
                t_one_minus_beta_max,
                log_max,
                xt_max,
                xt_alpha_max,
            ],
            dtype=torch.float64,
        )
        return mins, maxs

    def _apply_scale(self, features: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(self.scale_maxs - self.scale_mins, min=1e-12)
        normalized = (features - self.scale_mins) / denom
        return normalized * 2.0 - 1.0

    def _prepare_xy(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        return x, t

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x, t = self._prepare_xy(x, t)
        if self.memory_features == "none":
            return torch.cat([x, t], dim=-1)

        t_safe = torch.clamp(t, min=self.memory_epsilon)
        alpha = float(self.alpha)
        beta = float(self.beta)

        t_alpha = torch.pow(t_safe, alpha)
        features = torch.cat(
            [
                x,
                t,
                t_alpha,
                torch.pow(t_safe, 1.0 - alpha),
                torch.pow(t_safe, 1.0 - beta),
                torch.log1p(t_safe),
                x * t,
                x * t_alpha,
            ],
            dim=-1,
        )
        if self.memory_feature_normalization == "scale":
            features = self._apply_scale(features)
        return features

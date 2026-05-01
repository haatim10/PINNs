#!/usr/bin/env python3
"""Fail-fast validator for benchmark run configs.

Checks that every run config in a benchmark plan contains expected:
- problem.alpha
- problem.beta
- problem.solution.time_power

Also verifies required run names are present in the benchmark plan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List

import yaml


DEFAULT_REQUIRED_RUNS = [
    "classical_pi",
    "classical_memory_pi",
    "te_fixed_pi",
    "te_layernorm_post_quantum_pi",
    "te_memory_analytic_pi",
]


def load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"YAML root must be a mapping: {path}")
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate benchmark run config parameters.")
    parser.add_argument("--benchmark-config", required=True, help="Benchmark plan YAML path")
    parser.add_argument("--expected-alpha", type=float, required=True)
    parser.add_argument("--expected-beta", type=float, required=True)
    parser.add_argument("--expected-time-power", type=float, required=True)
    parser.add_argument(
        "--required-runs",
        default=",".join(DEFAULT_REQUIRED_RUNS),
        help="Comma-separated required run names",
    )
    return parser.parse_args()


def assert_close(name: str, observed: float, expected: float, cfg_path: Path):
    if abs(observed - expected) > 1e-12:
        raise ValueError(
            f"{cfg_path}: expected {name}={expected}, observed {observed}"
        )


def main() -> None:
    args = parse_args()
    benchmark_path = Path(args.benchmark_config).resolve()
    benchmark_cfg = load_yaml(benchmark_path)
    runs: List[Dict] = benchmark_cfg.get("runs", [])
    if not runs:
        raise ValueError(f"{benchmark_path}: missing or empty 'runs' list")

    required_runs = [item.strip() for item in args.required_runs.split(",") if item.strip()]
    observed_run_names = [str(run.get("name", "")).strip() for run in runs]
    missing_runs = [name for name in required_runs if name not in observed_run_names]
    if missing_runs:
        raise ValueError(
            f"{benchmark_path}: missing required run names: {missing_runs}; "
            f"observed={observed_run_names}"
        )

    for run in runs:
        run_name = str(run.get("name", "<unnamed>"))
        cfg_ref = run.get("config")
        if not cfg_ref:
            raise ValueError(f"{benchmark_path}: run '{run_name}' missing 'config'")
        cfg_ref_path = Path(str(cfg_ref))
        candidates = [
            cfg_ref_path if cfg_ref_path.is_absolute() else Path.cwd() / cfg_ref_path,
            benchmark_path.parent / cfg_ref_path,
        ]
        cfg_path = None
        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved.exists():
                cfg_path = resolved
                break
        if cfg_path is None:
            tried = ", ".join(str(path.resolve()) for path in candidates)
            raise FileNotFoundError(
                f"{benchmark_path}: config for run '{run_name}' not found. Tried: {tried}"
            )

        cfg = load_yaml(cfg_path)
        problem = cfg.get("problem", {})
        solution = problem.get("solution", {})
        alpha = float(problem.get("alpha"))
        beta = float(problem.get("beta"))
        time_power = float(solution.get("time_power"))

        assert_close("problem.alpha", alpha, args.expected_alpha, cfg_path)
        assert_close("problem.beta", beta, args.expected_beta, cfg_path)
        assert_close("problem.solution.time_power", time_power, args.expected_time_power, cfg_path)

    print(f"[OK] {benchmark_path} validated for alpha={args.expected_alpha}, beta={args.expected_beta}, time_power={args.expected_time_power}")
    print(f"[OK] required runs present: {required_runs}")


if __name__ == "__main__":
    main()

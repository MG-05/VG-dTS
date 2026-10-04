from __future__ import annotations

import argparse
from typing import Sequence

from src.adts.vgdts_config import make_benchmark_vgdts_v21_params, make_default_vgdts_v21_tuning_grid

from .pipeline import run_publication_bundle


def _parse_scalar(value: str) -> bool | int | float | str:
    lowered = value.strip().lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value.strip()


def _parse_vgdts_grid(
    items: list[str] | None,
) -> dict[str, list[bool | int | float | str]] | None:
    if not items:
        return None

    grid: dict[str, list[bool | int | float | str]] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --vgdts-grid {item!r}. Expected KEY=V1,V2,...")
        key, raw_values = item.split("=", 1)
        key = key.strip()
        values = [v.strip() for v in raw_values.split(",") if v.strip()]
        if not key:
            raise ValueError(f"Invalid --vgdts-grid {item!r}. Key cannot be empty.")
        if not values:
            raise ValueError(f"Invalid --vgdts-grid {item!r}. Provide at least one value.")
        grid.setdefault(key, []).extend(_parse_scalar(v) for v in values)

    return grid


def _parse_vgdts_params(items: list[str] | None) -> dict[str, bool | int | float | str]:
    overrides: dict[str, bool | int | float | str] = {}
    if not items:
        return overrides

    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --vgdts-param {item!r}. Expected KEY=VALUE.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid --vgdts-param {item!r}. Key cannot be empty.")
        if raw_value == "":
            raise ValueError(f"Invalid --vgdts-param {item!r}. Value cannot be empty.")
        overrides[key] = _parse_scalar(raw_value)

    return overrides


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate publication-ready VG-dTS comparison figures and summaries."
    )
    parser.add_argument("--output-dir", default="report/project_publication")
    parser.add_argument("--horizon", type=int, default=10_000)
    parser.add_argument("--n-arms", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--tuning-runs", type=int, default=40)
    parser.add_argument("--eval-runs", type=int, default=120)
    parser.add_argument(
        "--single-environment",
        default="fast",
        help=(
            "Environment for the standalone tuning plot: "
            "slow|fast|abrupt|mixed|random_breakpoints|random_drift|"
            "global_switching|per_arm_switching|random_signal"
        ),
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Shortcut preset: tuning-runs=10, eval-runs=20, horizon=min(horizon, 2500).",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Shortcut preset: tuning-runs=60, eval-runs=180, horizon=max(horizon, 10000).",
    )
    parser.add_argument(
        "--vgdts-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override benchmark VG-dTS base params used before the sweep.",
    )
    parser.add_argument(
        "--vgdts-grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2,...",
        help="Override VG-dTS sweep grid entries.",
    )
    parser.add_argument("--no-vgdts-sweep", action="store_true", help="Disable VG-dTS sweep and use base params.")
    parser.add_argument("--show", action="store_true", help="Display generated figures.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.quick and args.paper:
        parser.error("Use at most one preset: --quick or --paper.")

    horizon = args.horizon
    tuning_runs = args.tuning_runs
    eval_runs = args.eval_runs

    if args.quick:
        tuning_runs = 10
        eval_runs = 20
        horizon = min(horizon, 2500)
    elif args.paper:
        tuning_runs = 60
        eval_runs = 180
        horizon = max(horizon, 10_000)

    try:
        vgdts_params = make_benchmark_vgdts_v21_params(**_parse_vgdts_params(args.vgdts_param))
        cli_vgdts_grid = _parse_vgdts_grid(args.vgdts_grid)
    except ValueError as exc:
        parser.error(str(exc))

    if args.no_vgdts_sweep:
        vgdts_grid = None
    elif cli_vgdts_grid is None:
        vgdts_grid = make_default_vgdts_v21_tuning_grid()
    else:
        vgdts_grid = cli_vgdts_grid

    artifacts = run_publication_bundle(
        output_dir=args.output_dir,
        horizon=horizon,
        n_arms=args.n_arms,
        seed=args.seed,
        tuning_runs=tuning_runs,
        eval_runs=eval_runs,
        single_environment_key=args.single_environment,
        vgdts_params=vgdts_params,
        vgdts_grid=vgdts_grid,
        show=args.show,
    )

    print("Saved publication artifacts:")
    print(f"  single-env plot: {artifacts.single_env_plot}")
    print(f"  single-env summary: {artifacts.single_env_summary_json}")
    print(f"  original-env plot: {artifacts.original_envs_plot}")
    print(f"  original-env summary: {artifacts.original_envs_summary_json}")
    print(f"  random-signal plot: {artifacts.random_signal_plot}")
    print(f"  random-signal summary: {artifacts.random_signal_summary_json}")
    print("  oracle environment plots:")
    for key in sorted(artifacts.oracle_environment_plots):
        print(f"    {key}: {artifacts.oracle_environment_plots[key]}")
    print("  fixed-parameter environment plots:")
    for key in sorted(artifacts.fixed_environment_plots):
        print(f"    {key}: {artifacts.fixed_environment_plots[key]}")
    print(f"  optimized heatmap: {artifacts.optimized_heatmap_plot}")
    print(f"  optimized heatmap summary: {artifacts.optimized_heatmap_summary_json}")
    print(f"  fixed-parameter heatmap: {artifacts.fixed_heatmap_plot}")
    print(f"  fixed-parameter heatmap summary: {artifacts.fixed_heatmap_summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

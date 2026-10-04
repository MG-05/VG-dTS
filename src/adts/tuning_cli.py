from __future__ import annotations

import argparse
from typing import Sequence


def _parse_numeric(value: str) -> int | float:
    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError as exc:
        raise ValueError(f"Expected a numeric value, got {value!r}.") from exc


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


def _parse_env_kwargs(items: list[str] | None) -> dict[str, int | float] | None:
    if not items:
        return None

    kwargs: dict[str, int | float] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid --env-kw {item!r}. Expected KEY=VALUE.")

        key, raw_value = item.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not key:
            raise ValueError(f"Invalid --env-kw {item!r}. Key cannot be empty.")
        kwargs[key] = _parse_numeric(raw_value)

    return kwargs


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
        values = [value.strip() for value in raw_values.split(",") if value.strip()]
        if not key:
            raise ValueError(f"Invalid --vgdts-grid {item!r}. Key cannot be empty.")
        if not values:
            raise ValueError(f"Invalid --vgdts-grid {item!r}. Provide at least one value.")
        grid.setdefault(key, []).extend(_parse_scalar(value) for value in values)

    return grid


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run single-environment policy tuning and save the comparison plot."
    )
    parser.add_argument(
        "--environment",
        default="slow",
        help="Environment key: slow|fast|abrupt|mixed|random_breakpoints|random_drift.",
    )
    parser.add_argument("--n-runs", type=int, default=100, help="Evaluation runs per selected policy.")
    parser.add_argument(
        "--tuning-runs",
        type=int,
        default=None,
        help="Runs used during grid search (defaults to --n-runs).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base RNG seed.")
    parser.add_argument(
        "--lambda-grid",
        type=float,
        nargs="+",
        default=None,
        metavar="L",
        help="Optional lambda values for dTS/dOTS/Dynamic-TS tuning.",
    )
    parser.add_argument(
        "--rexp3-gamma-grid",
        type=float,
        nargs="+",
        default=None,
        metavar="G",
        help="Optional gamma values for REXP3 tuning.",
    )
    parser.add_argument(
        "--rexp3-delta-grid",
        type=int,
        nargs="+",
        default=None,
        metavar="D",
        help="Optional Delta values for REXP3 tuning.",
    )
    parser.add_argument(
        "--vgdts-grid",
        action="append",
        default=[],
        metavar="KEY=V1,V2,...",
        help=(
            "VG-dTS grid override. Repeat as needed, e.g. "
            "--vgdts-grid gamma_default=0.45,0.85 --vgdts-grid n0=0,25."
        ),
    )
    parser.add_argument(
        "--no-vgdts-tuning",
        action="store_true",
        help="Use the fixed benchmark VG-dTS parameters instead of the CLI sweep.",
    )
    parser.add_argument(
        "--env-kw",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Environment kwarg override. Repeat as needed, "
            "e.g. --env-kw horizon=1000 --env-kw period=100."
        ),
    )
    parser.add_argument(
        "--output-path",
        default="report/figures/single_env_lambda_tuning.png",
        help="Where to save the output figure.",
    )
    parser.add_argument("--show", action="store_true", help="Display the plot window after rendering.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        environment_kwargs = _parse_env_kwargs(args.env_kw)
        vgdts_grid = _parse_vgdts_grid(args.vgdts_grid)
    except ValueError as exc:
        parser.error(str(exc))

    from .experiments import run_single_env_lambda_tuning_experiment
    from .vgdts_config import make_default_vgdts_tuning_grid

    if args.no_vgdts_tuning:
        vgdts_grid = None
    elif vgdts_grid is None:
        vgdts_grid = make_default_vgdts_tuning_grid()

    plot_path, tuned = run_single_env_lambda_tuning_experiment(
        environment=args.environment,
        n_runs=args.n_runs,
        tuning_runs=args.tuning_runs,
        seed=args.seed,
        lambda_grid=args.lambda_grid,
        rexp3_gamma_grid=args.rexp3_gamma_grid,
        rexp3_delta_grid=args.rexp3_delta_grid,
        environment_kwargs=environment_kwargs,
        vgdts_grid=vgdts_grid,
        output_path=args.output_path,
        show=args.show,
    )

    print(f"Saved plot to: {plot_path}")
    print("Best parameters by policy:")
    for policy_name in sorted(tuned):
        result = tuned[policy_name]
        print(
            f"  {policy_name}: params={result.best_params}, "
            f"final_normalized_regret={result.final_normalized_regret:.6f}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

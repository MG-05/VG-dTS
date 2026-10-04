from __future__ import annotations

import argparse
from typing import Sequence

from .experiments import run_paper_algorithm_benchmarks
from .vgdts_config import make_benchmark_vgdts_params


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
        description=(
            "Run paper-algorithm benchmarks versus VG-dTS on existing and paper-specific environments."
        )
    )
    parser.add_argument(
        "--n-runs-nonstationary",
        type=int,
        default=20,
        help="Number of Monte-Carlo runs per policy/environment in non-stationary suite.",
    )
    parser.add_argument(
        "--n-runs-recovering",
        type=int,
        default=20,
        help="Number of Monte-Carlo runs per policy in recovering-bandit suite.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Base random seed.")
    parser.add_argument(
        "--output-dir",
        default="report",
        help="Output root directory for figures and JSON summary.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation and only write JSON summary.",
    )
    parser.add_argument(
        "--vgdts-param",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "Override benchmark VG-dTS parameters, e.g. "
            "--vgdts-param gamma_default=0.9 --vgdts-param n0=25."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        vgdts_params = make_benchmark_vgdts_params(**_parse_vgdts_params(args.vgdts_param))
    except ValueError as exc:
        parser.error(str(exc))

    nonstationary_plot, recovering_plot, summary_path, _ = run_paper_algorithm_benchmarks(
        n_runs_nonstationary=args.n_runs_nonstationary,
        n_runs_recovering=args.n_runs_recovering,
        seed=args.seed,
        output_dir=args.output_dir,
        vgdts_params=vgdts_params,
        make_plots=not args.no_plots,
    )

    if args.no_plots:
        print("Skipped plot generation (--no-plots).")
    else:
        print(f"Saved non-stationary benchmark plot to: {nonstationary_plot}")
        print(f"Saved recovering benchmark plot to: {recovering_plot}")
    print(f"Saved benchmark summary JSON to: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

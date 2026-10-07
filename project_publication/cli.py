"""Command-line entry point for the paper's experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .reproduce import reproduce, render, verify


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["paper", "retune", "smoke", "plot", "verify"], default="paper",
                        help="paper: replay saved parameters; retune: search original grids; smoke: tiny integration run; plot: redraw an existing run; verify: compare against paper JSON")
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/reproduction"))
    parser.add_argument("--reference-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-runs", type=int, default=1000)
    parser.add_argument("--tuning-runs", type=int, default=70)
    parser.add_argument("--workers", type=int, default=1, help="Independent environment jobs (use 2–4 on a laptop).")
    args = parser.parse_args(argv)
    if min(args.eval_runs, args.tuning_runs, args.workers) <= 0 or args.seed < 0:
        parser.error("Run counts and workers must be positive; seed must be nonnegative.")
    if args.mode == "verify":
        return 0 if verify(args.output_dir, args.reference_dir) else 1
    if args.mode == "plot":
        render(args.output_dir)
    else:
        reproduce(output_dir=args.output_dir, reference_dir=args.reference_dir,
                  mode=args.mode, seed=args.seed, tuning_runs=args.tuning_runs,
                  eval_runs=args.eval_runs, workers=args.workers)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

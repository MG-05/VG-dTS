"""Plot selected paper results for the README from a completed reproduction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .pipeline import ALGORITHM_COLORS, ALGORITHM_ORDER
from .reproduce import FIXED, _load_evaluations


SELECTED_ENVIRONMENTS = (
    ("fast", "Fast periodic drift"),
    ("abrupt", "Abrupt changes"),
    ("global_switching", "Global switching"),
    ("per_arm_switching", "Per-arm switching"),
)


def plot_selected_regret_curves(run_dir: Path, reference_dir: Path, output_path: Path) -> Path:
    """Show the four fixed-protocol wins, without smoothing or omitting baselines."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metadata = json.loads((run_dir / "metadata.json").read_text())
    if (metadata["status"] != "complete" or metadata["horizon"] != 5000
            or metadata["eval_runs"] != 1000 or metadata["seed"] != 0):
        raise ValueError("Use a completed paper reproduction: T=5000, seed=0, 1000 evaluation runs.")
    reference = json.loads((reference_dir / f"{FIXED}.json").read_text())["evaluations"]
    evaluations = _load_evaluations(run_dir, "fixed", [key for key, _ in SELECTED_ENVIRONMENTS])
    for key, result in evaluations.items():
        for name, policy in result.policies.items():
            expected = reference[key]["policies"][name]["eval_final_normalized_regret"]
            if not np.isclose(policy.avg_norm_regret_t[-1], expected, rtol=0, atol=1e-12):
                raise ValueError(f"The {key}/{name} curve does not match the archived paper result.")

    with plt.rc_context({"font.size": 11, "axes.titlesize": 13, "axes.labelsize": 11}):
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.6), sharex=True, sharey=True)
        handles = {}
        for ax, (key, title) in zip(axes.flat, SELECTED_ENVIRONMENTS):
            result = evaluations[key]
            # Draw VG-dTS last so its trajectory remains visible at intersections.
            for name in (*ALGORITHM_ORDER[1:], "VG-dTS"):
                values = result.policies[name].avg_norm_regret_t
                line, = ax.plot(np.arange(1, len(values) + 1), values,
                                color=ALGORITHM_COLORS[name],
                                linewidth=2.7 if name == "VG-dTS" else 1.25,
                                alpha=1.0 if name == "VG-dTS" else 0.85,
                                label=name)
                handles[name] = line
            final = result.policies["VG-dTS"].eval_final_normalized_regret
            ax.set_title(title, loc="left", fontweight="semibold", pad=10)
            ax.text(0.97, 0.91, f"VG-dTS final: {final:.4f}", transform=ax.transAxes,
                    ha="right", color=ALGORITHM_COLORS["VG-dTS"], fontweight="semibold")
            ax.set_xlim(0, 5000)
            ax.set_xticks([0, 1000, 2000, 3000, 4000, 5000])
            ax.grid(alpha=0.18, linewidth=0.7)
            ax.spines[["top", "right"]].set_visible(False)
        for ax in axes[-1]:
            ax.set_xlabel("Round")
        for ax in axes[:, 0]:
            ax.set_ylabel("Mean normalized regret")
        fig.suptitle("Selected fixed-parameter wins", fontsize=18, fontweight="semibold", y=0.98)
        fig.text(0.5, 0.931, "Same parameters across environments · 1,000 rollouts · Lower is better",
                 ha="center", fontsize=11, color="#454545")
        fig.legend([handles[name] for name in ALGORITHM_ORDER], list(ALGORITHM_ORDER),
                   loc="lower center", bbox_to_anchor=(0.5, 0.008), ncol=7,
                   frameon=False, handlelength=2.4, columnspacing=1.6)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.855, bottom=0.13, hspace=0.32, wspace=0.14)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=180, facecolor="white", bbox_inches="tight", pad_inches=0.15)
        plt.close(fig)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("artifacts/reproduction"))
    parser.add_argument("--reference-dir", type=Path, default=Path("results/paper"))
    parser.add_argument("--output-path", type=Path, default=Path("docs/figures/fixed_parameter_regret.png"))
    args = parser.parse_args()
    print(plot_selected_regret_curves(args.run_dir, args.reference_dir, args.output_path))


if __name__ == "__main__":
    main()

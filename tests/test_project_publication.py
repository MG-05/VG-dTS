"""Checks for paper scope, archival integrity, and the full reproduction path."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import re

import numpy as np
import pytest

from project_publication.cli import main
from project_publication.pipeline import (
    ALGORITHM_ORDER, build_publication_environment_suite,
    _build_policy_candidates_tuned, _build_policy_candidates_fixed,
    _compute_avg_curves_and_final_norm_regret,
)
from project_publication.reproduce import TUNED, FIXED, RANDOM, _saved_candidates

ROOT = Path(__file__).resolve().parents[1]


def test_paper_candidates_and_fixed_parameters():
    assert ALGORITHM_ORDER == ("VG-dTS", "dTS", "dOTS", "TS", "REXP3", "Dynamic TS", "Beta-SWTS")
    tuned = _build_policy_candidates_tuned(5000)
    assert [len(tuned[k]) for k in ALGORITHM_ORDER] == [64, 7, 7, 1, 15, 7, 5]
    saved = json.loads((ROOT / "results/paper" / f"{FIXED}.json").read_text())
    fixed = _build_policy_candidates_fixed(5000)
    for name in ALGORITHM_ORDER:
        assert fixed[name][0].best_params == saved["fixed_parameters"][name]
    for artifact in [TUNED, RANDOM]:
        payload = json.loads((ROOT / "results/paper" / f"{artifact}.json").read_text())
        for record in ([payload] if artifact == RANDOM else payload["evaluations"].values()):
            replay = _saved_candidates(record)
            for name in ALGORITHM_ORDER:
                assert any(asdict(c.params) == asdict(replay[name][0].params) for c in tuned[name])


def test_environment_suite_and_seed_determinism():
    a = build_publication_environment_suite(120, 4, 3)
    b = build_publication_environment_suite(120, 4, 3)
    assert [e.key for e in a] == ["slow", "fast", "abrupt", "mixed", "random_breakpoints", "random_drift", "global_switching", "per_arm_switching"]
    for x, y in zip(a, b):
        assert x.means_tk.shape == (120, 4)
        np.testing.assert_array_equal(x.means_tk, y.means_tk)
        assert np.all((x.means_tk >= 0) & (x.means_tk <= 1))


def test_realized_normalized_regret_can_be_negative():
    # This is realized oracle-minus-reward regret, not clipped pseudo-regret.
    mu = np.full((4, 3), 0.5)
    def all_successes(mu, params, rng):
        return np.ones(mu.shape[1])
    reward, regret, final = _compute_avg_curves_and_final_norm_regret(mu, all_successes, None, 2, 0)
    np.testing.assert_array_equal(reward, np.ones(3))
    np.testing.assert_array_equal(regret, np.full(3, -0.5))
    assert final == -0.5


def test_manuscript_and_original_figures_preserved():
    paper = ROOT / "report/ECE_270_Project_Publication"
    manifest = json.loads((paper / "OVERLEAF_MANIFEST.json").read_text())
    for name, digest in manifest["original_files_sha256"].items():
        assert hashlib.sha256((paper / name).read_bytes()).hexdigest() == digest, name
    refs = set()
    for name in ["main.tex", "appendix.tex"]:
        refs.update(re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", (paper / name).read_text()))
    assert refs == {p.relative_to(paper).as_posix() for p in paper.rglob("*.png")}
    assert len(refs) == 12


def test_end_to_end_smoke_and_plot_only(tmp_path):
    out = tmp_path / "run"
    assert main(["--mode", "smoke", "--output-dir", str(out), "--reference-dir", str(ROOT / "results/paper")]) == 0
    metadata = json.loads((out / "metadata.json").read_text())
    assert metadata["status"] == "complete"
    assert metadata["horizon"] == 120
    assert metadata["tuning_runs"] == 1 and metadata["eval_runs"] == 2
    assert len(list((out / "figures").rglob("*.png"))) == 12
    assert len(list((out / "runs").rglob("*.npz"))) == 17
    for name in [TUNED, FIXED, RANDOM]:
        assert (out / "results" / f"{name}.json").is_file()
    assert main(["--mode", "plot", "--output-dir", str(out)]) == 0
    with pytest.raises(FileExistsError):
        main(["--mode", "smoke", "--output-dir", str(out)])
    with pytest.raises(ValueError, match="Exact paper verification"):
        main(["--mode", "verify", "--output-dir", str(out)])


def test_cli_rejects_invalid_budgets():
    with pytest.raises(SystemExit):
        main(["--eval-runs", "0"])


def test_archived_breakpoint_realization():
    # NumPy 2.x changes multinomial draws here despite an identical seed.
    means = build_publication_environment_suite(5000, 4, 0)[4].means_tk
    assert means.max(axis=1).mean() == pytest.approx(0.7115935002265059, abs=1e-14, rel=0)


def test_full_reproduction_rejects_incompatible_numpy(tmp_path, monkeypatch):
    from project_publication.reproduce import reproduce
    monkeypatch.setattr(np, "__version__", "2.5.2")
    with pytest.raises(RuntimeError, match="numpy==1.26.4"):
        reproduce(output_dir=tmp_path, reference_dir=ROOT / "results/paper")

# Paper experiment pipeline

See the [root README](../README.md) for the paper overview and figures, and
[REPRODUCIBILITY.md](../REPRODUCIBILITY.md) for installation, replay, retuning,
plotting, manuscript builds, and exact experimental conventions.

```sh
python -m project_publication --help
```

The only policy families are VG-dTS, dTS, dOTS, TS, REXP3, Dynamic TS, and Beta-SWTS.
The runner generates only the figures used in the manuscript. `pipeline.py`
contains the recovered numerical/plotting implementation; `reproduce.py` records
run metadata and curves and compares new results against `results/paper/`.

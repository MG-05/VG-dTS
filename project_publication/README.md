# Paper experiment pipeline

See the [root README](../README.md) for installation, paper replay, retuning,
plotting, and manuscript build commands, and [REPRODUCIBILITY.md](../REPRODUCIBILITY.md)
for exact seeds and implementation conventions.

```sh
python -m project_publication --help
```

The only policy families are VG-dTS, dTS, dOTS, TS, REXP3, Dynamic TS, and Beta-SWTS.
The runner generates only the figures used in the manuscript. `pipeline.py`
contains the recovered numerical/plotting implementation; `reproduce.py` records
run metadata and curves and compares new results against `results/paper/`.

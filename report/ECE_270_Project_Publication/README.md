# ECE 270 project paper: VG-dTS

This folder contains the complete source of the **VG-dTS** project exported from
Overleaf on October 2, 2026:
https://www.overleaf.com/project/69a5f89cb76ba160fbed180d

The paper is titled *Variance-Gated Discounted Thompson Sampling for
Non-Stationary Bandits*. All 28 downloaded project files are preserved byte for
byte, including the original manuscript, appendix, bibliography, macros, package
configuration, NeurIPS 2025 style, and all 22 PNG figures. The figure folders and
relative paths are preserved. `OVERLEAF_MANIFEST.json` records the SHA-256 hash of
each downloaded file for checking against the export.

## Compile locally

Run from this folder:

```sh
./build.sh
```

The script uses pdfLaTeX and BibTeX from PATH, or `/Library/TeX/texbin` on macOS.
It runs pdfLaTeX once, BibTeX once, and pdfLaTeX three more times to resolve and
stabilize the bibliography, cross-references, and PDF outline. It disables shell
escape and stops on compilation errors. The result is `main.pdf` in this folder.
Generated auxiliary files are ignored by the folder's `.gitignore`.

The corresponding manual commands are:

```sh
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
pdflatex -interaction=nonstopmode -halt-on-error -no-shell-escape main.tex
```

A working TeX distribution is required. On a BasicTeX installation, first-time
compilation can generate missing font caches in the standard user TeX directory.

## Bundled dependencies

To support the installed BasicTeX distribution, this folder also includes unchanged
copies of `environ.sty`, `trimspaces.sty`, `nicefrac.sty`, `placeins.sty`,
`multirow.sty`, `threeparttable.sty`, `algorithm.sty`, and `algorithmic.sty`.
These are third-party LaTeX packages, with author and license notices in their
files; they are not manuscript contributions.

`nicefrac`, `placeins`, `multirow`, and `threeparttable` came from the official
TeX Live package archives at https://mirror.ctan.org/systems/texlive/tlnet/archive/.
`environ` and `trimspaces` were copied from the existing E-Values arXiv source
package in the original repository. `algorithm` and `algorithmic` were copied
from the installed user TeX tree at `~/Library/texmf/tex/latex/algorithms`.
Other standard packages and fonts are provided by the TeX distribution.

## Validation and existing warnings

Verified with pdfLaTeX and BibTeX on TeX Live 2026 Basic: the PDF has 17 pages,
and the final pass has no undefined citations or references and no overfull boxes.
The bibliography and appendix are included.

The original source still produces nine duplicate hyperlink-destination warnings
for figure/table anchors and two underfull-box warnings in the environment table.
The hyperlink warnings concern PDF destination identifiers, so links to those
figures/tables warrant checking. An epstopdf warning reports disabled shell escape;
all project figures are PNG files and do not need EPS conversion.
No manuscript or style changes were made to suppress these warnings.

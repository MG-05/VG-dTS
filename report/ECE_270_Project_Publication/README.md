# VG-dTS manuscript

The authoritative source is `main.tex`, with `appendix.tex`, `ref.bib`, `cmd.tex`,
and `pkg.tex`. The 12 retained PNG files are precisely the figures referenced by
the manuscript. Sources and retained figures are unchanged from the Overleaf
export; `OVERLEAF_MANIFEST.json` records their original SHA-256 hashes.

From the repository root, run:

```sh
sh report/ECE_270_Project_Publication/build.sh
```

The script uses pdfLaTeX and BibTeX from PATH or `/Library/TeX/texbin` on macOS,
disables shell escape, and stops on errors. A TeX distribution is required.
The output is `main.pdf` in this directory. Build products are ignored by Git.
See the [root README](../../README.md) to regenerate figures or compile a separate
manuscript copy using the regenerated figures.

The local `neurips_2025.sty` is part of the original paper source. The bundled
`environ.sty`, `trimspaces.sty`, `nicefrac.sty`, `placeins.sty`, `multirow.sty`,
`threeparttable.sty`, `algorithm.sty`, and `algorithmic.sty` support compilation
with BasicTeX. They retain their upstream author/license notices. The prior
export notes identify CTAN/TeX Live as the source for nicefrac, placeins,
multirow, and threeparttable; environ and trimspaces came from an existing arXiv
source bundle; algorithm and algorithmic came from the installed TeX tree.
Other standard packages and fonts come from the TeX distribution.

Validated during cleanup with TeX Live 2026 Basic: 17 pages, no undefined
citations/references and no overfull boxes. Existing duplicate hyperlink anchor
warnings and two underfull-box warnings remain. No manuscript changes were made
to suppress them. See [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) for scientific
and algorithm-description caveats that remain for author review.

# VG-dTS manuscript

The authoritative source is `main.tex`, with `appendix.tex`, `ref.bib`, `cmd.tex`,
and `pkg.tex`. The 12 retained PNG files are precisely the figures referenced by
the manuscript. Retained figures are unchanged from the Overleaf export.
`main.tex` and `appendix.tex` have been corrected against the experiments; see
[MANUSCRIPT_CHANGES.md](MANUSCRIPT_CHANGES.md). `OVERLEAF_MANIFEST.json` retains
the original export SHA-256 hashes as provenance, not current source checksums.

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

The original cleanup build used TeX Live 2026 Basic. Its historical validation
record in `results/paper/VALIDATION.json` describes the pre-correction manuscript.
See [REPRODUCIBILITY.md](../../REPRODUCIBILITY.md) for experimental conventions,
validation scope, and limitations. The revised manuscript uses preprint mode;
[CITATION.cff](../../CITATION.cff) identifies it as an unpublished manuscript.

The corrected manuscript was rebuilt with pdfLaTeX/BibTeX on 2026-10-07:
19 pages, no undefined citations/references, and no overfull boxes. Two underfull
box notices in the environment table and the expected shell-escape-disabled
package warning remain. All 23 repository tests pass. The citation file validates
against the official CFF 1.2.0 schema.

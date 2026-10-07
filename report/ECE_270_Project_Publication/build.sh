#!/bin/sh
set -eu
cd -- "$(dirname -- "$0")"

if command -v pdflatex >/dev/null 2>&1 && command -v bibtex >/dev/null 2>&1; then
    pdf_latex=$(command -v pdflatex)
    bib_tex=$(command -v bibtex)
elif [ -x /Library/TeX/texbin/pdflatex ] && [ -x /Library/TeX/texbin/bibtex ]; then
    pdf_latex=/Library/TeX/texbin/pdflatex
    bib_tex=/Library/TeX/texbin/bibtex
else
    echo 'Install a TeX distribution with pdfLaTeX and BibTeX before building.' >&2
    exit 1
fi

"$pdf_latex" -interaction=nonstopmode -halt-on-error -no-shell-escape -recorder main.tex
"$bib_tex" main
"$pdf_latex" -interaction=nonstopmode -halt-on-error -no-shell-escape -recorder main.tex
"$pdf_latex" -interaction=nonstopmode -halt-on-error -no-shell-escape -recorder main.tex
"$pdf_latex" -interaction=nonstopmode -halt-on-error -no-shell-escape -recorder main.tex
echo 'Built main.pdf'

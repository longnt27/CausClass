# CausClass thesis report

This is the complete source import of longnt27/CausClass-report at
`947714f3d1eb517c19e697f394763b1f7e8e9944`, with original history preserved.
The original report repository has not been deleted, archived or modified.
See PROVENANCE.md and ../docs/reproducibility.md for the relationship between
historical thesis results and current protocol-v2 software.

## Build from the main repository

On Debian/Ubuntu:

```bash
sudo apt-get update
sudo apt-get install latexmk biber texlive-latex-extra texlive-science \
  texlive-lang-other texlive-fonts-recommended
make report
```

Or from this directory:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -file-line-error -outdir=build main.tex
```

Output: `report/build/main.pdf`. pdfLaTeX, Vietnamese/VnTeX support and Biber are
required; BibTeX alone is insufficient. Builds run from report/ so all relative
figure, section and bibliography paths remain valid. The font-size and layout
choices of the imported thesis are retained.

CI uploads the PDF and complete log. It fails on unresolved references/citations,
not on every inherited typography warning. Review the rendered pages before a
formal submission; a successful compilation is not editorial or scientific review.
Generated auxiliary files and macOS metadata have been removed from the tracked
tree; all substantive source, bibliography, figures and appendices are preserved.

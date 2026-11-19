# Manuscript

`main.tex` is the LNCS entry point. Edit the files in `chapters/`, maintain
references in `references.bib`, and put manuscript figures in `figures/`.
Scientific commands shared by chapters belong in `shared/notation.tex`.

The main text has six chapter files and one appendix file. Only the introduction
and literature review currently contain text. Empty files are comments, so they
do not create empty sections. Use `\section` for each top-level heading and
`\subsection` / `\subsubsection` within it. The entry point switches to appendix
numbering before loading `chapters/appendix.tex`.

The abstract file should contain the complete `abstract` environment when ready.
Keep citation keys and labels stable across file moves.

From this directory, run:

```powershell
latexmk main.tex
```

Requires XeLaTeX, BibTeX, and latexmk. The bibliography style is `splncs04`.
The PDF and auxiliary files are written to `build/`. Source figures, including
PDF figures, are inputs and must not be removed as build products.

# Paper — Build Instructions

LaTeX source for the SDL paper. Compiled with [Tectonic](https://tectonic-typesetting.github.io/) from the `latex` conda env.

## Quick build

```bash
cd /raid/jupyter-linjimmy1003.md10/UCSD_stress/paper
make pdf
```

Output: `main.pdf`.

## Make targets

| Command        | What it does                                                         |
|----------------|----------------------------------------------------------------------|
| `make pdf`     | Compile `main.tex` once (default target).                            |
| `make clean`   | Remove all build artifacts (`main.pdf`, `.log`, `.aux`, `.bbl`, …). |
| `make watch`   | Auto-recompile on `.tex` / `.bib` / `.sty` changes (needs `entr`).   |
| `make view`    | Print the absolute path to `main.pdf` for JupyterLab.                |
| `make help`    | Show the command list.                                               |

Full rebuild from scratch:

```bash
make clean && make pdf
```

## Direct tectonic (no make)

```bash
conda run -n latex tectonic --synctex main.tex
```

## Source layout

```
paper/
├── main.tex              # document shell (preamble + \input of sections)
├── sections/             # one .tex per section
│   ├── abstract.tex
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── methods.tex
│   ├── results.tex
│   ├── discussion.tex
│   ├── limitations.tex
│   ├── conclusion.tex
│   └── appendix_psd_anchor.tex
├── figures/
│   ├── main/             # figures referenced from main text
│   └── supplementary/    # figures referenced only from appendices
├── references.bib        # BibTeX database
├── arxiv.sty             # venue style file (placeholder; swap for IEEEtran for TNSRE)
└── Makefile
```

## Notes

- First build runs BibTeX + multiple TeX passes (≈30–60 s). Subsequent builds without `.bib` changes take ≈10 s.
- Check page count: `pdfinfo main.pdf | grep Pages`.
- Tectonic auto-downloads missing LaTeX packages on first build — keep internet available the first time you run it on a new machine.
- If `conda run -n latex ...` fails with "environment not found", install with: `conda create -n latex -c conda-forge tectonic`.

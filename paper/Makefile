# Makefile for paper compilation
#
# Tectonic lives in the `latex` conda env (NOT timm_eeg — keep them separate
# to avoid clobbering pytorch). Activate with:
#     conda activate latex
# or just call via `conda run -n latex tectonic ...`.

TECTONIC := conda run -n latex tectonic
MAIN := main.tex
PDF := main.pdf

.PHONY: pdf clean watch view help

help:
	@echo "make pdf      — compile main.tex once"
	@echo "make watch    — recompile on file changes (requires entr/inotifywait)"
	@echo "make clean    — remove build artifacts"
	@echo "make view     — print path for JupyterLab to open"

pdf: $(PDF)

$(PDF): $(MAIN) sections/*.tex references.bib arxiv.sty
	$(TECTONIC) --keep-logs --synctex $(MAIN)

clean:
	rm -f main.pdf main.log main.aux main.bbl main.blg main.out main.synctex.gz
	rm -rf _minted-*

watch:
	@command -v entr >/dev/null 2>&1 || { \
	  echo "entr not installed. Install with: conda install -n latex -c conda-forge entr"; \
	  exit 1; }
	find . -name '*.tex' -o -name '*.bib' -o -name '*.sty' | entr -s 'make pdf'

view:
	@echo "Open in JupyterLab file browser: $(PWD)/$(PDF)"

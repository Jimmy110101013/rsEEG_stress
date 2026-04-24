"""Build Table 1 (master performance) — LP vs FT BA across 3 datasets × 3 FMs.

Output:
  paper/tables/table1_master_performance.tex  — LaTeX source (booktabs, 3-level header)
  paper/tables/table1_master_performance.pdf  — compiled standalone PDF
  paper/tables/_source/table1_master_performance.json  — raw numbers
"""
from __future__ import annotations
import json
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src import results  # noqa: E402

OUT_DIR = REPO / "paper/tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "_source").mkdir(parents=True, exist_ok=True)

FMS = ["labram", "cbramod", "reve"]
FM_PRETTY = {"labram": "LaBraM", "cbramod": "CBraMod", "reve": "REVE"}
# 2×2 factorial order: (within × coherent), (within × incoherent), (between × coherent), (between × absent)
DATASETS = ["eegmat", "sleepdep", "adftd", "stress"]
DS_PRETTY = {"eegmat": "EEGMAT", "sleepdep": "SleepDep", "adftd": "ADFTD", "stress": "Stress (DASS)"}
DS_N = {"eegmat": (72, 36), "sleepdep": (72, 36), "adftd": (82, 82), "stress": (70, 17)}

QUADRANT_GROUP = {
    "eegmat":   ("Within-subject × coherent",   0),
    "sleepdep": ("Within-subject × incoherent", 1),
    "adftd":    ("Between-subject × coherent",  2),
    "stress":   ("Between-subject × absent",    3),
}


lp_stats = results.lp_stats_3seed
ft_stats = results.ft_stats


def fmt_cell(stats):
    if stats is None:
        return "---"
    m = stats["mean"]
    if stats["std"] is None:
        # single-seed — mark with asterisk
        return f"${m:.3f}^{{*}}$"
    return f"${m:.3f} \\pm {stats['std']:.3f}$"


# -------- collect all data --------
rows = {}  # rows[(ds, fm)] = {"lp": ..., "ft": ...}
for ds in DATASETS:
    for fm in FMS:
        rows[(ds, fm)] = {"lp": lp_stats(ds, fm), "ft": ft_stats(ds, fm)}

# save raw
raw = {f"{ds}_{fm}": {"lp": rows[(ds,fm)]["lp"], "ft": rows[(ds,fm)]["ft"]}
       for ds in DATASETS for fm in FMS}
raw_path = OUT_DIR / "_source/table1_master_performance.json"
raw_path.write_text(json.dumps(raw, indent=2))
print(f"raw → {raw_path.relative_to(REPO)}")

# -------- LaTeX --------
# 3 datasets × 2 metrics (LP/FT) = 6 data columns; first col = Method
TEX = r"""\documentclass[10pt]{article}
\usepackage[a4paper,margin=1.5cm,landscape]{geometry}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{multicol}
\usepackage{siunitx}
\usepackage{array}
\usepackage{xcolor}
\pagestyle{empty}
\begin{document}

\begin{table}[htbp]
\centering
\footnotesize
\caption{\textbf{Subject-level 5-fold cross-validated balanced accuracy — linear probe (LP) vs full fine-tuning (FT) across the four-dataset 2$\times$2 factorial.}
Columns ordered by quadrant of (within- vs between-subject labelling) $\times$ (coherent vs absent/incoherent label signal).
Values are 3-seed mean $\pm$ sample standard deviation (ddof=1); seeds 42/123/2024.
LP = L2-penalised logistic regression on per-window frozen features, recording-level majority vote, \texttt{class\_weight=balanced}.
FT = full fine-tuning with unified recipe (lr=1e-5, wd=0.05, encoder\_lr\_scale=0.1); see Section~3.4 for HP-contamination caveat.
Stress uses the DASS per-recording binary label matching Komarov et al.\ (2020) and Wang et al.\ (2025); 14/17 subjects have consistent labels, 3/17 straddle both classes.
Cells marked with ${}^{*}$ are single-seed (reproduction pending).
Dataset sizes: EEGMAT 72 rec / 36 subj; SleepDep 72 / 36; ADFTD 82 / 82; Stress 70 / 17.}
\label{tab:master_performance}

\begin{tabular}{l cc cc cc cc}
\toprule
 & \multicolumn{2}{c}{\textbf{EEGMAT}} & \multicolumn{2}{c}{\textbf{SleepDep}} & \multicolumn{2}{c}{\textbf{ADFTD}} & \multicolumn{2}{c}{\textbf{Stress (DASS)}} \\
 & \multicolumn{2}{c}{\footnotesize within $\times$ coherent} & \multicolumn{2}{c}{\footnotesize within $\times$ incoherent} & \multicolumn{2}{c}{\footnotesize between $\times$ coherent} & \multicolumn{2}{c}{\footnotesize between $\times$ absent} \\
\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}
\textbf{Method} & LP & FT & LP & FT & LP & FT & LP & FT \\
\midrule
"""

for fm in FMS:
    parts = [FM_PRETTY[fm]]
    for ds in DATASETS:
        r = rows[(ds, fm)]
        parts += [fmt_cell(r["lp"]), fmt_cell(r["ft"])]
    TEX += " & ".join(parts) + r" \\" + "\n"

TEX += r"""\bottomrule
\end{tabular}
\end{table}

\end{document}
"""

tex_path = OUT_DIR / "table1_master_performance.tex"
tex_path.write_text(TEX)
print(f"tex → {tex_path.relative_to(REPO)}")

# -------- compile with tectonic --------
tectonic = Path("/raid/jupyter-linjimmy1003.md10/.conda/envs/latex/bin/tectonic")
if not tectonic.exists():
    print(f"WARN: tectonic not found at {tectonic}; skipping PDF compile")
else:
    res = subprocess.run(
        [str(tectonic), "--outdir", str(OUT_DIR), str(tex_path)],
        capture_output=True, text=True,
    )
    if res.returncode == 0:
        pdf_path = OUT_DIR / "table1_master_performance.pdf"
        print(f"pdf → {pdf_path.relative_to(REPO)}  ({pdf_path.stat().st_size/1024:.1f} KB)")
    else:
        print("TECTONIC FAILED:")
        print(res.stdout[-1500:])
        print(res.stderr[-1500:])

"""Merge all v2 section drafts into a single PDF for reading.

Injects relevant figures and tables into each section before rendering.
Uses markdown-pdf (markdown → HTML → PDF).
"""
from __future__ import annotations

import base64
import re
from pathlib import Path

from markdown_pdf import MarkdownPdf, Section


TITLE = "Subject-Dominance Limits (SDL) in Clinical Resting-State EEG Foundation Models"
SUBTITLE = "Draft compilation — 2026-04-20 — Path A (clinical/engineering framing)"

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")


# Section → list of (figure_path, caption, anchor_regex_or_None)
# anchor: regex pattern after which to insert the figure; None → append end
SECTION_FIGURES = {
    "docs/sdl_paper_draft_abstract_and_intro_v2.md": [],

    "docs/sdl_paper_draft_section_2_v2.md": [],

    "docs/sdl_paper_draft_section_3_v2.md": [],

    "docs/sdl_paper_draft_section_4.md": [
        ("paper/figures/main/fig_variance_atlas.png",
         "**Figure 4.1.** Variance atlas: frozen FM embedding variance decomposed "
         "into label-factor, subject-factor, and residual fractions across 3 FMs "
         "× 4 datasets. Subject-factor dominates (≥10× label in every cell).",
         r"## §4 Subject Atlas"),
    ],

    "docs/sdl_paper_draft_section_5_v2.md": [
        ("paper/figures/main/fig_honest_funnel.png",
         "**Figure 5.1.** Honest-evaluation funnel for UCSD Stress: "
         "published benchmark (0.90) → our trial-level reproduction (0.68) → "
         "subject-level CV (0.53) → permutation-null indistinguishability "
         "(p = 0.70) → classical band-power baseline (0.553).",
         r"## §5 Honest Evaluation Audit"),
        ("paper/figures/main/fig_perm_null_density.png",
         "**Figure 5.2.** Permutation-null distribution for LaBraM × Stress "
         "fine-tuning. Real-label BA (0.443 ± 0.083) overlaps the shuffled-label "
         "null distribution (0.497 ± 0.086), p = 0.70 one-sided.",
         r"### 5.3 Permutation-null comparison"),
    ],

    "docs/sdl_paper_draft_section_6_v2.md": [
        ("paper/figures/main/fig_paired_contrast.png",
         "**Figure 6.1.** EEGMAT vs UCSD Stress paired comparison: frozen LP BA "
         "under matched per-window protocol. EEGMAT separates the target label "
         "(0.72–0.74 across 3 FMs); Stress does not (0.43–0.53).",
         r"## §6 Representation-Level Evidence of Subject Dominance"),
        ("paper/figures/main/fooof_ablation.png",
         "**Figure 6.2.** FOOOF aperiodic/periodic causal ablation. "
         "**(A) EEGMAT subject-ID probe** — removing aperiodic 1/f component "
         "drops CBraMod subject-ID decoding by 14pp and REVE by 26pp; "
         "removing periodic peaks changes < 1pp. "
         "**(B) EEGMAT state-label probe** — state decoding survives both "
         "ablations (drops ≤ 2pp). "
         "**(C) Stress subject-ID probe** — same pattern at smaller magnitude "
         "consistent with Stress's noisier 17-subject probe.",
         r"### 6.5 Aperiodic 1/f carries subject identity"),
    ],

    "docs/sdl_paper_draft_section_7_v2.md": [
        ("paper/figures/main/fig_ceiling.png",
         "**Figure 7.1.** Architecture-independence of the UCSD Stress ceiling. "
         "Seven architectures spanning six orders of magnitude in parameter "
         "count — from 3 k-parameter EEGNet (2018) to 1.4 B-parameter REVE "
         "(2025) — all cluster in the 0.43–0.58 balanced-accuracy band under "
         "subject-disjoint 5-fold CV.",
         r"## §7 The ~0.55 Ceiling on UCSD Stress"),
        ("paper/figures/main/fig8c_non_fm_baselines.png",
         "**Figure 7.2.** Classical band-power baselines vs FM frozen LP on "
         "the same UCSD Stress 70-recording cohort. Class-balanced XGBoost "
         "(0.553) exceeds all three FM frozen LPs under matched per-window "
         "protocol.",
         r"### 7.2 What the ceiling is"),
    ],

    "docs/sdl_paper_draft_section_8_v2.md": [],  # Master table text-embedded below

    "docs/sdl_paper_draft_section_9_v2.md": [],

    "docs/sdl_paper_draft_section_10_v2.md": [],
}


SECTIONS = [
    ("Abstract + §1 Introduction", "docs/sdl_paper_draft_abstract_and_intro_v2.md"),
    ("§2 Related Work", "docs/sdl_paper_draft_section_2_v2.md"),
    ("§3 Methods", "docs/sdl_paper_draft_section_3_v2.md"),
    ("§4 Subject Atlas (v1)", "docs/sdl_paper_draft_section_4.md"),
    ("§5 Honest Evaluation Audit", "docs/sdl_paper_draft_section_5_v2.md"),
    ("§6 Representation-Level Evidence + FOOOF Ablation", "docs/sdl_paper_draft_section_6_v2.md"),
    ("§7 Architecture-Independent Ceiling", "docs/sdl_paper_draft_section_7_v2.md"),
    ("§8 Cross-Dataset FT vs LP", "docs/sdl_paper_draft_section_8_v2.md"),
    ("§9 Clinical Pre-Flight Checklist", "docs/sdl_paper_draft_section_9_v2.md"),
    ("§10 Limitations + Future Work", "docs/sdl_paper_draft_section_10_v2.md"),
]

OUT_PDF = "docs/sdl_paper_draft_full_v2.pdf"


def clean_draft_markers(text: str) -> str:
    text = re.sub(r"\n\*Draft status:.*?\*\n?$", "\n", text, flags=re.DOTALL)
    return text


def img_to_data_uri(path: Path) -> str:
    """Inline a PNG as base64 data URI so markdown-pdf embeds it reliably."""
    data = path.read_bytes()
    b64 = base64.b64encode(data).decode("ascii")
    return f"data:image/png;base64,{b64}"


def inject_figures(text: str, figures: list) -> str:
    """For each (img_path, caption, anchor_regex): insert an image block after
    the first match of anchor_regex (as its own line). If no anchor, append
    at end."""
    for img_path_str, caption, anchor in figures:
        img_path = ROOT / img_path_str
        if not img_path.exists():
            print(f"    MISSING figure: {img_path_str}")
            continue
        data_uri = img_to_data_uri(img_path)
        block = (
            f"\n\n<div style=\"page-break-inside: avoid; text-align: center; margin: 1em 0;\">"
            f"<img src=\"{data_uri}\" style=\"max-width: 95%; height: auto;\" />"
            f"<p style=\"font-size: 0.85em; text-align: left; margin-top: 0.5em;\">{caption}</p>"
            f"</div>\n\n"
        )
        if anchor is None:
            text = text + block
        else:
            # Insert after the anchor line
            m = re.search(anchor, text)
            if m:
                # Find end of the paragraph that contains the anchor match
                # Simpler: insert right after the anchor line
                end_of_line = text.find("\n", m.end())
                if end_of_line == -1:
                    end_of_line = len(text)
                text = text[:end_of_line + 1] + block + text[end_of_line + 1:]
            else:
                print(f"    Anchor not found ({anchor!r}); appending at end")
                text = text + block
    return text


def main():
    pdf = MarkdownPdf(toc_level=2, optimize=True)

    # Title page
    title_md = (
        f"# {TITLE}\n\n"
        f"*{SUBTITLE}*\n\n"
        f"---\n\n"
        f"**Contents**\n\n"
    )
    for i, (label, _) in enumerate(SECTIONS, 1):
        title_md += f"{i}. {label}\n"
    title_md += (
        "\n---\n\n"
        "**Source data and scripts**\n\n"
        "- Master FT-vs-LP table: `paper/figures/source_tables/master_frozen_ft_table_v2.md`\n"
        "- FOOOF ablation results: `results/studies/fooof_ablation/SUMMARY.md`\n"
        "- Per-window LP results: `results/studies/perwindow_lp_all/SUMMARY.md`\n"
        "- FT vs LP paired comparison: `results/studies/ft_vs_lp/comparison.json`\n"
        "- Strategy: `docs/paper_strategy_v3_minimum_viable.md`\n\n"
        "**Figures embedded**\n\n"
        "- Fig 4.1 Variance atlas (§4)\n"
        "- Fig 5.1 Honest-evaluation funnel (§5); Fig 5.2 Permutation-null density (§5.3)\n"
        "- Fig 6.1 Paired contrast (§6); Fig 6.2 FOOOF ablation (§6.5)\n"
        "- Fig 7.1 Architecture ceiling (§7); Fig 7.2 Classical baselines (§7.2)\n"
    )
    pdf.add_section(Section(title_md, toc=False))

    # Each draft section
    for label, path in SECTIONS:
        p = ROOT / path
        if not p.exists():
            print(f"  SKIP missing: {path}")
            continue
        text = p.read_text()
        text = clean_draft_markers(text)
        figures = SECTION_FIGURES.get(path, [])
        if figures:
            print(f"  Injecting {len(figures)} figure(s) into {label}")
            text = inject_figures(text, figures)
        # Embed master table content in §8
        if path.endswith("section_8_v2.md"):
            master_md_path = ROOT / "paper/figures/source_tables/master_frozen_ft_table_v2.md"
            if master_md_path.exists():
                print(f"  Embedding master table into §8")
                master = master_md_path.read_text()
                text += (
                    "\n\n---\n\n## Appendix 8.A — Full master table (FT vs frozen LP) "
                    "source reference\n\n"
                ) + master
        pdf.add_section(Section(text, toc=True))
        print(f"  Added: {label}")

    pdf.meta["title"] = TITLE
    pdf.meta["author"] = "Lin Jimmy (UCSD_stress project)"
    pdf.save(str(ROOT / OUT_PDF))
    print(f"\nSaved: {OUT_PDF}")


if __name__ == "__main__":
    main()

# Self-Review — SDL Paper Draft (2026-04-15)

Adversarial review of `paper/main.tex` using the paper-review skill protocol.
Pre-polish audit.

---

## §0 Reject-first simulation (mandatory first pass)

> "Authors claim discovery of an SDL two-regime diagnostic, but the entire
> work is built on two datasets (EEGMAT as anchored, UCSD Stress as
> bounded). This n=2 dataset-space observation is packaged as a
> field-level claim without cross-cohort independent validation. The
> primary Stress results (0.52-0.58 ceiling, 5pp FM-vs-classical bump,
> permutation p=0.70) all rest on a n=70 / 14-positives underpowered
> cohort; the authors' own disclosed seed envelope of ±5-10pp already
> consumes most of the claimed effect sizes. The §3.5 pre-registered
> group-level PSD anchoring analysis is promised in Methods but never
> executed in the paper. The Wang-2025 0.90→0.52 collapse claim is
> actually a mixture of pipeline mismatch plus CV correction, and the
> authors did not run Wang's code before concluding collapse. Novelty is
> diagnostic packaging rather than new methods or new data."

---

## Tier A — must fix before submission (reject risk)

### A1 · Stress group-level PSD anchoring never executed
**Location**: §3.5 (Methods), §6.1 (Results — Stress anchor).
**Problem**: Methods pre-registers a cluster-permutation PSD analysis as
the Stress contrast anchor. Results never runs it. This is the single
largest reject risk — pre-registered but not executed.
**Fix options**: (i) Actually run it. Output: supplementary Fig/Table
S-anchor. (ii) Or rewrite §3.5 second bullet to "pre-registered but
deferred to follow-up work" with explicit reasoning why current
evidence already suffices.

### A2 · Wang 0.90→0.52 collapse is overstated
**Location**: Abstract, §1 Introduction (second paragraph).
**Problem**: §5.1 concedes "our own trial-level numbers (0.56-0.68)
remain well below Wang's 0.90." That is, honest CV *alone* does not
collapse 40 pp; the remaining gap is attributed to undocumented pipeline
differences we cannot audit. But Abstract and §1 currently read as if a
single protocol correction absorbs the whole gap.
**Fix**: Abstract and §1 should explicitly say "honest CV plus
additional undisclosed pipeline differences jointly account for the gap;
the within-our-pipeline trial-vs-subject delta (1-15 pp) cleanly
isolates evaluation-design inflation as one necessary but not sufficient
component."

### A3 · Permutation-null claim too strong for underpowered test
**Location**: Abstract ("statistically indistinguishable from label-
permutation null"), §5.3.
**Problem**: Evidence is a Mann-Whitney between 3 real seeds and 10
permutation seeds. 3-vs-10 is not a test with real statistical power.
"Statistically indistinguishable" implies rigor the test does not have.
**Fix options**: (i) Increase permutation pool to ≥100 seeds
(1-2 GPU-weeks). (ii) Downgrade abstract language to factual description:
"real FT performance did not exceed label-shuffle null in our 3-seed ×
10-permutation comparison."

### A4 · Architecture-independence tested on one cohort only
**Location**: §7 (Results), §10 (Limitations).
**Problem**: §7's seven-architecture panel is entirely on UCSD Stress.
SDL is framed as a field-level diagnostic. Reviewer will ask: how do we
know the ceiling is not UCSD Stress-specific?
**Fix**: Add explicit scope note in §7 conclusion: architecture-
independence demonstrated on one bounded cohort; cross-cohort
replication is future work. Reference EEG-Bench and AdaBrain-Bench as
indirect aggregate-level corroboration.

### A5 · "FT does not exceed frozen LP" is paper's strongest within-
ceiling claim but visually buried
**Location**: §8.1, Table 8.1.
**Problem**: The 4-family comparison (classical 0.553 / frozen LP 0.605
/ FT 0.577 / CNN 0.557) is rhetorically central but appears only as a
table. `fig_ceiling` shows frozen LP only as a dotted line.
**Fix options**: (i) Add frozen LP as an explicit row in `fig_ceiling`.
(ii) Create a small new figure: 4-family BA bar chart with error bars.

---

## Tier B — significant score impact, not reject-level

### B1 · Claim-vs-evidence audit (6 items)
| Location | Claim | Issue |
|---|---|---|
| Abstract | "five orders of magnitude" | 3k → 1.4B = 5.7 orders OK, but §7 body says "three kilo- to one giga" = 5 orders. Reconcile. |
| §4.2 Table 4.1 | subject/label ratio ≥10× | Point estimates, no bootstrap CIs. |
| §5.2 | LaBraM frozen LP 0.605 ± 0.032 (8 seeds) | Why 8 seeds vs 3? Not justified — looks like seed-shopping. |
| §6.2 | "comparable to previously reported EEGMAT benchmarks" | No numerical citation. |
| §6.4 | band-stop "REVE alpha 0.150, 3× beta" | No CI, cross-seed stability unknown. |
| §8.3 | ADFTD "~0.60 / ~0.65 / 0.70" | Three "~" approximations, no source. |

### B2 · ADFTD does more rhetorical work than its evidence supports
**Location**: §4.5, §8.3.
**Problem**: ADFTD is (a) cross-subject, not within-subject paired, and
(b) not subjected to the same anchoring protocol as EEGMAT, and (c) the
+10.65 pp Δlabel-fraction is single-run with no seed variance.
**Fix**: Downgrade ADFTD framing from "confirming example" to
"suggestive but not directly comparable"; or run 3-seed ADFTD FT
variance decomposition to supply CIs.

### B3 · No Fig 1 / teaser figure for SDL thesis
**Location**: §1.
**Problem**: 11-paragraph text introduction with no visual thesis
anchor. Top-tier venues nearly always have a Fig 1 thesis schematic.
**Fix options**: (i) New SDL schematic (x = contrast strength, y = FM
rescue magnitude, mark EEGMAT / ADFTD / Stress). (ii) Promote
`fig_paired_contrast` to Fig 1 position.

### B4 · Related Work lacks positioning paragraph
**Location**: end of §2.
**Problem**: Six subsections introduce literature but no synthesis
paragraph stating explicitly how this work differs from each strand.
**Fix**: Add ~100-word positioning paragraph clarifying differences vs
(a) concurrent benchmark audits, (b) EEG biometric literature,
(c) small-N reliability critiques.

### B5 · Table/figure float placement is already triggering overfull
**Location**: §4–§8 collectively (6 tables, 4 figures).
**Problem**: One `Float too large for page by 44.98 pt` warning at
line 376.
**Fix**: Move less-central tables (e.g. Table 5.1 trial-vs-subject) to
supplementary; use `[!t]` / `[!b]` placement specifiers.

### B6 · Title is long and does not foreground SDL coinage
**Location**: Title.
**Candidates**:
- *"Subject-Dominance Limits: A Mechanistic Diagnostic for EEG Foundation Models"*
- *"Contrast or Capacity? A Subject-Dominance Diagnostic for EEG Foundation Models"*
- *"The Subject-Dominance Ceiling: Why EEG Foundation Models Need Neural Anchors"*

---

## Tier C — polish

- **C1** Introduction paragraph 5→6 transition abrupt.
- **C2** Abstract ~280 words; TNSRE cap is ~200.
- **C3** Conclusion and Discussion SDL-definition duplicate ~70%.
- **C4** `references.bib` has placeholder author lists ("Chen et al.", "Scherer et al.").
- **C5** Tables missing metric direction arrow (BA should be ↑).
- **C6** `fig_honest_funnel` caption too brief; should be self-contained.

---

## Counter-intuitive protocol outcomes

1. **Reject-first simulation**: §0 above.
2. **Delete one unsupported strong claim**: Abstract "statistically
   indistinguishable from label-permutation null" → weaker factual
   phrasing. (See A3.)
3. **Score trust, not only score gains**: Already applied — §8.2
   explicitly downgrades the 5 pp to "observation, consistent with"
   rather than strong claim.
4. **Promote one explicit limitation**: §10.1 ±5-10 pp seed envelope
   should move up to §1 paragraph 2-3 instead of being buried at the
   end.
5. **Attack novelty claim** — "Could a strong PhD derive SDL in one
   afternoon?" Half-yes: subject-dominance and trial-vs-subject gap are
   known. The novelty is primarily in **packaging the two-regime
   diagnostic as a pre-benchmark protocol**. Intro contribution order
   should front-load (c.4) the diagnostic protocol.

---

## Execution order

1. **A1** — run Stress group-level PSD anchoring (or rewrite §3.5 as deferred).
2. **A2, A3** — rewrite Abstract + §1 Wang collapse and permutation-null language.
3. **B2** — downgrade ADFTD role in §4.5 / §8.3.
4. **B4** — Related Work positioning paragraph.
5. **C2** — compress Abstract to 200 words.
6. **C4** — fill placeholder author lists in `references.bib`.
7. Rebuild PDF, re-verify figures, resolve overfull warnings.

---

*Review authored 2026-04-15 by Claude (self-review mode). Address items
tier-by-tier; update this document as fixes land.*

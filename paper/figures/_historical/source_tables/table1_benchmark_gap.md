# Table 1 — Benchmark gap (Intro §1.1 candidate)

**Purpose**: motivate Type C critique paper. Shows Frozen → FT balanced-accuracy trajectory across LaBraM / CBraMod / REVE under **explicit subject-level CV** (apples-to-apples comparison).

**Source**: Literature rows from EEG-FM-Bench (arXiv 2508.17742v2, 2025) Appendix 6: **Frozen from Table 4** (multi-task, avg-pool head) and **FT from Table 3** (multi-task FT, avg-pool head). This pairing is apples-to-apples (both multi-task). Column order in both tables: BENDR, BIOT, LaBraM, EEGPT, CBraMod, CSBrain, REVE. Individual FM papers (LaBraM, CBraMod, REVE) are mostly silent on split type (see `table1_excluded.md`). "Ours" rows from `results/studies/exp_newdata/` and `paper/figures/source_tables/master_frozen_ft_table.json`.

**Filter applied**: Only benchmarks with explicitly-documented **subject-level** train/val/test split are included in the main table. Benchmarks that use stratified / trial-level split in the literature (ADFTD-3cls, TUEV, Mimul-11, MentalArithmetic in EEG-FM-Bench) are **not** in this table — they go in `table1_excluded.md` with a note that published numbers for these are likely subject-leakage inflated.

---

## Table 1

| Benchmark | Design | n_cls | n_sub | LaBraM (F→FT) | CBraMod (F→FT) | REVE (F→FT) | Source |
|---|---|---:|---:|---|---|---|---|
| **Between-subject benchmarks (subject ≡ label)** | | | | | | | |
| TUAB (abnormal) | between | 2 | ~600 | 75.87 → 79.36 | 73.15 → 80.49 | 63.80 → 80.32 | EEG-FM-Bench T4/T3 |
| Siena (seizure) | between | 2 | 18 | 50.00 → 71.99 | 64.17 → 82.75 | 68.60 → 70.65 | EEG-FM-Bench T4/T3 |
| ADFTD (AD/FTD/CN, binary) | between | 2 | 65 | 69.5 → 70.9 | 55.8 → 53.7 | 69.2 → 65.8 | Ours (exp_30) |
| TDBRAIN (MDD/CN) | between | 2 | 359 | 67.9 → 66.5 | 56.4 → 48.9 | 54.4 → 48.8 | Ours (exp_30) |
| Meditation (expert/novice) | between | 2 | 24 | 47.3 → 51.5 | 71.0 → 68.3 | 53.8 → 43.3 | Ours (exp_30) |
| **Within-subject benchmarks (subject ⊥ label)** | | | | | | | |
| HMC (sleep staging) | within | 5 | 151 | 59.80 → 71.63 | 51.81 → 71.08 | 63.80 → 71.43 | EEG-FM-Bench T4/T3 |
| PhysioMI (motor imagery) | within | 4 | 109 | 29.63 → 43.19 | 26.90 → 31.15 | 27.37 → 30.63 | EEG-FM-Bench T4/T3 |
| BCIC-IV-2a (motor imagery) | within | 4 | 9 | 28.40 → 34.58 | 29.17 → 35.50 | 28.63 → 36.89 | EEG-FM-Bench T4/T3 |
| SEED-VII (emotion) | within | 7 | 20 | 23.23 → 26.13 | 19.43 → 26.05 | 20.57 → 20.76 | EEG-FM-Bench T4/T3 |
| Things-EEG-2 (tgt vs non-tgt) | within | 2 | 10 | 50.00 → 50.90 | 50.00 → 50.70 | 50.00 → 59.43 | EEG-FM-Bench T4/T3 |
| EEGMAT (task vs rest) | within | 2 | 36 | 67.1 → 73.1 | 73.1 → 62.0 | 67.1 → 72.7 | Ours (exp_30) |
| SleepDep (NS vs SD) | within | 2 | 36 | 50.0 → 53.2 | 55.7 → 55.6 | 54.4 → 54.2 | Ours (exp_30) |

**Notes**:
- All reported values are balanced accuracy (%). "F" = frozen backbone with linear/avg-pool head (equivalent to LP). "FT" = full-parameter fine-tuning.
- **Stress (UCSD) is excluded** because 3/17 subjects crossing DASS labels makes the design neither cleanly within- nor between-subject (see §2.1.1).
- Our datasets use subject-level 5-fold CV (subject-disjoint train/val/test); EEG-FM-Bench datasets use the split protocols documented in their Appendix B.3.
- `ADFTD` in our table is **binary** (AD vs CN) to match our main pipeline; EEG-FM-Bench uses **3-class** (AD/FTD/CN) with stratified split, so their numbers are not directly comparable.

---

## Derived statistics (for §1.1 text)

### Mean ΔBA (FT − Frozen) across the 3 FMs — **apples-to-apples T3+T4 pairing**

| Arm | Mean ΔBA | Range |
|---|---:|---|
| Between-arm (literature: TUAB, Siena) | **+11.7 BA** | [+2.1, +22.0] |
| Between-arm (ours: ADFTD, TDBRAIN, Meditation) | −3.1 BA | [−10.5, +4.2] |
| Within-arm (literature: HMC, PhysioMI, BCIC-IV-2a, SEED-VII, Things-EEG-2) | **+6.8 BA** | [+0.2, +19.3] |
| Within-arm (ours: EEGMAT, SleepDep) | +0.6 BA | [−11.1, +6.0] |

**Interpretation (data-driven, honest)**:
- Under apples-to-apples T3+T4 pairing, literature between-arm ΔBA (+11.7) does average **higher** than within-arm (+6.8). Earlier framing "comparable across arms" was based on mixed T2+T4 pairing and is retracted.
- BUT the arm-gap in literature is modest (~5 BA) and driven mainly by **TUAB, Siena** (between) vs **HMC** (within) — all other datasets show within or between ΔBA comparable.
- Our own datasets show much smaller ΔBA in both arms, with between actually **negative** on average (TDBRAIN and Meditation drive the negative). This is a combination of (a) small N, (b) no HP search, (c) harder benchmarks with less label signal.
- **C2 is a correlation claim, not a mean-magnitude claim**: in between-arm, ΔBA magnitude is predicted by subject-ID decodability (ρ = +0.50); in within-arm, ΔBA magnitude is NOT predicted by subject decodability (ρ wide CI). This holds regardless of whether between mean > within mean.
- The mechanism interpretation: between FT gain partly exploits subject fingerprints as label shortcut (label ≡ subject design); within FT gain comes from legitimate task signal (label ⊥ subject design). The magnitudes happen to differ but the mechanism claim does not depend on that.

### What this means for the critique

Literature under apples-to-apples T3+T4 shows a modest between > within gap (+11.7 vs +6.8 BA). Key observations:
1. **The gap is smaller than what a reader would infer** from mixed-CV tables in individual FM papers (which pool stratified/trial-level splits with subject-level splits). A naive reader seeing 90 % BA on SEED next to 80 % on TUAB may assume comparable protocols when actually the former is subject-dependent (trial leakage) and the latter is subject-level.
2. **Even on clean subject-level CV, the arm gap is task-dependent**. TUAB and HMC both give ~+10 BA gain; Things-EEG-2 and SEED-VII give ~0–4 BA. ΔBA is not monotonically predicted by arm.
3. **In our own data the arm gap flips** (within +0.6 vs between −3.1) because ADFTD / TDBRAIN / Meditation are harder benchmarks than TUAB / Siena.
4. **C2's claim is about correlation, not mean magnitude**: in between-arm, subject-ID decodability predicts which cells get big ΔBA (ρ=+0.50, n=9 lit+ours). In within-arm, no representation-level feature predicts ΔBA at current N. This is the novel contribution — the arm-gap in means is a known observation; the arm-gap in mechanism is new.

> **Core critique**: EEG-FM benchmark literature reports mixed split protocols in single summary tables, inflating apparent motor/emotion performance. Under matched subject-level CV, apparent "between wins" narrative shrinks, and the gain mechanism is revealed to differ by arm: between-arm gain is partly subject-leakage shortcut (predicted by subject_id_ba), within-arm gain is legitimate task learning (uncorrelated with subject_id_ba).

---

## Table 1 caption (draft)

> **Table 1.** Frozen → FT balanced accuracy for three EEG foundation models on benchmarks with explicit subject-level CV. Literature rows are drawn from EEG-FM-Bench v2 (arXiv 2508.17742, 2025) Appendix 6 — Frozen from Table 4 (multi-task, avg-pool head) and FT from Table 3 (multi-task FT, avg-pool head); this pairing is apples-to-apples (both multi-task). Rows labelled "ours" are from exp_30 under matched subject-level 5-fold CV (see §2.2). Between-arm datasets have one-subject-one-label design; within-arm have the same subjects contributing multiple labels. We exclude benchmarks that use stratified or trial-level splits in the source literature (ADFTD-3cls, TUEV, Mimul-11, MentalArithmetic in EEG-FM-Bench; see Appendix X for the list) because those numbers cannot be cleanly compared to subject-level numbers. Stress (UCSD) is excluded from both arms because only 3/17 subjects cross DASS labels (see §2.1.1).

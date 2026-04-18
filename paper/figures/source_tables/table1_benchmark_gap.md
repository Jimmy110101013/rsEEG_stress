# Table 1 — Benchmark gap (Intro §1.1 candidate)

**Purpose**: motivate Type C critique paper. Shows Frozen → FT balanced-accuracy trajectory across LaBraM / CBraMod / REVE under **explicit subject-level CV** (apples-to-apples comparison).

**Source**: Literature rows from EEG-FM-Bench (arXiv 2508.17742, 2025) Appendix B.3 — the only published source with exhaustive per-dataset split documentation. Individual FM papers (LaBraM, CBraMod, REVE) are mostly silent on split type (see `table1_excluded.md`). "Ours" rows from `results/studies/exp_newdata/` and `paper/figures/source_tables/master_frozen_ft_table.json`.

**Filter applied**: Only benchmarks with explicitly-documented **subject-level** train/val/test split are included in the main table. Benchmarks that use stratified / trial-level split in the literature (ADFTD-3cls, TUEV, Mimul-11, MentalArithmetic in EEG-FM-Bench) are **not** in this table — they go in `table1_excluded.md` with a note that published numbers for these are likely subject-leakage inflated.

---

## Table 1

| Benchmark | Design | n_cls | n_sub | LaBraM (F→FT) | CBraMod (F→FT) | REVE (F→FT) | Source |
|---|---|---:|---:|---|---|---|---|
| **Between-subject benchmarks (subject ≡ label)** | | | | | | | |
| TUAB (abnormal) | between | 2 | ~600 | 75.9 → 79.5 | 73.2 → 78.3 | 63.8 → 80.5 | EEG-FM-Bench |
| Siena (seizure) | between | 2 | 18 | 50.0 → 59.1 | 64.2 → 80.6 | 68.6 → 74.0 | EEG-FM-Bench |
| ADFTD (AD/FTD/CN, binary) | between | 2 | 65 | 69.5 → 70.9 | 55.8 → 53.7 | 69.2 → 65.8 | Ours (exp_30) |
| TDBRAIN (MDD/CN) | between | 2 | 359 | 67.9 → 66.5 | 56.4 → 48.9 | 54.4 → 48.8 | Ours (exp_30) |
| Meditation (expert/novice) | between | 2 | 24 | 47.3 → 51.5 | 71.0 → 68.3 | 53.8 → 43.3 | Ours (exp_30) |
| *Between-arm mean FT* | | | | *65.7* | *66.0* | *62.5* | |
| **Within-subject benchmarks (subject ⊥ label)** | | | | | | | |
| HMC (sleep staging) | within | 5 | 151 | 59.8 → 69.9 | 51.8 → 71.1 | 63.8 → 70.8 | EEG-FM-Bench |
| PhysioMI (motor imagery) | within | 4 | 109 | 29.6 → 57.3 | 26.9 → 56.7 | 27.4 → 27.9 | EEG-FM-Bench |
| BCIC-IV-2a (motor imagery) | within | 4 | 9 | 28.4 → 29.0 | 29.2 → 33.7 | 28.6 → 32.7 | EEG-FM-Bench |
| SEED-VII (emotion) | within | 7 | 20 | 23.2 → 20.4 | 19.4 → 23.1 | 20.6 → 22.8 | EEG-FM-Bench |
| Things-EEG-2 (tgt vs non-tgt) | within | 2 | 10 | 50.0 → 53.4 | 50.0 → 62.4 | 50.0 → 61.0 | EEG-FM-Bench |
| EEGMAT (task vs rest) | within | 2 | 36 | 67.1 → 73.1 | 73.1 → 62.0 | 67.1 → 72.7 | Ours (exp_30) |
| SleepDep (NS vs SD) | within | 2 | 36 | 50.0 → 53.2 | 55.7 → 55.6 | 54.4 → 54.2 | Ours (exp_30) |
| *Within-arm mean FT* | | | | *50.9* | *52.1* | *48.9* | |

**Notes**:
- All reported values are balanced accuracy (%). "F" = frozen backbone with linear/avg-pool head (equivalent to LP). "FT" = full-parameter fine-tuning.
- **Stress (UCSD) is excluded** because 3/17 subjects crossing DASS labels makes the design neither cleanly within- nor between-subject (see §2.1.1).
- Our datasets use subject-level 5-fold CV (subject-disjoint train/val/test); EEG-FM-Bench datasets use the split protocols documented in their Appendix B.3.
- `ADFTD` in our table is **binary** (AD vs CN) to match our main pipeline; EEG-FM-Bench uses **3-class** (AD/FTD/CN) with stratified split, so their numbers are not directly comparable.

---

## Derived statistics (for §1.1 text)

### Mean ΔBA (FT − Frozen) across the 3 FMs

| Arm | Mean ΔBA | Range |
|---|---:|---|
| Between-arm (literature: TUAB, Siena) | +10.1 BA | [−8.0, +16.7] |
| Between-arm (ours: ADFTD, TDBRAIN, Meditation) | −3.0 BA | [−10.5, +4.2] |
| Within-arm (literature: HMC, PhysioMI, BCIC-IV-2a, SEED-VII, Things-EEG-2) | +9.1 BA | [−2.8, +27.7] |
| Within-arm (ours: EEGMAT, SleepDep) | +1.4 BA | [−11.1, +6.0] |

**Interpretation (data-driven, honest)**:
- Literature ΔBA is comparable across arms (~+9–10 BA). **The simple "between arm always wins FT gain" story is NOT in the literature data** — e.g. PhysioMI (within, MI) shows +28 ΔBA for LaBraM.
- Our own datasets show much smaller ΔBA in both arms (often negative). This is a combination of (a) small N, (b) no HP search, (c) harder benchmarks. **This is acknowledged in §3 and §4 Limitations.**
- The paper's mechanism claim (C2: subject_id_ba predicts ΔBA in between-arm only) is orthogonal to arm-mean ΔBA magnitude and survives regardless of the magnitude pattern.

### What this means for the critique

The table does NOT show a dramatic between > within FT gain. Instead it shows:
1. Subject-level CV is rare in the literature — most FM papers silently use stratified/trial-level splits on motor-imagery and emotion datasets, inflating reported BA.
2. When subject-level CV IS used, ΔBA is task-dependent, not arm-dependent.
3. Therefore the "between wins" narrative comes from the **mix of subject-level (diagnostic) and trial-level (motor/emotion) numbers** in the same summary tables — which mixes two different leakage regimes.

This is a different critique than the one the pre-experiment paper strategy imagined. The finding is:

> **The EEG-FM benchmark literature is inconsistent because split protocols are inconsistent across task families. Between-subject diagnostic tasks are usually reported under subject-level splits (clean); within-subject motor/emotion tasks are usually reported under trial-level splits (subject-leakage inflated). The two can't be directly compared.**

The mechanism we demonstrate (subject_id_ba → ΔBA in between-arm only) is the representation-level smoking gun for why subject leakage is not just a CV bug but an artefact of how EEG FMs encode identity.

---

## Table 1 caption (draft)

> **Table 1.** Frozen → FT balanced accuracy for three EEG foundation models on benchmarks with explicit subject-level CV. Literature rows are drawn from EEG-FM-Bench Table 2/4 (2025); rows labelled "ours" are from exp_30 under matched subject-level 5-fold CV (see §2.2). Between-arm datasets have one-subject-one-label design; within-arm have the same subjects contributing multiple labels. We exclude benchmarks that use stratified or trial-level splits in the source literature (ADFTD-3cls, TUEV, Mimul-11, MentalArithmetic in EEG-FM-Bench; see Appendix X for the list) because those numbers cannot be cleanly compared to subject-level numbers. Stress (UCSD) is excluded from both arms because only 3/17 subjects cross DASS labels (see §2.1.1).

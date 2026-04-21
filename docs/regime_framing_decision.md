# Regime Framing Decision (2026-04-21)

Single source of truth for the strategic pivot from **Stress-centric** narrative
to **regime-level** narrative. Supersedes the "Stress as protagonist" language
scattered across older drafts. Complements `docs/TODO.md` and
`docs/methodology_notes.md#N-F22`.

---

## Motivation (why the pivot)

The Stress-centric framing concentrates all defensibility on a single dataset.
Attack surface:

1. **N=1 generalization** — 17 subjects × 70 recordings is one exhibit, not a phenomenon.
2. **Self-incrimination trap** — calling Stress a "bad dataset" reduces the contribution to "FMs fail on bad data, duh".
3. **Wang 2025 double-bind** — citing 90.47% as inflated baseline simultaneously implies the dataset is fit-able; we cannot both "it's a bad dataset" and "Wang inflated a legitimate benchmark".

The fix is to move the protagonist from *Stress* to *the class of datasets Stress belongs to*.

---

## The axis

**Regime** is binary:

| Regime | Definition | Example datasets |
|---|---|---|
| **Subject-label** | Label is a per-subject scalar; no within-subject contrast | Stress (DASS trait), DEAP, SEED, DREAMER, DASPS |
| **Within-subject paired** | Label is a paired within-subject contrast (state A vs state B in same person) | EEGMAT (rest vs arithmetic), SleepDep (normal vs sleep-deprived), HMC (sleep staging) |

Claim: FM-style representation learning is structurally **incompatible** with the
subject-label regime under small-N, and structurally **compatible** with the
within-subject regime. Stress demonstrates the failure mode; EEGMAT + SleepDep
demonstrate the success mode.

---

## What changes

### Core claim (new)
> There is a class of EEG experimental designs — per-subject scalar labels
> without within-subject contrast — that triggers a predictable FM failure
> mode. We characterise this mode using Stress as a canonical instance, contrast
> it with EEGMAT/SleepDep (within-subject regime), and provide a diagnostic
> framework (variance + RSA + FOOOF + band-stop) that generalises across regime.

### Stress's role
From protagonist → **representative instance** of the subject-label regime.
All Stress-specific mechanisms (per-subject label confound, subject shortcut,
0.43–0.58 architecture ceiling) are framed as *regime-level* predictions that
Stress happens to instantiate.

### SleepDep's role
From Fig 5 guest → **second datapoint in the within-subject regime**. Its
variance+RSA pattern (subject_frac rises under FT, matching EEGMAT) strengthens
the regime claim from N=1 per pole to N=1+2.

---

## Figure audit under regime framing

| Figure | Current | Under regime framing | Action |
|---|---|---|---|
| Fig 1 schematic | user-drawn | must show regime axis + 3 datasets on it | user drawing |
| Fig 2 geometry | 2 DS × variance + 6-pt RSA | **3 DS × variance (top) + RSA frozen→FT arrows (bottom)** | ✅ done 2026-04-21 |
| Fig 3 honest eval | Stress funnel + LaBraM×{Stress, EEGMAT} null | regime contrast already 2 vs 1, sufficient | keep as-is |
| Fig 4 anchoring | 2 DS × 6 trajectory cells | add SleepDep row if paired data exists | gated on data check |
| Fig 5 FOOOF | 3 DS × 3 panels | already 3-regime, good | keep |
| Fig 6 left ceiling | Stress-only | **add EEGMAT ceiling** to show "regime-dependent architecture" | open (TODO #12) |
| Fig 6 right drift | Stress + EEGMAT with shortcut/label/nodrift verdict | verdict logic flawed under regime framing | rewrite (TODO #11) |

---

## Subject frac is regime-conditional (critical reading note)

See `docs/methodology_notes.md#N-F22` for full detail. Summary:

- **Subject-label regime**: `subject_frac ↑` under FT = shortcut risk.
- **Within-subject regime**: `subject_frac ↑` under FT = healthy, because the
  label is defined relative to subject baseline and the FM needs subject
  identity to measure within-subject deviation.

The paper narrative cannot treat `subject_frac` as a universal quality signal.
Fig 2 caption must qualify direction by regime; Fig 6 verdict labels need
rewrite for the same reason.

Variance fraction and RSA can **disagree in direction** (Stress FT: variance
`subject_frac ↓` but `rsa_subject_r ↑`). They measure different things — the
former is absolute SS, the latter is rank correlation of pairwise distances.
Paper must present both as complementary, not substitutes.

---

## Low-cost reframe (what can be done without new experiments)

1. **Intro** — state regime taxonomy in §1; cite DEAP/SEED/DREAMER/DASPS for
   prevalence of subject-label regime. Zero experimental cost.
2. **Methods §3.1** — list three datasets with regime label each.
3. **Fig 2 caption** — qualify variance direction by regime.
4. **Fig 6 verdict rewrite** — drop or regime-condition.

## Higher-cost reframe (gated)

5. **Fig 4 SleepDep row** — needs paired protocol check.
6. **Fig 6 left EEGMAT ceiling** — needs 2 non-FM × EEGMAT × 3 seeds.
7. **Independent subject-label replication** (DEAP or DREAMER same pipeline) —
   strongest possible defence; only if reviewer forces it.

---

## Data artifacts added 2026-04-21

- `paper/figures/_historical/source_tables/sleepdep_variance_rsa.json` — frozen + FT variance fractions and RSA for SleepDep × 3 FMs
- `paper/figures/_historical/source_tables/ft_rsa_stress_eegmat.json` — FT RSA for Stress + EEGMAT × 3 FMs (completes the arrow endpoints)
- `scripts/analysis/compute_sleepdep_variance_rsa.py`
- `scripts/analysis/compute_ft_rsa.py`

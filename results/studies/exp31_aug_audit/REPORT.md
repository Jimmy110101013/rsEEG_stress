# exp31 — `aug_overlap` audit

**Date**: 2026-04-22
**Rationale**: `pipeline/dataset.py:196` hard-codes `aug_overlap` to oversample
windows of `label==1` only — an assumption baked in for UCSD Stress
(`label==1=increase` = minority). On other datasets the semantic of
`label==1` is arbitrary: for ADFTD it is **AD = majority** (108/87), and
for EEGMAT it is **task = window-minority** (task recordings are shorter
than rest, so task has ~3× fewer windows at the same window count).

s42-only seeds previously hinted (**ADFTD LaBraM no-aug 0.744 vs aug-on 0.709
= +3.5pp**) that the hard-coded aug might be a harmful artifact. This
15-seed audit (3 seeds × 3 FMs × relevant configs) tests that hypothesis.

## Results

### EEGMAT × LaBraM

| Config | Mean BA | SD | n | Δ vs aug=0.75 |
|---|---|---|---|---|
| **aug=0.75 (canonical)** | 0.7315 | 0.0212 | 3 | — |
| aug=0.5                  | 0.6713 | 0.0212 | 3 | **−6.0 pp** |
| no-aug                   | 0.7037 | 0.0349 | 3 | −2.8 pp (within 1 SD) |

Seeds: aug=0.5 → [0.667, 0.653, 0.694]; no-aug → [0.708, 0.736, 0.667].

**Finding**: aug=0.75 is the empirical optimum. aug=0.5 is substantially
worse (−3 SD), and no-aug lands close to baseline but with higher
variance. The non-monotonic sweep is consistent with aug acting as data
augmentation (regularization) on top of its class-balancing role.

### ADFTD × {LaBraM, CBraMod, REVE}

| FM | Config | Mean BA | SD | n | Δ vs aug=0.75 |
|---|---|---|---|---|---|
| LaBraM  | **aug=0.75 (canonical)** | 0.7086 | 0.0137 | 3 | — |
| LaBraM  | no-aug                   | 0.6959 | 0.0444 | 3 | −1.3 pp (within 1 SD; high variance) |
| CBraMod | **aug=0.75 (canonical)** | 0.5372 | 0.0266 | 3 | — |
| CBraMod | no-aug                   | 0.5039 | 0.0159 | 3 | **−3.3 pp** |
| REVE    | **aug=0.75 (canonical)** | 0.6577 | (n=1 perm) | 1 | — |
| REVE    | no-aug                   | 0.6376 | 0.0187 | 3 | −2.0 pp |

LaBraM seeds: [0.744, 0.656, 0.687]. The s42=0.744 outlier drove the
earlier "+3.5pp" signal; once s123 and s2024 landed back in the aug-on
band, the effect dissolves.

**Finding**: across all three FMs, aug=0.75 is either equal (LaBraM,
within seed noise) or mildly better than no-aug on ADFTD. The
methodological asymmetry (aug targets majority AD) does **not**
translate to measurable BA harm.

## Interpretation

The `aug_overlap` hard-coding to `label==1` is a latent implementation
caveat, not an active bug. Its effect varies by dataset:

| Dataset   | Raw window balance (no-aug) | Aug target `label==1` | Direction of aug | Empirical effect (mean BA) |
|-----------|-----------------------------|-----------------------|------------------|----------------------------|
| Stress    | ~3.9:1 class_0 majority     | minority (increase)   | ✅ corrects      | used as canonical |
| EEGMAT    | ~3:1 class_0 majority       | minority (task)       | ✅ direction OK, magnitude overshoots | aug=0.75 optimal |
| ADFTD     | ~1.26:1 class_1 majority    | majority (AD)         | ❌ amplifies     | aug=0.75 ≈ no-aug (LaBraM) or mildly better (CBraMod, REVE) |
| SleepDep  | 1:1 balanced                | (aug disabled)        | n/a              | correctly off |

Two mechanisms likely offset each other on ADFTD: aug increases the
AD:HC window ratio (harmful for class-balance reasons) but also
provides ~4× more training windows (beneficial as regularization).
Net effect is a wash.

## Decision

**Keep the canonical recipe on every dataset** (Stress/EEGMAT/ADFTD
aug=0.75, SleepDep no-aug). No real-FT or permutation-null rerun is
warranted by this audit — BA differences are within seed noise, and
no configuration convincingly beats the existing baseline.

## Paper methods note

A one-paragraph caveat will be added to the paper's methods / appendix:

> Window-level oversampling (`aug_overlap=0.75`) is applied to the
> `label==1` class in our training recipe. This is the minority class
> on UCSD Stress (for which the flag was originally designed) and the
> window-minority class on EEGMAT (task recordings are shorter than
> rest). On ADFTD, `label==1` corresponds to AD, which is the
> recording-majority; we audited the implication with a 9-seed
> no-augmentation sweep (LaBraM / CBraMod / REVE × 3 seeds) and
> observed no significant BA change (−1.3 to −3.3 pp, within 1–2 SD of
> the aug-on baseline). The canonical recipe is therefore retained
> across datasets and FMs for apples-to-apples comparison.

## Files produced

- `eegmat_aug50_s{42,123,2024}/`           — EEGMAT LaBraM aug=0.5 seeds
- `eegmat_noaug_s{42,123,2024}/`           — EEGMAT LaBraM no-aug seeds
- `adftd_noaug_s{42,123,2024}/`            — ADFTD LaBraM no-aug seeds
- `cbramod_adftd_noaug_s{42,123,2024}/`    — ADFTD CBraMod no-aug seeds
- `reve_adftd_noaug_s{42,123,2024}/`       — ADFTD REVE no-aug seeds
- `logs/_chain_*.log`                      — per-chain run logs

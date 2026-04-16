# WSCI — Within-Subject Contrast Index

**Status**: Exploratory design (2026-04-16). Not yet validated. Not in paper.
**Origin**: Discussion with NEH lecture inspiration; six-layer theoretical
walk-through with Jimmy on 2026-04-16. All design decisions ratified inline
below.
**Goal**: A model-agnostic, pre-fine-tuning diagnostic statistic that quantifies
the within-subject neural-contrast strength of a downstream EEG label, derived
from Holo-Hilbert Spectral Analysis (HHSA). Intended use: triage whether a new
clinical EEG dataset is *anchored* (FT will rescue) or *bounded* (subject
dominance ceiling will hold).

If validated against subject-level FT BA across ≥ 6 datasets with Spearman
$\rho > 0.7$, WSCI becomes the methodological contribution that lifts the
SDL paper from "diagnosis + checklist" to "diagnosis + actionable
quantitative protocol".

---

## Layer 0 — Why HHSA at all

Two failure modes of existing within-subject contrast measures motivate HHSA:

1. **Welch-PSD + cluster permutation** captures only stationary, narrow-band
   power differences. EEG cognitive contrasts (especially stress / arousal)
   often live in non-stationary cross-frequency *amplitude* coupling, which
   PSD misses entirely.
2. **Traditional PAC** (Tort MI, Canolty MVL) requires pre-specified
   bandpass for both phase and amplitude — which (a) imposes arbitrary
   band edges and (b) introduces filter ringing that fakes coupling on
   non-stationary signals (Aru et al. 2015 critique).

HHSA (Huang et al. 2016, Phil. Trans. R. Soc. A 374: 20150206) is a
data-adaptive cross-frequency representation with no bandpass step. The
hypothesis is: if a label modulates within-subject EEG via cross-frequency
AM coupling, HHSA will detect it where PSD and PAC do not.

---

## Layer 1 — EMD and IMF foundations

**IMF criterion** (Huang et al. 1998, Proc. R. Soc. A 454: 903):
1. $|N_\text{extrema} - N_\text{zero}| \leq 1$ over the signal
2. $(e_\max(t) + e_\min(t))/2 = 0$ at all $t$ (upper/lower cubic-spline
   envelopes have zero local mean)

**Sifting**:
```
h_0 ← x
repeat:
    locate maxima/minima of h_k
    cubic-spline envelopes e_max, e_min
    m_k ← (e_max + e_min)/2
    h_{k+1} ← h_k − m_k
until stop criterion
IMF_1 ← h_K; r_1 ← x − IMF_1; recurse on r_1 for IMF_2, ...
```

### Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Stop criterion | **S-number = 5** (Rilling & Flandrin 2003) | More stable than SD threshold on short EEG epochs |
| End-effect handling | **Boundary-point promotion** (Huang group's preferred rule, confirmed by NEH 2026-04 lecture): if endpoint $x(0)$ exceeds linear extrapolation of two nearest extrema, treat $x(0)$ as a maximum (symmetric for minima) | Avoids cubic-spline overshoot at boundaries; cheaper than wave-characteristic extension; no mirror-induced spurious low-frequency content on alpha-rich resting EEG |
| Max IMF count | **8** | 200 Hz × 5 s ⇒ ~10 dyadic levels theoretical; EEG has meaningful structure in first 5–7 |

**Implementation note**: `emd` package's `get_padded_extrema()` defaults to
mirror padding. Need to verify whether boundary-point promotion is
selectable; otherwise monkey-patch.

---

## Layer 2 — Mode mixing and noise-assisted decomposition

**Problem**: Plain EMD on EEG produces single IMFs that mix multiple
physical scales (intermittent bursts cause extrema to switch scales mid-signal).
Almost unusable on resting EEG without noise assistance.

**Algorithms**:
- **EEMD** (Wu & Huang 2009, Adv. Adapt. Data Anal. 1: 1): add Gaussian
  noise, run EMD $N$ times, average per-IMF. Suffers from residual noise
  leakage and inconsistent IMF count across runs.
- **CEEMDAN** (Torres et al. 2011, ICASSP-2011: 4144): extracts IMFs
  iteratively, each stage using residual + noise-derived component. Noise
  cancels per-IMF, IMF count is unique, decomposition is complete
  ($x = \sum \text{IMF}_k + r_K$ exact).

### Decisions

| Topic | Decision | Rationale |
|---|---|---|
| Algorithm | **CEEMDAN** | Unique IMF count enables cross-epoch alignment; cleaner residuals than EEMD |
| Noise amplitude $\sigma$ | **0.1 × std(x)** | EEG-HHSA literature median (e.g., Hsu et al. 2018, NeuroImage 165: 194). Smaller σ avoids drowning real alpha (~10 µV) in noise (~5 µV) |
| Ensemble size $N$ | **100** | $\sigma/\sqrt{N} \approx 0.01$ residual; CEEMDAN converges faster than EEMD |
| Noise seed | **`seed_master + ensemble_idx`** | Required for reproducibility consistent with `cudnn.deterministic=True` policy |

**EEG-specific risk to monitor**: if alpha IMFs show noise-pattern
artifacts in QC, drop to σ = 0.05 and re-run.

---

## Layer 3 — Hilbert transform, instantaneous frequency, Bedrosian

**Analytic signal**: $z(t) = x(t) + i\mathcal{H}\{x\}(t) = a(t) e^{i\phi(t)}$
gives $a(t) = |z|$, $\phi(t) = \arg z$, $f_\text{inst} = (2\pi)^{-1} d\phi/dt$.

**Bedrosian theorem** (Bedrosian 1963, Proc. IEEE 51: 868): for
$x = a(t)\cos\phi(t)$, $\mathcal{H}\{a\cos\phi\} = a\sin\phi$ requires
- $a$ and $\cos\phi$ have non-overlapping Fourier spectra (slow envelope,
  fast carrier)
- $a \geq 0$

**Why IMF criterion = Bedrosian sufficient condition**:

| IMF criterion | Bedrosian counterpart |
|---|---|
| Cond. 1: extrema ≈ zero crossings | Single-rhythm narrow-band carrier |
| Cond. 2: zero envelope mean | Slow envelope without DC bleed |

So sifting is precisely engineered to make Hilbert IF physically meaningful.
Skipping sifting and applying Hilbert directly to broadband EEG yields
nonsense IF.

### Decisions

| Topic | Decision | Rationale |
|---|---|---|
| IF method | **Direct quadrature / Normalized Hilbert (`method='nht'`)** (Huang et al. 2009, Adv. Adapt. Data Anal. 1) | More robust to transient than standard Hilbert; minimizes negative-IF artifacts |
| Phase unwrap | `np.unwrap` standard | No real choice |
| Boundary trust window | **Drop 1 s from each end** (= 1 cycle of lowest IMF at ~1 Hz) | FFT-based Hilbert has circular-convolution leakage at edges |
| Negative IF handling | **Set to NaN, exclude from holospectrum bin accumulation** | Avoids artificial fill-value contamination |

**Per (recording, channel, IMF)** output: $\{a_k(t), f_k(t)\}$ on the
trusted window (e.g., $t \in [1, 4]$ s for a 5 s epoch), 200 Hz sampling.

---

## Layer 4 — HHSA double decomposition

**Motivation**: First-order Hilbert spectrum
$H_1(\omega, t) = \sum_k a_k(t)\delta(\omega - f_k(t))$
captures FM (carrier frequency drift) but not AM (envelope's own frequency
structure). Huang 2016: a non-linear oscillator's signal has form
$\sum_k A_k(t)\cos(\int \omega_k d\tau)$ where $A_k(t)$ is itself a signal
with frequency content (e.g., theta amplitude modulated by 0.1 Hz
respiration). PSD, wavelet, and first-layer HHT all miss this — Huang's
"missing information".

**Construction**: apply EMD-Hilbert *to each $a_k(t)$*:
```
for k = 1..K_carrier:
    {AM-IMF_{k,1}, ..., AM-IMF_{k,M}} ← CEEMDAN(a_k)
    for j = 1..M:
        {b_{k,j}(t), g_{k,j}(t)} ← NHT(AM-IMF_{k,j})
```

**Holo-Hilbert spectrum** (4D):
$$H(\omega_F, \omega_A, t, E) = \sum_{k,j} b_{k,j}(t)\,
\delta(\omega_F - f_k(t))\,\delta(\omega_A - g_{k,j}(t))$$

Marginalize over $t$ for the standard 2D plot $H(\omega_F, \omega_A)$ —
"carrier frequency × AM frequency", read as "carrier $f^*$ is amplitude-
modulated at rate $g^*$".

### Decisions

| Topic | Decision | Rationale |
|---|---|---|
| AM-layer sift algorithm | **CEEMDAN** (same params as carrier layer) | Consistency; AM signals also non-stationary |
| AM max IMF per carrier | **5** | AM bandwidth limited by carrier Nyquist; few meaningful AM scales |
| Time marginalization | **Yes, output $H(\omega_F, \omega_A)$ only** | 5 s epoch too short for time-resolved AM (≥ 0.5 s per AM cycle) |
| Frequency grid | $\omega_F \in [0.5, 80]$ Hz × $\omega_A \in [0.05, 25]$ Hz, **log-spaced, 8 bins/octave** ⇒ 56 × 64 grid | Log scale matches EEG band structure |
| Per-recording aggregation | **Keep epoch-level distribution** (do not pre-average) | Enables epoch-as-replicate mixed-effect statistics later |

**Compute budget**: ~4.5 s per (recording, channel) on single CPU core;
70 rec × 30 ch ≈ 2.6 hr serial; ~10 min on 16 cores. Acceptable.

---

## Layer 5 — Relation to traditional PAC (Tort MI, Canolty MVL)

**Tort MI** (Tort et al. 2010, J. Neurophysiol. 104: 1195):
1. Bandpass-filter $x$ into phase band $[f_p^-, f_p^+]$ → $\phi(t)$
2. Bandpass-filter $x$ into amplitude band $[f_A^-, f_A^+]$ → $A(t)$
3. Bin $\phi$ into $N=18$ bins; compute mean $A$ per bin → $P(j)$
4. $\text{MI} = D_\text{KL}(P \| U) / \log N$

**Canolty MVL** (Canolty et al. 2006, Science 313: 1626):
$\text{MVL} = |T^{-1}\sum_t A(t) e^{i\phi(t)}|$

**HHSA vs PAC equivalence**: a horizontal slice of HHSA at fixed
$\omega_F = \omega_F^*$ along $\omega_A$ is approximately equivalent to
the amplitude spectrum of the bandpass-filtered signal at $\omega_F^*$.
HHSA is therefore a **continuous, adaptive comodulogram**.

**Three reasons HHSA is cleaner on EEG**:
1. **No bandpass artifacts**: Aru et al. 2015 documents PAC false-positive
   rate 5–10 % from filter ringing alone. HHSA has no bandpass step.
2. **No band-edge arbitrariness**: PAC requires choosing theta = 4–8?
   5–9? 6–10? Hülsemann et al. 2019 traces a chunk of PAC
   irreproducibility to this. HHSA bands emerge from sifting.
3. **Cross-band joint detection**: PAC comodulogram cells are independent
   bandpass runs; HHSA represents all carriers jointly, so co-modulation
   of multiple carriers by a shared AM signal is directly visible.

**Honest costs to disclose**:
- HHSA ≈ 100× MI compute per epoch
- HHSA spectrum is not a scalar ⇒ requires reduction (Layer 6)
- HHSA in EEG has < 50 papers vs PAC's > 4000 ⇒ "niche" critique risk

### Decisions

| Topic | Decision | Rationale |
|---|---|---|
| PAC-vs-HHSA validation experiment | **Include direct comparison in Appendix** | Sanity check + show added value |
| PAC method for comparison | **Tort MI** | More sensitive to amplitude distribution shape than MVL |
| PAC bands | **theta = 4–8 Hz, gamma = 30–80 Hz** (Tort 2010 standard) | No grid sweep; avoid drift into "PAC paper" |
| Methods justification scope | **All three reasons stated**, as different reviewers care about different angles | Don't pre-truncate the defense |

---

## Layer 6 — WSCI: Statistical design

### 6.1 From holospectrum to per-subject effect size

For subject $s$, condition $c \in \{0, 1\}$, epoch $i$, channel $\text{ch}$,
the holospectrum is $H_{s,c,i,\text{ch}}(\omega_F, \omega_A) \in
\mathbb{R}_{\geq 0}^{56 \times 64}$.

**Step 1** — Channel aggregation, geometric mean:
$$H_{s,c,i}(\omega_F, \omega_A) = \exp\!\left(\frac{1}{|\text{ch}|}\sum_\text{ch} \log H_{s,c,i,\text{ch}}\right)$$

**Step 2** — Per-bin paired effect size, Hedges' $g_z$ with small-sample
correction $J(n_s) \approx 1 - 3/(4n_s - 9)$:
$$g_z^{(s)}(\omega_F, \omega_A) = \frac{\bar{H}_{s,1} - \bar{H}_{s,0}}{\text{SD}_\text{paired}^{(s)}} \cdot J(n_s)$$

**Step 3** — 2D cluster-mass permutation (Maris & Oostenveld 2007,
J. Neurosci. Methods 164: 177):
- 1000 within-subject condition-label permutations
- Per permutation: build $g_z$ map, extract connected clusters where
  $|g_z| > 0.5$, record max cluster mass
- Real clusters with mass > 95th percentile of null are *surviving*

### 6.2 From surviving clusters to per-subject WSCI

$$\text{WSCI}_s = \frac{\sum_{C \in \mathcal{C}_s^*} \text{sign}(g_z^C)\,|C|}{\sum_{(\omega_F,\omega_A)} \bar{H}_s(\omega_F, \omega_A)}$$

where $|C|$ is cluster mass (sum of $|g_z|$ within the cluster) and
$\mathcal{C}_s^*$ is the set of surviving clusters.

**Interpretation**: fraction of holospectrum energy concentrated in
statistically significant cross-frequency coupling regions associated
with the label, signed by direction.

### 6.3 Dataset-level summary

$$\text{WSCI}_\text{dataset} = \text{median}_s \text{WSCI}_s, \quad
\text{CI}_{95\%} \text{ via subject-bootstrap, } B=1000$$

Median over mean to resist outlier subjects (epileptiform spikes etc.).

### 6.4 Cross-dataset validation

**Primary test**: Spearman rank correlation between WSCI_dataset and
subject-level FT BA across $\geq 6$ datasets:
- UCSD Stress, EEGMAT, TDBRAIN MDD, SAM40, EDPMSC + one more
- ADFTD excluded (single recording per subject ⇒ no within-subject
  contrast definable; this is itself diagnostic)
- Target: $\rho > 0.7$, one-sided $p$ honest given small $N$

**Backup**: subject-level mixed-effects regression predicting
held-out-subject FT BA from WSCI, scaling sample to ~100+ subjects
across datasets. Higher-cost (requires LOSO FT) but better-powered.

### 6.5 Pre-FT triage

```
new dataset → WSCI_dataset
   │
   ├─ WSCI > τ_high  → "Anchored": FT BA > 0.7 expected; train FM
   ├─ τ_low < WSCI < τ_high → "Intermediate": run LP first
   └─ WSCI < τ_low   → "Bounded": don't bother with FM, classical ML suffices
```

Thresholds $\tau_\text{high}, \tau_\text{low}$ selected via leave-one-
dataset-out CV maximizing 3-class accuracy.

### 6.6 Reporting (when ready for paper)

1. Table: per-dataset WSCI median + 95 % bootstrap CI
2. Figure: WSCI vs subject-level FT BA scatter (6 dataset points) +
   Spearman $\rho$
3. Appendix figure: per-subject WSCI distribution per dataset

### Decisions

| Topic | Decision |
|---|---|
| Channel aggregation | Geometric mean |
| Effect size | Hedges $g_z$ (paired, with small-sample correction) |
| Cluster threshold | $|g_z| > 0.5$, 1000 permutations |
| WSCI definition | Surviving cluster mass / total spectrum energy, signed |
| Dataset summary | Subject-median + bootstrap CI |
| Validation N target | 6 datasets minimum (Stress, EEGMAT, TDBRAIN, SAM40, EDPMSC + 1) |
| Threshold selection | LODO-CV |
| Paper placement (when validated) | Independent §5 "Pre-FT diagnostic protocol" |

---

## Implementation plan

### Phase 1 — Prototype (1 week)
1. `pipeline/hhsa.py`
   - `compute_imfs(x, fs, σ=0.1, N=100, max_imf=8)` → CEEMDAN
   - `instantaneous_amp_freq(imfs, fs)` → NHT, returns $\{a_k, f_k\}$
   - `holospectrum(x, fs, grid_F, grid_A)` → 2D array on (56, 64) log-grid
   - `holospectrum_recording(eeg, fs, ...)` → wraps over (channel, epoch)
2. `analysis/wsci.py`
   - `per_subject_gz(H_subject, conditions)` → 2D effect-size map
   - `cluster_mass_permutation(H_subject, conditions, n_perm=1000)` → surviving clusters
   - `wsci_subject(...)` → scalar
   - `wsci_dataset(wsci_subjects)` → median + bootstrap CI

### Phase 2 — Validation notebook (3 days)
Notebook `notebooks/wsci_validation.ipynb`:
- Load Stress (70 rec) + EEGMAT (72 rec)
- Compute holospectra, eyeball per-subject 2D plots
- Compute WSCI, plot per-subject distributions
- Compare WSCI(Stress) vs WSCI(EEGMAT) — sanity: EEGMAT should be much higher
- If pattern holds, expand to TDBRAIN, ADFTD (latter as null check)

### Phase 3 — Decision point
- If WSCI(EEGMAT) > WSCI(Stress) with non-overlapping CIs ⇒ proceed to
  multi-dataset rank correlation (Phase 4)
- Else ⇒ debug: parameter sweep on σ, N, AM max IMF; if still null,
  HHSA hypothesis fails for this label class — record outcome, don't
  publish

### Phase 4 — Multi-dataset (2-3 weeks)
- TDBRAIN, ADFTD, SAM40, EDPMSC
- Spearman rank correlation across 6 datasets
- LODO-CV for triage thresholds

### Phase 5 — Paper integration (when validated)
- Add §5 "Pre-FT diagnostic protocol"
- Add appendix PAC vs HHSA comparison
- Add appendix WSCI parameter sensitivity (σ, N)
- Position WSCI as 4th contribution in introduction

## Open questions

1. `emd` package's default sift padding — verify against boundary-promotion
   rule; may need monkey-patch.
2. Channel aggregation — geometric mean is one choice; alternatives
   (max, top-k mean) worth trying if geometric mean dilutes localized
   contrasts.
3. Cluster threshold $|g_z| > 0.5$ — somewhat arbitrary; try 0.3 in
   sensitivity analysis.
4. ADFTD has no within-subject contrast → WSCI undefined. This is
   itself the right answer (the dataset doesn't admit within-subject
   triage), but how to *report* this in the protocol diagram needs
   thought.
5. Lecture source — verify which specific Huang paper documents the
   boundary-point promotion rule for citation purposes.

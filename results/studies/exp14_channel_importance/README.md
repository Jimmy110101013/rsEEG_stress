# exp14: FM Neuroscience Interpretability

Three complementary analyses probing what FM frozen representations encode: spatial (which channels?), correlational spectral (which bands correlate?), and causal spectral (which band removal hurts most?).

---

## A. Channel Importance via Per-Channel Ablation

### Research question
Which EEG channels do FM representations rely on most?

### Method
**Per-channel zero ablation**: for each of 30 channels, zero it out → re-extract frozen features → measure cosine distance from original. Higher distance = more important.

- Model-agnostic (input-side perturbation, no attention map access needed)
- Dataset: Stress, 70 recordings, 30ch
- FMs: LaBraM, CBraMod, REVE (all frozen)

### Key findings
- **All 3 FMs prioritize posterior channels** (O2, OZ, PZ, P4) — consistent with resting-state alpha dominance
- **LaBraM** (max=0.0103): O2, OZ, T4 — matches F10 (classical RF right-hemisphere alpha)
- **CBraMod** (max=0.0033): 3-5× lower sensitivity, uniform distribution — criss-cross attention distributes across channels
- **REVE** (max=0.0154): posterior bias similar to LaBraM; CPZ also high

---

## B. Band-Specific RSA (correlational)

### Research question
Which frequency bands do FM representations correlate with? Does this differ between Stress and EEGMAT?

### Method
**RSA**: compute band power RDM (delta/theta/alpha/beta per channel) and FM feature RDM, correlate via Spearman r.

- Datasets: Stress (70 rec, 30ch), EEGMAT (72 rec, 19ch)
- FMs: LaBraM, CBraMod, REVE (all frozen)

### Key findings

**Stress** — no band selectivity:
| FM | Delta | Theta | Alpha | Beta |
|---|---|---|---|---|
| LaBraM | 0.274 | 0.172 | 0.180 | 0.270 |
| CBraMod | 0.373 | 0.310 | 0.340 | 0.285 |
| REVE | 0.327 | 0.249 | 0.272 | 0.164 |

All bands significant (p < 0.001) → FM captures broadband spectral structure (subject fingerprint)

**EEGMAT** — clear alpha/theta preference:
| FM | Delta | Theta | Alpha | Beta |
|---|---|---|---|---|
| LaBraM | 0.098 | **0.192** | **0.175** | 0.019 (ns) |
| CBraMod | 0.042* | 0.084 | **0.130** | **0.167** |
| REVE | −0.024 (ns) | 0.041* | **0.176** | **0.152** |

Band selectivity present → FM captures task-relevant frequency modulations

---

## C. Band-Stop Ablation (causal)

### Research question
Which frequency band does each FM causally depend on? Does this differ between Stress and EEGMAT?

### Method
**Band-stop filter ablation**: for each band (delta 1-4Hz, theta 4-8Hz, alpha 8-13Hz, beta 13-30Hz), apply Butterworth band-stop filter to raw EEG → re-extract frozen features → measure cosine distance. Higher = FM causally relies more on that band.

- Datasets: Stress (70 rec, 30ch), EEGMAT (72 rec, 19ch)
- FMs: LaBraM, CBraMod, REVE (all frozen)
- Filter: 4th-order Butterworth band-stop, zero-phase (sosfiltfilt)

### Key findings

**Stress:**
| FM | Delta | Theta | Alpha | **Beta** |
|---|---|---|---|---|
| LaBraM | 0.058 | 0.055 | 0.082 | **0.168** |
| CBraMod | 0.022 | 0.006 | 0.018 | 0.018 |
| REVE | 0.041 | 0.041 | 0.061 | 0.048 |

LaBraM most dependent on **beta** (0.168, 2× alpha, 3× delta/theta)

**EEGMAT:**
| FM | Delta | Theta | **Alpha** | Beta |
|---|---|---|---|---|
| LaBraM | 0.063 | 0.060 | **0.136** | 0.122 |
| CBraMod | 0.035 | 0.011 | **0.029** | 0.012 |
| REVE | 0.060 | 0.075 | **0.150** | 0.048 |

LaBraM and REVE shift to **alpha-dominant** (REVE: 0.150, 3× beta)

### Cross-dataset contrast

| FM | Stress dominant band | EEGMAT dominant band | Interpretation |
|---|---|---|---|
| **LaBraM** | Beta (arousal/subject ID) | Alpha (task-induced suppression) | FM adapts to available signal |
| **CBraMod** | Uniform (low sensitivity) | Delta (low sensitivity overall) | Criss-cross attention is band-agnostic |
| **REVE** | Weakly alpha | Strongly alpha | REVE has alpha affinity across tasks |

---

## Neuroscience Synthesis

The three analyses form a triangulated interpretability framework:

1. **Spatial** (channel ablation): FMs prioritize posterior/occipital channels — source region of alpha oscillations
2. **Correlational spectral** (band RSA): On Stress, FM RDMs correlate with all bands uniformly (subject fingerprint). On EEGMAT, alpha/theta selectivity emerges (task signal)
3. **Causal spectral** (band-stop): On Stress, LaBraM causally depends on beta (cortical arousal, high between-subject variability). On EEGMAT, alpha removal hurts most (alpha suppression during mental arithmetic is the known neural correlate)

**The core explanation**: EEGMAT has a frequency-specific neural correlate (alpha suppression) that FMs can capture and classify. Stress does not — FMs fall back to broadband subject-level spectral fingerprints (dominated by beta arousal patterns), which partially correlate with DASS labels but cannot generalize across subjects.

---

## Contents

| File | Description |
|---|---|
| `channel_importance.json` | Per-channel ablation importance for 3 FMs (Stress 30ch) |
| `channel_importance_topomap.{pdf,png}` | Scalp heatmaps |
| `band_rsa.json` | Band-specific RSA for Stress and EEGMAT |
| `band_rsa.{pdf,png}` | RSA bar chart |
| `band_stop_ablation.json` | Band-stop ablation for Stress and EEGMAT |
| `band_stop_ablation.{pdf,png}` | Band-stop bar chart |

## Scripts

```bash
# Channel ablation (requires GPU)
python scripts/channel_ablation_importance.py --device cuda:3

# Band RSA (CPU only)
python scripts/band_rsa_analysis.py

# Band-stop ablation (requires GPU)
python scripts/band_stop_ablation.py --device cuda:3
```

# HHT / Adaptive Signal Decomposition Integration Spec

## Motivation (from Prof. Norden Huang)

Current EEG FMs (LaBraM, REVE) treat EEG as 1D images — fixed-length patches + Fourier-based PE.
This assumes stationarity and linearity, which EEG violates fundamentally.

EMD/HHT decomposes adaptively into IMFs, giving instantaneous frequency, amplitude, and phase
without any basis function assumption. The Holo-spectrum H(w, t, phi) is a complete representation.

## Methods to Evaluate

| Method | Description | Status |
|---|---|---|
| EMD (Empirical Mode Decomposition) | Original Huang method, iterative sifting | Classic, well-established |
| EEMD / CEEMDAN | Noise-assisted EMD to fix mode mixing | Improved but slower |
| VMD (Variational Mode Decomposition) | Optimization-based, concurrent decomposition | Fixes mode mixing, faster, differentiable-friendly |
| VLMD (Variational Latent Mode Decomposition) | Latent-space VMD variant | Newer, needs investigation |
| HHT (Hilbert-Huang Transform) | Hilbert on IMFs → instantaneous freq/amp/phase | Core analysis tool |
| Holo-Spectrum | 3D: (time, freq, phase) → amplitude | Complete signal representation |
| eiPDF | Ensemble-averaged Instantaneous PDF | Statistical characterization of non-stationary signals |

## Key Research Questions

1. **VMD vs EMD**: VMD is variational (optimization-based), potentially differentiable. Better candidate for neural integration?
2. **Differentiability**: Can VMD/VLMD be made end-to-end differentiable as a tokenizer replacement?
3. **Fixed-K decomposition**: VMD requires specifying K modes upfront — solves the variable-dimension problem
4. **Phase-amplitude coupling**: Can Holo-spectrum features capture stress-related CFC (theta-gamma PAC)?

## Integration Paths

### Path A: Feature extraction preprocessing (immediate)
```
Raw EEG → VMD/EMD → IMFs → HHT → [inst. freq, amp, phase stats] → classifier
```
Physics-informed baseline. Compare against FM approach.

### Path B: Hybrid FM + HHT features (medium-term)
```
Raw EEG → LaBraM → FM embedding
Raw EEG → VMD+HHT → physics features
[concat] → classifier
```

### Path C: Learnable decomposition tokenizer (ambitious, future paper)
```
Raw EEG → Neural VMD layer (differentiable) → IMF tokens → Transformer → prediction
```
Replace dumb patch tokenizer with physics-informed decomposition.

## Python Libraries to Evaluate

- `PyEMD` — EMD/EEMD/CEEMDAN
- `vmdpy` — VMD
- `emd` (emd-signal) — EMD + Hilbert + holospectrum analysis
- Custom implementation may be needed for VLMD/eiPDF

## Notes

- Prof. Huang's critique: Fourier assumes signal = sum of sine waves. EEG is nonlinear, non-stationary, diffusive, dissipative.
- The mathematical foundation needs deeper review before implementation.
- VMD is likely the best starting point — fixed K, optimization-based, potentially differentiable.
- This is potentially a standalone paper: "Physics-informed tokenization for EEG Foundation Models"

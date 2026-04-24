# SDL Paper Draft — §3 Methods (v2, 2026-04-20)

**Revisions from v1**: adds (a) per-window matched-protocol LP description (§3.5); (b) canonical per-FM FT recipe description (§3.6); (c) FOOOF-based signal ablation pipeline (§3.7); (d) LOSO robustness check (§3.8); (e) permutation-null control procedure (§3.9). §3.1-§3.4 from v1 are lightly updated for consistency.

---

## §3 Methods

### 3.1 Datasets

We evaluate on four publicly available clinical rsEEG cohorts:

**UCSD Stress** [Komarov, Ko, Jung 2020, IEEE TNSRE 28(4):795]. 17 Taiwanese graduate students, 92 recordings with per-recording DASS-21 and Daily Stress Scale self-reports. We use the 70 recordings with both DASS-21 and DSS present (`data/comprehensive_labels.csv`). Per-recording binary label via DASS-21 threshold > 50 ("Group" column: 14 positive / 56 negative), matching the protocol in Wang et al. 2025. 5-second epochs at 200 Hz × 30 channels from the original Curry 8 cap.

**ADFTD** [Miltiadous et al. 2024, OpenNeuro ds004504, RS-only]. 65 subjects, 195 recordings. AD + FTD vs healthy controls (binary positive/negative: 108/87). 19 channels resampled to 200 Hz, 5-second epochs. Used here in the clinically deployable RS-only configuration (not the merged RS+PS corpus used by LEAD).

**TDBRAIN** [van Dijk et al. 2022]. 359 subjects, 734 recordings. MDD vs healthy controls (`target_dx="MDD"`, `condition="both"` combining EO and EC). Binary 640 positive / 94 negative. 26 channels downsampled to 19 for cross-FM compatibility (COMMON_19 subset), 200 Hz, 5-second epochs.

**EEGMAT** [Zyma et al. 2019, PhysioNet]. 36 healthy subjects × 2 recordings (resting baseline + serial-subtraction mental arithmetic; 72 recordings, 36 paired). 19 channels, 200 Hz, 5-second epochs.

All four datasets are accessed via `pipeline/*_dataset.py` adaptors that normalise to (M, C, T) epoch tensors with per-FM input norm (§3.4) applied.

### 3.2 Foundation models

Three pretrained EEG FMs:

- **LaBraM** [Jiang et al. 2024 ICLR; braindecode/labram-pretrained HuggingFace weights]: 5.8M-parameter (Base), 12-layer transformer with vector-quantised neural tokenizer and 128-slot channel position embedding. Output: (B, 200).
- **CBraMod** [Wang et al. 2024/2025 ICLR 2025; open weights `cbramod-base`]: criss-cross attention transformer. Output: (B, 200).
- **REVE** [Zhong et al. 2025; open weights `reve-base`]: ~1.4B-parameter transformer with linear patch embedding. Output: (B, 512).

All three are used in frozen mode and fine-tuning mode via a shared `baseline/abstract/factory.py` interface. Channel mapping for 19-channel datasets uses COMMON_19 subset (`pipeline/common_channels.py`).

### 3.3 Baseline architectures for cross-architecture robustness (§7)

For Stress cohort comparison: EEGNet [Lawhern 2018, ~3k params]; ShallowConvNet [Schirrmeister 2017, ~40k]; DeepConvNet [Schirrmeister 2017, ~300k]; EEGConformer [Song 2023, ~800k]. For classical band-power baselines (§5.2): Random Forest, LogReg-L2, SVM-RBF, XGBoost, all with class weight balancing (`class_weight='balanced'` for RF/LogReg/SVM, `sample_weight=compute_sample_weight('balanced', y_train)` for XGBoost).

### 3.4 Per-FM input normalisation

Input normalisation is **per-FM, not global**, following each FM's original pretraining convention (documented in `CLAUDE.md §2`):

- **LaBraM**: `norm='zscore'` per channel, per window. Matches the paper's fine-tuning recipe.
- **CBraMod**: `norm='none'`. The extractor internally applies `x / 100` in `cbramod_extractor.py:102`; pre-zscoring would double-scale input to ~0.
- **REVE**: `norm='none'`. The patch embedding is a `Linear(patch_size → embed_dim)` trained on µV-scale pretraining data.

A global `norm` flag across multiple FMs is **never** used; getting this wrong silently destroys the run. Cache directories encode norm: `data/cache` (zscore) and `data/cache_nnone` (no norm).

### 3.5 Matched per-window linear probing protocol

The frozen linear probe (LP) protocol used throughout §5–§8 is designed to be *apples-to-apples with fine-tuning*: both train on per-window samples, both pool predictions per recording for recording-level metrics. The alternative protocol — averaging window-level features per recording, then training a classifier on recording-level samples — artificially inflates LP accuracy relative to FT because it discards within-recording variance that FT uses during training. We avoid this protocol throughout.

Concretely, for each (FM, dataset) cell:

1. **Feature extraction (per window).** Load the dataset in its native configuration (channels, norm per §3.4, 5-s epochs). For each epoch, run the frozen FM to produce a per-window embedding of dimension 200 (LaBraM, CBraMod) or 512 (REVE). Save (features, window_rec_idx, window_labels, window_pids) to `results/features_cache/frozen_{model}_{dataset}_perwindow.npz`.

2. **Subject-level CV at recording level.** Use `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)` with `groups=patient_id` to partition **recordings** into 5 train-test folds stratified by recording label.

3. **Per-window training within each fold.** The training fold's recordings contribute all their windows as training samples; the test fold's recordings contribute all their windows as test samples. Feature preprocessing: per-dim 1-99 percentile clip (fit on train fold only), then `StandardScaler` (fit on train fold only). Classifier: `LogisticRegression(C=1.0, class_weight='balanced', solver='liblinear', max_iter=5000, tol=1e-3)`.

4. **Prediction pooling.** For each test-fold recording, average the per-window class-1 probabilities across all its windows → pooled probability. Threshold at 0.5 → recording-level prediction. This matches `train_ft.py`'s global prediction pooling.

5. **Recording-level balanced accuracy.** Compute `balanced_accuracy_score(rec_labels, rec_pred)` pooled across all 5 folds.

6. **Multi-seed aggregation.** Repeat (2)–(5) for 8 seeds: 42, 123, 2024, 7, 0, 1, 99, 31337. Report mean and sample std (ddof=1).

Scripts: `scripts/features/extract_frozen_all_perwindow.py`, `scripts/experiments/frozen_lp_perwindow_all.py`. Cache files in `results/features_cache/frozen_*_perwindow.npz`. Seed count (8) was chosen to tighten LP std below FT std (which uses 3 seeds due to compute); sensitivity analysis in §8.5 using the 3-seed subset (seeds 42, 123, 2024) confirms identical verdict on 11 of 12 cells.

### 3.6 Canonical per-FM fine-tuning recipe

All fine-tuning experiments in §5–§8 use the same per-FM recipe across all 4 datasets, following each FM's published default:

- **LaBraM**: `lr=1e-5, encoder_lr_scale=0.1, norm=zscore, loss=focal, epochs=50, patience=15, aug_overlap=0.75, warmup_freeze_epochs=1, weight_decay=0.05, batch_size=32`.
- **CBraMod**: `lr=1e-5, encoder_lr_scale=0.1, norm=none` (other settings identical).
- **REVE**: `lr=3e-5, encoder_lr_scale=0.1, norm=none` (other settings identical).

**No per-dataset HP tuning is performed.** This is a deliberate choice: using a per-dataset HP sweep on the same test cohort constitutes test-set contamination, and given our cohort sizes (17–359 subjects), HP selection noise is large enough that a "best HP" found on the test cohort is not a reliable estimate of deployment performance. On UCSD Stress we *do* have a 6-HP × 3-seed sweep; results in §8.5 robustness confirm that the best-HP BA on LaBraM (0.524) is still statistically indistinguishable from the frozen LP ceiling (0.525), so per-dataset HP tuning does not change our main conclusion on this cohort.

Fine-tuning sources: `exp03_stress_erosion/ft_real/` (Stress LaBraM canonical), `hp_sweep/20260410_dass/` Stress CBraMod/REVE canonical configs, `exp07_adftd_multiseed/`, `exp08_tdbrain_multiseed/`, `exp04_eegmat_feat_multiseed/` (EEGMAT LaBraM), `exp17_eegmat_cbramod_reve_ft/` (EEGMAT CBraMod/REVE).

Training determinism: `train_ft.py` sets `torch.backends.cudnn.deterministic=True` and `benchmark=False`. FT std is driven primarily by initialisation + data-ordering noise. We use 3 seeds for FT (42, 123, 2024) — compute-practical, matches LEAD 2025 and comparable FM benchmarks.

### 3.7 FOOOF-based aperiodic/periodic ablation (§6.5)

For the causal subject-identity test, we fit FOOOF [Donoghue et al. 2020] on recording-level averaged PSD per channel and reconstruct three ablated signal variants per channel × epoch.

**FOOOF fit**: For each recording and channel, compute Welch PSD (`nperseg=512`, `noverlap=256`, Hann window) across all epochs within that recording, average PSDs, then fit FOOOF on `fit_range=(1, 45) Hz`, `peak_width_limits=(1, 12)`, `max_n_peaks=6`, `min_peak_height=0.1`, `aperiodic_mode='fixed'`. Recording-level fitting (rather than per-epoch) yields higher R² (Stress 0.895 mean, EEGMAT 0.980 mean; 100 % fit success in both) and matches the trait-level interpretation of the 1/f aperiodic component as a stable within-recording property [Donoghue 2020]. Quality reported in `results/features_cache/fooof_ablation/{dataset}_norm_none.summary.json`.

**Signal reconstruction** (per channel × epoch, frequency domain):
- *Aperiodic-removed*: `X_flat(f) = X(f) / sqrt(10^(b - χ·log10(f)))`. Preserves phase, removes the FOOOF-modelled 1/f amplitude envelope. IFFT → `x_aperiodic_removed`.
- *Periodic-removed*: Subtract FOOOF Gaussian peak power from `|X(f)|^2`, clip to ≥ 10^-16, combine with preserved phase, IFFT → `x_periodic_removed`.
- *Both-removed*: Apply aperiodic division to the periodic-removed amplitude; acts as a joint ablation baseline.

**FM feature extraction on ablated signals**: Load ablated epochs, apply per-FM input norm (§3.4), pass through frozen FM, produce per-window embeddings matching the format of `frozen_{model}_{dataset}_perwindow.npz`. No re-extraction of original (un-ablated) features is needed — the §3.5 cache serves as the "Original" condition.

**Probe protocols**: Subject-ID probe uses multi-class `LogisticRegression` via `OneVsRestClassifier` (liblinear, class-balanced) with session-held-out splits (half of each subject's recordings for training, half for testing), 8 seeds, recording-level prediction pooling. State-label probe (EEGMAT only) uses the §3.5 matched LP protocol. Balanced accuracy chance level: 1/N_subjects for subject probe; 0.5 for binary state probe.

Scripts: `scripts/analysis/fooof_ablation.py`, `scripts/features/extract_fooof_ablated.py`, `scripts/experiments/fooof_ablation_probes.py`.

### 3.8 LOSO robustness check (§8.5)

As a robustness check against 5-fold partitioning on small cohorts, we additionally run leave-one-subject-out (LOSO, 17 folds for Stress) on the frozen LP arm. Each fold holds out all recordings of one subject. Otherwise the protocol matches §3.5 exactly. LOSO results are deterministic (no shuffle), so seed variance is essentially zero. Script: `scripts/experiments/stress_frozen_lp_loso.py`. Output: `results/studies/perwindow_lp_all/stress/{model}_loso.json`.

### 3.9 Permutation-null control (§5.3)

For the strongest cell in our FT evaluation (LaBraM × Stress), we run a permutation-null test: fine-tune under the same canonical recipe but with recording-level labels shuffled before CV. `train_ft.py --permute-labels <seed>` handles the shuffling internally. We use 10 permutation seeds (`scripts/experiments/run_perm_null.py`) and compare the resulting BA distribution to the 3-seed real-label FT distribution using a one-sided Mann-Whitney-like threshold: p-value estimated as `P(null ≥ real)`.

The p = 0.70 result for LaBraM × Stress in §5.3 means real-label FT is not distinguishable from shuffled-label FT at any conventional significance threshold, consistent with the representation containing no fine-tuning-exploitable state signal on this cohort × label combination. Extension of this control to CBraMod × Stress and REVE × Stress is pending compute (§10 open items).

---

*Draft status: §3 v2 ~1,500 words. All protocol descriptions now match scripts in the codebase.*

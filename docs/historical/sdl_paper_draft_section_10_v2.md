# SDL Paper Draft — §10 Limitations, Open Questions, and Future Work (v2, 2026-04-20)

**Role in SDL thesis**: *the honest accounting.* Every critique paper must itself be audit-able. §10 enumerates the claims we do not fully support, the experiments we deferred, and the dimensions along which the pre-flight checklist of §9 may need revision.

---

## §10 Limitations, Open Questions, and Future Work

### 10.1 Wang et al. 2025 reproduction is incomplete

Our §5.1 audit replicates the protocol-induced inflation on UCSD Stress (trial-level → subject-level CV accounts for approximately 20 pp of the 40 pp gap between Wang et al. 2025's reported 0.9047 BA and our honest-evaluation 0.443–0.577). The residual ~20 pp we attribute to unspecified design choices in the Wang pipeline (hyperparameter selection, label coding threshold, window policy, possibly features selection) that we cannot audit without full training logs. We do not claim full reproduction of the Wang pipeline. Our claim is narrower: under our controlled matched protocol, the within-pipeline trial-vs-subject inflation alone does not account for 0.9047-level accuracy, and no architecture we tested reaches 0.9047 under honest CV.

### 10.2 Stress CBraMod and REVE FT rescue — permutation-null control pending

Two of the three "FT > LP" cells in our §8 twelve-cell grid are on the small Stress cohort (CBraMod × Stress +11.8 pp, REVE × Stress +13.7 pp). Stress has 14 positive recordings distributed non-uniformly across 17 subjects, which creates a label-subject collinearity that could allow fine-tuning to exploit subject-identity shortcut rather than state-discriminative signal.

We have run permutation-null control for LaBraM × Stress (§5.3, p = 0.70 one-sided) — real-label FT is statistically indistinguishable from shuffled-label FT. We have *not* yet run the equivalent control for CBraMod × Stress and REVE × Stress. Our current reporting therefore treats these rescues as *mechanistically suspect* rather than confirmed shortcut. If a permutation-null test confirms that CBraMod/REVE FT on shuffled Stress labels still exceeds their own frozen LP, the "rescue" is mechanically identified as shortcut exploitation of label-subject collinearity; if not, the finding is genuine small-cohort FT-rescue and requires deeper investigation.

This test requires 10 permutation seeds × 2 FMs = 20 additional fine-tuning runs on Stress. Each run is compute-comparable to a single FT seed (~15 minutes on a single GPU). Total: ~5 GPU-hours. We defer this to the next revision.

### 10.3 Per-dataset HP sweep on ADFTD, TDBRAIN, EEGMAT not performed

We report FT under each FM's canonical per-paper recipe across all 4 datasets, *without* per-dataset HP tuning. This choice is deliberate (§3.6 rationale) but has a known cost: FM-×-dataset cells where FT degrades LP (6 of 12 cells) could conceivably show different results under a per-dataset HP sweep with nested CV.

On UCSD Stress, where we do have a 6-HP × 3-seed sweep, the best-HP LaBraM FT (0.524) is statistically indistinguishable from the frozen LP (0.525) — so at least on this cohort, per-dataset HP tuning does not overturn the main claim. Extending the HP sweep to ADFTD, TDBRAIN, and EEGMAT (54 runs per dataset × 3 datasets = 162 additional FT runs) would strengthen the claim, at ~1–2 GPU-days. We defer this.

Importantly, reviewers skeptical of the FT<LP claim on ADFTD/TDBRAIN/EEGMAT should note that the canonical per-FM recipe used here is the same recipe the original LaBraM, CBraMod, and REVE papers report for their own downstream evaluations. If our recipe is "not best", it is at least the recipe those benchmark tables were produced with.

### 10.4 Architecture panel is Stress-only and does not cover all 2017–2025 models

The §7 architecture-independence result is on UCSD Stress only. Extending it to ADFTD / TDBRAIN / EEGMAT would require multi-seed training of EEGNet, ShallowConvNet, and similar models on those cohorts, which is straightforward but was not part of the present scope. The currently-missing architectures from §7's panel — DeepConvNet and EEGConformer — were evaluated in an earlier archived run (results/archive/2026-04-09_subject_dass_cleanup/) under the deprecated `--label subject-dass` protocol and require rerunning under the current `--label dass` protocol (< 1 hour each).

### 10.5 FOOOF ablation is on Stress + EEGMAT only

§6.5 reports the FOOOF aperiodic/periodic ablation on Stress (17 subjects) and EEGMAT (36 subjects). Extending this to ADFTD (65 subjects) and TDBRAIN (359 subjects) would strengthen the causal claim and is methodologically identical (script `scripts/analysis/fooof_ablation.py` already handles generic datasets). The principal scientific question is whether LaBraM's reduced aperiodic-dependence (§6.5 Table 6.3: LaBraM EEGMAT aperiodic-removed Δ = −2.8 pp vs REVE's −26.0 pp) replicates on larger cohorts with different physiological signatures. Estimated compute: ~6 GPU-hours for the full ADFTD + TDBRAIN ablation + extraction + probe pipeline. Deferred to future work.

### 10.6 Small-cohort statistical reliability

UCSD Stress (70 recordings, 14 positive, 17 subjects) is the smallest cohort in our grid, and our §5 and §7 claims rest in part on its results. Multi-seed and multi-protocol (5-fold CV, LOSO) replication gives consistent BA estimates in the 0.43–0.58 band across all conditions we tested, and permutation-null control (on LaBraM) is not rejected at p = 0.05. However, the intrinsic variance of any DASS-related claim at N = 70 recordings is large, and no single study at this cohort size — including ours — produces confidence intervals tight enough to settle per-pp questions. The §5 claims should therefore be interpreted as "the reported 0.9047 is not reachable under honest evaluation", not as "the true ceiling is exactly 0.55".

Following the multisite reliability study of N = 2,874 rsEEG recordings [bioRxiv 2025.11.10.687610, 2025], results at N < 300 should be considered provisional.

### 10.7 Protocol assumptions

Our evaluations assume:
- **Subject-disjoint CV** as the honest evaluation protocol. This is standard in clinical EEG DL [Brookshire 2024] but is sometimes relaxed in emotion and cognitive-load benchmarks where within-subject performance is explicitly the target.
- **Recording-level labels** as the prediction target. Some clinical settings require window-level or sub-recording-level prediction (e.g., sleep staging). Our protocol does not directly address this regime.
- **A standard rsEEG montage** with ≥ 19 channels. Exotic montages or very low-density setups may behave differently.

### 10.8 LaBraM's reduced aperiodic-dependence — architectural explanation is speculative

§6.5 shows LaBraM's subject-ID probe drops only 2.8 pp under aperiodic removal, versus 14.2 pp (CBraMod) and 26.0 pp (REVE) on EEGMAT. We speculate in §6.5 that LaBraM's vector-quantised tokenizer combined with z-scored input normalisation may normalise aperiodic slope variation out of the tokens, leaving the discrete token distribution less aperiodic-dependent. This is a plausible architectural explanation but we have not performed the controlled ablation that would separate tokenizer vs preprocessing contributions. An architectural ablation study (VQ vs no-VQ, zscore vs raw input) is a future-work direction.

### 10.9 Clinical deployment: our checklist is necessary, not sufficient

§9's three-step pre-flight checklist is designed as a minimum screening procedure before investing fine-tuning compute. Passing all three checks does not guarantee clinical utility; it only rules out a set of known failure modes (trait-like labels without neural anchor, LP-unseparable representations, subject-shortcut-dominated rescues). Achieving clinical utility further requires:

- Generalisation to unseen cohorts (external validation set).
- Sensitivity to cases where the model's prediction is disease-relevant (calibration, not just accuracy).
- Robustness to device, montage, and preprocessing shifts not captured by a single-corpus evaluation.

Our checklist is necessary for a clinical researcher to not waste compute on a broken benchmark; it is not sufficient to produce a clinical product.

### 10.10 Future work — three concrete directions

1. **Causal FOOOF intervention during pretraining.** The §6.5 ablation shows aperiodic 1/f is the substrate of subject dominance in pretrained FM embeddings. A natural next experiment is to **pretrain an FM on FOOOF-detrended EEG** (aperiodic removed at preprocessing time) and measure whether the resulting model has reduced subject dominance without sacrificing task-discriminative capacity. This is an architectural change with clean expected comparison against an un-detrended baseline of identical size and corpus.

2. **Subject-shortcut regularisation.** If subject dominance is partially driven by label-subject collinearity in small cohorts (the Stress CBraMod/REVE rescue hypothesis in §10.2), explicit subject-adversarial regularisation during fine-tuning on small-N clinical cohorts may recover the apparent signal without the shortcut. Methods already exist [Özdenizci 2020, Zhang 2025]; they have not been integrated into EEG-FM fine-tuning pipelines.

3. **Within-subject longitudinal benchmarks.** Our pre-flight checklist targets cross-subject clinical prediction. A complementary research program would ask whether EEG FMs add value for within-subject longitudinal tracking (e.g., circadian drowsiness, pre/post-intervention state) where subject identity is fixed by design. The Stress cohort itself (longitudinal DSS per-recording trajectories) is a candidate; the 54-recording within-subject DSS-threshold analysis in our archive suggests within-subject FM performance is poor at N ≤ 10 recordings per subject, but this has not been systematically benchmarked.

---

*Draft status: §10 v2 ~1,700 words. Enumerates 10 specific limitations / open items with compute costs for deferrable extensions.*

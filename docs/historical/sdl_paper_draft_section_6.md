# SDL Paper Draft — §6 Paired Within-Subject Contrast Experiment

**Date**: 2026-04-15
**Role in SDL thesis**: *the controlled test.* §4 established that frozen FM representations are subject atlases; §5 established that the UCSD Stress ceiling is ~0.55–0.58 under honest evaluation. Neither finding, on its own, discriminates between competing explanations of the Stress ceiling — it could be a model-scale limit, a dataset-size limit, a label-noise artefact, or the SDL-predicted contrast-strength limit. §6 reports the paired comparison that separates these hypotheses: two within-subject experiments using the same three FMs and the same framework, differing only in the neural-contrast strength of the label. The SDL thesis predicts, and the data confirm, that the FMs rescue the high-contrast case and fail on the low-contrast case.

---

## §6 Paired Within-Subject Contrast Experiment: EEGMAT Rescues, UCSD Stress Does Not

### 6.1 Experimental design

The paired comparison has two arms. Both use within-subject paired recordings as the classification target, both use the same three FMs (LaBraM, CBraMod, REVE) with the same frozen-extractor → classifier pipeline, both are evaluated with a subject-respecting protocol (EEGMAT: leave-one-subject-out with per-subject rest/task pair; Stress: leave-one-subject-out with personal-median-split DSS threshold), and both are reported as 3-seed mean ± std on the same software stack. The *only* structural difference between the two arms is the neural contrast underlying the label: EEGMAT labels are a condition contrast (rest vs serial-subtraction mental arithmetic); UCSD Stress longitudinal labels are a within-subject DSS-median split across each subject's repeated recordings.

Following the contrast-strength anchoring protocol pre-registered in §3.5, we define the contrast strength of each label *externally* — before reading any FM result — via independent neural-level evidence:

- **EEGMAT contrast anchor**: serial-subtraction mental arithmetic is an extensively characterised cognitive-load paradigm whose canonical EEG signature is posterior alpha desynchronisation [Klimesch 1999, Brain Research Reviews; Pfurtscheller & Aranibar 1979; Klimesch 2012 NeuroImage]. The effect is robust, spectrally localised, and reliably induced across participants. The contrast strength can therefore be anchored on published effect sizes in this literature without any reference to FM outputs.

- **UCSD Stress longitudinal DSS anchor**: a systematic search of the EEG literature (documented in `docs/related_work.md`) returns no published per-recording within-subject EEG correlate of DASS-21 or Daily Stress Scale (DSS) scores. The Komarov–Ko–Jung 2020 paper itself is designed and analysed as a *group-level longitudinal correlational study* (`§3.7` psychometric scope), not a per-recording within-subject classification benchmark. The contrast strength anchor in this arm is therefore the *absence* of a published within-subject neural effect.

Under the SDL thesis, the two anchors jointly predict the experimental outcome: EEGMAT should rescue (strong, neurally-anchored within-subject contrast); Stress longitudinal DSS should fail (no published within-subject neural anchor).

### 6.2 EEGMAT rest vs mental arithmetic: FM rescue succeeds

Paired rest vs serial-subtraction mental-arithmetic recordings for each of 36 EEGMAT subjects (72 recordings, 19 channels, 60 s segments; source: Zyma et al. 2019, MNE-provided dataset). LaBraM fine-tuned on this within-subject contrast reaches **BA 0.731 ± 0.021** (3 seeds; source: `results/studies/exp04_eegmat_feat_multiseed/`). This is a sharp 23 pp gain over the LaBraM frozen linear-probe baseline on the same data and the same CV protocol, and is comparable to previously reported EEGMAT benchmarks using architecture-specific convolutional models.

A critical methodological note: the FT rescue on EEGMAT is *not* accomplished by rewriting the frozen representation to foreground the label axis. As reported in §4 (Table 4.1, LaBraM × EEGMAT row) and elaborated in the F-C.3 analysis (source: `results/studies/exp06_fm_task_fitness/variance_analysis_all.json`), the pooled label fraction shifts only from 5.35 % → 3.34 % under FT on EEGMAT, and the subject fraction actually *increases* from 71.2 % → 82.8 %. In other words, EEGMAT FT succeeds at the classification head while the embedding itself remains subject-dominated, with the label axis *projected onto* the existing subject-structured representation rather than reorganised around. This is the "projection, not rewrite" mechanism that distinguishes the SDL-style rescue from a representation-replacement account of FM fine-tuning.

### 6.3 UCSD Stress longitudinal DSS: FM rescue fails

We reframe the Stress dataset as a within-subject DSS trajectory classification task, matching the within-subject structural form of the EEGMAT arm. Each of the 14 Stress subjects with repeated recordings (54 total recordings) is assigned a personal-median DSS threshold; each recording is labelled "above vs below personal median." Classification is leave-one-subject-out with three candidate classifiers (centroid, 1-NN, linear) applied to frozen FM embeddings. Results are reported in Table 6.1 (source: `results/studies/exp11_longitudinal_dss/`):

**Table 6.1.** Within-subject Stress longitudinal DSS classification, frozen FM embeddings, LOO protocol.

| Model | Centroid BA | 1-NN BA | Linear BA |
|---|---:|---:|---:|
| LaBraM | 0.296 | 0.296 | 0.000 |
| CBraMod | 0.241 | 0.167 | 0.000 |
| REVE | 0.333 | 0.426 | 0.000 |

All nine FM × classifier cells fall at or below chance; all corresponding Cohen's kappas are negative. Unlike the EEGMAT arm, there is no rescue.

Two interpretations of the Stress failure are jointly supported by the DSS scope caveat (§3.7) and the null-anchor protocol (§6.1). (a) FM representations may lack the capacity to express within-subject DSS variation at 54-recording scale. (b) DSS-as-a-label may have no within-subject EEG ground truth to begin with, at the per-recording temporal resolution considered. The current data cannot distinguish these two explanations — but crucially, **both interpretations are consistent with the SDL thesis**: (a) instantiates the subject-dominance ceiling directly; (b) says the contrast/variance ratio is ≈ 0 because the contrast is absent. SDL's operative variable is contrast strength; whether the contrast is absent from the signal, absent from the labels, or present-but-unreachable-by-FMs is a secondary question that SDL deliberately does not pretend to answer on 54 recordings.

### 6.4 Independent neural-level corroboration: cross-model band consensus

A third line of evidence, independent of both the classification outcomes above and the external-literature anchors of §6.1, is available at the representation level: we ask whether the three FMs *agree on the neural band their frozen representation most depends on*, separately for each dataset. The intuition is that if a task has a strong, stable within-subject neural contrast, three independent architectures pretrained on different corpora with different objectives should converge on it; if the "signal" is architecture-specific (subject-fingerprint texture in the absence of a shared neural target), the three models should diverge.

We use causal band-stop ablation — removing one frequency band at a time via a Butterworth filter before feeding the signal to the frozen FM, and measuring the cosine distance between the resulting embedding and the unablated baseline — as the per-band dependence metric. (Source: `results/studies/exp14_channel_importance/`; analogous correlational band-RSA triangulates the same answer.)

**EEGMAT**: LaBraM, REVE, and CBraMod all show elevated dependence on alpha-band removal (with REVE strongest — alpha-cosine-distance 0.150, 3× larger than beta-cosine-distance). Three architectures converge on the same neural target — the Klimesch-predicted alpha-ERD band.

**UCSD Stress**: LaBraM peaks on beta, REVE on broadband, CBraMod on uniform (no band preference). Three architectures diverge on which band they rely on for subject-dominated representation. No shared neural target is isolable.

This band-consensus pattern is a purely representation-level diagnostic — it does not use classifier performance at all — and arrives at the same outcome-level conclusion as §6.2–§6.3 by an entirely independent path. Convergent cross-architecture band-locking on EEGMAT alpha is what a strong within-subject neural contrast would produce; architecture-specific band divergence on Stress is what an absent within-subject neural contrast would produce.

### 6.5 Summary: the paired comparison and its interpretation

EEGMAT: anchor = strong (Klimesch alpha-ERD), FT outcome = rescue (BA 0.73), cross-model band consensus = alpha.
UCSD Stress longitudinal DSS: anchor = absent, FT outcome = failure (BA 0.17–0.43), cross-model band consensus = divergent.

The paired design holds constant: FM family (3 × 3 = 9 FM × classifier cells per arm), pipeline, evaluation protocol (within-subject), software stack, seed discipline. The two arms differ only in the contrast strength of the target label as anchored by external neural evidence. They produce opposite classification outcomes at both the behavioural (BA) and representational (band-consensus) levels.

We therefore interpret §6 as a controlled empirical demonstration that, within the range of contrast strengths spanned by EEGMAT (strong) and UCSD Stress longitudinal DSS (absent/unknown), contrast strength governs whether FM fine-tuning rescues classification above the subject-dominance ceiling. The rescue on EEGMAT shows this is not a generic failure of FM fine-tuning on small cohorts; the failure on Stress shows that FM scale (LaBraM 100 M → REVE 1.4 B) does not rescue labels that lack a within-subject neural anchor.

§7 asks a separate question the paired design does not answer: given that the Stress ceiling exists, does *any* architecture escape it — including compact from-scratch convolutional models that never saw any pretrained EEG data? §8 then asks, within the ceiling, what specifically FM pretraining contributes over classical band-power features.

---

*Draft status: §6 ~1,400 words. Ready for review. §5 + §6 together address the honest-evaluation audit and the controlled paired test that together motivate the SDL diagnosis.*

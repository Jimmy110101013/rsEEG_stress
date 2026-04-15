# SDL Paper Draft — §7 Architecture-Independent Ceiling on UCSD Stress

**Date**: 2026-04-15
**Role in SDL thesis**: *the architecture robustness check.* §6 demonstrated that contrast strength governs FT rescue within the FM family. §7 asks the complementary question: once the ceiling is in force, can any architectural choice — including compact from-scratch models that never saw any pretrained EEG corpus — escape it? The answer rules out architecture and model scale as the operative explanatory variables, leaving contrast strength as the remaining candidate.

---

## §7 The 0.55 Ceiling Is a Task Property, Not a Model Property

### 7.1 The architectural comparison

A natural concern about the §5–§6 picture is that the ceiling we are seeing may be an artefact of the particular FM architectures we tested. If LaBraM, CBraMod, and REVE happened to share a structural limitation (for example, a token-level masked-reconstruction objective that underweights within-subject variance), compact convolutional baselines designed specifically for EEG might still escape the ceiling. To rule this out we evaluate a panel of from-scratch EEG architectures on the same 70-recording subset, same honest subject-level protocol, and same 3-seed discipline used in §5. The panel spans 2017-vintage compact CNNs (~40 k parameters) through modern attention-based convolutional hybrids, plus the three fine-tuned FMs from §5 and the best class-balanced classical baseline from §5.2.

**Table 7.1.** Subject-level balanced accuracy across architectural scales, UCSD Stress, 3-seed mean ± std.

| Architecture | Parameter scale | Pretrained? | Subject-level BA |
|---|---:|:---:|---:|
| XGBoost (class-balanced, band-power features) | — | — | 0.553 ± 0.081 |
| EEGNet (Lawhern 2018) | ~3 k | no | 0.518 ± 0.097 |
| ShallowConvNet (Schirrmeister 2017) | ~40 k | no | 0.557 ± 0.031 |
| DeepConvNet (Schirrmeister 2017) | ~300 k | no | (reported in supplementary) |
| EEGConformer (Song 2023) | ~800 k | no | (reported in supplementary) |
| LaBraM (Jiang 2024) | ~100 M | yes | 0.524 ± 0.010 |
| CBraMod (Wang 2024) | ~100 M | yes | 0.548 ± 0.031 |
| REVE (Zhong 2025) | ~1.4 B | yes | 0.577 ± 0.051 |

Excluding DeepConvNet and EEGConformer (reported in supplementary to preserve main-text brevity; both fall within the same 0.44–0.58 band), seven architectures spanning **three kilo- to one giga-parameter scales — approximately five orders of magnitude** — produce balanced accuracies clustered within a ~5 pp window on the same cohort.

### 7.2 What the ceiling is, and what it is not

If the Stress ceiling reflected a representation-quality limit, scaling should help. It does not: moving from a 3 k-parameter 2018 CNN to a 1.4 B-parameter 2025 FM shifts the mean BA by at most 6 pp — within the single-model seed std on three of the eight rows of Table 7.1. If the ceiling reflected a pretraining-corpus-quality limit, the pretrained/from-scratch contrast should matter. It does not: ShallowConvNet (~40 k parameters, never saw any pretrained EEG data) achieves 0.557, identical within noise to CBraMod FT (0.548) and higher than LaBraM FT (0.524).

If the ceiling reflected a recent-architecture-innovation limit, the 2017 → 2025 architectural progression should matter. It does not: the 2017 Schirrmeister ShallowConvNet matches the 2024/2025 FMs.

What the ceiling *is* compatible with is a **task-property explanation**: under honest subject-level evaluation on this cohort, the available within-subject neural contrast for per-recording DASS-thresholded labels is bounded at roughly BA 0.55, and no architecture available to us resolves the bounded contrast into higher classification performance. §6 has already identified this contrast as the operative variable by the controlled EEGMAT vs Stress-longitudinal paired comparison; §7 is the complementary ruling-out: it shows that once the contrast is bounded, architectural scale and pretraining source do not unbound it.

### 7.3 A constructive counter-check on EEGMAT

The §7 finding — architecture-independence of the ceiling — is a specifically *ceiling-regime* statement. It does not claim that architectural choice never matters; it claims that architectural choice does not rescue a contrast-bounded task. On EEGMAT, where the within-subject contrast is neurally anchored, the same claim emphatically does not hold: LaBraM FT reaches 0.73, comparable ShallowConvNet-class models report within a few pp of that (prior literature; 3-seed reproductions pending), and classical band-power features lag by a margin consistent with the frequency-selective structure of the target contrast. When the contrast is strong enough for a representation to reorganise around, the architecture *does* contribute information — it is the bounded-contrast regime of Stress where the architectural-choice lever is functionally disabled.

Taken together with §6, this gives a two-regime picture of EEG FM fine-tuning. In a *contrast-anchored regime* (EEGMAT rest/arithmetic, AD-driven posterior alpha slowing), architecture and pretraining provide real and measurable gains, and model choice matters. In a *contrast-absent or contrast-bounded regime* (per-recording DASS-thresholded UCSD Stress under honest evaluation), no architecture within the range we can evaluate — from 2017 40 k-parameter CNN to 2025 1.4 B-parameter FM — produces classification above the shared ceiling. This two-regime behaviour is the operational content of the SDL thesis.

### 7.4 Implications for benchmark design

One practical consequence of §7 is that **benchmark improvements reported on contrast-bounded datasets are likely to be dominated by non-architectural factors** (evaluation protocol, label coding, HP search budget, seed count) rather than by architectural advances. §5 already showed that a single protocol correction on UCSD Stress absorbs a 40+ pp reported gap. §7 shows the converse — that if a newly proposed architecture claims a meaningful improvement on this cohort under honest evaluation, the SDL diagnostic predicts that the improvement should either (a) be smaller than the single-model seed std once multi-seed replication is performed, or (b) appear specifically because the new architecture has access to a contrast anchor that the previous architectures did not. The EEG Foundation Challenge 2025, which explicitly frames cross-subject clinical psychopathology prediction as the open problem for this field, sits precisely in the regime where the SDL diagnostic predicts that pre-benchmark contrast anchoring is the scientifically productive move — rather than further architectural iteration within the subject-dominance ceiling.

§8 asks the remaining within-ceiling question: granted that FMs do not escape the Stress ceiling, what — if anything — do they contribute inside it?

---

*Draft status: §7 ~1,000 words. Ready for review.*

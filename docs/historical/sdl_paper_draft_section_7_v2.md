# SDL Paper Draft — §7 Architecture-Independent Ceiling on UCSD Stress (v2, 2026-04-20)

**Revisions from v1**: (a) removes DeepConvNet and EEGConformer rows from Table 7.1 because those results predate the 2026-04-09 `--label subject-dass` cleanup and are no longer valid under the current `--label dass` per-recording protocol (archived at `results/archive/2026-04-09_subject_dass_cleanup/`); (b) LaBraM FT updated to canonical recipe (0.443, exp03) matching §5 and §8; (c) LP numbers updated to per-window 8-seed.

---

## §7 The ~0.55 Ceiling on UCSD Stress is a Task Property, Not a Model Property

### 7.1 The architectural comparison

A natural concern about §5–§6's picture is that the ceiling may be an artefact of the specific FM architectures we tested. If LaBraM, CBraMod, and REVE happened to share a structural limitation — for example, a token-level masked-reconstruction objective that underweights within-subject variance — compact convolutional baselines designed specifically for EEG might still escape the ceiling. We rule this out by evaluating a panel of from-scratch EEG architectures on the same 70-recording subset, same honest subject-disjoint 5-fold CV protocol, and same multi-seed discipline. The panel spans 2017-vintage compact CNNs (~3k–40k parameters) through the three fine-tuned FMs plus the best class-balanced classical band-power baseline.

**Table 7.1.** Subject-level 5-fold CV balanced accuracy across architectural scales on UCSD Stress (70 recordings, DASS per-recording label, multi-seed).

| Architecture | Parameter scale | Pretrained? | Evaluation mode | Subject-level BA |
|---|---:|:---:|---|---:|
| XGBoost (class-balanced band-power) | — | — | classical, deterministic per-fold | 0.553 ± 0.081 |
| EEGNet (Lawhern 2018) | ~3k | no | trained from scratch, 3 seeds | 0.518 ± 0.097 |
| ShallowConvNet (Schirrmeister 2017) | ~40k | no | trained from scratch, 3 seeds | 0.557 ± 0.031 |
| LaBraM (Jiang 2024) | ~5.8M | yes | frozen LP, 8 seeds (per-window) | 0.525 ± 0.040 |
| LaBraM (Jiang 2024) | ~5.8M | yes | fine-tuned (canonical), 3 seeds | 0.443 ± 0.083 |
| CBraMod (Wang 2024) | ~100M | yes | frozen LP, 8 seeds (per-window) | 0.430 ± 0.033 |
| CBraMod (Wang 2024) | ~100M | yes | fine-tuned, 3 seeds | 0.548 ± 0.031 |
| REVE (Zhong 2025) | ~1.4B | yes | frozen LP, 8 seeds (per-window) | 0.441 ± 0.022 |
| REVE (Zhong 2025) | ~1.4B | yes | fine-tuned, 3 seeds | 0.577 ± 0.051 |

The panel spans approximately **six orders of magnitude in parameter count** (3 thousand → 1.4 billion). Under honest subject-disjoint 5-fold CV on the same 70-recording cohort, all nine entries produce balanced accuracies clustered in the **0.43–0.58 band**, a ~15 pp window. Within-band ranking is non-monotonic in parameter count (ShallowConvNet at 40 k matches the top of the band; REVE at 1.4 B is also at the top under FT; CBraMod under FT and ShallowConvNet are near-tied).

Compact archs (DeepConvNet, EEGConformer) were evaluated under our prior `--label subject-dass` OR-aggregation protocol before the 2026-04-09 cleanup reclassified that protocol as trait memorisation; those results are archived and omitted from Table 7.1. Rerunning them under the current `--label dass` per-recording protocol is straightforward (< 1 hour each) and is noted in §10 as an open item.

### 7.2 What the ceiling is, and what it is not

If the Stress ceiling reflected a representation-quality limit, architectural scale should help. It does not: moving from a 3 k-parameter 2018 CNN (0.518) to a 1.4 B-parameter 2025 FM (0.441–0.577) changes mean BA by at most 6 pp — within the seed std of single-model estimates.

If the ceiling reflected a pretraining-corpus-quality limit, the pretrained-vs-from-scratch contrast should matter. It does not: ShallowConvNet (~40 k parameters, no pretraining) achieves 0.557, identical within noise to CBraMod fine-tuned (0.548) and higher than LaBraM fine-tuned (0.443) or any of the three frozen FM LPs.

If the ceiling reflected a recent-architecture-innovation limit, the 2017-to-2025 progression should matter. It does not: the 2017 Schirrmeister ShallowConvNet matches the 2024/2025 FMs.

What the ceiling *is* consistent with is a **task-property explanation**: under honest subject-disjoint evaluation on this cohort, the available within-recording neural contrast for per-recording DASS-thresholded labels is bounded at roughly BA 0.55. No architecture available to us resolves the bounded contrast into higher classification performance, and no amount of fine-tuning on pretrained features either discovers additional signal or reliably harms it — different architectures sit at different points of the 0.43–0.58 band, and the within-band ordering is not driven by representation capacity.

§6 has already identified this contrast as the operative variable by the controlled EEGMAT vs Stress paired comparison; §7 is the complementary ruling-out: once the contrast is bounded, architectural scale and pretraining source do not unbound it.

### 7.3 A constructive counter-check on EEGMAT

The §7 finding — architecture-independence of the ceiling — is a specifically *ceiling-regime* statement. It does not claim that architectural choice never matters; it claims that architectural choice does not rescue a contrast-bounded task. On EEGMAT, where the within-subject contrast is neurally anchored, the same claim emphatically does not hold: the three frozen FM LPs reach 0.72–0.74 (§6.2), fine-tuning adds no more (Table 6.2), and the contrast-driven separability is at a qualitatively different operating point from the Stress ceiling.

Combining §6 (contrast-strength governs within-subject discrimination in frozen representations) with §7 (architecture does not bypass the ceiling once it is in force) gives a two-regime picture of EEG-FM behaviour on clinical rsEEG:

- **Contrast-anchored regime** (EEGMAT rest/arithmetic, ADFTD alpha-slowing): frozen representations already separate state; FT adds little; architecture and pretraining do contribute information but gains are modest on top of LP.
- **Contrast-absent or contrast-bounded regime** (per-recording DASS on UCSD Stress, MDD on TDBRAIN under CBraMod/REVE): no architecture within the range we can evaluate exceeds the cohort's intrinsic ceiling, and fine-tuning can degrade below frozen LP.

This two-regime behaviour is the operational content of the subject-dominance limit.

### 7.4 Implications for benchmark design

One practical consequence of §7 is that **benchmark improvements reported on contrast-bounded datasets are likely to be dominated by non-architectural factors** (evaluation protocol, label coding, HP search budget, seed count) rather than by architectural advances. §5 already showed that a single protocol correction on UCSD Stress absorbs a 20+ pp reported gap. §7 shows the converse: if a newly proposed architecture claims a meaningful improvement on a cohort that is structurally ceiling-bounded, the SDL diagnostic predicts that the improvement should either (a) be smaller than the single-model seed std once multi-seed replication is performed, or (b) appear specifically because the new architecture has access to a contrast anchor that the previous architectures did not.

The EEG Foundation Challenge 2025 [Truong et al., arXiv:2506.19141] frames cross-subject clinical psychopathology prediction as the open problem for this field. §7 suggests that pre-benchmark contrast anchoring (§9 Check 1) is a scientifically more productive move than further architectural iteration within the subject-dominance ceiling.

§8 asks the remaining within-ceiling question: granted that FMs do not escape the Stress ceiling on a single cohort, what does the picture look like across a broader 3-FM × 4-dataset grid under the same matched protocol?

---

*Draft status: §7 v2 ~1,200 words. Architecture-independence claim rests on 5 entries (XGBoost, EEGNet, ShallowConvNet + 3 FMs × {frozen LP, FT}) = 9 cells. DeepConvNet/EEGConformer rerun is a straightforward extension deferred to §10 open items.*

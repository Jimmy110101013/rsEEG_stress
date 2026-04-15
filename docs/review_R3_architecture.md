# Reviewer R3 — EEG Foundation Model / ML Architecture Review

**Reviewer scope:** ML architecture, FM pretraining methodology, fine-tuning dynamics.
**Target claim under attack:** **F-C** — "FT direction is a model × dataset interaction, not label biology."
**Venue calibration:** IEEE TNSRE / NeurIPS D&B. Bar is "mechanism defensible with controls," not "effect visible in one table."

**TL;DR:** F-C as currently written is **not defensible**. The three-FM "opposite direction" claim rides on per-model hand-tuned HP, a confounded 3-way architectural comparison (scale / corpus / tokenizer / patch / norm all covary), at least one acknowledged implementation drift (N-F21 LaBraM `self.norm`), and a variance structure on the flagship (CBraMod×ADFTD σ=3.35 > |μ|=0.83) that the authors themselves flag and then ignore. Pillar C reads as post-hoc pattern matching on a 3×2 table with 3 seeds per cell. At minimum, the authors must run a single-HP robustness check, a pretraining-corpus control, and re-establish LaBraM canonical numbers against the N-F21 fix before the "model × dataset interaction" language survives review.

---

## 1. Strongest architectural concern (flagship weakness)

**The F-C "direction flip" is confounded with per-model HP tuning, and no single-HP robustness check exists.**

From F-C.2 (`findings.md` L103–L108):

| Model    | Best FT HP                  | BA (3-seed)      |
|----------|-----------------------------|------------------|
| LaBraM   | lr=1e-4, **elrs=1.0**       | 0.524 ± 0.010    |
| CBraMod  | lr=1e-5, **elrs=0.1**       | 0.548 ± 0.031    |
| REVE     | lr=3e-5, **elrs=0.1**       | 0.577 ± 0.051    |

Each model's "winning" cell uses a **different** (lr, encoder-lr-scale) combination. The LaBraM canonical recipe (lr=1e-5, elrs=0.1 — the recipe CBraMod/REVE win under) gives LaBraM **0.443 ± 0.083** (F-C.2 footnote) — i.e. at shared HP, LaBraM's "erosion" is ~5pp deeper and variance explodes (σ=0.083 vs 0.010). The direction label ("erosion" vs "injection") is being read off the winner cell of a small grid.

This is the same failure mode as Dodge et al. 2019 "Show Your Work" and Musgrave et al. 2020 "Metric Learning Reality Check," both cited by the authors in `paper_strategy.md` §6 as *precedents they respect* — yet F-C.2 does exactly what those papers warn against: argmax over HP, then interpret the argmax cell as a mechanism. If the authors want "direction is architecture × dataset," they need to show the direction is **stable under HP**, not **selected by HP**.

The F-C.1 representation-level numbers (ADFTD +1.03 pp, TDBRAIN −1.56 pp for LaBraM; the opposite signs for REVE) are tighter but ride on the same FT HP — only the single best-HP training checkpoint enters the variance decomposition. There is no evidence in `findings.md` or `methodology_notes.md` that F-C.1 was repeated at a shared recipe across the 3 FMs.

**Required control (blocking):** Pick ONE HP config (e.g. lr=1e-5, elrs=0.1 — CBraMod/REVE's winner) and re-run all 3 FMs × 2 datasets × 3 seeds. If the LaBraM vs REVE direction flip survives, F-C.1 is real. If it collapses, the flagship claim is an HP-selection artifact and the paper must retreat to "LaBraM at its best HP injects; REVE at its best HP erodes" — a much weaker, less publishable claim.

---

## 2. Confounds not yet addressed

Ordered by severity. Each one alone would delay acceptance; the stack of them is fatal.

### C1. Architectural scale mismatch (14× params between REVE and the other two)

CBraMod ~100M, LaBraM ~100M, REVE ~1.4B (reviewer-supplied, consistent with public repos). A 14× scale gap means "architecture" is not the controlled variable. REVE having opposite FT direction from LaBraM could equally be (a) a scale effect — larger models have more redundant capacity for FT to redirect without degrading pretrained structure, (b) a different optimization basin under the same lr, (c) an optimizer-state interaction (AdamW β2 on 1.4B vs 100M models moves very differently at equal step count).

None of §6 or §9 acknowledges this. `paper_strategy.md` §9 speculates mechanistically ("CBraMod's criss-cross attention … REVE's linear patch embedding … LaBraM's neural tokenizer") but these are three correlated architectural axes, not a design.

### C2. Pretraining-corpus × downstream-dataset similarity is the un-measured factor

- LaBraM: TUH + ~16 other corpora, ~2500 hr
- CBraMod: TUEG, ~27000 hr (larger, narrower)
- REVE: BrainBridge mixture, ~2600 hr with substantially different channel/session distributions

ADFTD (dementia, clinical 19ch) and TDBRAIN (psychiatric, 26→19ch) have **very different similarity profiles** to TUH vs BrainBridge. The "model × dataset interaction" F-C names may simply be "pretraining-corpus × downstream-dataset similarity" — a much less architectural claim and one that has been shown by EEG-FM-Bench (Xiong 2025, `related_work.md` §2) to dominate transfer.

**Required control:** A frozen-LP pretrained-corpus proxy. E.g., measure representational similarity (RSA or CKA) between each FM's frozen embedding on a *held-out pretraining-corpus sample* and on each downstream dataset. If ADFTD similarity to LaBraM's pretraining > similarity to REVE's pretraining, the "LaBraM injects on ADFTD, REVE erodes" flip is explained without invoking architecture at all.

### C3. Tokenizer / patch-size / input-scale confounds are unmapped

From CLAUDE.md §2: LaBraM uses zscore input (neural tokenizer trained on zscored data), CBraMod uses none (extractor does `x/100` internally), REVE uses none (linear patch, µV-scale). Three different input representations reaching three different tokenizer designs (VQ discrete tokens, continuous patches via criss-cross, continuous patches via linear). The "direction flip" could be a tokenizer-level phenomenon (discrete VQ codebooks resist FT drift differently from continuous patches) rather than a scale-free "architecture" phenomenon.

None of this is ablated.

### C4. Window-size and SGD-steps confound (F-C.4 is worse than the authors admit)

REVE Δ on Stress is **+4.8 pp at 5s-matched** vs **+12.8 pp at 10s-matched** (F-C.4, methodology_notes N-F18). The authors frame this as "window choice inflates the gap" — but a 2.7× swing from a single preprocessing parameter means the "injection" magnitude is simply not robust. More damningly, 10s windows halve the number of training windows, which changes (a) SGD steps, (b) effective augmentation, (c) mini-batch variance. The direction-of-FT claim for REVE on Stress is therefore *jointly* determined by architecture, window length, and step count. F-C.4 quietly admits F-C.2 originally reported +8.3 pp from a **10s-FT + 5s-frozen mismatch** — that is a methodological error, not a caveat.

### C5. N-F21 LaBraM `self.norm` architecture drift

From `methodology_notes.md` N-F21: the authors found LaBraM applied both `self.norm` (loaded from pretrained CLS-head weights) and `self.fc_norm` in sequence, while official LaBraM under `use_mean_pooling=True` sets `self.norm = nn.Identity()`. Fix: exp24 vs exp23 shows **rec_BA mean moves −5.4 pp, with per-seed swings of ±25 pp that "cancel in mean."**

The authors' decision: keep the fix, do **not** re-run canonical experiments (exp03–08, exp11, exp12, exp18, frozen/ft_labram_* caches). The justification is "Wang 4-seed win_BA moves <1 pp." This is **not** adequate justification for keeping F-C canonical numbers:
- F-C.1 uses `ft_labram_adftd/` and `ft_labram_tdbrain/` caches — both pre-fix.
- F-C.2 LaBraM 0.524 is pre-fix.
- Per-seed swings of ±25 pp that cancel in a 4-seed mean are the **signature of an underdetermined model**, not of robustness. A 3-seed mean (which F-C.2 uses) is not protected by a 12-seed cancellation.

The ±25 pp per-seed swing is, by itself, evidence that the F-C.2 LaBraM 3-seed std of 0.010 is not the architecture's true std — it is 3 draws from a distribution whose per-seed variance is an order of magnitude larger. Paper claims built on a 0.010 std are overstated.

**Required control:** Re-run LaBraM F-C.1 and F-C.2 with the fixed model. 80 GPU-hr is not a good enough reason to skip this when it's the flagship claim. If the fixed LaBraM no longer injects on ADFTD, F-C.1 collapses.

### C6. CBraMod×ADFTD σ > |μ|

F-C.1 shows CBraMod ADFTD Δ = **+0.83 ± 3.35 pp** — i.e. the noise on the effect is **4× larger than the effect itself.** The authors flag this as "(unstable)" and move on. You cannot report σ > |μ| as an "injection" and then build a taxonomy that generalizes across FMs on that cell. CBraMod is either (a) not participating in F-C's mechanism, or (b) participating via a process whose sign is not determined at n=3. Either way, the cell is not evidence *for* the 3-FM generalization — at best it is silence, at worst it contradicts.

This isn't a minor caveat. It means the F-C.1 headline reduces to a 2-FM story (LaBraM vs REVE), and 2 models is not a taxonomy — it's an example and a counterexample.

### C7. Comparison to F-D.2 ShallowConvNet undermines the entire FM framing

From F-D.2 (`findings.md` L148–L161): **ShallowConvNet (~40k params, 2017 CNN, trained from scratch, no pretraining at all) reaches 0.557 ± 0.031 on Stress — inside the FM FT range (0.524–0.577).**

The authors' Pillar B sells "FM frozen LP beats classical features under honest labels" (LaBraM 0.605 vs classical RF 0.44). Fine. But Pillar C's injection/erosion framing implicitly assumes FMs encode something *worth injecting into* or *worth eroding*. If a 40k-param CNN from scratch, with zero pretraining, zero foundation-model machinery, reaches the same BA as a 1.4B-param REVE after FT — the entire "FT direction" framing has a trivial alternative explanation: **the three FMs are interchangeable task-agnostic feature extractors on Stress, and FT direction is reading off noise in the task head plus variance in optimizer trajectory, not a mechanism of the FM backbone.**

The authors partially acknowledge this in F-D §8 ("power-floor on Stress") but do not acknowledge the corrosive implication for F-C.2: if Stress is a power floor, then F-C.2 Stress row cannot be cited as independent evidence for F-C's mechanism. The paper's §2 Pillar C narrative treats C.1 (ADFTD/TDBRAIN) + C.2 (Stress) + C.3 (EEGMAT) as three independent triangulations. C.2 is not independent if §8 says Stress is noise; you cannot use a single dataset as both evidence *for* a taxonomy and evidence *that the taxonomy can't be measured on it.*

### C8. ADFTD window choice changes the effect magnitude 5×

A-F04 (archived, 10s ADFTD) had **+5 pp injection** for LaBraM. F-C.1 (current, 5s ADFTD) has **+1.03 pp**. That is a **5×** magnitude swing from preprocessing. The authors' rationale for preferring 5s is "cross-model matching" — but this means the "real" F-C.1 number depends on the matching convention, not the phenomenon. If ADFTD-LaBraM is +5pp at 10s and +1pp at 5s, a reviewer has no basis to believe the 5s number reflects anything architectural about LaBraM rather than anything about window-count-under-fixed-epochs.

---

## 3. Required ablations / controls (blocking)

In descending priority. The first three are, in my judgment, non-negotiable for a TNSRE or NeurIPS D&B acceptance.

1. **Single-HP robustness sweep (addresses §1, C4, C8).** Re-run F-C.1 (ADFTD, TDBRAIN, representation-level Δ) and F-C.2 (Stress, behavioral Δ) at (a) CBraMod/REVE's winning recipe (lr=1e-5, elrs=0.1) for all three FMs, (b) a mid-grid recipe (lr=3e-5, elrs=0.3), (c) LaBraM's winning recipe (lr=1e-4, elrs=1.0) for all three FMs. Report a 3×3 HP × direction matrix per dataset. If direction survives at ≥2/3 HP recipes for both LaBraM and REVE on both ADFTD and TDBRAIN, F-C.1 is real. Otherwise retreat.

2. **N-F21 LaBraM canonical re-run (addresses C5).** 80 GPU-hr on a 1.4B-scale cluster is a week. This is not a rounding error — it is the flagship model of the flagship claim under a fixed architecture. Re-run F-C.1 LaBraM and F-C.2 LaBraM at the fixed `self.norm = Identity`. Report exp24 numbers as canonical. If per-seed variance remains ±25 pp, the 3-seed regime is not sufficient; escalate to 8 seeds (matching Frozen LP protocol).

3. **Pretraining-corpus similarity control (addresses C2).** Compute CKA or RSA between each FM's frozen embedding of a held-out pretraining-corpus sample (use the original corpus if licenses permit, or a public proxy: TUEG for CBraMod, TUH-AB for LaBraM, open BrainBridge slice for REVE) and each downstream dataset. Report similarity correlation with FT-direction sign. If similarity predicts direction, rename F-C to "pretraining-domain × downstream-task interaction" and drop the architectural framing.

4. **Scale control (addresses C1).** Add a 4th FM at LaBraM/CBraMod scale OR a REVE-small variant OR a LaBraM-large variant. Without any scale-matched pair among {LaBraM, CBraMod, REVE}, the 3-way architectural comparison is unrunnable in principle.

5. **Input-scale ablation (addresses C3).** Run each FM at each input-norm convention (zscore, none, µV-clip) for one dataset. If input-norm changes direction sign, §2 CLAUDE.md's per-model norm policy is hiding a confound.

6. **Drop the CBraMod×ADFTD cell from F-C.1 headline (addresses C6).** σ > |μ| is not reportable as a direction. Either (a) extend to 8+ seeds until σ < |μ|/2, or (b) report it as "no detected direction," not "+0.83 pp injection."

7. **Window-matched REVE with matched training-step budget (addresses C4).** Run REVE FT at 10s with step budget truncated to match 5s-FT total steps. If the +12.8 pp vs +4.8 pp gap shrinks, F-C.4 is an SGD-step artifact. Report Δ at equal-step, not equal-epoch.

---

## 4. Verdict

**F-C as currently written is not defensible for TNSRE or NeurIPS D&B.**

The claim "FT direction is a model × dataset interaction, not label biology" requires a clean factorial design (model × dataset) where "model" is a controlled variable. The current design has every non-architectural factor covarying with architecture — scale (14×), pretraining corpus (TUH vs TUEG vs BrainBridge), tokenizer (VQ vs criss-cross vs linear), patch size, input norm convention, per-model HP — and then argmaxes over HP per cell, with one cell (CBraMod×ADFTD) showing σ > |μ|, another cell (LaBraM) built on an architectural variant the authors now know is buggy and chose not to re-run, and a third cell (Stress in F-C.2) that §8 of the same paper declares to be a statistical power floor.

What the evidence *can* support, after the controls above:
- "At its best HP, LaBraM FT injects on ADFTD and erodes on TDBRAIN" — a single-model cross-dataset observation. Defensible, matches A-F04 direction.
- "At its best HP, REVE FT shows the opposite pattern on both." — defensible *if* HP-invariance is shown and N-F21-analog confounds are ruled out (REVE audited clean per N-F21, so this is the more defensible half of C.1).
- "FT is not a universal mechanism — different FMs do different things to different datasets." — a qualitative existence claim, not a taxonomy.

What the evidence *cannot* support without the controls above:
- "Model × dataset interaction." (Needs HP robustness + scale control.)
- "Architecture matters." (Needs pretraining-corpus control.)
- "Three-FM triangulation." (Needs CBraMod cell with σ < |μ|.)
- Stress's F-C.2 row as evidence *for* C.1. (Power floor per §8; cannot play both sides.)

**Recommendation:** Major revision for TNSRE. Reject for NeurIPS D&B unless controls 1–3 are added in a revision within the cycle. The authors' own §8 power-floor honesty on Stress is admirable and should be extended to Pillar C: apply the same statistical rigor to the ADFTD/TDBRAIN cells that §8 applies to Stress. If that rigor survives, this is a strong TNSRE paper. If not, retreat from "taxonomy" to "case study" and publish accordingly.

---

*R3, 2026-04-15. Harsh reading deliberate. Pillars A, B, D (especially F-A subject dominance, F-B honest baselines, F-D ShallowConvNet ceiling) are strong and independent of these concerns; this review targets only F-C / Pillar C.*

# SDL Paper — Outline (IMRaD, 2×2 factorial, 4 datasets)

**Status (2026-04-23):** axis B renamed from "LP separable / null-indistinguishable" (observational only) to **task-substrate alignment (strong-aligned / weak-aligned)** (mechanistic), anchored on the completed 4-dataset permutation null (`paper/figures/main/fig3_honest_evaluation_4panel.pdf`). "Separable / null-indistinguishable" is retained as the operational diagnostic verdict within each cell. Framing pivoted to 2×2 factorial 2026-04-21; results chapter reorganised from *by-finding* to *by-diagnostic*. **Table placement deferred** — see TODO.
**Primary target:** Journal of Neural Engineering (JNE, IOP).
**Companion preprint:** TMLR.
**Fallback chain:** JBHI → J. Neuroscience Methods.
**Scope (4 datasets, one per cell):**

|  | **Strong-aligned task** (canonical neural signature; LP separable) | **Weak-aligned task** (behavioral / state summary; null-indistinguishable) |
|---|---|---|
| **Within-subject paired** | EEGMAT (rest vs arithmetic → theta/alpha) | SleepDep (NS vs SD state) |
| **Subject-label trait** | ADFTD (AD/FTD vs HC → 1/f aperiodic slope) | **Stress-DASS (representative failure case)** |

Regime labels follow `docs/master_results_table.md`. Axis B is **mechanistic** (does the label map to an EEG-identifiable neural pattern FM pretraining would encode?) and operationalised as a two-stage diagnostic: (i) frozen linear probe above chance under subject-disjoint CV across FMs (separable), (ii) real LaBraM FT BA clears a 30-seed permutation null (`results/studies/exp27_paired_null/*`). Strong-aligned = passes both; weak-aligned = fails both. The permutation null is the *primary* separator (Fig 3, `p ≤ 0.05` strong / `p > 0.1` weak).

TDBRAIN dropped from main text (duplicates ADFTD cell; retained as supplementary replication only).

**Framing constraint (locked 2026-04-21):** the **2×2 is descriptive** (n=1 per cell, no predictive claim for unseen datasets); the **diagnostic toolkit is reusable** (each tool has a defined scope and produces a cell-type verdict computable from EEG alone). What carries beyond this paper is the *toolkit and its scope conditions*, not the 2×2 itself. Previous framings (regime taxonomy / SDL ceiling / ceiling invariance) are superseded — see `docs/findings.md §Central thesis`.

**Stress's role:** representative case of the *subject-label × weak-aligned* cell, **not** the paper's protagonist. Wang et al. 2025's prior FM-evaluation on this dataset is what lets us populate this cell (no other small-N clinical rsEEG cohort has a comparable published FM baseline at the subject-label × weak-aligned corner). Our numbers complement Wang's under a different CV regime; this is a protocol distinction, not a refutation. If a reviewer says "Stress specifically is hard", the answer is "that's exactly what this cell exhibits under subject-disjoint evaluation — ADFTD in the adjacent strong-aligned cell behaves differently."

**One-sentence thesis:**
> *On small-N clinical rsEEG under subject-level CV, pretrained EEG foundation model behaviour is carved by **task-substrate alignment strength** — the degree to which a task's label maps to an EEG-identifiable neural pattern the FM learned during pretraining. Across a 2×2 factorial of (within-subject paired vs subject-label trait) × (strong-aligned vs weak-aligned task), the alignment column determines FM downstream success regardless of the CV regime row. We document per-cell outcomes across four datasets and provide a diagnostic toolkit (variance decomposition, permutation null, within-subject direction consistency, causal anchor ablation) that characterises each cell's dominant mechanism.*

---

## §1 Introduction
> 中文：為何單一資料集無法捕捉 FM 在臨床 rsEEG 上的行為 — 提出 2×2 factorial 與 diagnostic toolkit 的總論。

- 1.1 Clinical rsEEG + FM benchmark gap
  > 中文：現有 FM 在臨床 rsEEG 上的報告數字與 honest subject-CV 之間存在系統性缺口。
- 1.2 Why one dataset cannot settle the question
  > 中文：單一資料集反例易被 "one bad dataset" 論點打掉；需要跨 regime 的結構性證據 — motivates 一個 factorial framework。
- 1.3 The 2×2 factorial: (within-subject paired vs subject-label trait) × (strong-aligned vs weak-aligned task)
  > 中文：定義兩個軸與四個 cell — row 是 CV regime (是否 label 在同 subject 內對比)，column 是 task-substrate alignment (label 是否對應 FM pretrain 學到的 neural pattern)。column 是主軸（決定 FM 成敗），row 是結構變項（決定哪些 diagnostic 可跑）。
- 1.4 Dataset selection within the 2×2 framework
  > 中文：四個資料集 (EEGMAT / SleepDep / ADFTD / Stress) 各自填一格；特別說明 Stress 之所以是 between × null-indistinguishable cell 的代表，是因為 Wang et al. 2025 (trial-level CV, 0.9047 BA) 提供了該格唯一已發表的 EEG-FM reference point — Wang 的工作是讓我們能選這格的前提，非攻擊對象。
- 1.5 Paper contributions
  > 中文：三項貢獻 — (i) 2×2 populated with four datasets, (ii) diagnostic toolkit (4 工具配對 regime 的結構性組合), (iii) per-cell mechanistic characterisation + clinical pre-flight checklist。
  - 1.5.1 A 2×2 factorial populated with four small-N rsEEG datasets — to our knowledge, the first structured factorial comparison of EEG-FM behaviour across (CV regime × task-substrate alignment) axes
  - 1.5.2 A diagnostic toolkit organised against the four cells: each of the four individual tools (variance decomposition / permutation null / within-subject direction consistency / causal anchor ablation) is established in prior work, but each is also paired with a scope condition that states the cell types in which it runs. Within-subject direction consistency runs only in within-subject cells; causal anchor ablation returns a different anchor type per cell; variance decomposition returns a label-dominated vs subject-dominated reading anywhere. The contribution is this pairing — tool plus scope condition — rather than any individual tool in isolation.
  - 1.5.3 Descriptive characterisation of each cell's dominant mechanism + actionable pre-flight checklist
- 1.6 Paper roadmap
  > 中文：章節導覽。

## §2 Related Work
> 中文：定位本論文相對於 EEG-FM 文獻、subject leakage 討論、contrast anchoring、1/f 生理研究。

- 2.1 EEG foundation models (LaBraM / CBraMod / REVE)
  > 中文：三個 FM 的背景與 published benchmarks。
- 2.2 Subject leakage and evaluation protocols
  > 中文：trial-level vs subject-level CV 既有討論與未被解決的盲點。
- 2.3 Neural anchors for clinical rsEEG labels
  > 中文：臨床 rsEEG state 與 trait 標籤的 neural anchor 先前證據（以往文獻內部有時稱為 "contrast anchoring"，此處改用通用語避免 jargon）。
- 2.4 Aperiodic 1/f as subject identity signal
  > 中文：文獻支持 1/f 是 subject 指紋（Demuru 2020, Lanzone 2023 等）。
- 2.5 Factorial / taxonomy-style analyses in EEG benchmarking — and what is missing
  > 中文：既有 EEG-FM 評估多為單資料集 case study 或同 paradigm 多資料集 leaderboard；無 (subject-labelling × signal-coherence) 這類跨 paradigm 的 factorial 設計。本論文的貢獻正是填補這個空白。
  - Prior taxonomies focus on task category (motor imagery / SSVEP / affect / disease) or pretraining objective (masked / contrastive) — both orthogonal to our regime axes
  - BrainBench-style leaderboards aggregate across tasks without separating subject vs state labels or separable vs null-indistinguishable outcomes
  - Our 2×2 is explicitly designed to expose *why* FM behaviour fragments across tasks, not just *that* it does

## §3 Methods - Datasets and Protocol
> 中文：四個資料集各自的規格、2×2 落點理由，以及統一的 evaluation protocol。

- 3.1 2×2 cell assignment rationale
  > 中文：兩個軸的操作化定義，以及為何四資料集各自落在所屬 cell。
  - 3.1.1 Axis A — subject relation (within-subject paired vs between-subject label)
    > 中文：label 是否在同一 subject 內有對比 (rest vs task / NS vs SD)；between-subject 指 label 為 per-subject scalar。
  - 3.1.2 Axis B — task-substrate alignment (strong-aligned vs weak-aligned)
    > 中文：operational definition: label 是否對應 EEG-identifiable neural pattern 且 FM pretrain 已學到。Two-stage diagnostic — (i) frozen LP 在 subject-disjoint CV 下跨 ≥2/3 FM 顯著高於 chance (separable)，(ii) real LaBraM FT BA 清過 30-seed permutation null (`p ≤ 0.05`; ADFTD 用 subject-level perm)。strong-aligned = 兩者皆 pass，weak-aligned = 皆 fail。主要 separator 是 perm null（Fig 3 四資料集完整證據）。LP 程序 §3.4，perm-null 程序 §3.6.2，各 cell null 分布 §4.3。
  - 3.1.3 Cell assignments with one-line justification per dataset
    > 中文：每格為何選這個資料集（paradigm、cohort、label 類型）。
- 3.2 Per-dataset specs (source: `docs/master_results_table.md`)
  > 中文：四個資料集各自的 N / channels / epoch length / label rule，與對應的 LP/FT benchmark 數字。
  - 3.2.1 Stress-DASS (70 rec, 17 subj, 30ch, DASS per-recording) — subject-label × weak-aligned (LP 0.44–0.51)
  - 3.2.2 ADFTD (65 rec / 65 subj, 19ch, AD vs HC binary, split1) — subject-label × strong-aligned (LP 0.58–0.67)
  - 3.2.3 EEGMAT (72 rec, 36 subj, 19ch, rest vs arithmetic) — within-subject × strong-aligned (LP 0.72–0.74)
  - 3.2.4 SleepDep (72 rec, 36 subj, 19ch, NS vs SD) — within-subject × weak-aligned (LP 0.54–0.61, multi-FM inconsistent)
- 3.3 Foundation model feature extraction + per-FM input normalisation
  > 中文：FM 抽取流程與各模型必須使用的 input norm（LaBraM zscore / CBraMod/REVE none）。
- 3.4 Evaluation protocol
  > 中文：主 protocol 為 subject-level CV；within-subject cells 額外用 LOO 做 state tracking。
  - 3.4.1 Primary: subject-disjoint CV (per-window LP / FT-matched)
    > 中文：主評估 — subject-level 5-fold CV，避免 subject leakage。
  - 3.4.2 Secondary (within-subject cells only): LOO within-subject contrast
    > 中文：固定 subject 後測純 state signal。
  - 3.4.3 Multi-seed requirement (≥ 3 seeds) on small-N cells
    > 中文：small-N 下 seed variance ±5–10pp，所有 BA 聲明需 ≥3 seeds。
- 3.5 Fine-tuning canonical per-FM recipe
  > 中文：使用各 FM 原論文 canonical recipe，不做 per-dataset HP 調參以避免 test contamination。
- 3.6 Diagnostic toolkit — method specifications
  > 中文：§4 所有診斷工具的 method 層細節，結果層放 §4。
  - 3.6.1 Variance decomposition (recording-level)
    > 中文：Label / Subject / Residual 三項 variance 分解程序。
  - 3.6.2 Permutation-null test
    > 中文：label shuffle 建立 null 分布，作為 honest-eval 的統計錨。
  - 3.6.3 Within-subject direction consistency (scope: within-subject cells only by construction)
    > 中文：固定 subject 下 FM 對 label contrast 的方向一致性 metric。**在 between-subject cells 沒有 within-subject contrast 可測，因此此 diagnostic 不適用 — 這是 toolkit 的 *feature* 而非 bug**：每個 diagnostic 被設計為探測特定 regime 性質，哪些 diagnostic 能跑 本身就是 regime 資訊（"which diagnostics run" 告訴讀者 cell 落在 2×2 哪一側）。
  - 3.6.4 Causal anchor ablation (FOOOF aperiodic/periodic + band-stop)
    > 中文：以 FOOOF 拆解 1/f vs peaks + Butterworth band-stop，雙擾動攻擊 anchor 來源。
*(§3.7 benchmark landscape 已移至 §4.1 Results — Methods 只含程序定義。)*

## §4 Results
> 中文：Results 共 5 節 — §4.1 為 benchmark landscape (setup/entry point)，§4.2–§4.5 每節一個 diagnostic 掃過四 cell。

### 4.1 Benchmark landscape across the four cells (setup, not a finding)
> 中文：Results 入口 — 先給 4 cell × 3 FM 的 BA summary 定位每格 "起點"；BA 本身不是論文發現，而是後續 diagnostic 解讀的 context。數字來自 `docs/master_results_table.md`，由 **Tab 1** (`table1_master_performance.tex`) 承載，**不另做 figure**。

- Entry statement for each cell (LaBraM / CBraMod / REVE, LP→FT, all 3-seed; FT under per-FM canonical HP G-F09, source `results/final/{cell}/{model}/ft/`):
  - *Subject-label × weak-aligned (Stress-DASS)*: LP 0.51 / 0.44 / 0.46, FT 0.46 / 0.41 / 0.48 — FM stays in 0.41–0.48 band, **all three FMs ≤ chance**, classical LogReg 0.506 matches
  - *Subject-label × strong-aligned (ADFTD, split1 65/65)*: LP 0.64 / 0.58 / 0.67, FT 0.74 / 0.70 / 0.68 — LaBraM leads, all three FMs 0.68–0.74 tight band; classical SVM 0.647 and EEGNet 0.773 give a sizeable non-FM baseline for this cell (EEGNet matches the top FM)
  - *Within-subject × strong-aligned (EEGMAT)*: LP 0.74 / 0.72 / 0.74, FT 0.69 / 0.73 / 0.73 — strong LP signal, FT ≈ LP (saturated); all three FMs tight at ≈0.70; classical RF 0.889 beats all FMs
  - *Within-subject × weak-aligned (SleepDep)*: LP 0.61 / 0.55 / 0.54, FT 0.58 / 0.49 / 0.51 — FMs FT ≈ chance across all three; classical SVM 0.574 ≈ FM
- **Bridging statement**: BA alone does not explain why each cell arrives where it does — §4.2–§4.5 diagnostics characterise the mechanism behind each cell's benchmark landscape

### 4.2 Representation geometry across the 2×2 (Diagnostic 1: variance decomposition)
> 中文：每格 frozen FM representation 的 Label / Subject / Residual 三項 variance 分解，揭示 subject structure 主導的程度；FT 狀態先不畫 trajectory（LP→FT drift analysis 延後）。

- **Fig 4.2**  4-cell variance decomposition — stacked bars per cell (Label / Subject / Residual), frozen FM × 3 FM
- Per-cell reading: subject_frac 與 label_frac 的相對大小刻畫每格 frozen representation 的 "geometry budget"
  - *Between × null-indistinguishable (Stress)*: subject-dominant (label_frac ≈ 0)
  - *Between × separable (ADFTD)*: label_frac > 0 but still subject-dominated
  - *Within × separable (EEGMAT)*: label_frac 最高
  - *Within × null-indistinguishable (SleepDep)*: label_frac low, subject-dominated

### 4.3 Honest-evaluation calibration (Diagnostic 2: permutation null)
> 中文：用 label-shuffle null 逐 cell 確認觀察到的 BA 是真 signal；此節同時給出 Stress 在 subject-disjoint protocol 下的 reference numbers，與 Wang 2025 trial-level CV 報告做 protocol-consistent 對齊（非 gap-closing 論述）。
>
> **角色雙重性**：此 diagnostic 同時 (a) 支撐 §3.1.2 Axis B 的 separable/null-indistinguishable 分類（"≥ 2/3 FM p<0.05" 的證據來自 §4.3.1 的 null 密度），以及 (b) 作為 Results 的 diagnostic 2，報告每 cell 的 null 分布形狀與跨 FM 一致性。§4.3.2 用同一個 diagnostic 處理 Stress cell 的 Wang 2025 protocol 對比 — 這是此 diagnostic 的 natural use case，而非 §4 其他 diagnostic 的非對稱 carve-out。

- **§4.3.1 Null calibration across the 2×2**
  - **Fig 4.3**  Permutation-null density per cell (4 panels × 3 FM), observed BA marked
  - Stress (between × null-indistinguishable): null-indistinguishable across FMs, p ≈ 0.70
  - ADFTD (between × separable): observed BA p ≪ 0.05 for LaBraM, marginal for CBraMod/REVE
  - EEGMAT (within × separable): observed BA p ≪ 0.01 across FMs
  - SleepDep (within × null-indistinguishable): multi-FM inconsistency — LaBraM significant, CBraMod/REVE marginal
  - Take-home: the null test operationalises Axis B — separable cells show concentrated BA above the null; null-indistinguishable cells overlap with the null mass
- **§4.3.2 Stress cell: subject-disjoint reference numbers vs Wang 2025 trial-level CV**
  - Wang 2025 reports **0.9047 BA** on Stress (trial-level CV, LaBraM)
  - Under subject-disjoint 5-fold CV, 3 seeds (our protocol, matching §3.4.1): **0.44–0.58 BA** across 3 FMs
  - Classical LogReg baseline on the same cell (subject-disjoint): **0.506 ± 0.019 BA** — matches FM LP range
  - Interpretation: the two protocols measure different things — trial-level CV under within-subject labelling captures subject-identity structure in the test fold; subject-disjoint CV does not. The Stress cell's *subject-disjoint* BA (the regime this paper studies) is null-indistinguishable; our numbers and Wang's are consistent with each cell's expected behaviour under its respective CV rule
  - No claim is made against Wang's protocol — we simply operate in a different evaluation regime that the 2×2 framework foregrounds

### 4.4 Within-subject direction consistency (Diagnostic 3: longitudinal tracking)
> 中文：僅對 within-subject cells (EEGMAT, SleepDep) 適用 — 固定 subject 後測純 state signal 的方向一致性；為 SDL 最乾淨的 falsification 測試。

- **Fig 4.4**  Within-subject LOO + directional consistency (EEGMAT + SleepDep; Stress/ADFTD excluded by design)
- EEGMAT: LOO 0.64–0.68, directional consistency 0.07–0.15 across 3 FM
- SleepDep: LOO ≈ chance; directional consistency ≈ 0 — negative result motivates §4.5 FOOOF dissection
- Stress/ADFTD note: these two cells have no within-subject contrast by label design, so the diagnostic does not run here

### 4.5 Causal anchor ablation (Diagnostic 4: FOOOF + band-stop)
> 中文：以 FOOOF aperiodic/periodic 拆解 + Butterworth band-stop 兩種互補因果擾動，給出每格的 anchor 類型 — `absent` / `α-broadband` / `1/f-aperiodic`（不用 Type I/II/III 編號以免與統計學 Type I/II error 混淆）。
>
> **Band-stop metric note**：§4.5 量 **probe BA after band removal**（下游 decoding 的 causal effect）；Appendix B.2 量 **cosine distance between original and band-stopped representations**（representation geometry sensitivity）。同一擾動、互補 metric — §4.5 回答「model 有沒有在用這個頻段 decode」，B.2 回答「representation 對該頻段的 geometry 有多敏感」。

- **Fig 4.5**  Three-panel anchor dissection (PSD + FOOOF fit representative, FOOOF ablation scatter, band-stop line), 4 cells
- Per-cell anchor attribution (source: `results/studies/fooof_ablation/{stress,adftd,eegmat,sleepdep}_probes.json`):
  - *Stress (between × null-indistinguishable)*: **absent anchor** — no recoverable anchor in either intervention
  - *ADFTD (between × separable)*: LaBraM state probe drops 0.654→0.542 (−11.2 pp) under aperiodic removal, periodic removal barely moves it (0.654→0.655), REVE drops 0.652→0.614 (−3.8 pp) → **aperiodic-anchored trait signal**. CBraMod shows an **opposite-sign signature**: aperiodic removal *raises* state probe 0.571→0.692 (+12 pp) and subject probe 0.748→0.940 (+19 pp). Candidate explanation (anchor + model interaction): CBraMod's internal `x/100` input scaling is µV-calibrated and particularly sensitive to broadband amplitude; FOOOF-reconstruction on ADFTD (88 trait-heterogeneous subjects) acts as amplitude normalisation, *improving* subject and trait discriminability rather than removing a learned aperiodic anchor. This reinforces rather than overturns the taxonomy: ADFTD's trait signal co-varies with aperiodic 1/f for the two FMs that treat input as raw µV (LaBraM zscore, REVE raw µV), while CBraMod sees a different signal under its input pipeline. The opposite sign is a *model × preprocessing* artefact, not a violation of the anchor-type framework.
  - *EEGMAT (within × separable)*: **α-broadband anchor** (peak + in-band 1/f tail together)
  - *SleepDep (within × null-indistinguishable-but-causal)*: **1/f-aperiodic anchor** (FOOOF aperiodic removal collapses state probe) despite LOO ≈ chance
- Model-independent claim qualified: anchor taxonomy is computable from EEG alone, but *per-FM signature* can reverse when the FM's input normalisation interacts with the reconstruction. §3.2 per-FM norm table documents which FMs treat input as raw-µV vs z-scored.

## §5 Discussion
> 中文：整合四格觀察與 diagnostic toolkit 的輸出，給出 descriptive 敘事與 clinical checklist；明確聲明不做 predictive claim。

### 5.1 What the toolkit carries beyond this paper, and what it does not
> 中文：分清楚哪些東西可以帶出本論文 — toolkit 與其 scope conditions 可以 reuse，2×2 本身 (n=1 per cell) 不可以外推。

- **Reusable beyond this paper (the toolkit and its scope conditions)**:
  - Each of the four diagnostics has a defined input (raw EEG + label metadata), a defined output (a per-cell verdict), and a defined scope condition stating when the diagnostic runs
  - Within-subject direction consistency runs only when the label design supports a within-subject contrast; this scope condition is stated up-front, so a future user can tell without running anything whether their dataset qualifies
  - Causal anchor ablation produces an `absent` / `α-broadband` / `1/f-aperiodic` verdict computable from EEG alone before any FM is trained — the three anchor types (not the specific dataset assignments in this paper) are what carries over
  - Frozen variance decomposition produces label_frac vs subject_frac from frozen FM features — the label-dominated vs subject-dominated reading is what carries over
- **Not extrapolated beyond this paper (the 2×2 as a predictive map)**:
  - We do not claim that every future (between × null-indistinguishable) dataset will fail the way Stress does
  - We do not claim that every future (within × separable) dataset will decode the way EEGMAT does
  - With n=1 per cell, the 2×2 organises our own observations; it is not a map we ask readers to consult for unseen datasets
- The scientific contribution we ask readers to take away is the toolkit; the 2×2 is how we demonstrate the toolkit's coverage across complementary regimes

### 5.2 Per-cell mechanism synthesis
> 中文：整合四種 diagnostic 對每格的讀數，得出該 cell 的 dominant mechanism 描述。

- Stress (between × null-indistinguishable): subject-shortcut ceiling + null-indistinguishable benchmark BA
- ADFTD (between × separable): LaBraM FT 0.71 (best of 3 FM); aperiodic removal drops state probe −11 pp → 1/f-anchored trait signal with model-dependent aperiodic reliance
- EEGMAT (within × separable): α-broadband anchor + real state separability, FT LP-saturated
- SleepDep (within × null-indistinguishable): 1/f anchor detectable causally but not decodable within-subject (LOO ≈ chance)

### 5.3 Implications for benchmark design and FM deployment
> 中文：benchmark 設計應優先投資於 contrast anchoring 與 subject-disjoint protocol，而非架構疊代。

- Trial-level CV numbers on absent-signal cells are protocol artefacts
- Architectural iteration within an absent-signal cell is not the productive move
- Within-subject paired designs unlock contrast signal that between-subject designs cannot recover

### 5.4 Clinical pre-flight checklist (actionable output)
> 中文：提供 decision framework，讓臨床研究者在投入 FT compute 前用 diagnostic toolkit 做 cell 定位。

- Gate 1: Frozen LP under subject-disjoint CV — does FM separate label at all above chance?
- Gate 2: Permutation null — is the observed BA distinguishable from a label-shuffled distribution?
- Gate 3: Frozen variance decomposition — is representation label-dominated or subject-dominated at baseline? (label_frac ≪ subject_frac → cell is subject-geometry-bound; FT unlikely to rescue without a structural anchor)
- Gate 4: Causal anchor probe (FOOOF aperiodic/periodic ablation) — does the signal collapse under aperiodic removal (1/f-anchored), periodic removal (peak-anchored), both (broadband), or neither (absent anchor)?

## §6 Limitations and Future Work
> 中文：誠實列出未解決的 scope 限制與可擴展的 future work。

- 6.1 n=1 dataset per cell — the 2×2 is descriptive, not predictive
  > 中文：每格只有一個資料集，無法主張 generalisation；2×2 是 descriptive framework，不是 predictive rule。
- 6.2 Wang 2025 reproduction is partial (protocol-side closed; HP-side not)
  > 中文：Wang 的 HP-side reproduction 未完成；protocol-side 已關閉。
- 6.3 Per-dataset HP sweep not performed (canonical recipe rationale)
  > 中文：per-dataset HP sweep 只在 Stress 做過；其他 dataset 沿用 canonical recipe。
- 6.4 LP→FT drift analysis intentionally omitted from main text
  > 中文：frozen variance + perm-null + FOOOF 已建立每 cell 機制；drift 為低邊際資訊，主文不放。資料（Stress + EEGMAT）可提供 reviewer / reader 依需求檢閱。
- 6.5 Small-N statistical reliability (N=70 Stress, N=36 EEGMAT/SleepDep)
  > 中文：小樣本 N 的統計可靠性限制。
- 6.6 Protocol assumptions (subject-disjoint CV, recording-level labels, ≥ 19ch montage)
  > 中文：主文假設 subject-disjoint CV、recording-level label、≥19ch 電極。
- 6.7 LaBraM's reduced aperiodic-dependence — architectural explanation speculative
  > 中文：LaBraM 對 1/f 較不敏感的架構解釋尚屬推論。
- 6.8 Clinical deployment: checklist is necessary, not sufficient
  > 中文：checklist 是必要條件但不是充分條件。
- 6.9 Three concrete future directions
  > 中文：三個具體 future work。
  1. Pretrain an FM on FOOOF-detrended EEG
  2. Subject-adversarial regularisation for small-N FT
  3. Populate additional 2×2 cells with held-out datasets (promote to predictive claim)

---

## Appendix A — TDBRAIN replication on the between × separable cell
> 中文：TDBRAIN 作為 ADFTD cell 的 external replication，確認 between-subject trait cell 的 pattern 不是 ADFTD-specific。

### A.1 Variance atlas replication
> 中文：TDBRAIN 上 variance decomposition 與 LP→FT drift 是否重現 ADFTD pattern。

- **Fig A.1**  Variance atlas on TDBRAIN (3 FM)

### A.2 Positioning note
> 中文：TDBRAIN 為何只放 Appendix — 與 ADFTD 落在同一 cell，作為 breadth check 而非獨立 datapoint。

## Appendix C — Architecture ceiling across 4 cells
> 中文：4-cell × 4 架構類別（classical ML / non-FM deep / 3 FM FT）的 ceiling panel，驗證 §4.1 Tab 1 觀察到的 per-cell BA ceiling 是「cell 屬性」而非 FM-specific。主文 §4.2–§4.5 的四個 diagnostic 已先完整解釋 mechanism，此 panel 在最後作為 architecture breadth check，非 load-bearing step。

### C.1 Architecture panel × 4 cells
> 中文：classical ML (LogReg / SVM / RF / XGBoost) + non-FM deep (EEGNet / ShallowConvNet) + 3 FM FT (LaBraM / CBraMod / REVE)，跨約 6 個參數量級；每個 cell 的 BA 落在架構-不變的 band。

- **Fig 6** (planning label Fig C.1): 4-panel architecture ceiling, Subject-CV BA vs log(trainable params), one panel per cell (EEGMAT / SleepDep / ADFTD / Stress)
- Per-cell readings:
  - Stress (between × null-indistinguishable): ceiling 0.43–0.58, all architectures clustered near chance → cell property
  - SleepDep (within × null-indistinguishable): similar architecture-invariant near-chance band
  - ADFTD (between × separable): LaBraM + EEGNet both near 0.71–0.77, sizeable non-FM baseline
  - EEGMAT (within × separable): classical RF 0.89 above FM band, FM band 0.6–0.74

### C.2 Positioning note
> 中文：為何 ceiling 放在 Appendix — 主文 §4.1 Tab 1 已給 per-cell BA 的 setup，§4.2–§4.5 四個 diagnostic 已解釋 mechanism；Fig 6 在 Appendix 作為 architecture breadth 佐證，讓 reviewer 確認「更換架構不會逃離 cell-level ceiling」，不是 load-bearing 的論證步驟。原本規劃的 C.2 classical vs FM frozen LP 4-cell 比較已被 §4.1 Tab 1 的 classical / non-FM baseline 欄吸收，不另出圖。

- Why positioned in Appendix: Tab 1 and the four diagnostics already carry the main argument. Fig 6 supports the "ceiling is cell-property" reading by showing it persists across architectures spanning six orders of magnitude in trainable parameters, but the claim is not needed to establish any mechanism — it is a breadth check.

---

## Appendix B — Neuro-interpretability of FM frozen representations
> 中文：完整覆蓋 "FM 用哪些 component / 頻段 / 通道" 的三軸因果與相關分析；不屬主診斷工具但補充 spectral / spatial 落點。

### B.1 Channel ablation (spatial axis)
> 中文：30 通道 zero-out 後 cosine distance 的 topomap。

- **Fig B.1**  30-channel ablation topomap (3 FM × Stress 70 rec)

### B.2 Band-stop ablation per cell (frequency-band causal axis)
> 中文：對 delta/theta/alpha/beta 分別帶阻，看每格 FM cosine distance 落點。metric 為 **cosine distance**（representation geometry sensitivity），與 §4.5 band-stop 量 **probe BA**（decoding causal effect）互補 — 同一擾動、不同 metric，cross-reference 已在 §4.5 說明。

- **Fig B.2**  Band-stop cosine distance (3 FM × 4 datasets × 4 bands)

### B.3 Band RSA (correlational spectral axis)
> 中文：band power RDM 與 FM RDM 的 Spearman r per cell。

- **Fig B.3**  Per-band Spearman r matrix (3 FM × 4 datasets × 4 bands)

### B.4 Synthesis — 4-axis FM interpretability framework
> 中文：本論文涵蓋四個正交因果切片 — component (FOOOF)、frequency band、spatial channel、correlational spectrum。

- Cross-reference §4.4 (FOOOF aperiodic/periodic) as component axis
- Synthesis: subject dominance lives jointly in posterior channels × broadband × aperiodic 1/f

---

## Figure master list
> 中文：主文 5 圖（Fig 1 pipeline/teaser + Fig 2–5 四個 diagnostic） + Appendix 5 圖（A.1 + B.1–3 + C.1 = Fig 6 ceiling）。Tab 1 in §4.1 (placed)；Tab 2/3 proposed。

### Main text (4 diagnostic figures; benchmark landscape carried by Tab 1 in §4.1, no figure)
| Final Fig # | Planning label | Location | Content | Source file | Status |
|---|---|---|---|---|---|
| Fig 2 | Fig 4.2 | §4.2 | Variance decomposition 4 cells (stacked bars, frozen FM only) | `paper/figures/fig2/fig2_representation_2x2.{pdf,png}` | Existing — verify 4-cell extension |
| Fig 3 | Fig 4.3 | §4.3 | Permutation-null density 4 cells | `paper/figures/fig3/fig3_honest_evaluation.{pdf,png}` | Existing — verify 4-cell extension |
| Fig 4 (a/b/c) | Fig 4.4 | §4.4 | Within-subject LOO trajectory + direction consistency (EEGMAT + SleepDep) | `paper/figures/fig4/fig4{a,b,c}_*.{pdf,png}` | Existing — Stress/ADFTD excluded by design |
| Fig 5 (a/b/c) | Fig 4.5 | §4.5 | Causal anchor ablation (PSD+FOOOF fit, FOOOF scatter, band-stop line) | `paper/figures/fig5/fig5{a,b,c}_*.{pdf,png}` | Existing — verify 4-cell extension |

Note: **Fig 1 reserved for a planned pipeline/teaser schematic** (data → 2×2 cell assignment → diagnostic toolkit → per-cell verdict); not yet built. Benchmark landscape (setup, §4.1) is carried by Tab 1 only — no standalone figure.

### Appendix A (1 figure)
| # | Location | Content | Status |
|---|---|---|---|
| Fig A.1 | A.1 | TDBRAIN variance atlas (3 FM) | Existing — subset of old 12-cell |

### Appendix B (3 figures)
| # | Location | Content | Status |
|---|---|---|---|
| Fig B.1 | B.1 | 30-channel ablation topomap (Stress) | Existing (exp14) |
| Fig B.2 | B.2 | Band-stop cosine distance (3 FM × 4 DS × 4 bands) | Existing — extend to 4 DS |
| Fig B.3 | B.3 | Per-band Spearman r matrix (3 FM × 4 DS × 4 bands) | Existing — extend to 4 DS |

### Appendix C (1 figure)
| Final Fig # | Planning label | Location | Content | Source file | Status |
|---|---|---|---|---|---|
| Fig 6 | Fig C.1 | C.1 | 4-cell architecture ceiling × 4 architecture classes (classical / non-FM deep / FM FT) — breadth check that the cell-level BA ceiling observed in §4.1 Tab 1 is architecture-agnostic, not FM-specific | `paper/figures/main/Fig6_ceiling.png` (canonical) | Existing — panel titles need terminology update (`coherent/incoherent/absent` → `separable/null-indistinguishable`) |

Positioning note: Fig 6 enters *after* the four diagnostics (§4.2–§4.5) have explained why each cell sits at its BA band. It is a breadth check supporting "ceiling is cell property, not FM property", not a standalone finding. The original Appendix C.2 (classical XGBoost vs FM frozen LP, 4 cells) is subsumed by Tab 1's classical/non-FM baseline columns and is dropped.

### Tables
| Tab # | Location | Content | Source file | Status |
|---|---|---|---|---|
| Tab 1 | §4.1 | Master per-cell benchmark — LP/FT × 3 FM + classical-ML + non-FM deep baselines, subject-disjoint 3-seed | `paper/tables/table1_master_performance.tex` | Placed |
| Tab 2 | §3.1.3 | Cell-assignment summary — 4 rows × {dataset, axis A, axis B, cohort N, label rule} | TBD | Proposed |
| Tab 3 | §5.4 | Pre-flight checklist — 4 gates × {input, output, decision rule} | TBD | Proposed |

Note: per-cell drift summary dropped (LP→FT drift intentionally omitted from main text, §6.4).

---

## Writing protocol
> 中文：寫稿流程 — 先鎖 outline、再做圖表、文字一律用 /paper-writing skill、最後編譯。

1. Outline locked → this document
2. Figures built or regenerated first — pending: Fig 1 (pipeline/teaser, needs design); Fig 2 variance (verify 4-cell coverage); Fig 3 perm-null (verify 4-cell coverage); Fig 5 FOOOF (verify 4-cell coverage); Fig 6 ceiling (panel-title terminology update); Fig B.2–B.3 (extend to 4 DS)
3. Tables placed and built (deferred — see TODO)
4. For text: invoke **/paper-writing** skill per section; do not free-write
5. Final compile → new `docs/sdl_paper_full.pdf`

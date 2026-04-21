# SDL Paper — Outline (IMRaD, Stress+EEGMAT main)

**Status:** structure locked, text pending. Use `/paper-writing` skill when writing prose.
**Primary target:** Journal of Neural Engineering (JNE, IOP).
**Companion preprint:** TMLR (parallel submit for ML-community exposure).
**Fallback chain:** JBHI → J. Neuroscience Methods.
**Conditional upgrade:** Communications Medicine (Nature Portfolio) — requires FOOOF extension to ADFTD+TDBRAIN + full HP sweep first (§6.4, §6.2 open items).
**Scope:** Stress + EEGMAT as main paired-contrast datasets; ADFTD + TDBRAIN in Appendix A (external replication on disease cohorts).

**Framing constraint (locked):** framework-contribution, NOT audit/attack. Primary pitch is "SDL diagnostic framework + 3-gate pre-flight checklist". Wang 0.9047 is a worked example inside §4.2, not the paper's adversarial target. See `project_paper_journal_strategy.md` memory.

---

## §1 Introduction
> 中文：鋪陳臨床 rsEEG FM benchmark 的系統性失敗，提出 SDL 作為 diagnostic framework 與三路收斂證據的總論。

- 1.1 Clinical rsEEG FM benchmark gap
  > 中文：指出目前 FM 在臨床 rsEEG 上的報告數字與 honest eval 之間的缺口。
- 1.2 Motivating finding: Wang et al. 2025 reports 0.9047 BA on UCSD Stress under trial-level CV
  > 中文：以 Wang 0.9047 作為框架診斷的中性 motivating example，非攻擊對象。
- 1.3 Paper contribution: Subject Dominance Limits (SDL) — three converging lines of evidence on a paired within-cohort cognitive-state design (EEGMAT arithmetic state vs Stress DASS trait)
  > 中文：貢獻主軸 — SDL framework + 三路收斂證據（geometry / separability / causal）+ pre-flight checklist。
- 1.3.1 **Regime taxonomy (2026-04-21)**: FM failure is a property of *regime*, not dataset. Subject-label regime (Stress, DEAP, SEED, DREAMER, DASPS) triggers the failure mode; within-subject regime (EEGMAT, SleepDep, HMC) does not. Stress is the canonical instance, not the protagonist.
  > 中文：SDL 失敗不是 dataset 屬性，是 regime 屬性。subject-label regime 必然觸發失敗；within-subject paired regime 則否。Stress 是代表實例，不是故事主角。見 `docs/regime_framing_decision.md`。
- 1.4 Summary of findings + paper roadmap
  > 中文：主要發現摘要與章節導覽。

## §2 Related Work
> 中文：定位本論文相對於 EEG-FM 文獻、subject leakage 討論、contrast anchoring、1/f 生理研究。

- 2.1 EEG foundation models (LaBraM / CBraMod / REVE)
  > 中文：三個評估標的 FM 的背景與 published benchmarks。
- 2.2 Subject leakage and evaluation protocols
  > 中文：trial-level vs subject-level CV 既有討論與未被解決的盲點。
- 2.3 Contrast anchoring in clinical rsEEG
  > 中文：臨床 rsEEG state 與 trait 標籤的 neural anchor 先前證據。
- 2.4 Aperiodic 1/f as subject identity signal
  > 中文：文獻支持 1/f 是 subject 指紋（Demuru 2020, Lanzone 2023 等）。
- (ADFTD/TDBRAIN benchmark numbers as footnote, not spine)
  > 中文：ADFTD / TDBRAIN 只以 footnote 引用，不作為主文比對基準。

## §3 Methods
> 中文：定義資料集、FM 抽取、兩種 eval regime、FT、variance/RSA、FOOOF、perm-null（僅 §4.2 用）、representation drift 與架構 panel 的作法。

- 3.1 Datasets
  > 中文：主文 Stress+EEGMAT paired + SleepDep (§4.4 Type III case)；ADFTD+TDBRAIN 移 Appendix A。
  - 3.1.1 Stress (70 rec, 17 subj, DASS per-recording) — Type I (no recoverable state anchor)
    > 中文：Stress 資料集基本規格與 DASS per-recording label；Type I null-control。
  - 3.1.2 EEGMAT (36 subj, 19ch, rest/arithmetic) — Type II (α-broadband anchor)
    > 中文：EEGMAT 基本規格，rest vs 算數的 within-subject 設計；Type II α-broadband anchor（peak + in-band 1/f tail 一起）。
  - 3.1.3 SleepDep (ds004902, 36 subj NS/SD pairs, 19ch) — Type III (1/f-aperiodic anchor)
    > 中文：SleepDep 資料集基本規格；主文只進 §4.4 FOOOF 作為 Type III diffuse 1/f anchor 案例，不進 §4.1/§4.3/§4.6 paired 分析（within-subject LOO pairwise BA ≈ chance，見 Appendix note）。
  - 3.1.4 Paired-contrast rationale (same small-N rsEEG, differ in neural anchor type)
    > 中文：三資料集在 N 與 rsEEG protocol 匹配，只差在 anchor 種類（無/periodic/aperiodic）。
  - 3.1.5 External replication: ADFTD + TDBRAIN (Appendix A)
    > 中文：疾病 cohort 作為 external replication，說明 paradigm 差異。
- 3.2 Foundation model feature extraction + per-FM input normalisation
  > 中文：FM 抽取流程與各模型必須使用的 input norm（LaBraM zscore / CBraMod/REVE none）。
- 3.3 Evaluation regimes
  > 中文：論文並行採用 between-subject 與 within-subject 兩種 evaluation 設計。
  - 3.3.1 Between-subject: per-window LP + subject-disjoint CV (FT-matched)
    > 中文：與 FT 相匹配的 per-window LP + subject-disjoint 5-fold CV protocol。
  - 3.3.2 Within-subject: LOO with personal-median DSS threshold (Stress) / LOO rest-vs-task (EEGMAT)
    > 中文：固定 subject 身分下的 LOO 評估，測試純 state signal。
- 3.4 Fine-tuning canonical per-FM recipe
  > 中文：使用各 FM 原論文 canonical recipe，不做 per-dataset HP 調參以避免 test contamination。
- 3.5 Variance decomposition + RSA geometry (recording-level)
  > 中文：recording-level variance 分解與 RSA 幾何分析的程序。
- 3.6 FOOOF aperiodic/periodic ablation procedure
  > 中文：以 FOOOF 參數重建訊號並 selectively 去掉 aperiodic 或 periodic 成分，測量因果效應。
- 3.7 Permutation-null test (LaBraM Stress vs EEGMAT paired, §4.2 audit only)
  > 中文：以 label shuffle 建立 null 分布；用於 §4.2 LaBraM paired audit，不再用於 CBraMod/REVE 機制檢驗（改用 §3.8 drift）。
- 3.8 LP→FT representation drift analysis
  > 中文：比 frozen LP feature vs FT OOF feature 的 variance decomposition (label_frac, subject_frac) + LogME + CKA(LP, FT)，直接從 representation level 判定 FT 是 real-learning 還是 subject-shortcut。
- 3.9 Architecture comparison panel (7 architectures × Stress)
  > 中文：在 Stress 上評估 7 個跨 6 個量級的架構，檢驗 ceiling 是否為 task property。

## §4 Results
> 中文：六個 subsection 依序呈現 SDL 框架的三路收斂證據與 FT 診斷。

### 4.1 Representation geometry is subject-dominated across regime
> 中文：variance + RSA 顯示 frozen FM representation 幾何上由 subject 結構主導，但 FT 動向的意義是 **regime-conditional** — 同方向變化在 subject-label regime 與 within-subject regime 有相反意義（見 `docs/methodology_notes.md#N-F22`）。

- **Fig 4.1** (canonical Fig 2 in build) — 2 rows × 3 datasets (Stress, EEGMAT, SleepDep)
  - Row A: Variance decomposition stacked bars (Label / Subject / Residual), frozen vs FT per FM
  - Row B: RSA scatter with frozen→FT arrows (3 FM per panel), label-r vs subject-r
  - Caption must qualify `subject_frac ↑` direction by regime: shortcut in subject-label regime, healthy reference encoding in within-subject regime
- Regime axis introduced here: Stress = subject-label (1 datapoint); EEGMAT + SleepDep = within-subject paired (2 datapoints)
- See `docs/regime_framing_decision.md` for the full strategic pivot narrative

### 4.2 Honest evaluation closes the 40 pp gap on Stress
> 中文：Stress 上 0.9047 → 0.443-0.577 的差距在 honest protocol 下幾乎全部收斂。

- **Fig 4.2**  Honest-evaluation funnel (Wang 0.9047 → subject-CV 0.443–0.577 → permutation-null p=0.70 → classical 0.553)
- **Fig 4.3**  Permutation-null density, LaBraM × {Stress, EEGMAT} paired

### 4.3 Contrast-anchoring governs state separability — EEGMAT ↔ Stress paired, both eval regimes
> 中文：SDL 的 contrast-anchoring 預測在兩種 eval regime 皆成立 — EEGMAT 都 separate、Stress 都 chance。

The contrast-anchoring prediction of SDL is tested in two complementary evaluation regimes; SDL predicts divergent outcomes between EEGMAT and Stress in BOTH regimes (no escape by choice of CV unit).

#### 4.3.1 Between-subject generalization (per-window LP, subject-disjoint CV, FT-matched protocol)
> 中文：在 hold-out 新 subject 的設計下，EEGMAT 可分、Stress 不可分。

- **Fig 4.4 panel A**  EEGMAT 0.72–0.74 vs Stress 0.43–0.53 across 3 FMs
- Protocol matches FT evaluation in §4.6 (apples-to-apples LP vs FT)
- Clinical reading: "can the FM diagnose state in an unseen patient"

#### 4.3.2 Within-subject tracking (LOO, personal-median DSS threshold for Stress)
> 中文：固定 subject 後只測純 state signal，EEGMAT LOO 0.64-0.68、Stress DSS r≈0；為最乾淨的 falsification。

- **Fig 4.4 panel B**  EEGMAT 0.64–0.68 LOO BA vs Stress DSS Spearman r ≈ 0 across 3 FMs
- **Fig 4.4 panel C**  Within-subject directional consistency (exp11): EEGMAT 0.07–0.15 across 3 FMs vs Stress ≈ 0 (frozen and FT both); a second within-subject metric independent of LOO BA, confirming the same SDL pattern
- Holds subject identity fixed; isolates pure within-subject contrast signal from subject feature structure
- Source: exp11_longitudinal_dss (EEGMAT: LOO BA rest vs task; Stress: personal-median DSS trajectory Spearman; dir_consistency for both, frozen + FT)
- Clinical reading: "can the FM track an individual patient's state trajectory over time"
- This is the SDL framework's most direct falsification test — when subject identity is eliminated from the label-subject coupling, only pure state contrast remains

**Combined narrative (§4.3 take-home):** SDL prediction holds in both eval regimes — EEGMAT separates state in both; Stress fails in both. The difference is the presence of a neural anchor, not the evaluation protocol.

### 4.4 Causal anchor dissection — FOOOF ablation + band-stop sensitivity
> 中文：用 FOOOF 拆解 aperiodic/periodic + band-stop 兩種互補因果擾動，給出 Type I/II/III anchor 分類 — Stress (no anchor) / EEGMAT (α-broadband) / SleepDep (1/f-aperiodic)；subject ID 普遍住在 aperiodic 1/f。

- **Fig 4.5**  Three panels (PSD + FOOOF fit, FOOOF ablation scatter, band-stop profile)
  - **Top row**: PSD + FOOOF decomposition (aperiodic 1/f slope fit vs periodic peaks) for one representative recording per dataset — grounds what the two ablations actually remove
  - **Bottom-left**: FOOOF ablation signature (scatter, x = Δ subject-ID probe BA, y = Δ state probe BA); each point = dataset × ablation-condition; within-dataset lines connect −aperiodic → −periodic. Quadrant position encodes anchor type.
  - **Bottom-right**: Band-stop sensitivity (line plot, x = {δ, θ, α, β}, y = cosine distance between FM features pre/post Butterworth band-stop); FM-averaged; α band highlighted.
- **Take-home (two interventions, one taxonomy)**:
  - *FOOOF subject probe*: aperiodic removal drops subject ID in all 3 datasets (EEGMAT: CBraMod −14 pp / REVE −26 pp; Stress CBraMod −8.6 pp; SleepDep −2–4 pp); periodic removal leaves subject probe unchanged → **aperiodic 1/f is the universal subject substrate**.
  - *FOOOF state probe*: only SleepDep collapses under aperiodic removal (LaBraM 0.616→0.538, REVE 0.562→0.519; mean −4.5 pp) → **SleepDep state is 1/f-anchored (Type III)**. EEGMAT state survives both ablations (≤ 1.3 pp drop); Stress state is null throughout.
  - *Band-stop cosine distance*: EEGMAT FM features peak in α-band reliance (0.105, highest cell), Stress peak in β (0.078), SleepDep flat across all bands (≤ 0.012) → **EEGMAT state lives in the α band as broadband (peak + in-band 1/f tail together)**, not the peak alone. This resolves the apparent mismatch between FOOOF periodic-removal being weak on EEGMAT vs the α-band cosine-distance being strong: FOOOF removes peaks only; band-stop removes peak *and* in-band background.
  - **Integrated anchor taxonomy** (Type I Stress / Type II EEGMAT α-broadband / Type III SleepDep 1/f-aperiodic) is model-independent and computable from the EEG alone, before any FM training.
- **Interpretation note**: Cosine distance measures FM representation *reliance* on a band, not task-probe accuracy; the scatter (probe Δ) and the line (FM drift) must be read together to attribute anchor type.

### 4.5 Architecture-independent ceiling on Stress
> 中文：7 個跨 6 量級的架構在 Stress 全部停在 0.43-0.58 band，證明 ceiling 是 task property 而非架構限制。

- **Fig 4.6**  Seven architectures spanning six orders of magnitude → 0.43–0.58 BA band
- **Fig 4.7**  Classical band-power XGBoost vs FM frozen LP
- **Table 4.1**  Full architecture panel (EEGNet, ShallowConvNet, LaBraM, CBraMod, REVE × {frozen LP, FT}, XGBoost baseline)

### 4.6 FT rescue on Stress is subject-shortcut exploitation (representation drift)
> 中文：用 LP→FT 的 variance decomposition 直接看機制 — 三 FM 在 Stress 上 FT 後 label_frac 不增反減（−1～−2pp），subject_frac 大幅上升（+6～+25pp）；EEGMAT × CBraMod 的 +3/−24pp 真 label-learning signature 作為 positive control。

- **Fig 4.8**  LP→FT representation drift bar chart — 6 cells (3 FMs × {Stress, EEGMAT}); two bars per cell showing Δlabel_frac and Δsubject_frac, color-coded by mechanism verdict (shortcut / real / no-drift)
- **Table 4.2**  Per-cell drift summary: LP & FT label_frac, subject_frac, CKA(LP, FT), Δ values, mechanism verdict
- **Mechanism interpretation:**
  - Stress × LaBraM (canonical lr=1e-5): Δlabel −1.0pp / Δsubj **+24.6pp** / CKA 0.46 → shortcut
  - Stress × CBraMod: Δlabel −1.5pp / Δsubj **+19.5pp** / CKA 0.17 → shortcut
  - Stress × REVE: Δlabel +0.2pp / Δsubj **+6.2pp** / CKA 0.98 → shortcut (mild backbone change but subject-direction)
  - EEGMAT × CBraMod: Δlabel **+3.0pp** / Δsubj **−23.8pp** / CKA 0.14 → **real label-learning signature** (positive control)
  - EEGMAT × LaBraM / REVE: drift small or LP-saturated (FT BA ≈ LP BA)
- **Take-home:** all 3 FMs on Stress show consistent subject-shortcut signature regardless of FT BA outcome; EEGMAT × CBraMod confirms the analysis can detect real label-learning when present. This provides direct mechanistic evidence for SDL that does not depend on permutation-null statistical power.

## §5 Discussion
> 中文：總結 SDL 兩 regime 行為、對 benchmark 設計的啟示，並將 clinical checklist 作為可操作輸出。

### 5.1 Two-regime behaviour of EEG-FMs on clinical rsEEG
> 中文：EEG-FM 表現由 task contrast 決定 — anchored 與 bounded 兩種 regime；bounded 區 FT 在機制上是 subject-shortcut 而非 label learning。

- Contrast-anchored regime (EEGMAT): frozen LP separates state, FT adds little
- Contrast-bounded regime (Stress): all architectures ceiling at ~0.55; FT exploits subject identity (representation drift confirms +6 to +25 pp subject_frac increase across 3 FMs without label_frac gain)
- Critical implication: positive FT BA on contrast-bounded tasks requires representation-drift corroboration before being interpreted as real label learning

### 5.2 Implications for benchmark design
> 中文：benchmark 改善應投資於 contrast anchoring，而非架構疊代。

- Published trial-level CV numbers on subject-bounded tasks are protocol artefacts
- Architectural iteration within a contrast-bounded cohort is not the productive move

### 5.3 Clinical pre-flight checklist (actionable output)
> 中文：提供 3-gate checklist，讓臨床研究者在投入 FT compute 前做 diagnostic 篩檢。

- **Table 5.1**  3-gate decision framework before investing FT compute:
  1. Frozen LP under subject-disjoint CV — does FM separate state at all?
  2. **Representation drift test** — does FT increase label_frac (real learning) or only subject_frac (shortcut exploitation)?
  3. FOOOF detrend probe — is signal contrast-anchored or 1/f-parasitic?

## §6 Limitations and Future Work
> 中文：誠實列出本論文未解決的 scope 限制與可擴展的 future work 方向。

- 6.1 Wang 2025 reproduction is partial (protocol-side closed; HP-side not)
  > 中文：Wang 的 HP-side reproduction 未完成；protocol-side 已關閉。
- 6.2 Per-dataset HP sweep deferred (canonical recipe rationale; Stress-only sweep available)
  > 中文：per-dataset HP sweep 只在 Stress 做過；其他 dataset 沿用 canonical recipe。
- 6.3 Architecture panel is Stress-only; DeepConvNet / EEGConformer under old protocol (rerun deferred)
  > 中文：架構 panel 僅在 Stress 上做；DeepConvNet / EEGConformer 要在新 protocol 下補跑。
- 6.4 FOOOF ablation main analysis on 3/5 datasets (Stress + EEGMAT + SleepDep); ADFTD + TDBRAIN extension deferred
  > 中文：FOOOF ablation 目前含 Stress+EEGMAT+SleepDep 三個主文資料集；ADFTD/TDBRAIN 擴充為 future work。
- 6.5 Small-N statistical reliability (N=70 Stress)
  > 中文：N=70 的統計可靠性限制。
- 6.6 Protocol assumptions (subject-disjoint CV, recording-level labels, ≥ 19ch montage)
  > 中文：主文假設 subject-disjoint CV、recording-level label、≥19ch 電極。
- 6.7 LaBraM's reduced aperiodic-dependence — architectural explanation speculative
  > 中文：LaBraM 對 1/f 較不敏感的架構解釋尚屬推論。
- 6.8 Clinical deployment: checklist is necessary, not sufficient
  > 中文：checklist 是必要條件但不是充分條件。
- 6.9 Three concrete future directions:
  > 中文：三個具體 future work 方向。
  1. Pretrain an FM on FOOOF-detrended EEG
  2. Subject-adversarial regularisation for small-N FT
  3. Within-subject longitudinal benchmarks

---

## Appendix A — External replication on disease cohorts
> 中文：以 ADFTD + TDBRAIN 疾病 cohort 作為 subject-dominance 幾何層級的外部驗證。

### A.1 Variance atlas replication on ADFTD + TDBRAIN
> 中文：variance atlas 在兩個疾病 cohort 6-cell 重現 subject-dominant pattern。

- **Fig A.1**  Variance atlas 3-panel × 6-cell (3 FM × ADFTD/TDBRAIN)
- Text: subject dominance pattern holds under trait-label disease classification

### A.2 Frozen LP + FT master table on ADFTD + TDBRAIN
> 中文：ADFTD/TDBRAIN 的 FT vs Frozen LP 6-cell 比較表。

- **Table A.1**  6-cell FT vs frozen LP
- Text: Notes the regime shift — ADFTD/TDBRAIN are between-subject trait classification, not per-recording state; included to demonstrate SDL geometric claim is not paradigm-specific

### A.3 Positioning note
> 中文：說明 ADFTD/TDBRAIN 為何只放 Appendix — label paradigm (trait vs state)、cohort 規模、臨床問題都與主文不同。

- Why ADFTD/TDBRAIN are not in main Results: different label paradigm (trait vs state), different cohort size regime, different clinical question. The main paired contrast (Stress ↔ EEGMAT) tests the contrast-anchoring hypothesis directly; ADFTD/TDBRAIN serve as breadth check that subject-dominance pattern is not an artefact of the paired design.

---

## Appendix B — Neuro-interpretability of FM frozen representations
> 中文：完整覆蓋 "FM 用哪些 component / 頻段 / 通道" 的三軸因果與相關分析；不屬 SDL diagnostic framework，但說明 SDL 機制的 spectral / spatial 落點。

### B.1 Channel ablation (spatial axis)
> 中文：30 通道各 zero-out 後 cosine distance 揭示三 FM 都重視 posterior O2/Oz/Pz/P4（α 來源區）；LaBraM 最敏感、CBraMod 3-5× 較不敏感。

- **Fig B.1**  30-channel ablation topomap (3 FMs × Stress 70 rec)
- Source: exp14_channel_importance/channel_importance.json

### B.2 Band-stop ablation (frequency-band causal axis)
> 中文：對 delta/theta/alpha/beta 分別帶阻濾波再看 FM cosine distance；Stress 上 LaBraM 主要靠 beta（subject ID/arousal），EEGMAT 上 LaBraM/REVE 改靠 alpha（task α suppression）。

- **Fig B.2**  Band-stop cosine distance bar chart (3 FMs × {Stress, EEGMAT} × 4 bands)
- Source: exp14_channel_importance/band_stop_ablation.json

### B.3 Band RSA (correlational spectral axis)
> 中文：band power RDM 與 FM RDM 的 Spearman r — Stress 全 band 顯著（broadband subject fingerprint），EEGMAT 出現 alpha/theta selectivity（task signal）。

- **Table B.1**  Per-band Spearman r matrix (3 FMs × {Stress, EEGMAT} × 4 bands)
- Source: exp14_channel_importance/band_rsa.json

### B.4 Synthesis — 4-axis FM interpretability framework
> 中文：本論文涵蓋四個正交因果切片 — component (FOOOF, 主文 §4.4)、frequency band (band-stop)、spatial channel (channel ablation)、correlational spectrum (band RSA)；後三者放 Appendix B 不影響主文 SDL 診斷敘事。

- Cross-reference §4.4 (FOOOF aperiodic/periodic) as the 4th axis
- Synthesis: subject dominance lives jointly in posterior channels × broadband (esp. beta) × aperiodic 1/f component
- Practical takeaway: future FM pretraining should consider FOOOF-detrended input or spatial-spectral disentanglement to reduce subject leakage

---

## Figure/Table master list
> 中文：主文 9 圖 + 2 表；Appendix 1 圖 + 1 表。New 與 Redraw 狀態逐一標示。

### Main text (10 figures + 2 tables)
| # | Location | Content | Status |
|---|---|---|---|
| Fig 4.1 | §4.1 | Variance + RSA atlas, 2 row × 3 DS (Stress/EEGMAT/SleepDep, variance bars + RSA frozen→FT arrows) | ✅ Built 2026-04-21 (`fig2_representation_geometry`) |
| Fig 4.2 | §4.2 | Honest-evaluation funnel | Existing |
| Fig 4.3 | §4.2 | Perm-null density (LaBraM paired Stress/EEGMAT) | Existing |
| Fig 4.4A | §4.3.1 | Paired between-subject LP (EEGMAT vs Stress DASS) | Existing |
| Fig 4.4B | §4.3.2 | Paired within-subject LOO (EEGMAT rest/task vs Stress DSS) | **Redraw** from exp11 |
| Fig 4.4C | §4.3.2 | Within-subject directional consistency (frozen + FT, 3 FM × 2 DS) | **New** from exp11 dir_consistency |
| Fig 4.5 | §4.4 | Causal anchor dissection (PSD+FOOOF fit row + FOOOF scatter + band-stop line) | **New v6 — data ready**, scatter + line + PSD replaces 6-panel bar grid |
| Fig 4.6 | §4.5 | Architecture ceiling (7 arch × Stress) | Existing |
| Fig 4.7 | §4.5 | Classical band-power baselines | Existing |
| Fig 4.8 | §4.6 | LP→FT representation drift, 6 cells (3 FM × {Stress, EEGMAT}) | **New** — data ready |
| Table 4.1 | §4.5 | Architecture panel full | Existing |
| Table 4.2 | §4.6 | Per-cell drift summary (Δlabel_frac, Δsubject_frac, CKA, verdict) | **New** — data ready |
| Table 5.1 | §5.3 | 3-gate clinical pre-flight checklist | Existing (was §9) |

### Appendix A (1 figure + 1 table)
| # | Location | Content | Status |
|---|---|---|---|
| Fig A.1 | A.1 | Variance atlas, 6-cell (3 FM × ADFTD+TDBRAIN) | **Redraw** (subset of old 12-cell) |
| Table A.1 | A.2 | Master FT vs LP on ADFTD+TDBRAIN | Existing (subset of v2 12-cell) |

### Appendix B (2 figures + 1 table)
| # | Location | Content | Status |
|---|---|---|---|
| Fig B.1 | B.1 | 30-channel ablation topomap (3 FMs × Stress) | Existing (exp14) |
| Fig B.2 | B.2 | Band-stop cosine distance (3 FMs × {Stress, EEGMAT} × 4 bands) | Existing (exp14) |
| Table B.1 | B.3 | Band RSA per-band Spearman r matrix | Existing (exp14) |

---

## Writing protocol
> 中文：寫稿流程 — 先鎖 outline、再做圖表、文字一律用 /paper-writing skill、最後編譯。

1. Outline locked → this document
2. Figures/tables built or regenerated first
3. For text: invoke **/paper-writing** skill per section; do not free-write
4. Final compile → new `docs/sdl_paper_full.pdf`

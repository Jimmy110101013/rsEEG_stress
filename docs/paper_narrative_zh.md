# 論文完整論述（中文閱讀版）

**目的**：讓作者一次讀完整篇論文的敘事邏輯，確認故事連貫、每個段落都有據可查。
**英文版標題**：*Beyond Accuracy: What EEG Foundation Models Encode, and Why Fine-Tuning Direction Depends on Model × Dataset Interactions*
**日期**：2026-04-14
**對應文件**：`findings.md`（證據來源）、`paper_strategy.md`（結構決策）

---

## 摘要（一段話概述）

EEG 基礎模型（foundation models, FMs）近年在多個生理訊號分類任務上被宣稱達到接近飽和的表現，但這些數字多半建立在容易造成 subject leakage 的實驗設計上。我們對三個主流 EEG FM（LaBraM、CBraMod、REVE）在四個公開資料集（Stress、ADFTD、TDBRAIN、EEGMAT）上進行系統性再評估，得到三個層次的發現：**第一**，在 Stress 資料集上，文獻報告的 ~0.90 BA 在嚴格的 subject-level cross-validation + 誠實 per-recording 標籤下降至 0.45–0.60，顯示高分多半來自跨受試者洩漏；**第二**，凍結的 FM 表徵主要編碼 subject identity 而非診斷性訊號，跨 12 個 model × dataset 組合中，RSA 的 subject 相似度皆大於 label 相似度；**第三**，fine-tuning（FT）的效果方向並非固定的「有效」或「失效」，而是 model × dataset 的交互作用——LaBraM 在 ADFTD 上會強化 label 表徵、在 TDBRAIN 上會侵蝕之；REVE 則完全相反。我們同時以 Stress 資料集（70 recordings / 17 受試者）作為貫穿全文的 **statistical power floor 警世案例**：在此樣本規模下，cuDNN 非決定性造成 ±20pp 的單 seed 飄移，permutation null 無法與真實 label 區分，連 2017 年的 ShallowConvNet 都能追平 FM 的 FT 表現。論文據此提出三點建議：（1）subject-level CV 必須作為預設；（2）benchmark 文獻應強制報告 statistical power；（3）FT 配方的選擇是資料集特異性的，沒有「通用最佳」。

---

## 1. 引言：一個看似解決、實則未解的問題

近兩年 EEG 基礎模型（LaBraM、CBraMod、REVE 等）陸續發表，官方論文與第三方 benchmark 都報出相當亮眼的下游任務準確率。例如 Wang 等人（2025, arXiv:2505.23042）在 Stress 資料集上報告 90% balanced accuracy。若屬實，這意味著我們幾乎已經擁有一個可以從 resting-state EEG 偵測心理壓力的臨床工具。

但當我們嘗試重現這些結果時，第一個問題就浮現：**他們用的是 trial-level cross-validation，也就是同一個受試者的不同時段會同時出現在訓練與測試集**。EEG 訊號在個體內高度穩定，這種分割等於在測「模型能不能認出這個人」，而不是「能不能偵測壓力狀態」。我們改用 subject-level StratifiedGroupKFold——同一受試者只出現在一邊——並使用誠實的 per-recording DASS 標籤，三個 FM 的 frozen LP balanced accuracy 落在 0.45–0.60 區間、best-HP FT 落在 0.52–0.58 區間，與文獻報告的 0.90 相差 30 個百分點以上。這個落差完全來自 cross-validation 協議的選擇，跨三個不同架構的模型一致出現。

這個落差並不是新發現；Roy 等人在 2019 年的系統性回顧就指出 EEG 深度學習文獻普遍受此問題污染。2025–2026 年間，Brain4FMs、EEG-FM-Bench、AdaBrain-Bench 三份獨立的 benchmark 論文也都得出類似結論：EEG FM 在跨受試者的情感／認知任務上表現與傳統方法相當、甚至更差。我們的貢獻不是「再重複一次這個發現」，而是**提供他們缺少的機制解釋**：為什麼 FM 會失效，失效的結構是什麼，以及 fine-tuning 在什麼條件下能夠救回來。

我們因此提出三個具體問題：

1. **表徵診斷**：凍結的 FM 到底編碼了什麼？是診斷訊號、是受試者指紋、還是單純的訊號能量？
2. **Fine-tuning 交互作用**：FT 什麼時候會強化 label 表徵（injection）、什麼時候會反而侵蝕凍結的診斷訊號（erosion）？
3. **Benchmark 可信度**：在 70 recordings / 14 正樣本這類小樣本情境下，任何關於 FM 優越性的聲明，在統計上站得住腳嗎？

第一個問題的答案告訴我們 FM 為什麼難；第二個問題的答案告訴我們何時能改善它；第三個問題的答案告訴我們哪些現有宣稱不該被當真。

---

## 2. 方法：幾個看似細節但決定性的選擇

### 2.1 資料與標籤

Stress 資料集出自 Komarov、Ko、Jung（2020, IEEE TNSRE 28(4):795）的縱貫性研究，受試者為台灣研究生，每次錄音同時採集 EEG、DASS-21（Depression/Anxiety/Stress Scale）與 DSS（Daily Stress Score），設計本身即為 within-subject longitudinal。我們從原始 dataset 篩出**同時具備 DASS-21 與 DSS 標籤**的 70 段錄音（17 位受試者，14 段為正壓力樣本），作為全部 FM pipeline 的一致資料範圍。標籤使用 **per-recording DASS**（`--label dass`）——每段錄音根據當次的 DASS 自填分數分類，不做 within-subject aggregation，這與原研究「同一位受試者跨時間的情緒狀態會變化」的設計意圖一致。

### 2.2 Cross-validation 協議

**Subject-level StratifiedGroupKFold(k=5)** 作為主要指標，確保同一位受試者只出現在訓練或測試一側。**Trial-level StratifiedKFold(k=5)** 只用於與文獻對照，在論文中明確標示為「存在受試者洩漏」。

### 2.3 FM 特異性的輸入歸一化

這是一個容易被忽略但影響極大的細節。三個 FM 對輸入 scale 的需求不同：

- **LaBraM**：使用 zscore（符合原論文 FT 配方）
- **CBraMod**：使用 none（extractor 內部已有 `x/100` 的縮放；再 zscore 會把輸入推向 0，模型直接崩潰）
- **REVE**：使用 none（patch embedding 是線性層，在微伏 scale 上訓練；對 scale 敏感）

一次 multi-model sweep 用全域 `--norm` 會靜默毀掉其中兩個 FM 的訓練。我們在所有實驗中採用 per-model norm，並在論文方法章節明確列出。

**Window size**：LaBraM 與 CBraMod 使用 5 秒 windows（配合原作者 FT 配方），REVE 使用 10 秒 windows（配合其原生 patch 設計）。所有 frozen vs FT 比較皆 window-matched，避免 window-mismatch 造成的偽像；robustness check 置於 supplementary。

### 2.4 多 seed 與 cuDNN 決定性

在 Stress 的 70 rec / 14 positive regime 下，即使設定 `cudnn.deterministic=True`、`benchmark=False`，同一配方不同 init 仍可造成 10–20pp 的準確率飄移。我們因此規定**所有 Stress 數字都必須 ≥3 seeds**，且在全文不引用任何單 seed BA。

### 2.5 Permutation null

使用 `train_ft.py --permute-labels <seed>` 在 recording-level 洗亂標籤後重跑整個 CV pipeline。多次 null 構成分佈，對照真實 label 的分數判定「效果是否與隨機不可區分」。

### 2.6 Variance 分解

每個 model × dataset 的表徵上計算 **pooled label fraction**（以 label 為自變數的 η² 彙整），以及 subject 與 label 在 representation space 上的 RSA 相似度、mixed-effects ICC、cluster bootstrap CI，用於區分「受試者結構」與「標籤結構」。

---

## 3. 支柱 A：方法學陷阱（Methodology traps）

這一節用來建立信任——讀者必須先相信我們不是在用有偏頗的資料得到有趣的結果。

### 3.1 Subject leakage 造成大幅準確率膨脹（F01）

在 Stress 資料集上，文獻採用 trial-level CV 報告的數字可達 0.90（Wang 2025）。採用 subject-level StratifiedGroupKFold、誠實 per-recording DASS 標籤、`cudnn.deterministic=True`、3-seed 平均的協議下：三個 FM 的 frozen LP BA 落在 0.45–0.60（LaBraM 0.605 ± 0.032、CBraMod 0.452 ± 0.032、REVE 0.494 ± 0.018），best-HP FT 落在 0.52–0.58。兩個方案同一資料集，差距達 30 個百分點以上——這完全是 CV 協議與標籤處理的產物，而非模型能力差異。Roy 等人（2019）的系統性回顧已指出此現象在 EEG 深度學習文獻中普遍存在。

### 3.2 單 seed 數字不可信（F08）

在 70 rec / 14 positive 的小樣本 regime 下，即使開啟 `cudnn.deterministic=True` 與 `benchmark=False`，LaBraM canonical 配方 3-seed 仍給出 0.443 ± 0.083 的 std，對應 seed 間的範圍可以橫跨約 14 個百分點。這讓任何單 seed 報告都不可靠。我們因此規定 Stress 全部 BA 宣稱 **必須 ≥ 3 seeds**，並在全文不引用任何單 seed 數字。

### 3.3 古典特徵在誠實標籤下無法分辨壓力狀態（F16）

在誠實的 per-recording DASS 標籤下（與 FM pipeline 對齊的 70 recordings），所有古典方法——RF 0.44、LogReg 0.49、XGBoost 0.44、SVM 0.43——全部處於 chance 或以下，kappas 皆為負。同樣資料、同樣協議下，LaBraM frozen LP 仍保有 0.605 BA。這建立了 FM 在小樣本 EEG resting-state 分類上**相對古典 band-power 特徵的真實優勢**：不是天花板很高，但確實存在一個古典方法抓不到的可分類子空間。

---

## 4. 支柱 B：表徵診斷（Representation diagnosis）

這一節告訴讀者 FM 到底編碼了什麼。

### 4.1 Subject identity 主宰表徵（F02、F13）

在 12 個 frozen model × dataset 組合上，RSA 的 subject 相似度皆 > label 相似度。Mixed-effects 模型顯示，在 EEGMAT 上受試者身份解釋了 71% 的表徵變異，標籤只佔約 5%。這不是某一個 FM 的特定行為，而是**三個架構都表現出的普遍性質**：預訓練的 EEG FM 主要學到了個體 EEG 指紋，而不是任務相關的神經訊號。

### 4.2 但 FM frozen 仍勝過古典特徵（F03、F16）

在誠實標籤下，LaBraM frozen LP 還能達到 0.605 BA，而 RF/SVM/XGBoost 全部低於 chance。這說明 FM 雖然主要編碼 subject，但在這些 subject-specific embeddings 中，仍有一個可線性分類的 label 子空間存在——只是 signal-to-noise 比很低。古典的 band-power 特徵完全抓不到這個子空間。

### 4.3 Alpha 側化是古典 ML 的主要壓力特徵（F10）

Random Forest 的 feature importance 顯示 20 個最重要特徵中 13 個是 alpha 頻段，且集中在右半腦（T4、C4、CP4、P4）與額葉（Fp2、F8）。這與壓力相關的 alpha asymmetry 文獻一致，也作為後續 §6 神經科學可詮釋性分析的 baseline。

---

## 5. 支柱 C：FT 的 Cross-dataset × Cross-model Taxonomy（頭條結果）

這一節是論文的核心貢獻。

### 5.1 同樣的 label，不同方向的 FT 效應（F17）

我們在 ADFTD 與 TDBRAIN 上對三個 FM 分別執行 3-seed FT，並計算 pooled label fraction 的 FT − Frozen 差值（單位 pp，representation-level）：

| Model | ADFTD Δ | TDBRAIN Δ |
|---|---|---|
| LaBraM | **+1.03 ± 0.74** | **−1.56 ± 0.28** |
| CBraMod | +0.83 ± 3.35（不穩） | −0.02 ± 0.04 |
| REVE | **−1.53 ± 0.28** | +0.44 ± 0.32 |

**LaBraM 和 REVE 在兩個資料集上方向完全相反**。同樣的標籤、同樣的 pipeline，只是換了 backbone——FT 的效應就從 injection 翻成 erosion。這直接推翻了我們早期「FT 方向由 label biology 決定」的假設（F04 的 LaBraM-only 版本）。

CBraMod 在 TDBRAIN 上 FT-insensitive（Δ ≈ 0），在 ADFTD 上卻高度不穩（σ=3.35pp，最極端 seed 達 +4.68pp）。這可能與其 criss-cross attention 架構對 ADFTD 的光譜結構產生某種敏感性有關。

### 5.2 Behavioral-level 證據同樣支持 model × dataset 交互（F05 + F09）

Representation-level pp Δ 揭示表徵結構的重寫方向。Behavioral-level BA 則呈現相同的交互模式在下游分類器上的後果。下表統整三個 FM × 四個資料集的 frozen LP 與 fine-tuned subject-level BA（sample std，Stress n=8 frozen LP，其他 cell n=3）：

| Model | Phase | Stress | ADFTD | TDBRAIN | EEGMAT |
|---|---|---|---|---|---|
| LaBraM  | Frozen LP  | 0.605 ± 0.032 | 0.695 ± 0.006 | 0.679 ± 0.009 | 0.671 ± 0.056 |
|         | Fine-tuned | 0.524 ± 0.010 | 0.709 ± 0.014 | 0.665 ± 0.045 | 0.731 ± 0.021 |
| CBraMod | Frozen LP  | 0.452 ± 0.032 | 0.558 ± 0.022 | 0.564 ± 0.007 | **0.731 ± 0.053** |
|         | Fine-tuned | 0.548 ± 0.031 | 0.537 ± 0.027 | 0.488 ± 0.014 | 0.620 ± 0.058 |
| REVE    | Frozen LP  | 0.494 ± 0.018 | 0.692 ± 0.026 | 0.544 ± 0.009 | 0.671 ± 0.021 |
|         | Fine-tuned | 0.577 ± 0.051 | 0.658 ± 0.030 | 0.488 ± 0.019 | 0.727 ± 0.035 |

Δ (FT − Frozen) pp：

| Model | Stress | ADFTD | TDBRAIN | EEGMAT |
|---|---|---|---|---|
| LaBraM  | **−8.1** | +1.3 | −1.5 | **+6.0** |
| CBraMod | **+9.6** | −2.1 | −7.6 | **−11.1** |
| REVE    | **+8.3** | −3.5 | −5.6 | **+5.6** |

**五個觀察**：

1. **LaBraM 的 model×dataset 模式**：在 ADFTD（+1.3 pp）、EEGMAT（+6.0 pp）上 FT 為 injection，在 Stress（−8.1 pp）、TDBRAIN（−1.5 pp）上為 erosion——與 §5.1 的 representation-level pp Δ 方向一致。
2. **CBraMod / REVE 在 Stress 上表面 injection**（+9.6 / +8.3 pp）與它們在 TDBRAIN 的 erosion（−7.6 / −5.6 pp）方向相反，支持 model × dataset 是 FT 方向的決定性變因。
3. **CBraMod frozen LP 在 EEGMAT 達 0.731**（全表最高的 frozen），超越自身在其他資料集的 frozen/FT 所有 cell；但 FT 後 **反而降到 0.620（−11.1 pp）**，是整個表中最大的 erosion。這意味著 CBraMod 的 pre-trained 表徵對 EEGMAT rest/task 對比已經最優，FT 反而破壞這個結構——支持「FT 方向取決於 pre-trained 初始位置相對於資料集 label 結構的位置」的假設。
4. **REVE 在 EEGMAT 上也是 injection（+5.6 pp）**，與 LaBraM 方向一致但幅度較小；三個 FM 在 EEGMAT 上呈現 **兩 injection + 一 erosion** 的分歧，印證 §5.1 的 model-specific 結構。
5. **EEGMAT 成對設計下，LaBraM FT 0.731 與 frozen LP 0.671 差 +6 pp**；對照 F09 的 representation-level 分析（pooled label fraction 5.35% → 5.82%，bootstrap 噪聲內），顯示 BA 提升主要來自 projection 層重排，而非表徵重寫。

**但 Stress 的 BA Δ 是 surface-level 的警告訊號**：§7 將展示這些看似強的 ±8–10 pp 效應在 permutation null 下完全無法與隨機區分。此處列出只是為了完整呈現 model × dataset 交互在 behavioral-level 的存在，不作為獨立證據引用。

### 5.3 這個 taxonomy 不是樣本數產物（F04 matched-N）

我們擔心不同資料集的 N 不同會造成 FT 效應差異的假象，因此用 matched subsampling（100 draws/rung）把 ADFTD 與 TDBRAIN 壓到相同 N（例如 N=17 等於 Stress 規模），對照 LaBraM 的 FT 差異。結果在 N=17 時 ADFTD 仍有 +5.22 pp 的 injection，TDBRAIN 仍有 −1pp 左右的 erosion——**Taxonomy 是 N-invariant 的**。

### 5.4 Within-subject design 不是 subject-dominance 的萬用解（F09 + F14）

一個合理的反論是：既然 §4 指出 FM 表徵主要編碼 subject identity，那改用 within-subject design（同一受試者的兩個狀態對比）應該就能規避此問題。我們用兩個 within-subject 實驗檢驗這個假設：

- **EEGMAT 成對設計（F09）**：每位受試者同時錄製 rest 與 mental-arithmetic task，LaBraM FT 達 0.731 ± 0.021 BA（3-seed）——within-subject 對比**能**讓 FM 成功分類。
- **Stress within-subject DSS 縱貫（F14）**：以個體自身的 median DSS 為閾值，做 Leave-One-Out within-subject 分類。三個 FM × 三種分類器（centroid、1-NN、logistic）全部 ≤ chance，kappas 皆為負（supplementary Fig S4）。

**對比告訴我們什麼**：關鍵不在於「是否 within-subject」，而在於 **within-subject contrast 的性質**。EEGMAT 的 rest vs task 是 paradigm-driven 的強對比，對應明確的認知狀態切換；Stress 的 DSS trajectory 是連續、低訊號的時間漂移，同一受試者在不同日期填的 DASS 分數差異未必對應可偵測的 EEG 變化。這排除了「within-subject reframe 是普適解」的假設，也為 Stress 的 power floor 論點（§7）加上第五條獨立證據。

---

## 6. 神經科學可詮釋性（TNSRE 特色章節）

這一節是 TNSRE 讀者會最關心的：FM 到底是不是看到真實的生理結構？我們以三條獨立線索 + 一項跨資料集一致性檢驗建立證據鏈。**關鍵觀點**：真實神經訊號應該在不同 FM 架構上得出一致的 band importance；跨架構不一致則暗示 noise floor。

**空間線索（topomap）**：以 gradient-based channel importance 投影到電極配置上，右半腦與額葉節點貢獻最大，與 §4.3 的古典 alpha 側化一致。

**頻段相關性（per-band RSA）**：將輸入以 band-pass 拆解後，比較各頻段的 representation 與原始 representation 的相似度。Alpha 與 beta 頻段的相似度最高。

**頻段因果驗證（band-stop ablation）**：移除特定頻段後重新跑 frozen LP，觀察 representation 的偏移量。在 EEGMAT（清晰 rest/task 設計）上，**LaBraM 與 REVE 兩個架構都一致指向 alpha 為最受影響頻段**（LaBraM distance 0.136、REVE 0.150，皆為 4 個 bands 中最大）——跨架構收斂支持「alpha 是 EEGMAT 認知負荷任務中的真實神經簽名」此因果宣稱。

**跨架構收斂作為 signal-vs-noise 的標尺**：在 Stress 資料集上，同樣的 band-stop 分析得到**LaBraM 峰值為 beta（0.168）、REVE 峰值為 alpha（0.061），兩者不一致**。CBraMod 在兩個資料集皆峰值於 delta（可能是架構偏差，獨立議題）。按照「真實訊號應跨架構收斂」的標準，Stress 的跨模型分歧意味著：band-stop 訊號本身就已被噪聲稀釋到每個 model 抓到不同的殘餘結構，此資料集無法支持任何頻段特異性的因果宣稱。**這是論文中關於 Stress 是 statistical power floor 的第四條獨立證據**（接續 §3.2 cuDNN 飄移、§7.2 permutation null、§7.3 architecture-agnostic 三條）。

這三條線索合起來支持一個臨床有意義但 scope 明確的結論：**在 EEGMAT 上 FM 確實捕捉了跨架構一致的 alpha 頻段神經訊號**，不是單純的 subject fingerprint。§4.1 的 subject dominance 與此並不矛盾——FM 表徵中同時存在 subject 與 signal 兩個維度，前者變異大、後者 SNR 低；在 clean benchmark（EEGMAT）上後者能以 cross-model consistency 的方式被偵測出來，在 noisy benchmark（Stress）上則沉沒於 power floor 之下。

---

## 7. Stress 作為 Statistical Power Floor 警世案例

這一節貫穿前三個支柱，是論文的方法學 capstone。讀者此時已經相信我們的 subject-level 協議、相信 FM 有 representation-level 結構、相信 FT 存在 model × dataset 交互。現在我們要問：**如果只看 Stress 資料集，以上結論有哪些是可以直接重現的？**

答案是：**沒有一個**。

### 7.1 表面看起來很像「真實的 injection / erosion」（F05）

3 FM × 3 seeds 的 Stress best-HP FT 結果：

| Model | Frozen LP | Best FT | Δ | 表面模式 |
|---|---|---|---|---|
| LaBraM | 0.605 ± 0.032 | 0.524 ± 0.010 | **−8.1 pp** | erosion |
| CBraMod | 0.452 ± 0.032 | 0.548 ± 0.031 | **+9.6 pp** | injection |
| REVE | 0.494 ± 0.018 | 0.577 ± 0.051 | **+8.3 pp** | injection |

Bootstrap CI 非重疊，Cohen's d 全部 > 2.5（F15）。照統計報告格式寫下來會是「極強效應」。

### 7.2 但 permutation null 講了另一個故事（F06、F19）

- **LaBraM**：10 次 null permutation，real FT 0.443 ± 0.083 vs null FT 0.497 ± 0.086，one-sided p = 0.70。有兩個 null seed（s4=0.607、s8=0.643）高於所有真實 label 的 seed。完全無法與隨機區分。
- **CBraMod**：real 0.548 vs null 0.484，p = 0.100（10 perm 的底線）。
- **REVE**：real 0.577 vs null 0.486，p = 0.100。

三個 FM 的 null test 都落在「不顯著或邊界顯著」。結合 §7.1 的大 Cohen's d，這是典型的 **low-power regime 特徵**：效應量看起來大，但自由度太少無法把它與隨機分開。

### 7.3 連架構都不重要（F20）

我們訓練一個從頭開始的 ShallowConvNet（2017 年 Schirrmeister 的架構）與 EEGNet（Lawhern 2018），不使用任何預訓練權重。3-seed 結果：

| Model | 3-seed BA |
|---|---|
| ShallowConvNet | **0.557 ± 0.031** |
| EEGNet | 0.518 ± 0.097 |
| LaBraM FT (best HP) | 0.524 ± 0.010 |
| CBraMod FT (best HP) | 0.548 ± 0.031 |
| REVE FT (best HP) | 0.577 ± 0.051 |

ShallowConvNet 從零開始訓練 70 段錄音，達到 0.557，落在 FM FT 區間內。**FM 預訓練在 Stress 上沒有帶來任務特異的優勢**。這是「task property, not model property」的直接證據。

### 7.4 Within-subject 重新框架也救不了（F14）

§5.4 已經用 EEGMAT vs Stress 的對比建立一般原則（within-subject design 非普適解，contrast 強度才是關鍵）；在 Stress 自身 DSS trajectory 上的具體結果支撐該論點：三個 FM × 三種分類器全部 ≤ chance（詳 Fig S4）。這排除了論文策略早期最後一個「希望還活著」的方向。

### 7.5 本節結論

Stress 70rec / 14pos 在統計效力的地板之下。**五條獨立證據**匯聚至同一個結論：

1. **cuDNN 飄移**（§3.2）：單 seed 橫跨 14pp，同一配方不可重現
2. **Permutation null**（§7.2）：LaBraM p=0.70、CBraMod/REVE p=0.10（10-perm floor），real 與 null 不可區分
3. **Architecture-agnostic**（§7.3）：從頭訓練的 ShallowConvNet 0.557 與 FM FT 0.52–0.58 完全重疊，預訓練無可偵測優勢
4. **Band-stop 跨架構分歧**（§6）：LaBraM peak=beta 與 REVE peak=alpha 不一致；EEGMAT 上兩者皆收斂於 alpha，Stress 上則分歧——連頻段級的神經簽名都被噪聲稀釋
5. **Within-subject 縱貫失敗**（§5.4、§7.4）：即使控制 subject identity，DSS trajectory 的低訊號對比在三個 FM × 三個分類器上皆 ≤ chance；EEGMAT 的成功證明問題不在於 FM 而在於 Stress 的對比強度

在這個資料集上，任何關於 FM 優越性、FT 方向、架構選擇、頻段因果、within-subject 分析的聲明在統計上都無法與噪聲區分。這不是 FM 的失敗，也不是 Stress 的失敗——這是**一個小樣本 benchmark 無法支持任何 FM 相關科學宣稱**的實證。這個教訓對整個 EEG FM 領域都適用：benchmark 設計必須先報告 statistical power，才能談模型比較。

---

## 8. Discussion

### 8.1 為什麼 FT 方向會出現 model × dataset 交互？

一個可能的解釋是 backbone 架構對不同頻段與時間尺度的敏感度不同。CBraMod 的 criss-cross attention 把空間與時間的注意力拆開，與 ADFTD 的強頻段結構（delta 慢波）特別契合，但在 ADFTD 的 seed 不穩定暗示它可能過度依賴訓練軌跡。REVE 的 patch embedding 是線性的，對輸入 scale 敏感。LaBraM 的 neural tokenizer 把 EEG 離散化後再做 transformer，對 TDBRAIN 這類 MDD 資料的細微 spectral 差異反而會在 FT 時丟掉。這些都是假設，需要後續的機制研究。

### 8.2 幾條已經排除的路

- **Subject-adversarial losses**（GRL、DANN、LEAD）：λ 從 0.01 到 1.0 全面 sweep，全部劣於 baseline（F12）。移除 subject 變異會一併移除 label 訊號，因為兩者在表徵上高度糾纏。
- **Within-subject longitudinal DSS 重新框架**（F14）：三個 FM、三個分類器都失敗。
- **Sparse-label-subspace 假設**：FT 並不會把 label 訊號集中到更少的維度。這個假設被直接推翻。
- **Spectral-guided FiLM conditioning**：在預算與實驗優先級考量下已放棄。
- **Stress 作為主要證據**：如本文反覆強調，已降級為案例警示。

### 8.3 對臨床應用的意涵（TNSRE 角度）

即使 Stress benchmark 本身站不住腳，凍結的 FM 表徵仍保有 0.605 的 BA（LaBraM），這在臨床前期篩檢的情境下並非無用。但這份表徵中主導的是受試者身份而非病徵，這意味**任何臨床部署都必須做個體化校準**，不能期望一個通用分類器在新受試者身上直接可用。personalized fine-tuning 或 few-shot adaptation 是未來最務實的方向，但這需要比 70 rec 大得多的資料集才能系統性驗證——這也是我們提及 HNC（308+400 受試者的私有 Dementia+MDD 資料集）作為未來高效力對照的原因。

### 8.4 對整個領域的兩點建議

1. **Subject-level CV 必須作為 EEG 分類任務的預設**。Trial-level 只能用於與歷史文獻對照，且必須明確標示洩漏風險。
2. **報告 statistical power**。提供 n_subjects、n_positive、multi-seed std、permutation null、bootstrap CI。單 seed 數字在小樣本資料集上不可信。

---

## 9. Conclusion 與 Limitations

我們對三個主流 EEG FM 在四個資料集上進行系統性再評估，得到三個層次的發現：subject leakage 與 trial-level CV 導致文獻 BA 被系統性膨脹，在 Stress 資料集上誠實協議下的 FM 表現僅 0.45–0.60；FM 表徵主要編碼 subject identity，但仍保有古典特徵抓不到的可分類 label 子空間；FT 方向是 model × dataset 的交互作用，無普遍「有效／失效」結論。我們以 Stress 資料集作為貫穿式警世案例，證明 70 rec / 14 positive 的規模不足以支持任何 FM 優越性聲明。

**限制**：

- 跨 dataset × 3 FM 的完整矩陣只有 Stress 做完整；ADFTD 與 TDBRAIN 的 taxonomy 已跨三個模型驗證，EEGMAT 只做了 LaBraM。
- 我們提出的 model × dataset 交互的機制解釋（§8.1）目前是假設，需要後續的 architectural ablation 研究。
- HNC 私有資料集可作為未來的高效力驗證，但因為是加密資料，公開重現性受限。

---

## 10. 這份敘事自我檢查

閱讀時請確認以下幾點：

1. **三個支柱彼此獨立但互相支援**：A 建立方法學信任，B 描述表徵結構，C 是我們的核心科學貢獻。讀者可以只相信 A 就接受本文的實驗設計、只相信 A+B 就接受我們的診斷、完整相信 A+B+C 就接受我們的 taxonomy。
2. **Stress 的角色要一致**：Stress 不是主要證據，是貫穿全文的警示案例。任何時候引用 Stress 都要清楚說明「這是 power floor 下的結果」。
3. **每個宣稱都對應一個 finding ID**：全文主張可追溯到 `findings.md` 的 F01–F20。沒有 finding 支持的宣稱不應出現在論文。
4. **對 FM 的態度是中性的**：不捧也不踩。F02 說 FM 主要編碼 subject，F03 說 FM 還是比古典好，F17 說 FT 方向取決於配對——這三個加起來是一個「複雜但真實」的圖像，不是「FM 沒用」的 takedown。

---

## 11. 下一步

等 advisor 確認 D1（narrative framing）與 D2（REVE window）後，開始畫新圖（Fig 1 pipeline、Fig 2 trial-vs-subject、Fig 3 subject dominance composite、Fig 8c non-FM baselines），並依此敘事展開正式 LaTeX 草稿。

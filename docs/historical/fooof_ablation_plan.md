# FOOOF Ablation Experiment Plan — User Review Required

**Created**: 2026-04-20
**Status**: **AWAITING USER APPROVAL**. Do not execute until reviewed.
**Integration point**: Adds §6.5 to SDL paper draft, complementing existing §6.4 band-stop ablation.

---

## 一、為什麼要做這個實驗（motivation 白話版）

現有的 §6.4 用 **band-stop filter** 把 delta/theta/alpha/beta/gamma 某一段頻率移除，然後看 FM embedding 變化多少。這個答的問題是：

> 「FM 在意哪個**頻段**？」

但這還不夠 — 因為 EEG 信號裡，**同一個頻段**可以包含兩種完全不同性質的成分：

1. **Aperiodic (1/f 背景)**：整條光譜呈斜坡狀下降的部分。這是**受試者特徵**（skull 厚度、大腦 E/I balance），基本上是 trait level 的東西，個體間差異大、個體內穩定。Demuru 2020 證明單用 1/f 就能認出一個人。
2. **Periodic (震盪峰)**：疊在 1/f 背景上的突起（alpha 峰、beta 峰等）。這是**狀態相關**的 — alpha peak power 會隨閉眼/張眼、認知負荷變化；beta peak 隨肌肉/警覺度變化。

Band-stop 拆不出這兩類 — alpha band 既包含 1/f 在 10 Hz 的點，也包含 alpha peak。所以我們看到「alpha 重要」時，不知道 FM 看的是 trait (1/f @ 10Hz) 還是 state (alpha peak)。

**FOOOF ablation 回答**：
> 「FM 主要看的是 trait-like 的 1/f 背景，還是 state-like 的震盪峰？」

如果是前者 → 就是為什麼 subject dominant（SDL 的因果鏈）。
如果是後者 → 那 SDL 其實是資料的錯（受試者差異本來就主導），不是 FM 的錯。

---

## 二、核心假說

> **H1 主假說**：當 aperiodic 1/f 成分被抹除後，frozen FM embedding 對 subject 的可辨識度會大幅下降。
> 
> **量化標準**：受試者 ID 探針 balanced accuracy 從 original 基線下降 ≥ 15 pp（在 Stress + EEGMAT 至少一個資料集上）。
> 
> **白話**：1/f 抹掉 → 認不出人了 → 1/f 是指紋本體。

> **H2 對稱假說**：反過來，當 periodic 震盪峰被抹除後，subject 可辨識度**不應**大幅下降。
> 
> **量化標準**：受試者 ID 探針 bAcc 下降 < 5 pp。
> 
> **白話**：alpha/beta peak 抹掉 → 還認得出人 → 峰不是指紋。

> **H3 狀態訊號假說（選配，有風險）**：Aperiodic 抹除後，state label 的可解碼性**應該上升或持平**（因為遮蔽物消失）。
> 
> **量化標準**：EEGMAT 任務 bAcc ≥ 原始值。Stress DASS bAcc 無要求（我們已經證明基本上沒有可解碼信號）。
> 
> **白話**：1/f 是遮蔽物 → 掀開來 → 下面的狀態訊號露出來。
> 
> **風險**：signal reconstruction artifact 可能讓 FM 輸出亂跳，若沒乾淨效果不要硬報。

---

## 三、資料集選擇

只在**兩個**資料集上做，不擴張：

| 資料集 | 用來測什麼 | 為什麼選 |
|---|---|---|
| **UCSD Stress** (70 rec, 17 subj) | H1, H2（subject-ID 探針）。H3 不測（已知 DASS 無狀態訊號） | SDL 論文主角 |
| **EEGMAT** (72 rec, 36 subj) | H1, H2, H3（完整三條假說） | 正面對照；已有 alpha-ERD 強狀態訊號 |

**不加 ADFTD / TDBRAIN** — 保持實驗 scope 窄，避免 2 週內跑不完。

---

## 四、信號處理管線（技術細節）

這一部分**最容易翻車**。先寫清楚步驟，user review 可以對每一步挑錯。

### 4.1 每個 epoch 的 FOOOF fit

每段 5 秒 × 200 Hz = 1000 sample 的 epoch（或 EEGMAT 60 秒），**每個 channel 獨立**做：

```
1. Welch PSD:
   - nperseg = 512（for 5s epoch: 2.56 sec window, 約 5 次平均）
   - noverlap = 256
   - window = 'hann'
   - frequency range: 1-45 Hz (FOOOF 標準設定)

2. FOOOF fit:
   - peak_width_limits = [1, 12]
   - max_n_peaks = 6
   - min_peak_height = 0.1
   - peak_threshold = 2.0
   - aperiodic_mode = 'fixed' (slope + offset, 不用 knee)
   - fit_range = [1, 45]  Hz

3. 保存 per-channel fit:
   - aperiodic params: offset b, slope χ
   - peak params: list of (f_c, amp, bw)
   - R² goodness-of-fit
   - fit_success: True / False
```

**品管指標（必須報告）**：
- % channel-epochs with fit_success
- % fit with R² > 0.9
- Channel-epochs with R² < 0.7 flagged, 可選擇 drop 或用原始值代替

### 4.2 Aperiodic-removed 信號重建

**兩種做法，都試，選乾淨的那個**：

**方法 A — 頻域除法（建議首選）**：
```
對原始 epoch x:
  X(f) = FFT(x)
  power_aper(f) = 10^(b - χ · log10(f))        # FOOOF 的 1/f 模型
  amp_aper(f) = sqrt(power_aper(f))
  X_flat(f) = X(f) / amp_aper(f)                # 保留相位，除掉 1/f 幅度
  x_aperiodic_removed = IFFT(X_flat)
```

**方法 B — 線性濾波逼近**：
用 IIR filter 擬合 1/f slope（設計一個 lowpass 補償）。比較粗糙但對 FM 較友善。

**已知風險**：
- FOOOF 在 < 2Hz 和 > 40Hz 擬合不穩 → 限制 fit range 在 [1, 45] Hz，其他頻段不動
- FM 沒見過扁平化信號 → out of distribution，embedding 可能亂跳
- 解決方案：**報告 embedding norm 的變化**，若 norm 崩壞 > 10× 就標 failure mode，不硬算下游

### 4.3 Periodic-removed 信號重建（對照）

```
對原始 epoch x:
  X(f) = FFT(x)
  |X_new(f)|² = |X(f)|² - sum_peaks(gaussian(f; f_c_i, amp_i, bw_i))
  X_new(f) = sqrt(|X_new(f)|²) · exp(i · phase(X(f)))  # 保留相位
  x_periodic_removed = IFFT(X_new)
```

關鍵：**保留 phase，只減 power**，避免 phase artifact。若 PSD 減去 peak 後變負 → clip to 0（會產生少量失真，要報告 clip %）。

### 4.4 Both-removed (control) + Swap (optional)

**Both-removed**：aperiodic 和 periodic 都抹除 — 只留下白雜訊。此時 FM 應該什麼都認不出來。若還能認出 subject → 說明 FM 在看 phase / 其他超出 PSD 的東西，論點要重寫。

**Swap（可選，有加分）**：subject A 的 aperiodic 移植到 subject B 的信號上：
```
X_swap(f) = X_B(f) / amp_aper_B(f) · amp_aper_A(f)
```
餵給 FM，看 subject-ID 探針把它預測成 A 還是 B。若 > 80% 預測成 A → 因果證據最強。

**建議**：pilot 先不做 swap，主 H1/H2 過了再補。

---

## 五、實驗設計總表

| Condition | 目的 | 關鍵測試 |
|---|---|---|
| Original | baseline | — |
| Aperiodic-removed | H1 | subject-ID bAcc 大降？ |
| Periodic-removed | H2 對照 | subject-ID bAcc 不降？ |
| Both-removed | control | subject-ID bAcc ≈ chance？ |
| Swap A→B | 因果（選配） | FM 預測 A 比例？ |

**Datasets × FMs × Conditions × Probes**:
- 2 datasets (Stress, EEGMAT)
- 3 FMs (LaBraM, CBraMod, REVE)
- 4 conditions (主 pipeline) + 1 swap (選配)
- 2 probes (subject-ID, state-label — state-label 只在 EEGMAT 測 H3)

**主 pipeline 總 run 數**：2 × 3 × 4 = 24 個 frozen feature 抽取 job + 24 個 probe evaluation。

---

## 六、Probe 協議（對齊 §5.2 新標準）

繼承今天才鎖定的 per-window LP 協議（避免再踩 apples-to-oranges 坑）：

**Subject-ID 探針**：
- Label = patient_id (17 人 Stress, 36 人 EEGMAT)
- Multi-class LogReg，one-vs-rest
- **Split**：session-level — 每個 subject 的 recording 分 train/test 各半（subject 內 held-out session），重複 8 seed
- 報 balanced accuracy (chance = 1/17 ≈ 5.9% for Stress, 1/36 ≈ 2.8% for EEGMAT)

**State-label 探針（僅 EEGMAT）**：
- Label = rest vs arithmetic (binary, 36 subjects × 2)
- Per-window LogReg + prediction pooling（同今天 §5.2 新協議）
- Subject-level 5-fold GroupKFold，8 seed
- 報 recording-level bAcc

---

## 七、已知技術坑（白話）

| 坑 | 風險 | 緩解 |
|---|---|---|
| FOOOF fit 失敗 | 某些 channel fit 不出來，報錯或給爛 params | 先測 Stress 上的 fit success rate；若 < 90% 要先調整 fit 參數 |
| Flatten 後 FM 崩潰 | FM 沒見過扁平信號，embedding 變 NaN 或 norm 爆炸 | 先 extract 看 embedding norm 分布；若 > 10× 原始 → 標 failure，不硬算 |
| Phase 重建產生 ringing | 頻域除法可能在頻段邊界造 artifact | 用 cosine-tapered window；或在 time domain 做 detrend + highpass 逼近 |
| Periodic-removed 的 peak 殘留 | Gaussian fit 可能欠擬合，實際信號 peak 形狀非純 Gaussian | 報告 residual PSD 統計；可選用 median-smoothed baseline 替代 FOOOF fit |
| Swap 的頻帶交界 | A 和 B 的 fit range 不一樣時，交界會有 spike | 統一所有 fit range = [1, 45] Hz，且 amp_aper(f) 在範圍外設 1（不做變換） |

---

## 八、驗證步驟（一定要做，不要跳）

**Day 0 Pilot（0.5 天）**：
1. 隨機抽 3 個 Stress recording（包含 1 high-stress, 1 low-stress, 1 mid）
2. 對每個 recording 的每個 channel 做 FOOOF fit
3. **人工目視**所有 30 channel × 3 rec = 90 個 PSD + FOOOF fit 是否合理
4. 報 R² 分布、failure rate
5. 隨機抽 1 channel × 1 recording，做 aperiodic-removed 信號
6. 畫時域前後對比 + 頻域前後對比
7. **user review** — 看起來對嗎？

若 pilot 通過 → 繼續。若 pilot 不對 → 回頭調 FOOOF 參數或信號重建邏輯。

---

## 九、時程（2 週框架內）

```
Day 0 (0.5 天)  Pilot FOOOF fit + 信號重建
                USER REVIEW ← 這裡要確認才繼續
Day 1 (1 天)    Stress + EEGMAT 全部 channel-epoch FOOOF fit cache
                Aperiodic-removed / periodic-removed / both 信號 cache
Day 2 (1 天)    三個 FM × 兩個 dataset × 四個 condition frozen feature 抽取
                Embedding 品質檢查（norm、NaN）
Day 3 (0.5 天)  Subject-ID 探針 + state-label 探針
                產出 table + figure
Day 4 (0.5 天)  Swap 實驗（若主 pipeline 乾淨）
                寫 §6.5 draft
```

**總共 ~3.5 天**。若 pilot 翻車可能 +2 天 debug，仍在 2 週預算內。

---

## 十、產出（文件與圖表）

**主文**：
- §6.5 新章節 (~500 字)
- Fig 6 (新): 2 dataset × 3 FM × 4 condition subject-ID probe bar chart
- Supplementary table: 原始數字

**Supplementary**:
- FOOOF fit 品質報告
- Signal reconstruction 前後對比圖（2-3 個 sample）
- Embedding norm / NaN rate per condition

---

## 十一、失敗模式與 backup plan

| Pilot 結果 | 判定 | 對策 |
|---|---|---|
| FOOOF R² > 0.9 on > 90% channels | 🟢 繼續 | Day 1-4 全部做 |
| FOOOF R² > 0.9 on 70-90% | 🟡 繼續但標 caveat | 同上，但 results 加品質欄位 |
| FOOOF R² > 0.9 on < 70% | 🔴 暫停 | 調 fit 參數；若仍不行 → 放棄這個 experiment |
| Aperiodic-removed embedding norm 爆炸 | 🔴 暫停 | 改用 method B（linear filter）；若還不行 → 放棄 |
| 主 pipeline 做完但結果雜亂（H1/H2 都不顯著） | 🟡 降級 | 改成 supplementary，不進主文 |
| 主 pipeline 給乾淨 H1+H2 但 H3 亂 | 🟢 正常 | H3 當選配，不報 |
| 全部 H1+H2+H3 都乾淨 | 🟢 最佳 | 進主文，可能升級 Fig 6 為 money shot |

---

## 十二、Review Checkpoints（必要通過點）

User 必須在以下三個 checkpoint 簽核才繼續：

- **CP1（Day 0 結束）**：看 pilot PSD + FOOOF fit + 重建信號圖，確認品質
- **CP2（Day 1 結束）**：看 FM embedding norm 分布，確認沒崩潰
- **CP3（Day 3 結束）**：看探針結果，決定進主文還是放 supp

如果你（user）任何一個 checkpoint 覺得「算了不做」 — 可以直接 drop，不會影響其他已完成的部分（這是我特別把實驗設計成 self-contained 的原因）。

---

## 十三、審查問題清單（請你回答）

1. **Scope**：只做 Stress + EEGMAT 兩個資料集可以嗎？要不要加 ADFTD？
2. **FOOOF 參數**：peak_width_limits 和 max_n_peaks 可以依我寫的預設嗎？或你有臨床經驗傾向？
3. **信號重建方法**：優先 method A (頻域除法) 對嗎？還是你覺得 method B (IIR filter) 更穩？
4. **Swap 實驗**：主 pipeline 過了再做 swap，還是根本不做？
5. **Subject-ID 探針的 split**：per-subject session-level held-out 可以嗎？或你想要別的 CV 設計？
6. **Time budget**：3.5 天有沒有太緊？要不要把預算拉到 5 天？

---

**結案**：這份規劃是你說「不要偷懶」的產物 — 我列出了每個技術步驟、每個失敗模式、每個 checkpoint。你逐條 review，任何不同意的部分我修。同意後我才開始 Day 0 pilot。

# CLAUDE.md (Master Architecture Blueprint)

## 1. Project Overview & Core Philosophy

This project aims to decode resting-state EEG (5-minute recordings @ 200Hz) into clinical stress markers using **EEG Foundation Models (e.g., REVE, LaBraM)**.

**[MAJOR ARCHITECTURAL DECISIONS]**:
1. **No Resampling Needed**: The raw data is at 200Hz, which natively aligns with REVE's pre-training specifications.
2. **Decoupled Ordinal MTL**: Instead of standard classification, we use a dual-branch Multi-Task Learning architecture:
   - *Branch A*: Chronic/Baseline Stress Prediction (Classification to handle 8:2 class imbalance via Focal Loss).
   - *Branch B*: State Stress Severity (Ordinal Regression using MSE/Coral Loss to capture the monotonic Theta-band physiological effect).
3. **Evaluation Strategy**: We employ two distinct evaluation pipelines:
   - *Global Zero-Shot*: Leave-One-Subject-Out (LOSO) to prove cross-subject generalization.
   - *Longitudinal Few-Shot*: Using 1-shot historical data to calibrate individual baseline shifts and track subjective state fluctuations.

---

## 2. Directory Structure & Modular Design

We adopt a highly modular, config-driven design to seamlessly swap Foundation Models without rewriting the training loops.

### A. Foundation Model Extractors (`models/extractors/`)
*Strict Rule: All FM wrappers must output a unified feature tensor of shape `(Batch, Embed_Dim)`.*
* `base.py`: Defines the `BaseExtractor` interface.
* `mock_fm.py`: A dummy extractor used **strictly for MVP debugging** to avoid downloading massive weights initially.
* `reve_wrapper.py` / `labram_wrapper.py`: Wrappers for actual FMs (to be implemented via git submodules or cherry-picking).

### B. Feature Pipeline (`data/`)
* `dataset.py`: Loads `.set` files (MNE), applies Z-score normalization, and implements sliding window epoching (e.g., slicing 5 mins into `M` 10-second non-overlapping epochs).
* `samplers.py`: Handles complex subject-wise splits (LOSO) and chronological longitudinal splits (Few-Shot).

### C. Core Logic & Architecture (`models/`)
* `aggregators.py`: Temporal aggregation layer (e.g., `AdaptiveAvgPool1d(1)`) to compress `M` epoch features into a single subject-level vector `v`.
* `heads.py`: The Dual-branch MTL prediction heads (Branch A: Classification, Branch B: Ordinal Regression).

### D. Engine (`engine/`)
* `trainer.py`: PyTorch training loop with support for BF16 Mixed Precision (crucial for A100 GPUs) and W&B logging.
* `losses.py`: Custom losses (Focal Loss for Branch A, Ordinal/MSE for Branch B).

---

## 3. Development Phases (Strict Execution Order)

**AI Assistant Instructions**: Please follow these phases strictly. Do not jump to Phase 2 until Phase 1 is verified.

### Phase 1: The MVP (Minimum Viable Product)
**Goal**: Verify the plumbing (Data Loading $\rightarrow$ Epoching $\rightarrow$ Forward Pass $\rightarrow$ Dual-Branch Loss $\rightarrow$ Backward Pass) without complexities.
1. Implement basic `data/dataset.py` (MNE loading, 200Hz epoching).
2. Implement `models/extractors/mock_fm.py` (outputs random tensors of correct shape).
3. Implement the `DecoupledStressModel` with temporal aggregation and dual heads.
4. Write a simple `train_mvp.py` using a naive 80/20 random split (ignore subject-wise rules for now) and standard Loss functions to ensure gradients flow correctly.

### Phase 2: Foundation Model Integration
**Goal**: Replace the Mock FM with the real REVE model.
1. Integrate REVE weights and implement `reve_wrapper.py`.
2. Ensure the 4D Fourier Positional Encoding correctly maps the 30 channel coordinates.
3. Test a single forward pass with real data.

### Phase 3: Rigorous Evaluation Pipelines
**Goal**: Implement the clinical validation protocols.
1. Implement LOSO (Leave-One-Subject-Out) cross-validation in `data/samplers.py`.
2. Implement the Longitudinal Few-Shot Calibration script (Personalized Baseline Shift calculation).
3. Integrate advanced metrics and BF16 Mixed Precision for A100 optimization.
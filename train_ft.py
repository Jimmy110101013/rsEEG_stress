"""Finetuning with Subject-Level Stratified Group K-Fold CV.

Supports LP (frozen backbone), LoRA, and full FT modes.
Hyperparameters aligned with REVE reference (EEG-FM-Bench reve_unified.yaml).

Usage:
    # Linear probing (classification only)
    python train_ft.py --mode lp --extractor reve --folds 5

    # Full fine-tuning with subject-level DASS labels
    python train_ft.py --mode ft --extractor labram --label subject-dass \
        --aug-overlap 0.75 --folds 5 --lr 1e-5 --epochs 50 --device cuda:4

    # LoRA finetuning + MTL
    python train_ft.py --mode lora --extractor reve --folds 5 --mtl --norm none
"""

import argparse
import copy
import csv
import json
import os
import time
from collections import deque
from datetime import datetime
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
)
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader, Subset, TensorDataset

# Register extractors
import baseline.mock_fm  # noqa: F401
import baseline.reve  # noqa: F401
import baseline.labram  # noqa: F401
import baseline.cbramod  # noqa: F401
import baseline.eegnet  # noqa: F401
import baseline.shallowconvnet  # noqa: F401
import baseline.deepconvnet  # noqa: F401
import baseline.eegconformer  # noqa: F401
from baseline.abstract import create_extractor
from pipeline.dataset import StressEEGDataset, WindowDataset, stress_collate_fn, window_collate_fn
from src.loss import FocalLoss, MTLLoss, PairwiseRankingLoss
from src.model import DecoupledStressModel

# ──────────────────── Defaults (aligned with REVE reference) ─────
CSV_PATH = "data/comprehensive_labels_stress.csv"
DATA_ROOT = "data"
BATCH_SIZE = 4
N_EPOCHS = 50
PATIENCE = 15
EMBED_DIM = 512
SMA_WINDOW = 3
# ──────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="EEG Stress Finetuning (LP / LoRA)")
    # Mode
    p.add_argument("--mode", choices=["lp", "lora", "ft"], default="lp")
    p.add_argument("--extractor", default="mock_fm")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=None,
                   help="Max epochs (default: 50 for LP, 200 for LoRA)")
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--lr", type=float, default=None,
                   help="Learning rate (default: 5e-3 for LP, 5e-5 for LoRA)")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-id", default=None,
                   help="Override auto-generated run directory name "
                        "(used by HP sweeps to guarantee unique per-run dirs).")
    # Label source
    p.add_argument("--label", choices=["dass", "subject-dass", "dss"], default="dass",
                   help="dass=file DASS, subject-dass=subject-level DASS, dss=score threshold")
    p.add_argument("--threshold", type=float, default=50,
                   help="DSS threshold for --label dss (default: 50)")
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--no-bf16", action="store_true")
    # Loss / MTL
    p.add_argument("--loss", choices=["focal", "ce"], default="focal")
    p.add_argument("--mtl", action="store_true", help="Enable MTL with regression branch")
    p.add_argument("--mtl-alpha", type=float, default=1.0, help="Classification loss weight")
    p.add_argument("--mtl-beta", type=float, default=0.3, help="Regression loss weight")
    # Model
    p.add_argument("--head-hidden", type=int, default=128,
                   help="Hidden dim for MLP head (0=bare Linear)")
    p.add_argument("--dropout", type=float, default=None,
                   help="Dropout (default: 0.05 for LP, 0.5 for LoRA)")
    p.add_argument("--freeze-cls", action="store_true",
                   help="Freeze REVE cls_query_token (unfrozen by default per REVE ref)")
    p.add_argument("--norm", choices=["zscore", "none"], default="zscore")
    p.add_argument("--csv", default=CSV_PATH,
                   help="Path to labels CSV (default: comprehensive_labels_stress.csv)")
    p.add_argument("--max-duration", type=float, default=None,
                   help="Filter out recordings longer than this (seconds). Paper uses 400.")
    p.add_argument("--window-sec", type=float, default=None,
                   help="Override window size in seconds (default: from extractor config)")
    # Optimizer (REVE reference: StableAdamW betas=[0.92, 0.999], eps=1e-9, wd=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--adam-beta2", type=float, default=0.999)
    p.add_argument("--grad-clip", type=float, default=2.0,
                   help="Max grad norm (0=off)")
    # Scheduler
    p.add_argument("--warmup-epochs", type=int, default=None,
                   help="Warmup epochs (default: 3 for LP, 5 for LoRA)")
    p.add_argument("--no-scheduler", action="store_true")
    # LoRA
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--warmup-freeze-epochs", type=int, default=1,
                   help="Epochs to freeze encoder before unfreezing LoRA (LoRA mode only)")
    # Augmentation
    p.add_argument("--stride", type=float, default=None)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    p.add_argument("--aug-overlap", type=float, default=None,
                   help="Overlap fraction for increase-class augmentation (e.g. 0.75)")
    # Regularization (FT mode)
    p.add_argument("--llrd", type=float, default=1.0,
                   help="Layer-wise LR decay factor (1.0=off, 0.85 typical)")
    p.add_argument("--label-smoothing", type=float, default=0.0,
                   help="Label smoothing (0.0=off, 0.1 typical)")
    p.add_argument("--ema", type=float, default=0.0,
                   help="EMA decay (0=off, 0.999 typical)")
    p.add_argument("--channel-drop", type=float, default=0.0,
                   help="Channel dropout probability (0=off, 0.1 typical)")
    p.add_argument("--time-shift", type=int, default=0,
                   help="Max time shift in samples (0=off, 100 typical)")
    p.add_argument("--subject-loss-weight", type=float, default=0.0,
                   help="Weight for recording-level aggregated loss (LEAD-style). "
                        "0=off, 0.5 typical. Final loss = (1-w)*window_loss + w*recording_loss")
    p.add_argument("--adv-weight", type=float, default=0.0,
                   help="Max adversarial lambda for subject-adversarial training (GRL). "
                        "0=off, 0.1 recommended. Ramps from 0 via DANN sigmoid schedule.")
    # Memory
    p.add_argument("--grad-accum", type=int, default=0,
                   help="Gradient accumulation steps (0=auto: 4 for LoRA, 1 for LP)")
    # Cross-dataset support
    p.add_argument("--dataset",
                   choices=["stress", "adftd", "tdbrain", "dementia", "mdd", "eegmat",
                            "sam40", "meditation", "sleepdep"],
                   default="stress",
                   help="Dataset to use (stress=UCSD, adftd=Alzheimer's, "
                        "tdbrain=Brainclinics MDD, dementia=HNC dementia, mdd=HNC MDD, "
                        "eegmat=PhysioNet mental arithmetic, sam40=SAM40 stress, "
                        "meditation=OpenNeuro meditation, sleepdep=sleep deprivation)")
    p.add_argument("--hnc-data-root", default="data/hnc",
                   help="Directory containing HNC dementia/MDD HDF5 + .pkl files")
    p.add_argument("--hnc-channels", default=None,
                   help="Comma-separated list of 30 channel names in HDF5 axis-1 "
                        "order, e.g. 'Fp1,Fp2,F3,F4,...'. Required when "
                        "--dataset is 'dementia' or 'mdd'.")
    p.add_argument("--n-splits", type=int, default=3,
                   help="Pseudo-recording splits for ADFTD (default: 3)")
    # Feature extraction
    p.add_argument("--save-features", action="store_true",
                   help="Save test-fold features from best model for eta-squared analysis")
    # Permutation null — shuffle labels before CV to build a null distribution
    # of BA under a label-free representation.
    p.add_argument("--permute-labels", type=int, default=-1,
                   help="If >=0, permute labels with this RNG seed before CV "
                        "(permutation-null mode). -1 disables (default).")
    p.add_argument("--permute-level", choices=["recording", "subject"], default="recording",
                   help="Level at which to permute labels. 'recording' (default) shuffles "
                        "per-recording labels — correct for paired/within-subject labels "
                        "(e.g. EEGMAT, SleepDep) and per-recording scores (e.g. Stress DASS). "
                        "'subject' shuffles per-subject labels with all of a subject's "
                        "recordings sharing the shuffled label — correct for subject-trait "
                        "labels (e.g. ADFTD AD/HC). Ignored if --permute-labels < 0.")
    args = p.parse_args()

    # Mode-dependent defaults (aligned with REVE stress task config)
    if args.epochs is None:
        args.epochs = {"lp": 50, "lora": 200, "ft": 50}[args.mode]
    if args.lr is None:
        args.lr = {"lp": 5e-3, "lora": 5e-5, "ft": 1e-5}[args.mode]
    if args.dropout is None:
        args.dropout = {"lp": 0.05, "lora": 0.5, "ft": 0.0}[args.mode]
    if args.warmup_epochs is None:
        args.warmup_epochs = {"lp": 3, "lora": 5, "ft": 3}[args.mode]

    return args


# ──────────────────── Helpers ─────────────────────


def extract_test_features(model, dataset, test_idx, device, use_amp=True, ch_select_idx=None):
    """Extract pooled features from test set using the fine-tuned model.

    Returns:
        features: (N_test, embed_dim) numpy array
    """
    model.eval()
    all_feats = []
    with torch.no_grad():
        for i in test_idx:
            item = dataset[i]
            epochs = item[0]  # (M, C, T)
            if ch_select_idx is not None:
                epochs = epochs[:, ch_select_idx, :]
            M = epochs.shape[0]
            epoch_feats = []
            for start in range(0, M, 16):
                batch = epochs[start:start+16].to(device)
                with torch.autocast('cuda', dtype=torch.bfloat16, enabled=use_amp):
                    feats = model.extractor(batch)  # (sub_B, embed_dim)
                epoch_feats.append(feats.float().cpu())
            epoch_feats = torch.cat(epoch_feats, dim=0)
            pooled = epoch_feats.mean(dim=0).numpy()
            all_feats.append(pooled)
    return np.stack(all_feats)


def print_split_info(name: str, indices, labels: np.ndarray, patient_ids: np.ndarray):
    split_labels = labels[indices]
    split_pids = np.unique(patient_ids[indices])
    n0 = int((split_labels == 0).sum())
    n1 = int((split_labels == 1).sum())
    print(f"  {name:>5s}: {len(indices):>3d} samples, "
          f"{len(split_pids):>2d} subjects {sorted(split_pids.tolist())}, "
          f"label_0={n0}, label_1={n1}")


def mixup_batch(
    x: torch.Tensor, y: torch.Tensor, alpha: float
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Mixup augmentation. Returns (mixed_x, y_a, y_b, lam)."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def apply_lora(model, args):
    """Wrap REVE transformer layers with LoRA adapters."""
    from peft import LoraConfig, get_peft_model

    if not hasattr(model.extractor, "reve"):
        # MockExtractor has no transformer — apply LoRA to the whole extractor
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["proj.1"],  # MockExtractor linear layer
            bias="none",
        )
        model.extractor = get_peft_model(model.extractor, lora_config)
    else:
        lora_config = LoraConfig(
            r=args.lora_r, lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=["to_qkv", "to_out"],
            bias="none",
        )
        model.extractor.reve = get_peft_model(model.extractor.reve, lora_config)
    n_lora = sum(p.numel() for p in model.extractor.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.extractor.parameters())
    print(f"  LoRA: {n_lora:,} trainable / {n_total:,} total encoder params "
          f"({100*n_lora/n_total:.2f}%)")
    return model


def build_optimizer(model, args):
    """Build AdamW optimizer with separate LR for encoder vs head params.

    When args.llrd < 1.0, applies layer-wise learning rate decay:
    earlier layers get smaller LR (lr * encoder_lr_scale * llrd^(depth-i)).
    """
    head_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head_cls" in name or "head_reg" in name or "head_subj" in name:
            head_params.append(param)

    param_groups = [{"params": head_params, "lr": args.lr}]

    if args.llrd < 1.0 and hasattr(model.extractor, "get_layer_groups"):
        # LLRD: one param group per layer, exponentially decaying LR
        layer_groups = model.extractor.get_layer_groups()
        n_layers = len(layer_groups)
        encoder_base_lr = args.lr * args.encoder_lr_scale
        for i, group_params in enumerate(layer_groups):
            trainable = [p for p in group_params if p.requires_grad]
            if trainable:
                layer_lr = encoder_base_lr * (args.llrd ** (n_layers - 1 - i))
                param_groups.append({"params": trainable, "lr": layer_lr})
        print(f"  LLRD: {n_layers} layer groups, LR range "
              f"[{encoder_base_lr * args.llrd**(n_layers-1):.2e}, {encoder_base_lr:.2e}]")
    else:
        # Flat encoder LR
        encoder_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if "head_cls" not in name and "head_reg" not in name and "head_subj" not in name:
                encoder_params.append(param)
        if encoder_params:
            param_groups.append({
                "params": encoder_params,
                "lr": args.lr * args.encoder_lr_scale,
            })

    return torch.optim.AdamW(
        param_groups,
        betas=(0.92, args.adam_beta2),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )


def precompute_features(model, loader, device, use_amp) -> TensorDataset:
    """Run frozen backbone once, return TensorDataset of (pooled, labels, scores, pids)."""
    model.eval()
    all_pooled, all_labels, all_scores, all_pids = [], [], [], []

    with torch.no_grad():
        for epochs_batch, labels, scores, mask, pids in loader:
            epochs_batch = epochs_batch.to(device)
            mask = mask.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pooled = model.extract_pooled(epochs_batch, mask)

            all_pooled.append(pooled.float().cpu())
            all_labels.append(labels)
            all_scores.append(scores)
            all_pids.append(pids)

    return TensorDataset(
        torch.cat(all_pooled),
        torch.cat(all_labels),
        torch.cat(all_scores),
        torch.cat(all_pids),
    )


# ──────────────────── Core ────────────────────────


def train_one_fold(
    fold_idx: int,
    train_labels: np.ndarray,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Train a single fold. Returns (y_true_test, y_pred_test, best_epoch)."""
    use_amp = not args.no_bf16 and device.type == "cuda"

    # Fresh model per fold (embed_dim auto-detected from extractor)
    _extractor_kwargs = {}
    if args.dataset in ("adftd", "tdbrain", "dementia", "mdd", "eegmat",
                        "sam40", "meditation", "sleepdep") \
            and args.extractor in ("eegnet", "shallowconvnet"):
        _extractor_kwargs["n_channels"] = 19
    extractor = create_extractor(args.extractor, **_extractor_kwargs)
    # Override channel mapping for 19ch datasets
    if args.dataset in ("adftd", "tdbrain", "dementia", "mdd", "eegmat",
                        "sam40", "meditation", "sleepdep"):
        from pipeline.common_channels import COMMON_19
        if hasattr(extractor, "input_chans"):
            from baseline.labram.channel_map import get_input_chans
            extractor.input_chans = get_input_chans(COMMON_19)
        if hasattr(extractor, "set_channels"):
            extractor.set_channels(COMMON_19)
    embed_dim = extractor.embed_dim
    # Adversarial only for FT mode (LP/LoRA freeze backbone — no gradients to reverse)
    model = DecoupledStressModel(
        extractor, embed_dim=embed_dim,
        dropout=args.dropout, head_hidden=args.head_hidden,
    ).to(device)

    # Freeze / LoRA setup
    if args.mode == "lp":
        model.freeze_backbone(unfreeze_cls_query=not args.freeze_cls)
    elif args.mode == "lora":
        # Freeze everything first, then apply LoRA (which makes LoRA params trainable)
        model.freeze_backbone(unfreeze_cls_query=not args.freeze_cls)
        model = apply_lora(model, args)
        # Warmup-freeze: keep LoRA frozen for initial epochs
        if args.warmup_freeze_epochs > 0:
            for name, param in model.extractor.named_parameters():
                if param.requires_grad and "head" not in name:
                    param.requires_grad = False
            print(f"  Warmup-freeze: encoder/LoRA frozen for {args.warmup_freeze_epochs} epoch(s)")

    # Print param summary
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    if fold_idx == 0:
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Trainable: {len(trainable)} params ({n_train:,} values) | Frozen: {len(frozen)}")
        for n in trainable[:20]:
            print(f"    [trainable] {n}")
        if len(trainable) > 20:
            print(f"    ... and {len(trainable) - 20} more")

    # Loss
    counts = np.bincount(train_labels)
    class_weights = torch.tensor(
        len(train_labels) / (len(counts) * counts), dtype=torch.float32
    ).to(device)
    if args.loss == "focal":
        cls_criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    else:
        cls_criterion = nn.CrossEntropyLoss(weight=class_weights)

    if args.mtl:
        reg_criterion = PairwiseRankingLoss(margin=0.1)
        criterion = MTLLoss(cls_criterion, reg_criterion,
                            alpha=args.mtl_alpha, beta=args.mtl_beta)
    else:
        criterion = cls_criterion

    # Gradient accumulation for VRAM management (LoRA needs small effective batch)
    grad_accum = args.grad_accum if args.grad_accum > 0 else (4 if args.mode == "lora" else 1)

    # Feature caching for LP mode (frozen backbone, no noise, no unfrozen cls query)
    use_feature_cache = (args.mode == "lp" and args.noise == 0
                         and args.freeze_cls)
    if use_feature_cache:
        t_cache = time.time()
        print("  Precomputing features (frozen backbone)...", end=" ", flush=True)
        train_feat_ds = precompute_features(model, train_loader, device, use_amp)
        val_feat_ds = precompute_features(model, val_loader, device, use_amp)
        test_feat_ds = precompute_features(model, test_loader, device, use_amp)
        # Replace loaders with lightweight cached versions
        train_loader = DataLoader(train_feat_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_feat_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_feat_ds, batch_size=args.batch_size, shuffle=False)
        print(f"done ({time.time() - t_cache:.1f}s)")

    # Optimizer & scheduler
    optimizer = build_optimizer(model, args)

    warmup_scheduler = None
    if args.warmup_epochs > 0:
        # Exponential warmup per REVE reference: (10^(step/total) - 1) / 9
        total_warmup = args.warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (10 ** (min(epoch, total_warmup) / total_warmup) - 1) / 9,
        )
    plateau_scheduler = None
    if not args.no_scheduler:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
        )

    best_raw_bal_acc = 0.0
    best_state = None
    best_epoch = 0
    val_history = deque(maxlen=SMA_WINDOW)
    best_smoothed = 0.0
    no_improve = 0
    encoder_unfrozen = False
    curves = []  # (epoch, train_loss, val_loss, val_bal_acc, lr)

    for epoch in range(1, args.epochs + 1):
        # ── Warmup-freeze stage transition (LoRA mode) ──
        if (args.mode == "lora" and not encoder_unfrozen
                and epoch > args.warmup_freeze_epochs):
            for name, param in model.named_parameters():
                if "lora_" in name or "cls_query_token" in name:
                    param.requires_grad = True
            encoder_unfrozen = True
            # Add encoder params to existing optimizer (preserves head momentum)
            new_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and "head_cls" not in n and "head_reg" not in n and "head_subj" not in n]
            optimizer.add_param_group({
                "params": new_params,
                "lr": args.lr * args.encoder_lr_scale,
            })
            # Recreate schedulers on same optimizer
            if args.warmup_epochs > 0 and epoch <= args.warmup_epochs:
                remaining = max(1, args.warmup_epochs - epoch + 1)
                warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
                    optimizer,
                    lr_lambda=lambda e, r=remaining: (10 ** (min(e, r) / r) - 1) / 9,
                )
            if not args.no_scheduler:
                plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
                )
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  >>> Epoch {epoch}: encoder/LoRA unfrozen ({n_train:,} trainable values)")

        # ── Train ──
        model.train()
        train_loss, n_steps = 0.0, 0
        t0 = time.time()
        optimizer.zero_grad()

        # Epoch-wide accumulators for within-subject ranking loss
        # (pooled features are tiny — ~36 x 512 floats per epoch, safe to store)
        epoch_pooled, epoch_scores, epoch_pids = [], [], []

        for batch_idx, batch in enumerate(train_loader):
            if use_feature_cache:
                pooled, labels, scores, batch_pids = (b.to(device) for b in batch)
            else:
                epochs_batch, labels, scores, mask, batch_pids = batch
                epochs_batch = epochs_batch.to(device)
                labels = labels.to(device)
                scores = scores.to(device)
                mask = mask.to(device)
                batch_pids = batch_pids.to(device)
                if args.noise > 0:
                    epochs_batch = epochs_batch + args.noise * torch.randn_like(epochs_batch)

            use_mixup = args.mixup > 0

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if not use_feature_cache:
                    pooled = model.extract_pooled(epochs_batch, mask)

                if use_mixup:
                    lam = np.random.beta(args.mixup, args.mixup)
                    idx = torch.randperm(pooled.size(0), device=device)
                    mixed_pooled = lam * pooled + (1 - lam) * pooled[idx]
                    cls_logits, _ = model.classify(mixed_pooled)
                    loss_cls = lam * criterion.cls_criterion(cls_logits, labels) \
                             + (1 - lam) * criterion.cls_criterion(cls_logits, labels[idx]) \
                             if args.mtl else \
                             lam * criterion(cls_logits, labels) \
                             + (1 - lam) * criterion(cls_logits, labels[idx])
                else:
                    cls_logits, reg_output = model.classify(pooled)
                    if args.mtl:
                        loss_cls = criterion.cls_criterion(cls_logits, labels)
                    else:
                        loss_cls = criterion(cls_logits, labels)

            # Backward cls loss immediately (frees backbone activations)
            (loss_cls / grad_accum).backward()

            # Save detached pooled features for epoch-end ranking loss
            if args.mtl:
                epoch_pooled.append(pooled.detach())
                epoch_scores.append(scores.detach())
                epoch_pids.append(batch_pids.detach())

            train_loss += loss_cls.item()
            n_steps += 1

            # Step optimizer every grad_accum batches (or at end of epoch)
            is_accum_step = (batch_idx + 1) % grad_accum == 0 or (batch_idx + 1) == len(train_loader)
            if is_accum_step:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()

        # ── Epoch-end within-subject ranking loss ──
        # Computed on ALL training samples at once so every within-subject pair
        # contributes (not split across random batches).
        if args.mtl and len(epoch_pooled) > 1:
            cat_pooled = torch.cat(epoch_pooled).requires_grad_(True)
            cat_scores = torch.cat(epoch_scores)
            cat_pids = torch.cat(epoch_pids)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                _, reg_out = model.classify(cat_pooled)
                loss_reg = criterion.reg_criterion(
                    reg_out, cat_scores, patient_ids=cat_pids)
            if loss_reg.item() > 0:
                (criterion.beta * loss_reg).backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss_reg.item() * criterion.beta

        train_loss /= max(n_steps, 1)

        # ── Val ──
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp,
                               args.mtl, use_feature_cache)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"  Fold {fold_idx+1} | Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_bal_acc={val_metrics['bal_acc']:.4f} | "
            f"lr={current_lr:.1e} | {elapsed:.1f}s"
        )
        curves.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_metrics["loss"], "val_bal_acc": val_metrics["bal_acc"],
            "lr": current_lr,
        })

        # Scheduler
        if warmup_scheduler is not None and epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        elif plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics["bal_acc"])

        # Checkpoint
        if val_metrics["bal_acc"] > best_raw_bal_acc:
            best_raw_bal_acc = val_metrics["bal_acc"]
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        # Smoothed early stopping
        val_history.append(val_metrics["bal_acc"])
        smoothed = np.mean(val_history)
        if smoothed > best_smoothed:
            best_smoothed = smoothed
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                break

    # Reload best checkpoint
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("  [WARN] No improvement during training, using last model")

    test_results = predict(model, test_loader, device, use_amp, use_feature_cache)
    return test_results, best_epoch, curves


def evaluate(model, loader, criterion, device, use_amp,
             use_mtl=False, use_feature_cache=False) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_steps = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            if use_feature_cache:
                pooled, labels, scores, _pids = (b.to(device) for b in batch)
            else:
                epochs_batch, labels, scores, mask, _pids = batch
                epochs_batch = epochs_batch.to(device)
                labels = labels.to(device)
                scores = scores.to(device)
                mask = mask.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if use_feature_cache:
                    cls_logits, reg_output = model.classify(pooled)
                else:
                    cls_logits, reg_output = model(epochs_batch, mask)

                if use_mtl:
                    loss, _, _ = criterion(cls_logits, reg_output, labels, scores)
                else:
                    loss = criterion(cls_logits, labels)

            total_loss += loss.item()
            n_steps += 1
            all_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)

    return {
        "loss": total_loss / max(n_steps, 1),
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def predict(model, loader, device, use_amp, use_feature_cache=False) -> dict:
    """Run inference. Returns dict with y_true, y_pred, reg, scores, and window metrics."""
    model.eval()
    all_preds, all_labels, all_reg, all_scores = [], [], [], []
    all_win_correct, all_win_total = [], []

    with torch.no_grad():
        for batch in loader:
            if use_feature_cache:
                pooled, labels, scores, _pids = batch
                pooled = pooled.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    cls_logits, reg_output = model.classify(pooled)
                # No window metrics in cache mode
                all_win_correct.append(np.full(len(labels), np.nan))
                all_win_total.append(np.full(len(labels), np.nan))
            else:
                epochs_batch, labels, scores, mask, _pids = batch
                epochs_batch = epochs_batch.to(device)
                mask = mask.to(device)

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    win_logits, cls_logits, reg_output = model.predict_windows(
                        epochs_batch, mask)

                # Window-level metrics per sample
                win_preds = win_logits.argmax(dim=-1)  # (B, M)
                for i in range(len(labels)):
                    n_valid = int(mask[i].sum().item())
                    wpreds = win_preds[i, :n_valid].cpu().numpy()
                    correct = int((wpreds == labels[i].item()).sum())
                    all_win_correct.append(np.array([correct]))
                    all_win_total.append(np.array([n_valid]))

            all_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
            all_reg.append(reg_output.squeeze(-1).float().cpu().numpy())
            all_scores.append(scores.numpy())

    return {
        "y_true": np.concatenate(all_labels),
        "y_pred": np.concatenate(all_preds),
        "reg_pred": np.concatenate(all_reg),
        "scores": np.concatenate(all_scores),
        "win_correct": np.concatenate(all_win_correct),
        "win_total": np.concatenate(all_win_total),
    }


# ──────────────────── FT Mode (window-level) ────


class EMA:
    """Exponential Moving Average of model weights."""

    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.clone().detach() for k, v in model.state_dict().items()}
        self.backup = {}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v, alpha=1.0 - self.decay)

    def apply(self, model):
        self.backup = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        model.load_state_dict(self.backup)
        self.backup = {}


def evaluate_windows_ft(model, loader, criterion, device, use_amp) -> dict:
    """Window-level evaluation for FT mode."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for windows, labels, _pids, _rec_idxs in loader:
            windows, labels = windows.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                features = model.extractor(windows)
                cls_logits = model.head_cls(features)
                loss = criterion(cls_logits, labels)
            total_loss += loss.item()
            all_preds.append(cls_logits.argmax(1).cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    return {
        "loss": total_loss / max(len(loader), 1),
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
    }


def evaluate_recording_level(model, loader, device, use_amp) -> dict:
    """Aggregate window predictions to recording level via majority vote."""
    from collections import Counter
    model.eval()
    win_preds, win_labels = {}, {}

    with torch.no_grad():
        for windows, labels, _pids, rec_idxs in loader:
            windows = windows.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                features = model.extractor(windows)
                cls_logits = model.head_cls(features)
            preds = cls_logits.argmax(1).cpu().numpy()
            rec_idxs_np = rec_idxs.numpy()
            labels_np = labels.numpy()
            for i in range(len(preds)):
                ri = int(rec_idxs_np[i])
                if ri not in win_preds:
                    win_preds[ri] = []
                    win_labels[ri] = int(labels_np[i])
                win_preds[ri].append(int(preds[i]))

    y_true, y_pred = [], []
    for ri in sorted(win_preds.keys()):
        y_true.append(win_labels[ri])
        y_pred.append(Counter(win_preds[ri]).most_common(1)[0][0])

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "y_true": y_true, "y_pred": y_pred,
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def train_one_fold_ft(
    fold_idx, train_idx, val_idx, test_idx, trial_labels,
    dataset, args, device,
):
    """Full fine-tuning with window-level classification and subject-level CV."""
    import math
    use_amp = not args.no_bf16 and device.type == "cuda"

    # Build window-level datasets
    print("  Building window datasets...")
    train_win_ds = WindowDataset(dataset, train_idx, aug_overlap=args.aug_overlap,
                                  label_override=trial_labels)
    val_win_ds = WindowDataset(dataset, val_idx, aug_overlap=None,
                                label_override=trial_labels)
    test_win_ds = WindowDataset(dataset, test_idx, aug_overlap=None,
                                 label_override=trial_labels)

    if args.subject_loss_weight > 0:
        from pipeline.dataset import RecordingGroupSampler
        train_sampler = RecordingGroupSampler(
            train_win_ds, batch_size=args.batch_size,
            drop_last=True, seed=args.seed)
        train_loader = DataLoader(train_win_ds, batch_size=args.batch_size,
                                  sampler=train_sampler,
                                  collate_fn=window_collate_fn, num_workers=0)
    else:
        train_sampler = None
        train_loader = DataLoader(train_win_ds, batch_size=args.batch_size, shuffle=True,
                                  collate_fn=window_collate_fn, num_workers=0, drop_last=True)
    val_loader = DataLoader(val_win_ds, batch_size=args.batch_size, shuffle=False,
                            collate_fn=window_collate_fn, num_workers=0)
    test_loader = DataLoader(test_win_ds, batch_size=args.batch_size, shuffle=False,
                             collate_fn=window_collate_fn, num_workers=0)

    # Fresh model — all params trainable
    _extractor_kwargs = {}
    if args.dataset in ("adftd", "tdbrain", "dementia", "mdd", "eegmat",
                        "sam40", "meditation", "sleepdep") \
            and args.extractor in ("eegnet", "shallowconvnet"):
        _extractor_kwargs["n_channels"] = 19
    extractor = create_extractor(args.extractor, **_extractor_kwargs)
    # Override channel mapping for 19ch datasets (ADFTD, TDBRAIN, HNC dementia/MDD, EEGMAT)
    if args.dataset in ("adftd", "tdbrain", "dementia", "mdd", "eegmat",
                        "sam40", "meditation", "sleepdep"):
        from pipeline.common_channels import COMMON_19
        if hasattr(extractor, "input_chans"):
            from baseline.labram.channel_map import get_input_chans
            extractor.input_chans = get_input_chans(COMMON_19)
        if hasattr(extractor, "set_channels"):
            extractor.set_channels(COMMON_19)
    embed_dim = extractor.embed_dim
    n_subjects = len(np.unique(dataset.get_patient_ids())) if args.adv_weight > 0 else 0
    model = DecoupledStressModel(
        extractor, embed_dim=embed_dim,
        dropout=args.dropout, head_hidden=args.head_hidden,
        n_subjects=n_subjects,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if fold_idx == 0:
        print(f"  Trainable parameters: {n_params:,}")

    # Class weights from training windows
    from collections import Counter
    label_counts = Counter(train_win_ds.labels)
    n_total = len(train_win_ds)
    class_weights = torch.tensor([
        n_total / (2.0 * max(label_counts.get(c, 1), 1)) for c in range(2)
    ], dtype=torch.float32).to(device)
    if fold_idx == 0:
        print(f"  Class weights: {class_weights.tolist()}")

    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=class_weights,
                              label_smoothing=args.label_smoothing)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights,
                                         label_smoothing=args.label_smoothing)

    # Subject-adversarial setup
    pid_to_idx = None
    if args.adv_weight > 0:
        unique_pids = np.unique(dataset.get_patient_ids())
        pid_to_idx = {int(pid): i for i, pid in enumerate(unique_pids)}
        if fold_idx == 0:
            print(f"  Adversarial: {len(unique_pids)} subjects, max_lambda={args.adv_weight}")

    # EMA
    ema = EMA(model, decay=args.ema) if args.ema > 0 else None

    # Optimizer with dual LR
    optimizer = build_optimizer(model, args)

    # Grad accumulation
    grad_accum = args.grad_accum if args.grad_accum > 0 else 1

    # Cosine annealing with warmup
    total_steps = len(train_loader) * args.epochs // grad_accum
    warmup_steps = len(train_loader) * args.warmup_epochs // grad_accum

    def lr_lambda(step):
        if step < warmup_steps:
            return max(1e-2, step / max(warmup_steps, 1))
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_bal_acc, best_state, best_epoch = 0.0, None, 0
    val_history = deque(maxlen=SMA_WINDOW)
    best_smoothed, no_improve = 0.0, 0
    curves = []
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        train_loss, n_correct, n_total_train = 0.0, 0, 0
        t0 = time.time()
        optimizer.zero_grad()

        for step, (windows, labels, pids, rec_idxs) in enumerate(train_loader):
            windows, labels = windows.to(device), labels.to(device)
            pids = pids.to(device)
            rec_idxs = rec_idxs.to(device)

            # Channel dropout: zero random channels
            if args.channel_drop > 0:
                ch_mask = (torch.rand(windows.shape[0], windows.shape[1], 1,
                                      device=device) > args.channel_drop).float()
                windows = windows * ch_mask

            # Time shift: circular shift per sample
            if args.time_shift > 0:
                shifts = torch.randint(-args.time_shift, args.time_shift + 1,
                                       (windows.shape[0],))
                windows = torch.stack([w.roll(s.item(), dims=-1)
                                       for w, s in zip(windows, shifts)])

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                features = model.extractor(windows)
                cls_logits = model.head_cls(features)
                window_loss = criterion(cls_logits, labels)

                # Subject-adversarial loss (GRL)
                if args.adv_weight > 0:
                    from src.loss import adv_lambda_schedule
                    lambda_adv = adv_lambda_schedule(epoch, args.epochs, args.adv_weight)
                    subj_logits = model.classify_subject(features, lambda_adv)
                    subj_targets = torch.tensor(
                        [pid_to_idx[p.item()] for p in pids],
                        device=device, dtype=torch.long)
                    loss_adv = F.cross_entropy(subj_logits, subj_targets)
                    # Scale forward loss by lambda too (GRL only handles backward)
                    window_loss = window_loss + lambda_adv * loss_adv

                # LEAD-style recording-level aggregated loss
                if args.subject_loss_weight > 0:
                    unique_recs = rec_idxs.unique()
                    if len(unique_recs) > 1:
                        rec_logits_list, rec_labels_list = [], []
                        for rid in unique_recs:
                            mask = rec_idxs == rid
                            rec_logits_list.append(cls_logits[mask].mean(dim=0))
                            rec_labels_list.append(labels[mask][0])
                        rec_logits = torch.stack(rec_logits_list)
                        rec_labels = torch.stack(rec_labels_list)
                        rec_loss = criterion(rec_logits, rec_labels)
                        w = args.subject_loss_weight
                        loss = ((1 - w) * window_loss + w * rec_loss) / grad_accum
                    else:
                        loss = window_loss / grad_accum
                else:
                    loss = window_loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0 or (step + 1) == len(train_loader):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if ema is not None:
                    ema.update(model)
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            train_loss += loss.item() * grad_accum
            n_correct += (cls_logits.argmax(1) == labels).sum().item()
            n_total_train += labels.shape[0]

        train_loss /= max(len(train_loader), 1)
        train_acc = n_correct / max(n_total_train, 1)

        # Val — swap in EMA weights for evaluation
        if ema is not None:
            ema.apply(model)
        val_win = evaluate_windows_ft(model, val_loader, criterion, device, use_amp)
        val_rec = evaluate_recording_level(model, val_loader, device, use_amp)
        if ema is not None:
            ema.restore(model)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"  Fold {fold_idx+1} | Ep {epoch:>3}/{args.epochs} | "
              f"t_loss={train_loss:.4f} t_acc={train_acc:.3f} | "
              f"v_win_bal={val_win['bal_acc']:.3f} v_rec_bal={val_rec['bal_acc']:.3f} | "
              f"lr={lr:.1e} | {elapsed:.1f}s")
        curves.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_win_bal_acc": val_win["bal_acc"],
            "val_bal_acc": val_rec["bal_acc"], "lr": lr,
        })

        # Model selection on recording-level bal_acc
        val_bal = val_rec["bal_acc"]
        if val_bal > best_bal_acc:
            best_bal_acc = val_bal
            if ema is not None:
                best_state = copy.deepcopy(ema.shadow)
            else:
                best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

        val_history.append(val_bal)
        smoothed = np.mean(val_history)
        if smoothed > best_smoothed:
            best_smoothed = smoothed
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch}")
                break

    if best_state:
        model.load_state_dict(best_state)

    # Test — recording-level
    test_rec = evaluate_recording_level(model, test_loader, device, use_amp)

    # Extract test-fold features if requested
    ft_features = None
    if args.save_features:
        print(f"  Extracting test-fold features ({len(test_idx)} recordings)...")
        ft_features = extract_test_features(model, dataset, test_idx, device, use_amp)
        print(f"  Features shape: {ft_features.shape}")

    # Return in the same format as predict() for compatibility with main()
    scores = np.zeros(len(test_rec["y_true"]))
    if hasattr(dataset, 'records') and len(dataset.records) > 0:
        if "stress_score" in dataset.records[0]:
            scores = np.array([dataset.records[int(i)]["stress_score"] for i in test_idx])
    return {
        "y_true": test_rec["y_true"],
        "y_pred": test_rec["y_pred"],
        "reg_pred": np.zeros(len(test_rec["y_true"])),
        "scores": scores,
        "win_correct": np.full(len(test_rec["y_true"]), np.nan),
        "win_total": np.full(len(test_rec["y_true"]), np.nan),
        "ft_features": ft_features,
    }, best_epoch, curves


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # cuDNN determinism — needed for reproducible BA on the 70-recording
    # Stress regime, where non-deterministic conv kernels produce +/-10 pp
    # BA swings at identical HP+seed. See studies/exp03_stress_erosion/
    # ft_drift_check/ for the 0.656 -> 0.450 evidence.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print(f"Mode: {args.mode.upper()} | Extractor: {args.extractor} | Folds: {args.folds}")
    print(f"LR: {args.lr} | WD: {args.weight_decay} | Beta2: {args.adam_beta2} | "
          f"Grad clip: {args.grad_clip}")
    print(f"Head: hidden={args.head_hidden}, dropout={args.dropout} | "
          f"Norm: {args.norm} | Loss: {args.loss}")
    if args.mtl:
        print(f"MTL: alpha={args.mtl_alpha}, beta={args.mtl_beta}")
    if args.mode == "lora":
        print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, "
              f"dropout={args.lora_dropout}, encoder_lr_scale={args.encoder_lr_scale}")
        print(f"Warmup-freeze: {args.warmup_freeze_epochs} epoch(s)")
        # Force batch_size=1 for LoRA to avoid OOM, compensate with grad accumulation
        if args.batch_size > 1:
            print(f"LoRA: overriding batch_size {args.batch_size} -> 1 "
                  f"(grad_accum={args.grad_accum if args.grad_accum > 0 else 4} "
                  f"for effective batch={args.batch_size})")
            if args.grad_accum == 0:
                args.grad_accum = args.batch_size
            args.batch_size = 1
    if args.mode == "ft":
        ft_info = [f"encoder_lr_scale={args.encoder_lr_scale}"]
        if args.llrd < 1.0:
            ft_info.append(f"LLRD={args.llrd}")
        if args.label_smoothing > 0:
            ft_info.append(f"label_smooth={args.label_smoothing}")
        if args.ema > 0:
            ft_info.append(f"EMA={args.ema}")
        if args.channel_drop > 0:
            ft_info.append(f"ch_drop={args.channel_drop}")
        if args.time_shift > 0:
            ft_info.append(f"time_shift={args.time_shift}")
        print(f"Full FT: {', '.join(ft_info)}")
        if args.aug_overlap:
            print(f"Augmentation: {args.aug_overlap*100:.0f}% overlap for increase class")
    if args.stride:
        print(f"Stride: {args.stride}s")
    if args.noise > 0:
        print(f"Noise: std={args.noise}")
    if args.mixup > 0:
        print(f"Mixup: alpha={args.mixup} (embedding-level)")
    print()

    # Results directory
    label_tag = f"dss{int(args.threshold)}" if args.label == "dss" else args.label.replace("-", "")
    aug_tag = f"_aug{int(args.aug_overlap*100)}" if args.aug_overlap else ""
    ds_tag = f"_{args.dataset}" if args.dataset != "stress" else ""
    feat_tag = "_feat" if args.save_features else ""
    if args.run_id:
        run_id = args.run_id
    else:
        run_id = f"{datetime.now():%Y%m%d_%H%M}_{args.mode}_{label_tag}{aug_tag}_{args.extractor}{ds_tag}{feat_tag}"
    results_dir = os.path.join("results", run_id)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Results → {results_dir}/")

    # Window size: 5s for LaBraM, 10s for REVE/mock (from extractor config)
    # --window-sec overrides the extractor default (useful for cross-model comparison)
    from baseline.abstract.factory import EXTRACTOR_REGISTRY
    _, config_cls = EXTRACTOR_REGISTRY[args.extractor]
    window_sec = args.window_sec if args.window_sec else config_cls().window_sec

    ch_select_idx = None  # For 19ch selection when using ADFTD extractor on stress

    if args.dataset == "adftd":
        from pipeline.adftd_dataset import ADFTDDataset
        from pipeline.common_channels import COMMON_19
        from baseline.labram.channel_map import get_input_chans
        dataset = ADFTDDataset(
            "data/adftd", binary=True,
            window_sec=window_sec,
            cache_dir=f"data/cache_adftd_split3{'_nnone' if args.norm == 'none' else ''}",
            n_splits=args.n_splits,
            norm=args.norm,
        )
        patient_ids = dataset.get_patient_ids()
        labels = dataset.get_labels()
        n1, n0 = int(labels.sum()), int(len(labels) - labels.sum())
        print(f"ADFTD: {len(dataset)} recordings, {len(np.unique(patient_ids))} subjects "
              f"(AD={n1}, HC={n0})")
        # Override label arg since ADFTD has its own labels
        args.label = "adftd"
    elif args.dataset == "eegmat":
        from pipeline.eegmat_dataset import EEGMATDataset
        dataset = EEGMATDataset(
            "data/eegmat",
            target_sfreq=200.0,
            window_sec=window_sec,
            norm=args.norm,
            cache_dir="data/cache_eegmat",
        )
        patient_ids = dataset.get_patient_ids()
        labels = dataset.get_labels()
        n1, n0 = int(labels.sum()), int(len(labels) - labels.sum())
        print(f"EEGMAT: {len(dataset)} recordings, {len(np.unique(patient_ids))} subjects "
              f"(task={n1}, rest={n0})")
        args.label = "eegmat"
    elif args.dataset == "tdbrain":
        from pipeline.tdbrain_dataset import TDBRAINDataset
        # TDBRAIN already has natural multi-recording structure (EO+EC, multi-session)
        # so we don't create pseudo-recording splits — always n_splits=1
        dataset = TDBRAINDataset(
            "data/tdbrain",
            target_sfreq=200.0,
            window_sec=window_sec,
            norm=args.norm,
            condition="both",
            target_dx="MDD",
            cache_dir="data/cache_tdbrain",
            n_splits=1,
        )
        patient_ids = dataset.get_patient_ids()
        labels = dataset.get_labels()
        args.label = "tdbrain"
    elif args.dataset in ("dementia", "mdd"):
        from pipeline.hnc_dataset import HNCDataset
        # HNC private datasets — Dementia (3-class collapsed to Control vs
        # Dementia) and MDD (binary). Provider train/valid/test splits are
        # concatenated and re-folded subject-level by the CV loop below.
        if not args.hnc_channels:
            raise SystemExit(
                "--hnc-channels is required when --dataset is dementia/mdd. "
                "Pass the 30 channel names from the data owner in HDF5 axis-1 "
                "order, e.g. --hnc-channels 'Fp1,Fp2,F3,F4,F7,F8,Fz,...'"
            )
        hnc_channel_names = [c.strip() for c in args.hnc_channels.split(",")]
        dataset = HNCDataset(
            name=args.dataset,
            data_root=args.hnc_data_root,
            channel_names=hnc_channel_names,
            target_sfreq=200.0,
            window_sec=window_sec,
            norm=args.norm,
            binary=True,
            cache_dir=f"data/cache_hnc_{args.dataset}",
        )
        patient_ids = dataset.get_patient_ids()
        labels = dataset.get_labels()
        args.label = f"hnc-{args.dataset}"
    elif args.dataset == "sam40":
        from pipeline.sam40_dataset import SAM40Dataset
        dataset = SAM40Dataset(
            "data/sam40/Data/filtered_data",
            target_sfreq=200.0,
            window_sec=window_sec,
            stride_sec=2.5,
            norm=args.norm,
            cache_dir=f"data/cache_sam40{'_nnone' if args.norm == 'none' else ''}",
        )
        patient_ids = dataset.get_patient_ids()
        labels = dataset.get_labels()
        n1, n0 = int(labels.sum()), int(len(labels) - labels.sum())
        print(f"SAM40: {len(dataset)} recordings, {len(np.unique(patient_ids))} subjects "
              f"(stress={n1}, relax={n0})")
        args.label = "sam40"
        # Pre-build cache (WindowDataset in FT mode needs it)
        for rec in dataset.records:
            dataset._preprocess(rec)
    elif args.dataset == "meditation":
        from pipeline.meditation_dataset import MeditationDataset
        dataset = MeditationDataset(
            "data/meditation",
            target_sfreq=200.0,
            window_sec=window_sec,
            norm=args.norm,
            cache_dir=f"data/cache_meditation{'_nnone' if args.norm == 'none' else ''}",
            crop_sec=300.0,
        )
        patient_ids = dataset.get_patient_ids()
        labels = dataset.get_labels()
        n1, n0 = int(labels.sum()), int(len(labels) - labels.sum())
        print(f"Meditation: {len(dataset)} recordings, {len(np.unique(patient_ids))} subjects "
              f"(expert={n1}, novice={n0})")
        args.label = "meditation"
        for rec in dataset.records:
            dataset._preprocess(rec)
    elif args.dataset == "sleepdep":
        from pipeline.sleepdep_dataset import SleepDepDataset
        dataset = SleepDepDataset(
            "data/sleep_deprivation",
            target_sfreq=200.0,
            window_sec=window_sec,
            norm=args.norm,
            cache_dir=f"data/cache_sleepdep{'_nnone' if args.norm == 'none' else ''}",
        )
        patient_ids = dataset.get_patient_ids()
        labels = dataset.get_labels()
        n1, n0 = int(labels.sum()), int(len(labels) - labels.sum())
        print(f"SleepDep: {len(dataset)} recordings, {len(np.unique(patient_ids))} subjects "
              f"(SD={n1}, NS={n0})")
        args.label = "sleepdep"
        for rec in dataset.records:
            dataset._preprocess(rec)
    else:
        dataset = StressEEGDataset(args.csv, DATA_ROOT, window_sec=window_sec,
                                   stride_sec=args.stride, norm=args.norm,
                                   max_duration=args.max_duration)
        patient_ids = dataset.get_patient_ids()

        if args.label == "dss":
            threshold_norm = args.threshold / 100.0
            labels = np.array([
                1 if r["stress_score"] >= threshold_norm else 0
                for r in dataset.records
            ])
            n_high, n_low = int(labels.sum()), int(len(labels) - labels.sum())
            print(f"DSS labels (>={args.threshold}): high={n_high}, low={n_low}")
        elif args.label == "subject-dass":
            increase_pids = set(
                r["patient_id"] for r in dataset.records if r["baseline_label"] == 1
            )
            labels = np.array([
                1 if r["patient_id"] in increase_pids else 0
                for r in dataset.records
            ])
            n1, n0 = int(labels.sum()), int(len(labels) - labels.sum())
            print(f"Subject-DASS labels: increase={n1} ({len(increase_pids)} subjects), normal={n0}")
        else:
            labels = dataset.get_labels()

    if args.permute_labels >= 0:
        rng = np.random.default_rng(args.permute_labels)
        pid_arr = np.asarray(patient_ids)
        if args.permute_level == "subject":
            # Subject-trait null (e.g. ADFTD AD/HC): each subject carries one
            # trait label shared by all their recordings; shuffling must happen
            # at subject level so a subject's splits stay label-consistent.
            unique_pids = np.unique(pid_arr)
            subj_labels = np.empty(len(unique_pids), dtype=labels.dtype)
            for i, pid in enumerate(unique_pids):
                mask = pid_arr == pid
                rec_labels = labels[mask]
                if len(np.unique(rec_labels)) != 1:
                    raise ValueError(
                        f"--permute-level=subject requires each subject to have a "
                        f"single trait label, but subject {pid} has "
                        f"{np.unique(rec_labels).tolist()}."
                    )
                subj_labels[i] = rec_labels[0]
            shuffled = subj_labels[rng.permutation(len(unique_pids))]
            pid_to_label = dict(zip(unique_pids.tolist(), shuffled.tolist()))
            labels = np.array([pid_to_label[pid] for pid in pid_arr.tolist()],
                              dtype=labels.dtype)
            print(f"[permute-labels] Subject-level shuffle with rng seed "
                  f"{args.permute_labels} over {len(unique_pids)} subjects. "
                  f"Class balance preserved: {int(labels.sum())} pos / "
                  f"{int(len(labels)-labels.sum())} neg")
        else:
            labels = labels[rng.permutation(len(labels))]
            print(f"[permute-labels] Recording-level shuffle with rng seed "
                  f"{args.permute_labels}. Class balance preserved: "
                  f"{int(labels.sum())} pos / {int(len(labels)-labels.sum())} neg")

    outer_cv = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )

    global_results = []

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(dataset)), labels, groups=patient_ids)
    ):
        inner_cv = StratifiedGroupKFold(
            n_splits=args.folds - 1, shuffle=True, random_state=args.seed
        )
        train_inner_idx, val_inner_idx = next(
            inner_cv.split(
                np.zeros(len(trainval_idx)),
                labels[trainval_idx],
                groups=patient_ids[trainval_idx],
            )
        )
        train_idx = trainval_idx[train_inner_idx]
        val_idx = trainval_idx[val_inner_idx]

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{args.folds}")
        print(f"{'='*60}")
        print_split_info("Train", train_idx, labels, patient_ids)
        print_split_info("Val", val_idx, labels, patient_ids)
        print_split_info("Test", test_idx, labels, patient_ids)

        for name, idx in [("Val", val_idx), ("Test", test_idx)]:
            if len(np.unique(labels[idx])) < 2:
                print(f"  [WARN] {name} set has only one class")

        if args.mode == "ft":
            test_results, best_epoch, curves = train_one_fold_ft(
                fold_idx, train_idx, val_idx, test_idx, labels,
                dataset, args, device,
            )
        else:
            train_loader = DataLoader(
                Subset(dataset, train_idx),
                batch_size=args.batch_size, shuffle=True,
                collate_fn=stress_collate_fn, num_workers=0,
            )
            val_loader = DataLoader(
                Subset(dataset, val_idx),
                batch_size=args.batch_size, shuffle=False,
                collate_fn=stress_collate_fn, num_workers=0,
            )
            test_loader = DataLoader(
                Subset(dataset, test_idx),
                batch_size=args.batch_size, shuffle=False,
                collate_fn=stress_collate_fn, num_workers=0,
            )

            test_results, best_epoch, curves = train_one_fold(
                fold_idx, labels[train_idx],
                train_loader, val_loader, test_loader,
                args, device,
            )
        global_results.append(test_results)

        # Save test-fold features if extracted
        if test_results.get("ft_features") is not None:
            feat_path = os.path.join(results_dir, f"fold{fold_idx+1}_features.npz")
            np.savez_compressed(
                feat_path,
                features=test_results["ft_features"],
                labels=labels[test_idx],
                patient_ids=patient_ids[test_idx],
                test_idx=test_idx,
            )
            print(f"  Saved features → {feat_path}")

        # Save curves
        curves_path = os.path.join(results_dir, f"curves_fold{fold_idx+1}.csv")
        with open(curves_path, "w", newline="") as f:
            if curves:
                w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
                w.writeheader()
                w.writerows(curves)

        # Save per-sample predictions
        test_pids = patient_ids[test_idx]
        pred_path = os.path.join(results_dir, "predictions.csv")
        write_header = fold_idx == 0
        with open(pred_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["fold", "patient_id", "y_true", "y_pred", "stress_score", "reg_pred"])
            for i in range(len(test_results["y_true"])):
                w.writerow([
                    fold_idx + 1, int(test_pids[i]),
                    int(test_results["y_true"][i]), int(test_results["y_pred"][i]),
                    f"{test_results['scores'][i]:.4f}", f"{test_results['reg_pred'][i]:.4f}",
                ])

        # Save window metrics
        win_path = os.path.join(results_dir, "window_metrics.csv")
        with open(win_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["fold", "patient_id", "y_true", "subject_pred",
                            "n_windows", "win_correct", "win_acc"])
            for i in range(len(test_results["y_true"])):
                wc = test_results["win_correct"][i]
                wt = test_results["win_total"][i]
                wacc = wc / wt if wt > 0 and not np.isnan(wt) else np.nan
                w.writerow([
                    fold_idx + 1, int(test_pids[i]),
                    int(test_results["y_true"][i]), int(test_results["y_pred"][i]),
                    int(wt) if not np.isnan(wt) else "", int(wc) if not np.isnan(wc) else "",
                    f"{wacc:.4f}" if not np.isnan(wacc) else "",
                ])

        fold_acc = accuracy_score(test_results["y_true"], test_results["y_pred"])
        fold_bal = balanced_accuracy_score(test_results["y_true"], test_results["y_pred"])
        fold_msg = f"  -> Test (best @ epoch {best_epoch}): acc={fold_acc:.4f}, bal_acc={fold_bal:.4f}"

        # Window BA for this fold
        wc_all = test_results["win_correct"]
        wt_all = test_results["win_total"]
        if not np.any(np.isnan(wt_all)):
            total_wc = int(wc_all.sum())
            total_wt = int(wt_all.sum())
            fold_msg += f", win_acc={total_wc/total_wt:.4f} ({total_wc}/{total_wt})"

        if args.mtl:
            from scipy.stats import kendalltau
            tau, _ = kendalltau(test_results["scores"], test_results["reg_pred"])
            fold_msg += f", tau={tau:.4f}"
        print(fold_msg)

    # ── Global Aggregation ──
    y_true_all = np.concatenate([r["y_true"] for r in global_results])
    y_pred_all = np.concatenate([r["y_pred"] for r in global_results])

    print(f"\n{'='*60}")
    print(f"Global Results ({args.folds}-Fold, subject-level CV)")
    print(f"Mode: {args.mode.upper()} | Extractor: {args.extractor} | Loss: {args.loss}"
          f"{' + MTL' if args.mtl else ''}")
    print(f"{'='*60}")

    subj_acc = accuracy_score(y_true_all, y_pred_all)
    subj_bal = balanced_accuracy_score(y_true_all, y_pred_all)
    subj_f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    subj_kappa = cohen_kappa_score(y_true_all, y_pred_all)

    print(f"  {'acc':>15s}: {subj_acc:.4f}")
    print(f"  {'bal_acc':>15s}: {subj_bal:.4f}")
    print(f"  {'f1':>15s}: {subj_f1:.4f}")
    print(f"  {'kappa':>15s}: {subj_kappa:.4f}")

    # Window-level global metrics
    wc_global = np.concatenate([r["win_correct"] for r in global_results])
    wt_global = np.concatenate([r["win_total"] for r in global_results])
    yt_global = np.concatenate([r["y_true"] for r in global_results])
    if not np.any(np.isnan(wt_global)):
        # Compute window-level balanced accuracy
        win_preds_flat, win_labels_flat = [], []
        for i in range(len(yt_global)):
            n = int(wt_global[i])
            c = int(wc_global[i])
            # c windows predicted correct (=label), n-c predicted wrong (=1-label)
            win_preds_flat.extend([int(yt_global[i])] * c + [1 - int(yt_global[i])] * (n - c))
            win_labels_flat.extend([int(yt_global[i])] * n)
        win_bal = balanced_accuracy_score(win_labels_flat, win_preds_flat)
        win_acc = sum(wc_global) / sum(wt_global)
        print(f"  {'window_acc':>15s}: {win_acc:.4f} ({int(sum(wc_global))}/{int(sum(wt_global))})")
        print(f"  {'window_bal_acc':>15s}: {win_bal:.4f}")

    if args.mtl:
        from scipy.stats import kendalltau
        reg_all = np.concatenate([r["reg_pred"] for r in global_results])
        scores_all = np.concatenate([r["scores"] for r in global_results])
        tau, p_val = kendalltau(scores_all, reg_all)
        print(f"  {'kendall_tau':>15s}: {tau:.4f} (p={p_val:.4f})")
    print(f"  Total predictions: {len(y_true_all)} (should equal dataset size: {len(dataset)})")

    # ── Save summary.json ──
    summary = {
        "subject_acc": round(subj_acc, 4),
        "subject_bal_acc": round(subj_bal, 4),
        "subject_f1": round(subj_f1, 4),
        "subject_kappa": round(subj_kappa, 4),
        "n_samples": len(y_true_all),
    }
    if not np.any(np.isnan(wt_global)):
        summary["window_acc"] = round(float(win_acc), 4)
        summary["window_bal_acc"] = round(float(win_bal), 4)
        summary["total_windows"] = int(sum(wt_global))
    if args.mtl:
        summary["kendall_tau"] = round(float(tau), 4)

    # Per-subject breakdown from predictions.csv
    import pandas as pd
    pred_df = pd.read_csv(os.path.join(results_dir, "predictions.csv"))
    subj_df = pred_df.groupby("patient_id").apply(
        lambda g: pd.Series({
            "n_samples": len(g),
            "n_correct": int((g["y_true"] == g["y_pred"]).sum()),
            "acc": round((g["y_true"] == g["y_pred"]).mean(), 4),
        }), include_groups=False
    ).reset_index()
    subj_df.to_csv(os.path.join(results_dir, "subject_breakdown.csv"), index=False)
    summary["per_subject"] = subj_df.to_dict(orient="records")

    with open(os.path.join(results_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nResults saved to {results_dir}/")


if __name__ == "__main__":
    main()

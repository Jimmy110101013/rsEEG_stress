"""Finetuning with Subject-Level Stratified Group K-Fold CV.

Supports LP (frozen backbone) and LoRA modes, with optional MTL regression branch.
Hyperparameters aligned with REVE reference (EEG-FM-Bench reve_unified.yaml).

Usage:
    # Linear probing (classification only)
    python train_ft.py --mode lp --extractor reve --folds 5

    # LP + MTL (both classification and regression branches)
    python train_ft.py --mode lp --extractor reve --folds 5 --mtl

    # LoRA finetuning + MTL
    python train_ft.py --mode lora --extractor reve --folds 5 --mtl --norm none
"""

import argparse
import copy
import time
from collections import deque
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
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
from baseline.abstract import create_extractor
from pipeline.dataset import StressEEGDataset, stress_collate_fn
from src.loss import FocalLoss, MTLLoss, PairwiseRankingLoss
from src.model import DecoupledStressModel

# ──────────────────── Defaults (REVE reference) ─────
CSV_PATH = "data/comprehensive_labels.csv"
DATA_ROOT = "data"
BATCH_SIZE = 4
LR = 2.4e-4
N_EPOCHS = 50
PATIENCE = 15
EMBED_DIM = 512
SMA_WINDOW = 3
# ──────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="EEG Stress Finetuning (LP / LoRA)")
    # Mode
    p.add_argument("--mode", choices=["lp", "lora"], default="lp")
    p.add_argument("--extractor", default="mock_fm")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
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
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--unfreeze-cls", action="store_true",
                   help="Unfreeze REVE cls_query_token")
    p.add_argument("--norm", choices=["zscore", "none"], default="zscore")
    # Optimizer (REVE reference)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--adam-beta2", type=float, default=0.95)
    p.add_argument("--grad-clip", type=float, default=1.0,
                   help="Max grad norm (0=off)")
    # Scheduler
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--no-scheduler", action="store_true")
    # LoRA
    p.add_argument("--lora-r", type=int, default=8)
    p.add_argument("--lora-alpha", type=int, default=16)
    p.add_argument("--lora-dropout", type=float, default=0.0)
    p.add_argument("--encoder-lr-scale", type=float, default=0.1)
    p.add_argument("--warmup-freeze-epochs", type=int, default=1,
                   help="Epochs to freeze encoder before unfreezing LoRA (LoRA mode only)")
    # Augmentation
    p.add_argument("--stride", type=float, default=None)
    p.add_argument("--noise", type=float, default=0.0)
    p.add_argument("--mixup", type=float, default=0.0)
    return p.parse_args()


# ──────────────────── Helpers ─────────────────────


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
    """Build AdamW optimizer with separate LR for encoder vs head params."""
    head_params = []
    encoder_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "head_cls" in name or "head_reg" in name:
            head_params.append(param)
        else:
            encoder_params.append(param)

    param_groups = [{"params": head_params, "lr": args.lr}]
    if encoder_params:
        param_groups.append({
            "params": encoder_params,
            "lr": args.lr * args.encoder_lr_scale,
        })

    return torch.optim.AdamW(
        param_groups,
        betas=(0.9, args.adam_beta2),
        eps=1e-9,
        weight_decay=args.weight_decay,
    )


def precompute_features(model, loader, device, use_amp) -> TensorDataset:
    """Run frozen backbone once, return TensorDataset of (pooled, labels, scores)."""
    model.eval()
    all_pooled, all_labels, all_scores = [], [], []

    with torch.no_grad():
        for epochs_batch, labels, scores, mask in loader:
            epochs_batch = epochs_batch.to(device)
            mask = mask.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pooled = model.extract_pooled(epochs_batch, mask)

            all_pooled.append(pooled.float().cpu())
            all_labels.append(labels)
            all_scores.append(scores)

    return TensorDataset(
        torch.cat(all_pooled),
        torch.cat(all_labels),
        torch.cat(all_scores),
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

    # Fresh model per fold
    extractor = create_extractor(args.extractor, embed_dim=EMBED_DIM)
    model = DecoupledStressModel(
        extractor, embed_dim=EMBED_DIM,
        dropout=args.dropout, head_hidden=args.head_hidden,
    ).to(device)

    # Freeze / LoRA setup
    if args.mode == "lp":
        model.freeze_backbone(unfreeze_cls_query=args.unfreeze_cls)
    elif args.mode == "lora":
        # Freeze everything first, then apply LoRA (which makes LoRA params trainable)
        model.freeze_backbone(unfreeze_cls_query=args.unfreeze_cls)
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

    # Feature caching for LP mode (frozen backbone, no noise, no unfrozen cls query)
    use_feature_cache = (args.mode == "lp" and args.noise == 0
                         and not args.unfreeze_cls)
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
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0,
            total_iters=args.warmup_epochs,
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

    for epoch in range(1, args.epochs + 1):
        # ── Warmup-freeze stage transition (LoRA mode) ──
        if (args.mode == "lora" and not encoder_unfrozen
                and epoch > args.warmup_freeze_epochs):
            for name, param in model.named_parameters():
                if "lora_" in name or "cls_query_token" in name:
                    param.requires_grad = True
            encoder_unfrozen = True
            # Rebuild optimizer with encoder params now active
            optimizer = build_optimizer(model, args)
            if warmup_scheduler is not None and epoch <= args.warmup_epochs:
                warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                    optimizer, start_factor=0.1, end_factor=1.0,
                    total_iters=max(1, args.warmup_epochs - epoch + 1),
                )
            if plateau_scheduler is not None:
                plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer, mode="max", factor=0.5, patience=5, min_lr=1e-6,
                )
            n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"  >>> Epoch {epoch}: encoder/LoRA unfrozen ({n_train:,} trainable values)")

        # ── Train ──
        model.train()
        train_loss, n_steps = 0.0, 0
        t0 = time.time()

        for batch in train_loader:
            if use_feature_cache:
                pooled, labels, scores = (b.to(device) for b in batch)
            else:
                epochs_batch, labels, scores, mask = batch
                epochs_batch = epochs_batch.to(device)
                labels = labels.to(device)
                scores = scores.to(device)
                mask = mask.to(device)
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

                    if args.mtl:
                        # Ranking loss on unmixed features (mixup destroys ordinal signal)
                        _, reg_output = model.classify(pooled)
                        loss_reg = criterion.reg_criterion(reg_output, scores)
                        loss = criterion.alpha * loss_cls + criterion.beta * loss_reg
                    else:
                        loss = loss_cls
                else:
                    cls_logits, reg_output = model.classify(pooled)
                    if args.mtl:
                        loss, _, _ = criterion(cls_logits, reg_output, labels, scores)
                    else:
                        loss = criterion(cls_logits, labels)

            optimizer.zero_grad()
            loss.backward()

            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            train_loss += loss.item()
            n_steps += 1

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
    return test_results, best_epoch


def evaluate(model, loader, criterion, device, use_amp,
             use_mtl=False, use_feature_cache=False) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_steps = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            if use_feature_cache:
                pooled, labels, scores = (b.to(device) for b in batch)
            else:
                epochs_batch, labels, scores, mask = batch
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
    """Run inference. Returns dict with y_true, y_pred, and optionally reg/scores."""
    model.eval()
    all_preds, all_labels, all_reg, all_scores = [], [], [], []

    with torch.no_grad():
        for batch in loader:
            if use_feature_cache:
                pooled, labels, scores = batch
                pooled = pooled.to(device)
            else:
                epochs_batch, labels, scores, mask = batch
                epochs_batch = epochs_batch.to(device)
                mask = mask.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                if use_feature_cache:
                    cls_logits, reg_output = model.classify(pooled)
                else:
                    cls_logits, reg_output = model(epochs_batch, mask)

            all_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
            all_labels.append(labels.numpy())
            all_reg.append(reg_output.squeeze(-1).float().cpu().numpy())
            all_scores.append(scores.numpy())

    return {
        "y_true": np.concatenate(all_labels),
        "y_pred": np.concatenate(all_preds),
        "reg_pred": np.concatenate(all_reg),
        "scores": np.concatenate(all_scores),
    }


def main():
    args = parse_args()
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
    if args.stride:
        print(f"Stride: {args.stride}s")
    if args.noise > 0:
        print(f"Noise: std={args.noise}")
    if args.mixup > 0:
        print(f"Mixup: alpha={args.mixup} (embedding-level)")
    print()

    dataset = StressEEGDataset(CSV_PATH, DATA_ROOT, stride_sec=args.stride, norm=args.norm)
    labels = dataset.get_labels()
    patient_ids = dataset.get_patient_ids()

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

        test_results, best_epoch = train_one_fold(
            fold_idx, labels[train_idx],
            train_loader, val_loader, test_loader,
            args, device,
        )
        global_results.append(test_results)

        fold_acc = accuracy_score(test_results["y_true"], test_results["y_pred"])
        fold_bal = balanced_accuracy_score(test_results["y_true"], test_results["y_pred"])
        fold_msg = f"  -> Test (best @ epoch {best_epoch}): acc={fold_acc:.4f}, bal_acc={fold_bal:.4f}"
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
    print(f"  {'acc':>12s}: {accuracy_score(y_true_all, y_pred_all):.4f}")
    print(f"  {'bal_acc':>12s}: {balanced_accuracy_score(y_true_all, y_pred_all):.4f}")
    print(f"  {'f1':>12s}: {f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0):.4f}")
    print(f"  {'kappa':>12s}: {cohen_kappa_score(y_true_all, y_pred_all):.4f}")
    if args.mtl:
        from scipy.stats import kendalltau
        reg_all = np.concatenate([r["reg_pred"] for r in global_results])
        scores_all = np.concatenate([r["scores"] for r in global_results])
        tau, p_val = kendalltau(scores_all, reg_all)
        print(f"  {'kendall_tau':>12s}: {tau:.4f} (p={p_val:.4f})")
    print(f"  Total predictions: {len(y_true_all)} (should equal dataset size: {len(dataset)})")


if __name__ == "__main__":
    main()

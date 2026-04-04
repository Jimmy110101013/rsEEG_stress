"""Trial-Level Classification with Full Fine-Tuning Support.

Supports two modes:
  lp  — Linear probing (frozen backbone, precomputed features)
  ft  — Full fine-tuning (end-to-end, window-level training)

Labels:
  dass — DASS-21 group labels (increase=1, normal=0)
  dss  — DSS score threshold (Stress_Score >= threshold)

Paper reproduction (arXiv 2505.23042):
  python train_trial.py --mode ft --extractor labram --label dass --folds 5 \
      --aug-overlap 0.75 --lr 1e-5 --weight-decay 0.05 --epochs 50 \
      --batch-size 32 --warmup-epochs 3 --device cuda:4

Usage:
  # LP baseline
  python train_trial.py --mode lp --extractor labram --label dass --folds 5 --device cuda:4

  # Full FT (paper reproduction)
  python train_trial.py --mode ft --extractor labram --label dass --folds 5 \
      --aug-overlap 0.75 --lr 1e-5 --epochs 50 --batch-size 32 --device cuda:4
"""

import argparse
import copy
import csv
import json
import math
import os
import time
from collections import Counter, deque
from datetime import datetime
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
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset, TensorDataset

# Register extractors
import baseline.mock_fm  # noqa: F401
import baseline.reve  # noqa: F401
import baseline.labram  # noqa: F401
import baseline.cbramod  # noqa: F401
from baseline.abstract import create_extractor
from pipeline.dataset import (
    StressEEGDataset,
    WindowDataset,
    stress_collate_fn,
    window_collate_fn,
)
from src.loss import FocalLoss
from src.model import DecoupledStressModel

# ──────────────────── Defaults ────────────────────
CSV_PATH = "data/comprehensive_labels_stress.csv"
DATA_ROOT = "data"
SMA_WINDOW = 3
# ──────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Trial-Level Classification")
    p.add_argument("--mode", choices=["lp", "ft"], default="ft",
                   help="lp=frozen backbone, ft=full fine-tuning")
    p.add_argument("--label", choices=["dass", "subject-dass", "dss"], default="dass",
                   help="dass=file-path DASS, subject-dass=subject-level DASS, dss=score threshold")
    p.add_argument("--extractor", default="labram")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--loss", choices=["focal", "ce"], default="focal")
    p.add_argument("--norm", choices=["zscore", "none"], default="zscore")
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-epochs", type=int, default=3)
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--threshold", type=float, default=60,
                   help="DSS score threshold (only used with --label dss)")
    p.add_argument("--aug-overlap", type=float, default=None,
                   help="Overlap fraction for increase-class augmentation (e.g. 0.75)")
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--subject-loss-weight", type=float, default=0.0,
                   help="Weight for recording-level aggregated loss (LEAD-style). "
                        "0=off, 0.5 typical. Final loss = (1-w)*window_loss + w*recording_loss")
    p.add_argument("--csv", default=CSV_PATH,
                   help="Path to labels CSV (default: comprehensive_labels_stress.csv)")
    p.add_argument("--max-duration", type=float, default=None,
                   help="Filter out recordings longer than this (seconds). Paper uses 400.")
    return p.parse_args()


# ──────────────────── Helpers ─────────────────────


def print_split_info(name: str, indices, labels: np.ndarray):
    split_labels = labels[indices]
    n0 = int((split_labels == 0).sum())
    n1 = int((split_labels == 1).sum())
    print(f"  {name:>5s}: {len(indices):>3d} trials, label_0={n0}, label_1={n1}")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_labels(dataset, label_mode, threshold_norm):
    """Get trial-level labels from dataset records."""
    if label_mode == "dass":
        return np.array([r["baseline_label"] for r in dataset.records])
    elif label_mode == "subject-dass":
        # Subject-level DASS: if ANY recording from a subject is in increase group,
        # ALL recordings from that subject get label=1
        increase_pids = set(
            r["patient_id"] for r in dataset.records if r["baseline_label"] == 1
        )
        return np.array([
            1 if r["patient_id"] in increase_pids else 0
            for r in dataset.records
        ])
    else:
        return np.array([
            1 if r["stress_score"] >= threshold_norm else 0
            for r in dataset.records
        ])


# ──────────────────── LP Mode ────────────────────


def precompute_features(model, loader, device, use_amp, labels_array):
    """Run frozen backbone once, return TensorDataset of (pooled, labels)."""
    model.eval()
    all_pooled, all_labels = [], []

    with torch.no_grad():
        for epochs_batch, baseline_labels, stress_scores, mask, _pids in loader:
            epochs_batch = epochs_batch.to(device)
            mask = mask.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pooled = model.extract_pooled(epochs_batch, mask)

            all_pooled.append(pooled.float().cpu())
            all_labels.append(baseline_labels)

    return TensorDataset(torch.cat(all_pooled), torch.cat(all_labels))


def train_one_fold_lp(
    fold_idx, train_idx, val_idx, test_idx, trial_labels,
    dataset, args, device, threshold_norm,
):
    """LP mode: frozen backbone, precompute features, train head only."""
    use_amp = not args.no_bf16 and device.type == "cuda"

    extractor = create_extractor(args.extractor)
    embed_dim = extractor.embed_dim
    model = DecoupledStressModel(extractor, embed_dim=embed_dim, dropout=args.dropout).to(device)
    model.freeze_backbone()

    if fold_idx == 0:
        trainable = [n for n, p in model.named_parameters() if p.requires_grad]
        print(f"  Trainable params: {len(trainable)}")

    # Dataloaders for feature extraction
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=args.batch_size,
                              shuffle=False, collate_fn=stress_collate_fn)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=args.batch_size,
                            shuffle=False, collate_fn=stress_collate_fn)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=args.batch_size,
                             shuffle=False, collate_fn=stress_collate_fn)

    print("  Precomputing features...", end=" ", flush=True)
    t0 = time.time()
    train_feat_ds = precompute_features(model, train_loader, device, use_amp, trial_labels[train_idx])
    val_feat_ds = precompute_features(model, val_loader, device, use_amp, trial_labels[val_idx])
    test_feat_ds = precompute_features(model, test_loader, device, use_amp, trial_labels[test_idx])
    print(f"done ({time.time() - t0:.1f}s)")

    # Replace with lightweight loaders
    train_loader = DataLoader(train_feat_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_feat_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_feat_ds, batch_size=args.batch_size, shuffle=False)

    # Class weights
    train_labels = trial_labels[train_idx]
    counts = np.bincount(train_labels, minlength=2)
    class_weights = torch.tensor(
        len(train_labels) / (2.0 * counts.clip(min=1)), dtype=torch.float32
    ).to(device)

    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.01, end_factor=1.0, total_iters=args.warmup_epochs,
        )
    plateau_scheduler = None
    if not args.no_scheduler:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6,
        )

    best_bal_acc, best_state, best_epoch = 0.0, None, 0
    val_history = deque(maxlen=SMA_WINDOW)
    best_smoothed, no_improve = 0.0, 0
    curves = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss, n_steps = 0.0, 0
        t0 = time.time()

        for pooled_feats, labels in train_loader:
            pooled_feats, labels = pooled_feats.to(device), labels.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                cls_logits, _ = model.classify(pooled_feats)
                loss = criterion(cls_logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_steps += 1

        train_loss /= max(n_steps, 1)

        # Val
        model.eval()
        val_preds, val_labels_list, val_loss = [], [], 0.0
        with torch.no_grad():
            for pooled_feats, labels in val_loader:
                pooled_feats, labels = pooled_feats.to(device), labels.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    cls_logits, _ = model.classify(pooled_feats)
                    loss = criterion(cls_logits, labels)
                val_loss += loss.item()
                val_preds.append(cls_logits.argmax(1).cpu().numpy())
                val_labels_list.append(labels.cpu().numpy())
        val_loss /= max(len(val_loader), 1)
        vt, vp = np.concatenate(val_labels_list), np.concatenate(val_preds)
        val_bal = balanced_accuracy_score(vt, vp)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"  Fold {fold_idx+1} | Epoch {epoch:>3}/{args.epochs} | "
              f"t_loss={train_loss:.4f} | v_loss={val_loss:.4f} | "
              f"v_bal={val_bal:.4f} | lr={lr:.1e} | {elapsed:.1f}s")
        curves.append({"epoch": epoch, "train_loss": train_loss,
                        "val_loss": val_loss, "val_bal_acc": val_bal, "lr": lr})

        if warmup_scheduler and epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        elif plateau_scheduler:
            plateau_scheduler.step(val_bal)

        if val_bal > best_bal_acc:
            best_bal_acc = val_bal
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

    # Test
    model.eval()
    test_preds, test_labels_list = [], []
    with torch.no_grad():
        for pooled_feats, labels in test_loader:
            pooled_feats = pooled_feats.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                cls_logits, _ = model.classify(pooled_feats)
            test_preds.append(cls_logits.argmax(1).cpu().numpy())
            test_labels_list.append(labels.cpu().numpy())

    y_true = np.concatenate(test_labels_list)
    y_pred = np.concatenate(test_preds)
    return y_true, y_pred, best_epoch, curves


# ──────────────────── FT Mode ────────────────────


def train_one_fold_ft(
    fold_idx, train_idx, val_idx, test_idx, trial_labels,
    dataset, args, device, threshold_norm,
):
    """FT mode: end-to-end fine-tuning with window-level classification."""
    use_amp = not args.no_bf16 and device.type == "cuda"
    label_mode = args.label

    # Build window-level datasets
    print("  Building window datasets...")
    train_win_ds = WindowDataset(dataset, train_idx, aug_overlap=args.aug_overlap,
                                  label_mode=label_mode, threshold_norm=threshold_norm,
                                  label_override=trial_labels)
    val_win_ds = WindowDataset(dataset, val_idx, aug_overlap=None,
                                label_mode=label_mode, threshold_norm=threshold_norm,
                                label_override=trial_labels)
    test_win_ds = WindowDataset(dataset, test_idx, aug_overlap=None,
                                 label_mode=label_mode, threshold_norm=threshold_norm,
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

    # Fresh model
    extractor = create_extractor(args.extractor)
    embed_dim = extractor.embed_dim
    model = DecoupledStressModel(extractor, embed_dim=embed_dim, dropout=args.dropout).to(device)
    # Don't freeze — full fine-tuning

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if fold_idx == 0:
        print(f"  Trainable parameters: {n_params:,}")

    # Class weights from training windows
    train_label_counts = Counter(train_win_ds.labels)
    n_total = len(train_win_ds)
    class_weights = torch.tensor([
        n_total / (2.0 * max(train_label_counts.get(c, 1), 1))
        for c in range(2)
    ], dtype=torch.float32).to(device)
    print(f"  Class weights: {class_weights.tolist()}")

    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, betas=(0.9, 0.999),
    )

    # Cosine annealing with warmup
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = len(train_loader) * args.warmup_epochs // args.grad_accum

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

        for step, (windows, labels, _pids, rec_idxs) in enumerate(train_loader):
            windows, labels = windows.to(device), labels.to(device)
            rec_idxs = rec_idxs.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                features = model.extractor(windows)          # (B, embed_dim)
                cls_logits = model.head_cls(features)         # (B, 2)
                window_loss = criterion(cls_logits, labels)

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
                        loss = ((1 - w) * window_loss + w * rec_loss) / args.grad_accum
                    else:
                        loss = window_loss / args.grad_accum
                else:
                    loss = window_loss / args.grad_accum

            loss.backward()

            if (step + 1) % args.grad_accum == 0 or (step + 1) == len(train_loader):
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            train_loss += loss.item() * args.grad_accum
            n_correct += (cls_logits.argmax(1) == labels).sum().item()
            n_total_train += labels.shape[0]

        train_loss /= max(len(train_loader), 1)
        train_acc = n_correct / max(n_total_train, 1)

        # Val — window-level metrics
        val_metrics = evaluate_windows(model, val_loader, criterion, device, use_amp)
        # Val — recording-level metrics
        val_rec_metrics = evaluate_recording_level(model, val_loader, dataset, val_idx,
                                                    device, use_amp, label_mode, threshold_norm)

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']
        print(f"  Fold {fold_idx+1} | Ep {epoch:>3}/{args.epochs} | "
              f"t_loss={train_loss:.4f} t_acc={train_acc:.3f} | "
              f"v_loss={val_metrics['loss']:.4f} v_bal={val_metrics['bal_acc']:.3f} | "
              f"v_rec_bal={val_rec_metrics['bal_acc']:.3f} | "
              f"lr={lr:.1e} | {elapsed:.1f}s")
        curves.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": round(train_acc, 4),
            "val_loss": val_metrics["loss"],
            "val_win_bal_acc": val_metrics["bal_acc"],
            "val_rec_bal_acc": val_rec_metrics["bal_acc"],
            "lr": lr,
        })

        # Use recording-level bal_acc for model selection
        val_bal = val_rec_metrics["bal_acc"]

        if val_bal > best_bal_acc:
            best_bal_acc = val_bal
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

    # Test — recording-level predictions
    test_rec_metrics = evaluate_recording_level(model, test_loader, dataset, test_idx,
                                                 device, use_amp, label_mode, threshold_norm)
    y_true = test_rec_metrics["y_true"]
    y_pred = test_rec_metrics["y_pred"]

    return y_true, y_pred, best_epoch, curves


def evaluate_windows(model, loader, criterion, device, use_amp) -> dict:
    """Window-level evaluation."""
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


def evaluate_recording_level(model, loader, dataset, rec_indices, device, use_amp,
                              label_mode, threshold_norm) -> dict:
    """Aggregate window predictions to recording level via majority vote."""
    model.eval()
    # Collect per-window predictions with recording index
    win_preds = {}  # rec_idx → list of predictions
    win_labels = {}  # rec_idx → label

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

    # Majority vote per recording
    y_true, y_pred = [], []
    for ri in sorted(win_preds.keys()):
        y_true.append(win_labels[ri])
        votes = Counter(win_preds[ri])
        y_pred.append(votes.most_common(1)[0][0])

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


# ──────────────────── Main ───────────────────────


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    threshold_norm = args.threshold / 100.0

    print(f"Mode: {args.mode.upper()} | Extractor: {args.extractor} | Label: {args.label}")
    print(f"Device: {device} | Folds: {args.folds} | Loss: {args.loss}")
    if args.label == "dss":
        print(f"Threshold: Stress_Score >= {args.threshold}")
    if args.aug_overlap:
        print(f"Augmentation: {args.aug_overlap*100:.0f}% overlap for increase class")
    print(f"LR: {args.lr} | WD: {args.weight_decay} | BS: {args.batch_size} | "
          f"Grad-accum: {args.grad_accum} | Grad-clip: {args.grad_clip}")
    print()

    # Results directory
    label_tag = args.label.replace("-", "") if args.label.startswith("dass") or args.label.startswith("subject") else f"t{int(args.threshold)}"
    aug_tag = f"_aug{int(args.aug_overlap*100)}" if args.aug_overlap else ""
    run_id = f"{datetime.now():%Y%m%d_%H%M}_trial_{args.mode}_{label_tag}{aug_tag}_{args.extractor}"
    results_dir = os.path.join("results", run_id)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"Results -> {results_dir}/")

    # Dataset
    from baseline.abstract.factory import EXTRACTOR_REGISTRY
    _, config_cls = EXTRACTOR_REGISTRY[args.extractor]
    window_sec = config_cls().window_sec
    print(f"Window: {window_sec}s")

    dataset = StressEEGDataset(args.csv, DATA_ROOT, window_sec=window_sec, norm=args.norm,
                               max_duration=args.max_duration)

    # Trial-level labels
    trial_labels = get_labels(dataset, args.label, threshold_norm)
    n1 = int(trial_labels.sum())
    n0 = len(trial_labels) - n1
    print(f"Labels ({args.label}): class_0={n0}, class_1={n1} (total={len(trial_labels)})")
    print()

    # Trial-based CV (no subject grouping — matches paper)
    outer_cv = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)

    global_y_true, global_y_pred = [], []
    all_test_pids, all_test_scores = [], []

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(dataset)), trial_labels)
    ):
        # Inner split for validation
        inner_cv = StratifiedKFold(n_splits=max(args.folds - 1, 2), shuffle=True, random_state=args.seed)
        train_inner_idx, val_inner_idx = next(
            inner_cv.split(np.zeros(len(trainval_idx)), trial_labels[trainval_idx])
        )
        train_idx = trainval_idx[train_inner_idx]
        val_idx = trainval_idx[val_inner_idx]

        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{args.folds}")
        print(f"{'='*60}")
        print_split_info("Train", train_idx, trial_labels)
        print_split_info("Val", val_idx, trial_labels)
        print_split_info("Test", test_idx, trial_labels)

        if args.mode == "ft":
            y_true, y_pred, best_epoch, curves = train_one_fold_ft(
                fold_idx, train_idx, val_idx, test_idx, trial_labels,
                dataset, args, device, threshold_norm,
            )
        else:
            y_true, y_pred, best_epoch, curves = train_one_fold_lp(
                fold_idx, train_idx, val_idx, test_idx, trial_labels,
                dataset, args, device, threshold_norm,
            )

        global_y_true.append(y_true)
        global_y_pred.append(y_pred)

        # Save curves
        with open(os.path.join(results_dir, f"curves_fold{fold_idx+1}.csv"), "w", newline="") as f:
            if curves:
                w = csv.DictWriter(f, fieldnames=list(curves[0].keys()))
                w.writeheader()
                w.writerows(curves)

        # Save predictions
        test_pids = np.array([dataset.records[i]["patient_id"] for i in test_idx])
        test_scores = np.array([dataset.records[i]["stress_score"] for i in test_idx])
        all_test_pids.extend(test_pids.tolist())
        all_test_scores.extend(test_scores.tolist())

        pred_path = os.path.join(results_dir, "predictions.csv")
        write_header = fold_idx == 0
        with open(pred_path, "a" if not write_header else "w", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["fold", "patient_id", "y_true", "y_pred", "stress_score"])
            for i in range(len(y_true)):
                w.writerow([fold_idx + 1, int(test_pids[i]), int(y_true[i]),
                            int(y_pred[i]), f"{test_scores[i]:.4f}"])

        fold_acc = accuracy_score(y_true, y_pred)
        fold_bal = balanced_accuracy_score(y_true, y_pred)
        print(f"  -> Test (best @ epoch {best_epoch}): acc={fold_acc:.4f}, bal_acc={fold_bal:.4f}")

    # ── Global Aggregation ──
    y_true_all = np.concatenate(global_y_true)
    y_pred_all = np.concatenate(global_y_pred)

    acc = accuracy_score(y_true_all, y_pred_all)
    bal_acc = balanced_accuracy_score(y_true_all, y_pred_all)
    f1 = f1_score(y_true_all, y_pred_all, average="weighted", zero_division=0)
    kappa = cohen_kappa_score(y_true_all, y_pred_all)

    print(f"\n{'='*60}")
    print(f"Global Results ({args.folds}-Fold, trial-level CV, {args.mode.upper()})")
    print(f"Extractor: {args.extractor} | Label: {args.label} | Loss: {args.loss}")
    if args.aug_overlap:
        print(f"Augmentation: {args.aug_overlap*100:.0f}% overlap for increase class")
    print(f"{'='*60}")
    print(f"  {'acc':>12s}: {acc:.4f}")
    print(f"  {'bal_acc':>12s}: {bal_acc:.4f}")
    print(f"  {'f1':>12s}: {f1:.4f}")
    print(f"  {'kappa':>12s}: {kappa:.4f}")
    print(f"  Predictions: {len(y_true_all)} (dataset: {len(dataset)})")

    # Save summary
    summary = {
        "mode": args.mode,
        "label": args.label,
        "threshold": args.threshold if args.label == "dss" else None,
        "aug_overlap": args.aug_overlap,
        "acc": round(acc, 4),
        "bal_acc": round(bal_acc, 4),
        "f1": round(f1, 4),
        "kappa": round(kappa, 4),
        "n_samples": len(y_true_all),
        "n_class_1": int(trial_labels.sum()),
        "n_class_0": int(len(trial_labels) - trial_labels.sum()),
    }

    # Per-subject breakdown
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

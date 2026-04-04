"""Linear Probing with Subject-Level Stratified Group K-Fold CV.

Global prediction pooling: predictions from each fold's held-out test set are
concatenated and metrics are computed once at the end (no per-fold averaging).

Usage:
    conda run -n timm_eeg python train_lp.py --extractor reve --folds 5 --loss focal
    conda run -n timm_eeg python train_lp.py --extractor mock_fm --folds 5 --epochs 5
    conda run -n timm_eeg python train_lp.py --extractor reve --stride 5.0 --noise 0.1 --mixup 0.2
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
import baseline.labram  # noqa: F401
import baseline.cbramod  # noqa: F401
from baseline.abstract import create_extractor
from pipeline.dataset import StressEEGDataset, stress_collate_fn
from src.loss import FocalLoss
from src.model import DecoupledStressModel

# ──────────────────── Defaults (aligned with REVE reference) ─────
CSV_PATH = "data/comprehensive_labels.csv"
DATA_ROOT = "data"
BATCH_SIZE = 4
LR = 5e-3
N_EPOCHS = 50
PATIENCE = 15
EMBED_DIM = 512
SMA_WINDOW = 3
# ──────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Linear Probing with Subject-Level K-Fold CV")
    p.add_argument("--extractor", default="mock_fm")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--loss", choices=["focal", "ce"], default="focal")
    p.add_argument("--norm", choices=["zscore", "none"], default="zscore",
                   help="EEG normalization (zscore=per-epoch z-score, none=raw µV)")
    p.add_argument("--dropout", type=float, default=0.05,
                   help="Dropout rate before classification head")
    p.add_argument("--freeze-cls", action="store_true",
                   help="Freeze REVE cls_query_token (unfrozen by default per REVE ref)")
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--warmup-epochs", type=int, default=3,
                   help="Exponential LR warmup epochs (0=off)")
    p.add_argument("--no-scheduler", action="store_true",
                   help="Disable ReduceLROnPlateau scheduler")
    # Augmentation
    p.add_argument("--stride", type=float, default=None,
                   help="Epoch stride in seconds (default=window_sec, no overlap)")
    p.add_argument("--noise", type=float, default=0.0,
                   help="Gaussian noise std added during training (0=off)")
    p.add_argument("--mixup", type=float, default=0.0,
                   help="Mixup alpha parameter (0=off)")
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


# ──────────────────── Core ────────────────────────


def precompute_features(model, loader, device, use_amp):
    """Run frozen backbone once, return TensorDataset of (pooled, labels)."""
    model.eval()
    all_pooled, all_labels = [], []

    with torch.no_grad():
        for epochs_batch, labels, _scores, mask, _pids in loader:
            epochs_batch = epochs_batch.to(device)
            mask = mask.to(device)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                pooled = model.extract_pooled(epochs_batch, mask)

            all_pooled.append(pooled.float().cpu())
            all_labels.append(labels)

    return TensorDataset(torch.cat(all_pooled), torch.cat(all_labels))


def train_one_fold(
    fold_idx: int,
    train_labels: np.ndarray,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    args,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Train a single fold. Returns (y_true_test, y_pred_test) arrays."""
    use_amp = not args.no_bf16 and device.type == "cuda"

    # Fresh model per fold
    extractor = create_extractor(args.extractor)
    embed_dim = extractor.embed_dim
    model = DecoupledStressModel(extractor, embed_dim=embed_dim, dropout=args.dropout).to(device)
    model.freeze_backbone(unfreeze_cls_query=not args.freeze_cls)

    # Verify freeze
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    if fold_idx == 0:
        print(f"  Trainable params: {len(trainable)} | Frozen: {len(frozen)}")
        for n in trainable:
            print(f"    [trainable] {n}")

    # Feature caching: run backbone once, skip if noise or cls_query is trainable
    use_feature_cache = (args.noise == 0 and args.freeze_cls)
    if use_feature_cache:
        t_cache = time.time()
        print("  Precomputing features (frozen backbone)...", end=" ", flush=True)
        train_feat_ds = precompute_features(model, train_loader, device, use_amp)
        val_feat_ds = precompute_features(model, val_loader, device, use_amp)
        test_feat_ds = precompute_features(model, test_loader, device, use_amp)
        print(f"done ({time.time() - t_cache:.1f}s)")
        train_loader = DataLoader(train_feat_ds, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_feat_ds, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_feat_ds, batch_size=args.batch_size, shuffle=False)

    counts = np.bincount(train_labels)
    class_weights = torch.tensor(len(train_labels) / (len(counts) * counts), dtype=torch.float32).to(device)
    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0, alpha=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, betas=(0.92, 0.999), eps=1e-9,
        weight_decay=args.weight_decay,
    )
    warmup_scheduler = None
    if args.warmup_epochs > 0:
        # Exponential warmup per REVE reference: (10^(step/total) - 1) / 9
        total_warmup = args.warmup_epochs
        warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: (10 ** (min(epoch, total_warmup) / total_warmup) - 1) / 9,
        )
    if not args.no_scheduler:
        plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-6,
        )
    else:
        plateau_scheduler = None

    best_raw_bal_acc = 0.0
    best_state = None
    best_epoch = 0
    # Smoothed early stopping
    val_history = deque(maxlen=SMA_WINDOW)
    best_smoothed = 0.0
    no_improve = 0
    grad_verified = False

    for epoch in range(1, args.epochs + 1):
        # ── Train ──
        model.train()
        train_loss, n_steps = 0.0, 0
        t0 = time.time()

        if use_feature_cache:
            for pooled_feats, labels in train_loader:
                pooled_feats, labels = pooled_feats.to(device), labels.to(device)

                use_mixup = args.mixup > 0
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    if use_mixup:
                        pooled_feats, y_a, y_b, lam = mixup_batch(pooled_feats, labels, args.mixup)
                        cls_logits, _ = model.classify(pooled_feats)
                        loss = lam * criterion(cls_logits, y_a) + (1 - lam) * criterion(cls_logits, y_b)
                    else:
                        cls_logits, _ = model.classify(pooled_feats)
                        loss = criterion(cls_logits, labels)

                optimizer.zero_grad()
                loss.backward()

                if not grad_verified:
                    _verify_gradients(model)
                    grad_verified = True

                optimizer.step()
                train_loss += loss.item()
                n_steps += 1
        else:
            for epochs_batch, labels, _scores, mask, _pids in train_loader:
                epochs_batch, labels, mask = (
                    epochs_batch.to(device),
                    labels.to(device),
                    mask.to(device),
                )

                # Training-time augmentation: Gaussian noise
                if args.noise > 0:
                    epochs_batch = epochs_batch + args.noise * torch.randn_like(epochs_batch)

                use_mixup = args.mixup > 0

                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    pooled = model.extract_pooled(epochs_batch, mask)
                    if use_mixup:
                        pooled, y_a, y_b, lam = mixup_batch(pooled, labels, args.mixup)
                        cls_logits, _ = model.classify(pooled)
                        loss = lam * criterion(cls_logits, y_a) + (1 - lam) * criterion(cls_logits, y_b)
                    else:
                        cls_logits, _ = model.classify(pooled)
                        loss = criterion(cls_logits, labels)

                optimizer.zero_grad()
                loss.backward()

                if not grad_verified:
                    _verify_gradients(model)
                    grad_verified = True

                optimizer.step()
                train_loss += loss.item()
                n_steps += 1

        train_loss /= max(n_steps, 1)

        # ── Val ──
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp, use_feature_cache)
        elapsed = time.time() - t0

        current_lr = optimizer.param_groups[0]['lr']
        print(
            f"  Fold {fold_idx+1} | Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_bal_acc={val_metrics['bal_acc']:.4f} | "
            f"lr={current_lr:.1e} | {elapsed:.1f}s"
        )

        if warmup_scheduler is not None and epoch <= args.warmup_epochs:
            warmup_scheduler.step()
        elif plateau_scheduler is not None:
            plateau_scheduler.step(val_metrics["bal_acc"])

        # Checkpoint on raw bal_acc improvement
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

    # Reload best checkpoint and evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("  [WARN] No improvement during training, using last model")

    y_true, y_pred = predict(model, test_loader, device, use_amp, use_feature_cache)
    return y_true, y_pred, best_epoch


def evaluate(model, loader, criterion, device, use_amp, cached=False) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_steps = 0.0, 0

    with torch.no_grad():
        if cached:
            for pooled_feats, labels in loader:
                pooled_feats, labels = pooled_feats.to(device), labels.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    cls_logits, _ = model.classify(pooled_feats)
                    loss = criterion(cls_logits, labels)
                total_loss += loss.item()
                n_steps += 1
                all_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        else:
            for epochs_batch, labels, _scores, mask, _pids in loader:
                epochs_batch, labels, mask = (
                    epochs_batch.to(device), labels.to(device), mask.to(device),
                )
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    cls_logits, _ = model(epochs_batch, mask)
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


def predict(model, loader, device, use_amp, cached=False) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred) arrays."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        if cached:
            for pooled_feats, labels in loader:
                pooled_feats, labels = pooled_feats.to(device), labels.to(device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    cls_logits, _ = model.classify(pooled_feats)
                all_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())
        else:
            for epochs_batch, labels, _scores, mask, _pids in loader:
                epochs_batch, labels, mask = (
                    epochs_batch.to(device), labels.to(device), mask.to(device),
                )
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                    cls_logits, _ = model(epochs_batch, mask)
                all_preds.append(cls_logits.argmax(dim=1).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

    return np.concatenate(all_labels), np.concatenate(all_preds)


def _verify_gradients(model):
    for name, param in model.extractor.named_parameters():
        assert param.grad is None, f"Backbone param {name} has gradient — not frozen!"
    head_grads = sum(
        1 for p in model.head_cls.parameters() if p.grad is not None
    )
    assert head_grads > 0, "Head has no gradients!"
    print("  ✓ Freeze verified: backbone frozen, head has gradients")


def seed_everything(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Extractor: {args.extractor} | Folds: {args.folds} | Loss: {args.loss}")
    print(f"Norm: {args.norm} | Dropout: {args.dropout} | WD: {args.weight_decay} | Scheduler: {not args.no_scheduler}")
    if not args.freeze_cls:
        print("CLS query token: unfrozen")
    if args.stride:
        print(f"Stride: {args.stride}s (overlapping epochs)")
    if args.noise > 0:
        print(f"Gaussian noise: std={args.noise}")
    if args.mixup > 0:
        print(f"Mixup: alpha={args.mixup} (embedding-level)")
    print()

    dataset = StressEEGDataset(CSV_PATH, DATA_ROOT, stride_sec=args.stride, norm=args.norm)
    labels = dataset.get_labels()
    patient_ids = dataset.get_patient_ids()

    # Outer CV: subject-level stratified split → Test set
    outer_cv = StratifiedGroupKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )

    global_y_true, global_y_pred = [], []

    for fold_idx, (trainval_idx, test_idx) in enumerate(
        outer_cv.split(np.zeros(len(dataset)), labels, groups=patient_ids)
    ):
        # Inner split: hold out ~1 group from trainval as Val
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

        # Warn if any split has only one class
        for name, idx in [("Val", val_idx), ("Test", test_idx)]:
            if len(np.unique(labels[idx])) < 2:
                print(f"  [WARN] {name} set has only one class — metrics may be unreliable")

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

        y_true, y_pred, best_epoch = train_one_fold(
            fold_idx, labels[train_idx],
            train_loader, val_loader, test_loader,
            args, device,
        )
        global_y_true.append(y_true)
        global_y_pred.append(y_pred)

        fold_acc = accuracy_score(y_true, y_pred)
        fold_bal = balanced_accuracy_score(y_true, y_pred)
        print(
            f"  -> Test (best @ epoch {best_epoch}): "
            f"acc={fold_acc:.4f}, bal_acc={fold_bal:.4f}"
        )

    # ── Global Aggregation ──
    y_true_all = np.concatenate(global_y_true)
    y_pred_all = np.concatenate(global_y_pred)

    print(f"\n{'='*60}")
    print(f"Global LP Results ({args.folds}-Fold, subject-level CV)")
    print(f"Extractor: {args.extractor} | Loss: {args.loss}")
    if args.stride:
        print(f"Stride: {args.stride}s | ", end="")
    if args.noise > 0:
        print(f"Noise: {args.noise} | ", end="")
    if args.mixup > 0:
        print(f"Mixup: {args.mixup} | ", end="")
    print(f"{'='*60}")

    print(f"  {'acc':>12s}: {accuracy_score(y_true_all, y_pred_all):.4f}")
    print(f"  {'bal_acc':>12s}: {balanced_accuracy_score(y_true_all, y_pred_all):.4f}")
    print(f"  {'f1':>12s}: {f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0):.4f}")
    print(f"  {'kappa':>12s}: {cohen_kappa_score(y_true_all, y_pred_all):.4f}")
    print(f"  Total predictions: {len(y_true_all)} (should equal dataset size: {len(dataset)})")


if __name__ == "__main__":
    main()

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
from torch.utils.data import DataLoader, Subset

# Register extractors
import baseline.mock_fm  # noqa: F401
import baseline.reve  # noqa: F401
from baseline.abstract import create_extractor
from pipeline.dataset import StressEEGDataset, stress_collate_fn
from src.loss import FocalLoss
from src.model import DecoupledStressModel

# ──────────────────── Defaults ────────────────────
CSV_PATH = "data/comprehensive_labels.csv"
DATA_ROOT = "data"
BATCH_SIZE = 4
LR = 1e-3
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
    extractor = create_extractor(args.extractor, embed_dim=EMBED_DIM)
    model = DecoupledStressModel(extractor, embed_dim=EMBED_DIM).to(device)
    model.freeze_backbone()

    # Verify freeze
    trainable = [n for n, p in model.named_parameters() if p.requires_grad]
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    if fold_idx == 0:
        print(f"  Trainable params: {len(trainable)} | Frozen: {len(frozen)}")
        for n in trainable:
            print(f"    [trainable] {n}")

    if args.loss == "focal":
        criterion = FocalLoss(gamma=2.0)
    else:
        counts = np.bincount(train_labels)
        weights = torch.tensor(len(train_labels) / (len(counts) * counts), dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

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

        for epochs_batch, labels, _scores, mask in train_loader:
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
                if use_mixup:
                    mixed_x, y_a, y_b, lam = mixup_batch(epochs_batch, labels, args.mixup)
                    cls_logits, _ = model(mixed_x, mask)
                    loss = lam * criterion(cls_logits, y_a) + (1 - lam) * criterion(cls_logits, y_b)
                else:
                    cls_logits, _ = model(epochs_batch, mask)
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
        val_metrics = evaluate(model, val_loader, criterion, device, use_amp)
        elapsed = time.time() - t0

        print(
            f"  Fold {fold_idx+1} | Epoch {epoch:>3}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_bal_acc={val_metrics['bal_acc']:.4f} | "
            f"{elapsed:.1f}s"
        )

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

    y_true, y_pred = predict(model, test_loader, device, use_amp)
    return y_true, y_pred, best_epoch


def evaluate(model, loader, criterion, device, use_amp) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_steps = 0.0, 0

    with torch.no_grad():
        for epochs_batch, labels, _scores, mask in loader:
            epochs_batch, labels, mask = (
                epochs_batch.to(device),
                labels.to(device),
                mask.to(device),
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


def predict(model, loader, device, use_amp) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference and return (y_true, y_pred) arrays."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for epochs_batch, labels, _scores, mask in loader:
            epochs_batch, labels, mask = (
                epochs_batch.to(device),
                labels.to(device),
                mask.to(device),
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


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Extractor: {args.extractor} | Folds: {args.folds} | Loss: {args.loss}")
    if args.stride:
        print(f"Stride: {args.stride}s (overlapping epochs)")
    if args.noise > 0:
        print(f"Gaussian noise: std={args.noise}")
    if args.mixup > 0:
        print(f"Mixup: alpha={args.mixup}")
    print()

    dataset = StressEEGDataset(CSV_PATH, DATA_ROOT, stride_sec=args.stride)
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
            f"  → Test (best @ epoch {best_epoch}): "
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

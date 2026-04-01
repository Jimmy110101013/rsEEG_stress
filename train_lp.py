"""Linear Probing with Stratified K-Fold CV.

Usage:
    conda run -n timm_eeg python train_lp.py --extractor reve --folds 6 --loss focal
    conda run -n timm_eeg python train_lp.py --extractor mock_fm --folds 3 --epochs 5
"""

import argparse
import time

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
PATIENCE = 10
EMBED_DIM = 512
# ──────────────────────────────────────────────────


def parse_args():
    p = argparse.ArgumentParser(description="Linear Probing with K-Fold CV")
    p.add_argument("--extractor", default="mock_fm")
    p.add_argument("--folds", type=int, default=6)
    p.add_argument("--epochs", type=int, default=N_EPOCHS)
    p.add_argument("--patience", type=int, default=PATIENCE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:6")
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--loss", choices=["focal", "ce"], default="focal")
    return p.parse_args()


def train_one_fold(
    fold_idx: int,
    train_labels: np.ndarray,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args,
    device: torch.device,
) -> dict:
    """Train and evaluate a single fold. Returns best validation metrics."""
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

    best = {"bal_acc": 0.0, "epoch": 0}
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

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
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

        if val_metrics["bal_acc"] > best["bal_acc"]:
            best = {**val_metrics, "epoch": epoch}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  Early stop at epoch {epoch} (patience={args.patience})")
                break

    return best


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
    print()

    dataset = StressEEGDataset(CSV_PATH, DATA_ROOT)
    labels = dataset.get_labels()

    skf = StratifiedKFold(
        n_splits=args.folds, shuffle=True, random_state=args.seed
    )

    fold_results = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.zeros(len(dataset)), labels)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx+1}/{args.folds}  |  train={len(train_idx)}, val={len(val_idx)}")
        print(f"{'='*60}")

        train_loader = DataLoader(
            Subset(dataset, train_idx),
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=stress_collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            Subset(dataset, val_idx),
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=stress_collate_fn,
            num_workers=0,
        )

        fold_best = train_one_fold(fold_idx, labels[train_idx], train_loader, val_loader, args, device)
        fold_results.append(fold_best)
        print(
            f"  → Best @ epoch {fold_best['epoch']}: "
            f"bal_acc={fold_best['bal_acc']:.4f}, "
            f"f1={fold_best['f1']:.4f}, "
            f"kappa={fold_best['kappa']:.4f}"
        )

    # ── Aggregate ──
    print(f"\n{'='*60}")
    print(f"{args.folds}-Fold LP Results  |  Extractor: {args.extractor}  |  Loss: {args.loss}")
    print(f"{'='*60}")
    for metric in ["acc", "bal_acc", "f1", "kappa"]:
        vals = [r[metric] for r in fold_results]
        print(f"  {metric:>12s}: {np.mean(vals):.4f} ± {np.std(vals):.4f}")


if __name__ == "__main__":
    main()

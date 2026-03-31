"""MVP Training Script — Verify end-to-end architecture plumbing.

Usage:
    conda run -n timm_eeg python train_mvp.py
"""

import sys
import time

import torch
from torch.utils.data import DataLoader, random_split

# Register mock extractor
import baseline.mock_fm  # noqa: F401
from baseline.abstract import create_extractor
from pipeline.dataset import StressEEGDataset, stress_collate_fn
from src.model import DecoupledStressModel
from src.loss import MTLLoss

# ──────────────────── Config ────────────────────
CSV_PATH = "data/comprehensive_labels.csv"
DATA_ROOT = "data"
BATCH_SIZE = 4
LR = 1e-3
N_EPOCHS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXTRACTOR_NAME = "mock_fm"
EMBED_DIM = 512
# ────────────────────────────────────────────────


def main():
    print(f"Device: {DEVICE}")
    print(f"Extractor: {EXTRACTOR_NAME}")
    print()

    # 1. Dataset
    dataset = StressEEGDataset(CSV_PATH, DATA_ROOT)

    # 2. 80/20 random split (MVP — no subject-wise rules)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )
    print(f"Split: {n_train} train, {n_val} val\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=stress_collate_fn,
        num_workers=0,  # MNE loading is not fork-safe
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=stress_collate_fn,
        num_workers=0,
    )

    # 3. Model
    extractor = create_extractor(EXTRACTOR_NAME, embed_dim=EMBED_DIM)
    model = DecoupledStressModel(extractor, embed_dim=EMBED_DIM).to(DEVICE)
    criterion = MTLLoss(alpha=1.0, beta=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print()

    # 4. Training loop
    grad_checked = False
    for epoch in range(1, N_EPOCHS + 1):
        # ── Train ──
        model.train()
        train_loss_sum, train_steps = 0.0, 0
        t0 = time.time()

        for step, (epochs_batch, labels, scores, mask) in enumerate(train_loader, 1):
            epochs_batch = epochs_batch.to(DEVICE)
            labels = labels.to(DEVICE)
            scores = scores.to(DEVICE)
            mask = mask.to(DEVICE)

            cls_logits, reg_output = model(epochs_batch, mask)
            total_loss, loss_a, loss_b = criterion(
                cls_logits, reg_output, labels, scores
            )

            optimizer.zero_grad()
            total_loss.backward()

            # One-time gradient check
            if not grad_checked:
                n_with_grad = sum(
                    1 for p in model.parameters() if p.grad is not None
                )
                n_total = sum(1 for _ in model.parameters())
                print(f"Gradient check: {n_with_grad}/{n_total} parameters have gradients")
                if n_with_grad == n_total:
                    print("✓ All parameters have gradients\n")
                else:
                    print("⚠ Some parameters missing gradients!\n")
                grad_checked = True

            optimizer.step()

            train_loss_sum += total_loss.item()
            train_steps += 1
            print(
                f"  [Epoch {epoch}/{N_EPOCHS}] Step {step}/{len(train_loader)} | "
                f"Loss: {total_loss.item():.4f} "
                f"(A: {loss_a.item():.4f}, B: {loss_b.item():.4f})"
            )

        train_avg = train_loss_sum / max(train_steps, 1)
        elapsed = time.time() - t0

        # ── Val ──
        model.eval()
        val_loss_sum, val_steps = 0.0, 0
        with torch.no_grad():
            for epochs_batch, labels, scores, mask in val_loader:
                epochs_batch = epochs_batch.to(DEVICE)
                labels = labels.to(DEVICE)
                scores = scores.to(DEVICE)
                mask = mask.to(DEVICE)

                cls_logits, reg_output = model(epochs_batch, mask)
                total_loss, _, _ = criterion(cls_logits, reg_output, labels, scores)
                val_loss_sum += total_loss.item()
                val_steps += 1

        val_avg = val_loss_sum / max(val_steps, 1)
        print(
            f"\n  Epoch {epoch}/{N_EPOCHS} Summary: "
            f"Train Loss={train_avg:.4f} | Val Loss={val_avg:.4f} | "
            f"Time={elapsed:.1f}s\n"
        )

    print("MVP training complete. Architecture plumbing verified.")


if __name__ == "__main__":
    main()

"""Sanity reproduction: LEAD (arXiv:2502.01678v4) ADFTD 3-class baseline numbers.

Purpose
-------
Before trusting any of our exp_newdata FT numbers — which used lr=1e-5 layer_decay=off
wd=0.01 loss=focal (10-80x off every FM's official recipe) — we must verify our
pipeline can reproduce a published subject-level benchmark under per-FM official HP.

Target (LEAD v4 Table 2 + Appendix G.1 Table 8, ADFTD 3-class HC vs AD vs FTD,
88 subjects / 167,083 samples, subject-independent 8:1:1 x 5 seeds 41-45):

              Sample-level         Subject-level
  LaBraM  :   F1=75.64+/-4.68     F1=91.14+/-8.64
              AUROC=91.22+/-2.72  AUROC=93.77+/-6.16
  CBraMod :   F1=68.33+/-4.53     F1=82.21+/-6.30
              AUROC=86.95+/-2.89  AUROC=87.10+/-3.77

Exit criteria
-------------
  PASS:  our LaBraM sample-level F1 within [70, 81]  -> pipeline healthy
  FAIL:  sample-level F1 < 65 or > 85                -> pipeline has structural issue
  NOISE: sample-level F1 in [65, 70] or [81, 85]     -> borderline, inspect per-fold

Does NOT modify train_ft.py, DecoupledStressModel, or any production code.
All HP / head / classifier are defined inline here so that when this driver
gets deleted post-sanity-check, nothing breaks.

Usage
-----
  stress_py scripts/experiments/run_sanity_lead_adftd.py --fm labram --seed 41 --device cuda:0
  stress_py scripts/experiments/run_sanity_lead_adftd.py --fm cbramod --seed 42 --device cuda:1
  stress_py scripts/experiments/run_sanity_lead_adftd.py --fm reve --seed 43 --device cuda:2
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, roc_auc_score,
)
from torch.utils.data import DataLoader, Dataset

ROOT = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
sys.path.insert(0, str(ROOT))

import baseline.labram   # noqa: F401
import baseline.cbramod  # noqa: F401
import baseline.reve     # noqa: F401
from baseline.abstract.factory import create_extractor
from pipeline.adftd_dataset import ADFTDDataset
from pipeline.common_channels import COMMON_19


# =====================================================================
# Per-FM HP recipes — sourced from official repos + papers
# =====================================================================
# NOTE: these deviate from our exp_newdata unified-HP (lr=1e-5, wd=0.01,
# llrd=off, loss=focal). Each row cites its source.
# Per-FM official HP — sourced from G-F09 methodology policy
RECIPES = {
    "labram": dict(                     # LaBraM README + run_class_finetuning.py
        lr=5e-4, weight_decay=0.05, layer_decay=0.65,
        label_smoothing=0.1, head_hidden=0, drop_path=0.1,
        encoder_lr_scale=1.0, warmup_epochs=5,
        optimizer_betas=(0.9, 0.999), norm="zscore",
    ),
    "cbramod": dict(                    # CBraMod finetune_main.py (multi_lr)
        lr=5e-4, weight_decay=0.05, layer_decay=1.0,
        label_smoothing=0.1, head_hidden=0, drop_path=0.0,
        encoder_lr_scale=0.2, warmup_epochs=0,      # encoder 1e-4 = 5e-4 * 0.2
        optimizer_betas=(0.9, 0.999), norm="none",
    ),
    "reve": dict(                       # reve_unified.yaml
        lr=2.4e-4, weight_decay=0.01, layer_decay=1.0,
        label_smoothing=0.0, head_hidden=0, drop_path=0.0,
        encoder_lr_scale=0.1, warmup_epochs=2,
        optimizer_betas=(0.9, 0.95), norm="none",
    ),
}

EPOCHS = 200
PATIENCE = 15
BATCH_SIZE = 256          # LEAD LaBraM-S-1-Multi.sh
GRAD_CLIP = 4.0           # LEAD exp_supervised.py line 269
USE_SWA = True            # LEAD enables --swa for all supervised runs
VAL_FRAC = 0.1
TEST_FRAC = 0.1
N_CLASSES = 3             # HC=0, AD=1, FTD=2


# =====================================================================
# Data
# =====================================================================
class FlatWindowDataset(Dataset):
    """Flatten ADFTDDataset into per-window samples.

    Each item = (window: (C, T) float32, label: int, subject_id: int, rec_idx: int).
    """

    def __init__(self, base: ADFTDDataset, indices: list[int]):
        self.items: list[tuple[torch.Tensor, int, int, int]] = []
        for i in indices:
            epochs, label, n_ep, pid = base[i]
            for w in range(epochs.shape[0]):
                self.items.append((epochs[w], int(label), int(pid), int(i)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


def window_collate(batch):
    xs, ys, pids, rids = zip(*batch)
    return (torch.stack(xs), torch.tensor(ys, dtype=torch.long),
            torch.tensor(pids), torch.tensor(rids))


def subject_level_split(patient_ids: np.ndarray, labels: np.ndarray,
                        seed: int, val_frac: float, test_frac: float
                        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stratified subject-independent split (8:1:1 by subject).

    LEAD paper §3.1: Monte Carlo CV with subject-independent 8:1:1 split,
    stratified by subject label (each subject has a single label).
    """
    rng = np.random.RandomState(seed)
    # subject -> label (take first occurrence; ADFTD has one label per subject)
    subj_to_label: dict[int, int] = {}
    for pid, lab in zip(patient_ids, labels):
        subj_to_label.setdefault(int(pid), int(lab))
    subjects = np.array(sorted(subj_to_label.keys()))
    subj_labels = np.array([subj_to_label[s] for s in subjects])

    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(subj_labels):
        class_subj = subjects[subj_labels == c]
        perm = rng.permutation(len(class_subj))
        n_test = max(1, int(round(len(class_subj) * test_frac)))
        n_val = max(1, int(round(len(class_subj) * val_frac)))
        test_s = class_subj[perm[:n_test]]
        val_s = class_subj[perm[n_test:n_test + n_val]]
        train_s = class_subj[perm[n_test + n_val:]]
        for rec_i, pid in enumerate(patient_ids):
            if int(pid) in set(test_s):
                test_idx.append(rec_i)
            elif int(pid) in set(val_s):
                val_idx.append(rec_i)
            elif int(pid) in set(train_s):
                train_idx.append(rec_i)
    return (np.array(sorted(set(train_idx))),
            np.array(sorted(set(val_idx))),
            np.array(sorted(set(test_idx))))


# =====================================================================
# Model
# =====================================================================
class SimpleFM3Class(nn.Module):
    """Minimal 3-class head on top of any EEG-FM extractor.

    Per-window forward: (B, C, T) -> extractor -> (B, embed_dim) -> head -> (B, 3).
    """

    def __init__(self, fm_name: str, embed_dim: int, head_hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fm_name = fm_name
        extractor = create_extractor(fm_name)
        if fm_name == "labram":
            from baseline.labram.channel_map import get_input_chans
            extractor.input_chans = get_input_chans(COMMON_19)
        if fm_name == "reve":
            extractor.set_channels(COMMON_19)
        self.extractor = extractor
        if head_hidden > 0:
            self.head = nn.Sequential(
                nn.Dropout(dropout), nn.Linear(embed_dim, head_hidden),
                nn.GELU(), nn.Dropout(dropout), nn.Linear(head_hidden, N_CLASSES),
            )
        else:
            # LaBraM-style: single Linear, init_scale 0.001 for classification layer
            lin = nn.Linear(embed_dim, N_CLASSES)
            nn.init.trunc_normal_(lin.weight, std=0.001)
            nn.init.zeros_(lin.bias)
            self.head = lin

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.extractor(x)      # (B, embed_dim)
        return self.head(feats)


# =====================================================================
# LLRD builder (LaBraM only)
# =====================================================================
def build_optimizer(model: SimpleFM3Class, recipe: dict) -> torch.optim.Optimizer:
    head_params = [p for n, p in model.named_parameters()
                   if p.requires_grad and n.startswith("head")]
    groups = [{"params": head_params, "lr": recipe["lr"]}]

    if recipe["layer_decay"] < 1.0 and hasattr(model.extractor, "get_layer_groups"):
        layer_groups = model.extractor.get_layer_groups()
        n_layers = len(layer_groups)
        encoder_base = recipe["lr"] * recipe["encoder_lr_scale"]
        for i, group_params in enumerate(layer_groups):
            trainable = [p for p in group_params if p.requires_grad]
            if trainable:
                lr_i = encoder_base * (recipe["layer_decay"] ** (n_layers - 1 - i))
                groups.append({"params": trainable, "lr": lr_i})
        print(f"  LLRD active: {n_layers} layer groups, "
              f"LR range [{encoder_base * recipe['layer_decay']**(n_layers-1):.2e}, "
              f"{encoder_base:.2e}]")
    else:
        encoder_params = [p for n, p in model.named_parameters()
                          if p.requires_grad and not n.startswith("head")]
        if encoder_params:
            groups.append({"params": encoder_params,
                           "lr": recipe["lr"] * recipe["encoder_lr_scale"]})

    return torch.optim.AdamW(
        groups,
        betas=recipe["optimizer_betas"],
        eps=1e-8,
        weight_decay=recipe["weight_decay"],
    )


# =====================================================================
# Training loop
# =====================================================================
@torch.no_grad()
def evaluate(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns per-window (y_true, y_pred, y_proba) and per-window (pid, rid)."""
    model.eval()
    all_y, all_pred, all_proba, all_pid, all_rid = [], [], [], [], []
    for x, y, pid, rid in loader:
        x = x.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        proba = F.softmax(logits.float(), dim=-1)
        all_y.append(y.numpy())
        all_pred.append(logits.argmax(-1).cpu().numpy())
        all_proba.append(proba.cpu().numpy())
        all_pid.append(pid.numpy())
        all_rid.append(rid.numpy())
    return (np.concatenate(all_y), np.concatenate(all_pred), np.concatenate(all_proba),
            np.concatenate(all_pid), np.concatenate(all_rid))


def metrics_sample(y_true, y_pred, y_proba) -> dict:
    m = {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
    }
    try:
        m["auroc"] = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro")
    except ValueError:
        m["auroc"] = float("nan")
    return m


def metrics_subject(y_true_w, y_pred_w, y_proba_w, pid_w) -> dict:
    """Majority-vote window -> subject label; mean-proba for AUROC."""
    subj_true, subj_pred, subj_proba = {}, {}, {}
    subj_proba_sum: dict[int, np.ndarray] = {}
    subj_proba_cnt: dict[int, int] = {}
    for i in range(len(y_true_w)):
        s = int(pid_w[i])
        subj_true[s] = int(y_true_w[i])
        subj_pred.setdefault(s, []).append(int(y_pred_w[i]))
        if s not in subj_proba_sum:
            subj_proba_sum[s] = y_proba_w[i].copy()
            subj_proba_cnt[s] = 1
        else:
            subj_proba_sum[s] += y_proba_w[i]
            subj_proba_cnt[s] += 1
    subjects = sorted(subj_true.keys())
    yt = np.array([subj_true[s] for s in subjects])
    yp = np.array([Counter(subj_pred[s]).most_common(1)[0][0] for s in subjects])
    yprob = np.stack([subj_proba_sum[s] / subj_proba_cnt[s] for s in subjects])
    out = {
        "acc": accuracy_score(yt, yp),
        "bal_acc": balanced_accuracy_score(yt, yp),
        "f1": f1_score(yt, yp, average="macro", zero_division=0),
    }
    try:
        out["auroc"] = roc_auc_score(yt, yprob, multi_class="ovr", average="macro")
    except ValueError:
        out["auroc"] = float("nan")
    return out


def run_one_seed(fm: str, seed: int, device: str, out_dir: Path, mode: str = "ft") -> dict:
    recipe = RECIPES[fm]
    print(f"\n=== {fm} seed {seed} mode={mode} ===")
    print(f"  recipe: {recipe}")

    # --- data
    cache = f"data/cache_adftd_3cls_split1_n{recipe['norm']}"
    base = ADFTDDataset(str(ROOT / "data/adftd"), binary=False,
                         window_sec=5.0, cache_dir=str(ROOT / cache),
                         n_splits=1, norm=recipe["norm"])
    pids = np.array(base.get_patient_ids())
    labels = np.array(base.get_labels())
    print(f"  ADFTD 3-class: {len(base)} recordings, {len(np.unique(pids))} subjects, "
          f"label counts={dict(zip(*np.unique(labels, return_counts=True)))}")

    train_idx, val_idx, test_idx = subject_level_split(pids, labels, seed, VAL_FRAC, TEST_FRAC)
    print(f"  split: train rec={len(train_idx)}/{len(np.unique(pids[train_idx]))} subj, "
          f"val={len(val_idx)}/{len(np.unique(pids[val_idx]))}, "
          f"test={len(test_idx)}/{len(np.unique(pids[test_idx]))}")

    train_ds = FlatWindowDataset(base, train_idx.tolist())
    val_ds = FlatWindowDataset(base, val_idx.tolist())
    test_ds = FlatWindowDataset(base, test_idx.tolist())
    print(f"  windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              collate_fn=window_collate, num_workers=2, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            collate_fn=window_collate, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             collate_fn=window_collate, num_workers=2)

    # --- model + optim
    torch.manual_seed(seed); np.random.seed(seed); torch.cuda.manual_seed_all(seed)
    extractor_stub = create_extractor(fm)
    embed_dim = extractor_stub.embed_dim
    del extractor_stub
    model = SimpleFM3Class(fm, embed_dim, recipe["head_hidden"]).to(device)

    # Freeze backbone if mode == "frozen" (head-only LP)
    if mode == "frozen":
        for p in model.extractor.parameters():
            p.requires_grad = False
        head_only_lr = 5e-3   # typical LP head LR, not recipe["lr"]
        frozen_recipe = {**recipe, "lr": head_only_lr, "layer_decay": 1.0,
                         "encoder_lr_scale": 0.0, "warmup_epochs": 0}
        optimizer = build_optimizer(model, frozen_recipe)
        print(f"  mode=frozen — backbone frozen, head-only LR={head_only_lr}")
    else:
        optimizer = build_optimizer(model, recipe)
        print(f"  mode=ft — full fine-tuning per recipe")

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = len(train_loader) * recipe["warmup_epochs"]

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return max(1e-2, step / warmup_steps)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(1e-2, 0.5 * (1 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = nn.CrossEntropyLoss(label_smoothing=recipe["label_smoothing"])

    # --- SWA (LEAD enables for all supervised runs; eval uses SWA weights)
    swa_model = torch.optim.swa_utils.AveragedModel(model) if USE_SWA else None
    eval_model = swa_model if USE_SWA else model

    # --- train
    best_f1, best_state, best_epoch = 0.0, None, 0
    no_improve = 0
    global_step = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        t0 = time.time()
        train_loss = 0.0
        for x, y, _, _ in train_loader:
            x, y = x.to(device), y.to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            train_loss += loss.item()
            global_step += 1
        train_loss /= max(len(train_loader), 1)

        if swa_model is not None:
            swa_model.update_parameters(model)

        yt, yp, ypr, pidw, _ = evaluate(eval_model, val_loader, device)
        v_sample = metrics_sample(yt, yp, ypr)
        v_subj = metrics_subject(yt, yp, ypr, pidw)
        cur_lr = optimizer.param_groups[0]["lr"]
        print(f"  Ep {epoch:>3}/{EPOCHS} | train_loss={train_loss:.4f} | "
              f"val_sample_f1={v_sample['f1']:.3f} auroc={v_sample['auroc']:.3f} | "
              f"val_subj_f1={v_subj['f1']:.3f} | lr={cur_lr:.1e} | {time.time()-t0:.1f}s")

        if v_sample["f1"] > best_f1:
            best_f1 = v_sample["f1"]
            best_state = copy.deepcopy(eval_model.state_dict())
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"  early stop @ ep {epoch}")
                break

    if best_state is not None:
        eval_model.load_state_dict(best_state)

    yt, yp, ypr, pidw, _ = evaluate(eval_model, test_loader, device)
    test_sample = metrics_sample(yt, yp, ypr)
    test_subj = metrics_subject(yt, yp, ypr, pidw)
    print(f"\n  TEST  sample: f1={test_sample['f1']:.4f} acc={test_sample['acc']:.4f} "
          f"auroc={test_sample['auroc']:.4f} bal_acc={test_sample['bal_acc']:.4f}")
    print(f"  TEST  subject: f1={test_subj['f1']:.4f} acc={test_subj['acc']:.4f} "
          f"auroc={test_subj['auroc']:.4f} bal_acc={test_subj['bal_acc']:.4f}")

    result = {
        "fm": fm, "seed": seed, "mode": mode, "best_epoch": best_epoch,
        "recipe": {k: v if not isinstance(v, tuple) else list(v) for k, v in recipe.items()},
        "test_sample": test_sample,
        "test_subject": test_subj,
        "n_train_rec": int(len(train_idx)),
        "n_val_rec": int(len(val_idx)),
        "n_test_rec": int(len(test_idx)),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{fm}_{mode}_seed{seed}.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fm", required=True, choices=list(RECIPES.keys()))
    p.add_argument("--seed", type=int, required=True,
                   help="LEAD used seeds 41-45")
    p.add_argument("--mode", choices=["ft", "frozen"], default="ft",
                   help="ft=full fine-tuning (recipe LR); frozen=head-only LP (head LR=5e-3)")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--out-dir", default="results/studies/sanity_lead_adftd")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = ROOT / args.out_dir
    run_one_seed(args.fm, args.seed, device, out_dir, mode=args.mode)


if __name__ == "__main__":
    main()

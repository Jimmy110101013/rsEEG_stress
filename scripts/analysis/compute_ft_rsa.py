"""Compute FT RSA (label-r, subject-r) for stress + eegmat (3 FMs each).
Writes to results/final/source_tables/ft_rsa_stress_eegmat.json."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr

REPO = Path("/raid/jupyter-linjimmy1003.md10/UCSD_stress")
CACHE = REPO / "results/features_cache"
OUT = REPO / "results/final/source_tables/ft_rsa_stress_eegmat.json"


def compute_rsa(features, labels, pids):
    rdm = squareform(pdist(features, metric="cosine"))
    n = len(features)
    lrdm = (labels[:, None] != labels[None, :]).astype(float)
    srdm = (pids[:, None] != pids[None, :]).astype(float)
    tri = np.triu_indices(n, k=1)
    r_l, _ = spearmanr(rdm[tri], lrdm[tri])
    r_s, _ = spearmanr(rdm[tri], srdm[tri])
    return float(r_l), float(r_s)


def main():
    out = {}
    for fm in ["labram", "cbramod", "reve"]:
        for ds in ["stress", "eegmat"]:
            d = CACHE / f"ft_{fm}_{ds}"
            feats, labels, pids = [], [], []
            for k in range(1, 6):
                p = d / f"fold{k}_features.npz"
                if not p.exists():
                    break
                f = np.load(p, allow_pickle=True)
                feats.append(f["features"])
                labels.append(f["labels"])
                pids.append(f["patient_ids"])
            if not feats:
                print(f"MISSING {fm}_{ds}")
                continue
            X = np.concatenate(feats)
            y = np.concatenate(labels).astype(int)
            s = np.concatenate(pids).astype(int)
            rl, rs = compute_rsa(X, y, s)
            out[f"{fm}_{ds}"] = {
                "ft_rsa_label_r": round(rl, 4),
                "ft_rsa_subject_r": round(rs, 4),
                "n_windows": int(len(X)),
            }
            print(f"{fm}_{ds}  FT: rsa_label_r={rl:+.4f}  rsa_subject_r={rs:+.4f}  (n={len(X)})")
    OUT.write_text(json.dumps(out, indent=2))
    print(f"wrote {OUT.relative_to(REPO)}")


if __name__ == "__main__":
    main()

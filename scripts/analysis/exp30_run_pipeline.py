"""exp_30 SDL-vs-Between pipeline — single orchestrator.

Tests the v2 "SDL as benchmark critique" framing (docs/paper_strategy_sdl_critique.md)
by assembling a master table across 6 datasets x 3 FMs = 18 cells, split into
a within-subject arm (Stress, EEGMAT, SleepDep) and a between-subject arm
(ADFTD, TDBRAIN, Meditation).

Stages A-K each write a checkpoint CSV/JSON into
  results/studies/exp_30_sdl_vs_between/tables/
Figure (J) writes to figures/; REPORT (K) writes to the study root.

Run:
    /raid/jupyter-linjimmy1003.md10/.conda/envs/stress/bin/python \
        scripts/analysis/exp30_run_pipeline.py

Fail-soft on missing data: NaN rather than crash. No GPU needed.
"""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.variance_analysis import (  # noqa: E402
    cluster_bootstrap,
    nested_ss,
    omega_squared_from_ss,
    subject_level_permanova,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------
STUDY_DIR = REPO_ROOT / "results" / "studies" / "exp_30_sdl_vs_between"
TABLES = STUDY_DIR / "tables"
FIGURES = STUDY_DIR / "figures"
LOGS = STUDY_DIR / "logs"
for d in (TABLES, FIGURES, LOGS):
    d.mkdir(parents=True, exist_ok=True)

FEATURES_CACHE = REPO_ROOT / "results" / "features_cache"
EXP_NEWDATA = REPO_ROOT / "results" / "studies" / "exp_newdata"
SOURCE_TABLES = REPO_ROOT / "paper" / "figures" / "source_tables"
HHSA_ROOT = REPO_ROOT / "results" / "hhsa"

DATASETS = ["stress", "eegmat", "sleepdep", "adftd", "tdbrain", "meditation"]
WITHIN = {"stress", "eegmat", "sleepdep"}
BETWEEN = {"adftd", "tdbrain", "meditation"}
FMS = ["labram", "cbramod", "reve"]
SEEDS = [42, 123, 2024]

# Dataset channel suffix for frozen feature files.
DS_SUFFIX = {
    "stress": "30ch",
    "eegmat": "19ch",
    "sleepdep": "19ch",
    "adftd": "19ch",
    "tdbrain": "19ch",
    "meditation": "19ch",
}

# Arm-design mapping (v2 convention: within => crossed SS, between => nested).
# crossed: each subject has multiple labels (label is a within-subject factor)
# nested:  each subject has exactly one label (label is subject-level)
DESIGN = {
    "stress": "nested",      # per-recording DASS, but subjects usually pure
    "eegmat": "crossed",     # rest/task within subject
    "sleepdep": "crossed",   # NS/SD within subject
    "adftd": "nested",
    "tdbrain": "nested",
    "meditation": "nested",  # 2 sessions, same label
}

# For the paper_strategy_sdl_critique tex claim, arm overrides any design
# coincidence (Stress is within-subject even if nested_ss works on a purged
# subset).
ARM = {d: ("within" if d in WITHIN else "between") for d in DATASETS}

# HHSA dataset name mapping in results/hhsa/07_full_holospectrum/
HHSA_DIR_NAME = {
    "stress": "stress",
    "eegmat": "eegmat",
    "sleepdep": "sleep_deprivation",
    "meditation": "meditation_expert_novice",
    # adftd/tdbrain: no HHSA outputs
}


def _log(stage: str, msg: str) -> None:
    print(f"[exp30 {stage}] {msg}", flush=True)


def _save_csv_json(df: pd.DataFrame, name: str) -> None:
    csv_path = TABLES / f"{name}.csv"
    json_path = TABLES / f"{name}.json"
    df.to_csv(csv_path, index=False)
    df.to_json(json_path, orient="records", indent=2)
    _log("IO", f"saved {csv_path.name} (rows={len(df)})")


def _load_frozen(fm: str, dataset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Load frozen features. Returns (features, subject_ids, labels) or None."""
    path = FEATURES_CACHE / f"frozen_{fm}_{dataset}_{DS_SUFFIX[dataset]}.npz"
    if not path.is_file():
        return None
    d = np.load(path)
    return d["features"], d["patient_ids"], d["labels"]


# ----------------------------------------------------------------------
# Stage A — FM performance (FT BA, LP BA per seed/mean)
# ----------------------------------------------------------------------
def collect_fm_performance() -> pd.DataFrame:
    _log("A", "collecting FM performance")
    rows: list[dict] = []

    # (1) SleepDep / Meditation — per-seed from exp_newdata/<ds>_<mode>_<fm>_s<seed>/summary.json
    for dataset in ("sleepdep", "meditation"):
        for fm in FMS:
            for mode in ("ft", "lp"):
                for seed in SEEDS:
                    summary_path = EXP_NEWDATA / f"{dataset}_{mode}_{fm}_s{seed}" / "summary.json"
                    if not summary_path.is_file():
                        rows.append({
                            "dataset": dataset, "fm": fm, "mode": mode,
                            "seed": seed, "bal_acc": np.nan,
                            "source": "missing",
                        })
                        continue
                    try:
                        with open(summary_path) as fh:
                            js = json.load(fh)
                        ba = float(js.get("subject_bal_acc", np.nan))
                    except Exception as exc:
                        _log("A", f"  parse-err {summary_path}: {exc}")
                        ba = np.nan
                    rows.append({
                        "dataset": dataset, "fm": fm, "mode": mode,
                        "seed": seed, "bal_acc": ba,
                        "source": f"exp_newdata/{dataset}_{mode}_{fm}_s{seed}",
                    })

    # (2) Stress / EEGMAT / ADFTD / TDBRAIN — from master_frozen_ft_table.json
    mf_path = SOURCE_TABLES / "master_frozen_ft_table.json"
    if mf_path.is_file():
        with open(mf_path) as fh:
            mf = json.load(fh)
        tbl = mf.get("table", {})
        for dataset in ("stress", "eegmat", "adftd", "tdbrain"):
            for fm in FMS:
                cell = tbl.get(fm, {}).get(dataset, {})
                if not cell:
                    for mode in ("ft", "lp"):
                        rows.append({
                            "dataset": dataset, "fm": fm, "mode": mode,
                            "seed": None, "bal_acc": np.nan,
                            "source": "missing_master_table",
                        })
                    continue
                # frozen_mean maps to LP (linear probe on frozen)
                rows.append({
                    "dataset": dataset, "fm": fm, "mode": "lp",
                    "seed": None,
                    "bal_acc": float(cell.get("frozen_mean", np.nan)),
                    "bal_acc_std": float(cell.get("frozen_std", np.nan)),
                    "n": int(cell.get("frozen_n", 0)),
                    "source": "master_frozen_ft_table.json:frozen",
                })
                rows.append({
                    "dataset": dataset, "fm": fm, "mode": "ft",
                    "seed": None,
                    "bal_acc": float(cell.get("ft_mean", np.nan)),
                    "bal_acc_std": float(cell.get("ft_std", np.nan)),
                    "n": int(cell.get("ft_n", 0)),
                    "source": "master_frozen_ft_table.json:ft",
                })
    else:
        _log("A", f"  WARN: {mf_path} missing")

    df = pd.DataFrame(rows)
    _save_csv_json(df, "fm_performance")
    return df


# ----------------------------------------------------------------------
# Stage B — Variance decomposition (label_frac / subject_frac / residual)
# ----------------------------------------------------------------------
def _crossed_ss_fractions(
    features: np.ndarray, subject: np.ndarray, label: np.ndarray
) -> dict[str, float]:
    """Two-factor SS fractions for CROSSED designs (each subject has multiple
    labels). Not orthogonal (unbalanced) so we report sum-of-squares fractions
    computed as marginal effects:
        SS_label   = sum_l n_l * (mean_l - grand)^2
        SS_subject = sum_s n_s * (mean_s - grand)^2
    These do NOT partition SS_total (overlap exists when design is unbalanced)
    but each is a legitimate "fraction of total variance explainable by factor
    X alone" — matches how variance_analysis_all.json EEGMAT rows were computed.
    """
    f = np.asarray(features, dtype=np.float64)
    s = np.asarray(subject)
    y = np.asarray(label)
    grand = f.mean(axis=0, keepdims=True)
    ss_total = ((f - grand) ** 2).sum()

    ss_label = 0.0
    for lab in np.unique(y):
        m = y == lab
        n_l = m.sum()
        if n_l < 1:
            continue
        mean_l = f[m].mean(axis=0)
        ss_label += n_l * float(((mean_l - grand.squeeze()) ** 2).sum())

    ss_subject = 0.0
    for sid in np.unique(s):
        m = s == sid
        n_s = m.sum()
        if n_s < 1:
            continue
        mean_s = f[m].mean(axis=0)
        ss_subject += n_s * float(((mean_s - grand.squeeze()) ** 2).sum())

    total_safe = max(ss_total, 1e-18)
    frac_label = float(ss_label / total_safe)
    frac_subject = float(ss_subject / total_safe)
    frac_residual = float(max(1.0 - frac_label - frac_subject, 0.0))
    return {
        "frac_label": frac_label * 100.0,   # paper convention: percent
        "frac_subject": frac_subject * 100.0,
        "frac_residual": frac_residual * 100.0,
    }


def _nested_ss_fractions(
    features: np.ndarray, subject: np.ndarray, label: np.ndarray
) -> dict[str, float]:
    """Nested SS fractions using src.variance_analysis.nested_ss."""
    ss = nested_ss(features, subject, label)
    total = max(float(ss["total"].sum()), 1e-18)
    frac_label = float(ss["label"].sum() / total) * 100.0
    frac_subject = float(ss["subject_within_label"].sum() / total) * 100.0
    frac_residual = float(ss["residual"].sum() / total) * 100.0
    return {
        "frac_label": frac_label,
        "frac_subject": frac_subject,
        "frac_residual": frac_residual,
    }


def _compute_variance(
    fm: str, dataset: str
) -> dict[str, Any] | None:
    data = _load_frozen(fm, dataset)
    if data is None:
        return None
    f, s, y = data
    design = DESIGN[dataset]
    try:
        if design == "crossed":
            fracs = _crossed_ss_fractions(f, s, y)
        else:
            # Nested — may fail if subjects are mixed-label (Stress has this).
            # Drop mixed-label subjects first, matching run_variance_analysis.py.
            pids = np.unique(s)
            keep_sub = np.array([
                len(np.unique(y[s == p])) == 1 for p in pids
            ])
            mixed = set(pids[~keep_sub].tolist())
            if mixed:
                mask = np.array([p not in mixed for p in s])
                f, s, y = f[mask], s[mask], y[mask]
            fracs = _nested_ss_fractions(f, s, y)
    except Exception as exc:
        _log("B", f"  {fm}/{dataset} variance err: {exc}")
        return None
    fracs["design"] = design
    fracs["source"] = "computed_from_frozen"
    return fracs


def load_variance_decomposition() -> pd.DataFrame:
    _log("B", "loading / computing variance decomposition")
    # (1) Pull Stress/EEGMAT/ADFTD/TDBRAIN from variance_analysis_all.json (authoritative)
    var_path = SOURCE_TABLES / "variance_analysis_all.json"
    precomputed: dict[tuple[str, str], dict] = {}
    if var_path.is_file():
        with open(var_path) as fh:
            va_all = json.load(fh)
        for key, v in va_all.items():
            model = v.get("model")
            ds = v.get("dataset")
            if model is None or ds is None:
                continue
            precomputed[(model, ds)] = {
                "design": v.get("design", "nested"),
                "frac_label": float(v.get("frozen_label_frac", np.nan)),
                "frac_subject": float(v.get("frozen_subject_frac", np.nan)),
            }

    rows: list[dict] = []
    for dataset in DATASETS:
        for fm in FMS:
            row = {
                "dataset": dataset, "fm": fm,
                "design": DESIGN[dataset],
                "frozen_label_frac": np.nan,
                "frozen_subject_frac": np.nan,
                "frozen_residual_frac": np.nan,
                "subject_to_label_ratio": np.nan,
                "source": None,
            }
            key = (fm, dataset)
            if key in precomputed and not (
                np.isnan(precomputed[key]["frac_label"])
                or np.isnan(precomputed[key]["frac_subject"])
            ):
                pc = precomputed[key]
                row["frozen_label_frac"] = pc["frac_label"]
                row["frozen_subject_frac"] = pc["frac_subject"]
                row["frozen_residual_frac"] = max(
                    100.0 - pc["frac_label"] - pc["frac_subject"], 0.0
                )
                row["design"] = pc["design"]
                row["source"] = "variance_analysis_all.json"
            else:
                # Compute on-the-fly from frozen features.
                res = _compute_variance(fm, dataset)
                if res is not None:
                    row["frozen_label_frac"] = res["frac_label"]
                    row["frozen_subject_frac"] = res["frac_subject"]
                    row["frozen_residual_frac"] = res["frac_residual"]
                    row["design"] = res["design"]
                    row["source"] = res["source"]

            if (
                not np.isnan(row["frozen_label_frac"])
                and not np.isnan(row["frozen_subject_frac"])
                and row["frozen_label_frac"] > 0
            ):
                row["subject_to_label_ratio"] = (
                    row["frozen_subject_frac"] / row["frozen_label_frac"]
                )
            rows.append(row)

    df = pd.DataFrame(rows)
    _save_csv_json(df, "variance_decomposition")
    return df


# ----------------------------------------------------------------------
# Stage C — RSA
# ----------------------------------------------------------------------
def _rsa_spearman(feature_rdm: np.ndarray, target_rdm: np.ndarray) -> float:
    from scipy.stats import spearmanr
    # Both arrays are upper-triangle vectors from pdist.
    if len(feature_rdm) != len(target_rdm) or len(feature_rdm) < 3:
        return float("nan")
    rho, _ = spearmanr(feature_rdm, target_rdm)
    return float(rho) if np.isfinite(rho) else float("nan")


def compute_rsa() -> pd.DataFrame:
    _log("C", "computing RSA (frozen feature RDM vs subject/label RDMs)")
    from scipy.spatial.distance import pdist
    rows: list[dict] = []
    for dataset in DATASETS:
        for fm in FMS:
            data = _load_frozen(fm, dataset)
            if data is None:
                rows.append({
                    "dataset": dataset, "fm": fm,
                    "rsa_subject": np.nan, "rsa_label": np.nan,
                    "n": 0, "source": "missing",
                })
                continue
            f, s, y = data
            # Feature RDM — correlation distance between recordings.
            try:
                feat_rdm = pdist(f, metric="correlation")
            except Exception as exc:
                _log("C", f"  {fm}/{dataset} pdist err: {exc}")
                rows.append({
                    "dataset": dataset, "fm": fm,
                    "rsa_subject": np.nan, "rsa_label": np.nan,
                    "n": len(f), "source": "pdist_err",
                })
                continue
            # Subject RDM: 0 if same subject, 1 otherwise (upper-triangle vec).
            N = len(s)
            iu = np.triu_indices(N, k=1)
            subj_rdm = (s[iu[0]] != s[iu[1]]).astype(float)
            lab_rdm = (y[iu[0]] != y[iu[1]]).astype(float)
            rsa_s = _rsa_spearman(feat_rdm, subj_rdm)
            rsa_l = _rsa_spearman(feat_rdm, lab_rdm)
            rows.append({
                "dataset": dataset, "fm": fm,
                "rsa_subject": rsa_s, "rsa_label": rsa_l,
                "n": N, "source": "computed",
            })
    df = pd.DataFrame(rows)
    _save_csv_json(df, "rsa")
    return df


# ----------------------------------------------------------------------
# Stage D — Subject decodability
# ----------------------------------------------------------------------
def _run_probe(f: np.ndarray, s: np.ndarray, probe_kind: str,
               cv_name: str, cv_splits) -> float:
    """Fit probe (LR or MLP) under the given CV and return balanced accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import balanced_accuracy_score

    preds = np.empty(len(s), dtype=s.dtype)
    used = np.zeros(len(s), dtype=bool)
    for tr, te in cv_splits:
        if probe_kind == "lr":
            clf = LogisticRegression(solver="saga", max_iter=2000, n_jobs=1)
        else:
            clf = MLPClassifier(
                hidden_layer_sizes=(64,), max_iter=400,
                early_stopping=False, random_state=42,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(f[tr], s[tr])
            pr = clf.predict(f[te])
        preds[te] = pr if cv_name != "loo" else pr[0]
        used[te] = True
    return float(balanced_accuracy_score(s[used], preds[used]))


def compute_subject_decodability() -> pd.DataFrame:
    """Compute subject-ID decodability via BOTH linear (LR) and MLP probes.

    LR = Hewitt-Liang-style linear probe (standard for representation analysis).
    MLP (64 hidden) = robustness check — captures non-linearly separable subject
    structure.
    """
    _log("D", "computing subject-ID decodability (LR + MLP probes)")
    from sklearn.model_selection import LeaveOneOut, StratifiedKFold

    rows: list[dict] = []
    for dataset in DATASETS:
        for fm in FMS:
            data = _load_frozen(fm, dataset)
            if data is None:
                rows.append({
                    "dataset": dataset, "fm": fm,
                    "subject_id_ba_lr": np.nan, "subject_id_ba_mlp": np.nan,
                    "n_subjects": 0, "n_samples": 0, "cv": "none",
                    "source": "missing",
                })
                continue
            f, s, _y = data
            n_subj = len(np.unique(s))
            N = len(s)

            _unique, counts = np.unique(s, return_counts=True)
            min_per = int(counts.min())

            # Pick CV scheme
            ba_lr, ba_mlp = float("nan"), float("nan")
            cv_name = "err"
            try:
                if min_per >= 2 and N >= 15:
                    k = min(5, int(min_per))
                    if k < 2:
                        raise ValueError("k<2")
                    cv_name = f"skf{k}"
                    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
                    splits = list(skf.split(f, s))
                else:
                    if N > 300:
                        raise ValueError(f"skipping LOO for N={N}")
                    cv_name = "loo"
                    splits = list(LeaveOneOut().split(f))

                # LR probe
                try:
                    ba_lr = _run_probe(f, s, "lr", cv_name, splits)
                except Exception as exc_lr:
                    _log("D", f"  {fm}/{dataset} LR err: {exc_lr}")

                # MLP probe
                try:
                    ba_mlp = _run_probe(f, s, "mlp", cv_name, splits)
                except Exception as exc_mlp:
                    _log("D", f"  {fm}/{dataset} MLP err: {exc_mlp}")
            except Exception as exc:
                _log("D", f"  {fm}/{dataset} probe err: {exc}")

            rows.append({
                "dataset": dataset, "fm": fm,
                "subject_id_ba_lr": ba_lr, "subject_id_ba_mlp": ba_mlp,
                "n_subjects": n_subj, "n_samples": N,
                "cv": cv_name, "source": "computed",
            })
            _log(
                "D", f"  {fm}/{dataset}: lr={ba_lr:.3f} mlp={ba_mlp:.3f} "
                     f"(n_subj={n_subj}, n={N}, cv={cv_name})"
            )

    df = pd.DataFrame(rows)
    # Keep legacy single-value column for back-compat with downstream builders
    df["subject_id_ba"] = df["subject_id_ba_lr"]
    _save_csv_json(df, "subject_decodability")
    return df


# ----------------------------------------------------------------------
# Stage E — PERMANOVA
# ----------------------------------------------------------------------
def compute_permanova() -> pd.DataFrame:
    _log("E", "collecting / computing PERMANOVA")
    # Pre-populate from variance_analysis_all.json where available (4 datasets).
    var_path = SOURCE_TABLES / "variance_analysis_all.json"
    pre: dict[tuple[str, str], dict] = {}
    if var_path.is_file():
        with open(var_path) as fh:
            va_all = json.load(fh)
        for _k, v in va_all.items():
            fm = v.get("model"); ds = v.get("dataset")
            if fm is None or ds is None:
                continue
            if "permanova_frozen" in v:
                pre[(fm, ds)] = v["permanova_frozen"]

    rows: list[dict] = []
    for dataset in DATASETS:
        for fm in FMS:
            key = (fm, dataset)
            if key in pre:
                d = pre[key]
                rows.append({
                    "dataset": dataset, "fm": fm,
                    "permanova_pseudo_F": float(d["pseudo_F"]),
                    "permanova_R2": float(d["R2"]),
                    "permanova_p": float(d.get("p_value", np.nan)),
                    "source": "variance_analysis_all.json",
                })
                continue
            data = _load_frozen(fm, dataset)
            if data is None:
                rows.append({
                    "dataset": dataset, "fm": fm,
                    "permanova_pseudo_F": np.nan, "permanova_R2": np.nan,
                    "permanova_p": np.nan, "source": "missing",
                })
                continue
            f, s, y = data
            try:
                # Subject-level PERMANOVA requires each subject to have a
                # single label. For crossed designs (EEGMAT/SleepDep), the
                # subject-level permanova is not well-defined, so we fall back
                # to a plain label permutation via the same function — still
                # reports a valid pseudo-F even if permutation p is not
                # strictly the right null.
                # For crossed designs we simply run it and label source.
                res = subject_level_permanova(f, s, y, n_perm=199, seed=42)
                rows.append({
                    "dataset": dataset, "fm": fm,
                    "permanova_pseudo_F": float(res["pseudo_F"]),
                    "permanova_R2": float(res["R2"]),
                    "permanova_p": float(res["p_value"]),
                    "source": "computed",
                })
            except Exception as exc:
                _log("E", f"  {fm}/{dataset} permanova err: {exc}")
                rows.append({
                    "dataset": dataset, "fm": fm,
                    "permanova_pseudo_F": np.nan, "permanova_R2": np.nan,
                    "permanova_p": np.nan, "source": "err",
                })
    df = pd.DataFrame(rows)
    _save_csv_json(df, "permanova")
    return df


# ----------------------------------------------------------------------
# Stage F — HHSA contrast (collection-only)
# ----------------------------------------------------------------------
def compute_hhsa_contrast() -> pd.DataFrame:
    _log("F", "collecting existing HHSA contrast outputs")
    rows: list[dict] = []

    # Try cross_dataset_comparison summary first — has mean_t for 4 datasets.
    summary_path = HHSA_ROOT / "cross_dataset_comparison" / "summary_stats.npz"
    mean_t_map: dict[str, float] = {}
    max_t_map: dict[str, float] = {}
    if summary_path.is_file():
        try:
            s = np.load(summary_path, allow_pickle=True)
            names = [str(x).lower() for x in s["datasets"]]
            for i, n in enumerate(names):
                # Normalise names to our dataset keys.
                if "eegmat" in n:
                    key = "eegmat"
                elif "stress" in n:
                    key = "stress"
                elif "sleep" in n:
                    key = "sleepdep"
                elif "meditation" in n:
                    key = "meditation"
                else:
                    continue
                mean_t_map[key] = float(s["mean_t"][i])
                max_t_map[key] = float(s["max_t"][i])
        except Exception as exc:
            _log("F", f"  summary_stats load err: {exc}")

    # Fall back to per-dataset 07_full_holospectrum/{name}/condition_contrast.npz
    # for any dataset not in the summary. We derive median |t| as the contrast
    # statistic (analogous to median g used elsewhere).
    def _load_per_dataset(ds_key: str) -> tuple[float, float] | None:
        name = HHSA_DIR_NAME.get(ds_key)
        if name is None:
            return None
        path = HHSA_ROOT / "07_full_holospectrum" / name / "condition_contrast.npz"
        if not path.is_file():
            return None
        try:
            z = np.load(path)
            t_map = z["t_map"]
            finite = t_map[np.isfinite(t_map)]
            if finite.size == 0:
                return None
            return float(np.median(np.abs(finite))), float(np.max(np.abs(finite)))
        except Exception as exc:
            _log("F", f"  {ds_key} contrast load err: {exc}")
            return None

    for dataset in DATASETS:
        median_g = np.nan
        max_t = np.nan
        if dataset in mean_t_map:
            median_g = mean_t_map[dataset]
            max_t = max_t_map[dataset]
            source = "cross_dataset_comparison/summary_stats.npz:mean_t"
        else:
            per = _load_per_dataset(dataset)
            if per is not None:
                median_g, max_t = per
                source = "07_full_holospectrum/condition_contrast.npz:median|t|"
            else:
                source = "missing"
        rows.append({
            "dataset": dataset,
            "hhsa_median_g": median_g,
            "hhsa_max_t": max_t,
            "hhsa_wsci": np.nan,     # not collected here
            "hhsa_ci_low": np.nan,
            "hhsa_ci_high": np.nan,
            "source": source,
        })
    df = pd.DataFrame(rows)
    _save_csv_json(df, "hhsa_contrast")
    return df


# ----------------------------------------------------------------------
# Stage G — Master table
# ----------------------------------------------------------------------
def build_master_table(
    fm_perf: pd.DataFrame, variance_df: pd.DataFrame, rsa_df: pd.DataFrame,
    subj_df: pd.DataFrame, perm_df: pd.DataFrame, hhsa_df: pd.DataFrame,
) -> pd.DataFrame:
    _log("G", "building master table")
    # Aggregate FM performance to (dataset, fm) ft_mean / lp_mean.
    def _agg_mode(df: pd.DataFrame, mode: str, col: str) -> pd.DataFrame:
        sub = df[df["mode"] == mode]
        # For source master_frozen_ft_table rows: seed=None, bal_acc_mean already.
        # For per-seed rows: average across seeds.
        grouped = (
            sub.groupby(["dataset", "fm"])["bal_acc"]
            .mean()
            .reset_index()
            .rename(columns={"bal_acc": col})
        )
        return grouped

    ft_df = _agg_mode(fm_perf, "ft", "ft_ba_mean")
    lp_df = _agg_mode(fm_perf, "lp", "lp_ba_mean")

    # Start with all 18 cells.
    grid = [{"dataset": d, "fm": f} for d in DATASETS for f in FMS]
    master = pd.DataFrame(grid)

    master = master.merge(ft_df, on=["dataset", "fm"], how="left")
    master = master.merge(lp_df, on=["dataset", "fm"], how="left")
    master = master.merge(
        variance_df[[
            "dataset", "fm", "design",
            "frozen_label_frac", "frozen_subject_frac",
            "frozen_residual_frac", "subject_to_label_ratio",
        ]],
        on=["dataset", "fm"], how="left",
    )
    master = master.merge(
        rsa_df[["dataset", "fm", "rsa_subject", "rsa_label"]],
        on=["dataset", "fm"], how="left",
    )
    # Pull both probe variants: subject_id_ba_lr (primary, linear probe) and
    # subject_id_ba_mlp (robustness probe). `subject_id_ba` is aliased to _lr.
    subj_cols = ["dataset", "fm", "n_subjects"]
    for c in ("subject_id_ba_lr", "subject_id_ba_mlp", "subject_id_ba"):
        if c in subj_df.columns:
            subj_cols.append(c)
    master = master.merge(subj_df[subj_cols], on=["dataset", "fm"], how="left")
    master = master.merge(
        perm_df[["dataset", "fm", "permanova_pseudo_F", "permanova_R2"]],
        on=["dataset", "fm"], how="left",
    )
    master = master.merge(
        hhsa_df[["dataset", "hhsa_median_g", "hhsa_wsci"]],
        on="dataset", how="left",
    )
    master["arm"] = master["dataset"].map(ARM)
    master["delta_ba"] = master["ft_ba_mean"] - master["lp_ba_mean"]

    # Reorder columns for readability.
    preferred = [
        "dataset", "fm", "arm", "design",
        "lp_ba_mean", "ft_ba_mean", "delta_ba",
        "frozen_label_frac", "frozen_subject_frac",
        "frozen_residual_frac", "subject_to_label_ratio",
        "rsa_subject", "rsa_label",
        "subject_id_ba_lr", "subject_id_ba_mlp", "subject_id_ba",
        "n_subjects",
        "permanova_pseudo_F", "permanova_R2",
        "hhsa_median_g", "hhsa_wsci",
    ]
    master = master[[c for c in preferred if c in master.columns]]
    _save_csv_json(master, "master_table")
    return master


# ----------------------------------------------------------------------
# Stage H — Correlations (Spearman ρ(predictor, ΔBA) per arm, bootstrap CI)
# ----------------------------------------------------------------------
def compute_correlations(master: pd.DataFrame) -> pd.DataFrame:
    _log("H", "computing Spearman correlations + bootstrap CIs")
    from scipy.stats import spearmanr

    predictors = [
        "frozen_label_frac", "frozen_subject_frac", "subject_to_label_ratio",
        "rsa_subject", "rsa_label",
        "subject_id_ba_lr", "subject_id_ba_mlp",
        "permanova_pseudo_F", "hhsa_median_g",
    ]

    rng = np.random.default_rng(0)
    n_boot = 10000
    rows: list[dict] = []

    # Stress has only 3/17 subjects crossing DASS classes — its within-subject
    # design is degenerate. within_strict = EEGMAT + SleepDep only (n=6 cells).
    for pred in predictors:
        if pred not in master.columns:
            continue
        if master[pred].notna().sum() == 0:
            continue
        for arm in ("within", "within_strict", "between", "pooled"):
            if arm == "pooled":
                sub = master
            elif arm == "within_strict":
                sub = master[(master["arm"] == "within")
                             & (master["dataset"] != "stress")]
            else:
                sub = master[master["arm"] == arm]
            # Keep rows with finite predictor and delta_ba.
            x = sub[pred].to_numpy(dtype=float)
            ymat = sub["delta_ba"].to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(ymat)
            xf, yf = x[mask], ymat[mask]
            n = int(mask.sum())
            if n < 3:
                rows.append({
                    "predictor": pred, "arm": arm, "n": n,
                    "rho": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                    "p": np.nan,
                })
                continue
            rho, p = spearmanr(xf, yf)
            # Bootstrap: resample rows with replacement; compute ρ each time.
            boot = np.empty(n_boot)
            for b in range(n_boot):
                idx = rng.integers(0, n, size=n)
                try:
                    if np.all(xf[idx] == xf[idx][0]) or np.all(yf[idx] == yf[idx][0]):
                        boot[b] = np.nan
                        continue
                    r, _ = spearmanr(xf[idx], yf[idx])
                    boot[b] = r
                except Exception:
                    boot[b] = np.nan
            boot = boot[np.isfinite(boot)]
            if len(boot) >= 2:
                lo, hi = np.percentile(boot, [2.5, 97.5])
            else:
                lo, hi = np.nan, np.nan
            rows.append({
                "predictor": pred, "arm": arm, "n": n,
                "rho": float(rho) if np.isfinite(rho) else np.nan,
                "ci_low": float(lo) if np.isfinite(lo) else np.nan,
                "ci_high": float(hi) if np.isfinite(hi) else np.nan,
                "p": float(p) if np.isfinite(p) else np.nan,
            })

    df = pd.DataFrame(rows)
    _save_csv_json(df, "correlations")
    return df


# ----------------------------------------------------------------------
# Stage I — Figure
# ----------------------------------------------------------------------
def make_figure(master: pd.DataFrame, corr: pd.DataFrame) -> None:
    _log("I", "rendering 2-panel figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fm_colors = {"labram": "#1f77b4", "cbramod": "#d62728", "reve": "#2ca02c"}
    ds_markers = {
        "stress": "o", "eegmat": "s", "sleepdep": "^",
        "adftd": "D", "tdbrain": "v", "meditation": "P",
    }

    fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
    panels = [
        ("within", "frozen_label_frac", axes[0, 0], "Panel A: Within-subject arm"),
        ("between", "frozen_label_frac", axes[0, 1], "Panel B: Between-subject arm"),
        ("within", "subject_id_ba", axes[1, 0], "Panel C: ΔBA vs subject_id_ba (within)"),
        ("between", "subject_id_ba", axes[1, 1], "Panel D: ΔBA vs subject_id_ba (between)"),
    ]

    for arm, predictor, ax, title in panels:
        sub = master[master["arm"] == arm]
        for _, row in sub.iterrows():
            x = row.get(predictor)
            y = row.get("delta_ba")
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            ax.scatter(
                x, y,
                color=fm_colors.get(row["fm"], "gray"),
                marker=ds_markers.get(row["dataset"], "o"),
                s=110, edgecolor="black", linewidth=0.6, alpha=0.85,
                label=f"{row['fm']}/{row['dataset']}",
            )
        ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
        # Overlay ρ + CI.
        crow = corr[(corr["predictor"] == predictor) & (corr["arm"] == arm)]
        if len(crow) == 1 and np.isfinite(crow.iloc[0]["rho"]):
            r = crow.iloc[0]
            text = (
                f"Spearman ρ={r['rho']:.2f}\n"
                f"95% CI [{r['ci_low']:.2f}, {r['ci_high']:.2f}]\n"
                f"n={int(r['n'])}"
            )
            ax.text(
                0.02, 0.98, text, transform=ax.transAxes,
                fontsize=9, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85),
            )
        ax.set_xlabel(predictor.replace("_", " "))
        ax.set_ylabel("ΔBA (FT − LP)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)

    # Single combined legend (FM x dataset) on right.
    handles = []
    labels = []
    for fm, col in fm_colors.items():
        for ds, mk in ds_markers.items():
            if ARM.get(ds) is None:
                continue
            handles.append(plt.Line2D(
                [0], [0], marker=mk, color="w",
                markerfacecolor=col, markeredgecolor="black",
                markersize=8,
            ))
            labels.append(f"{fm} / {ds}")
    # Too many entries for a tidy legend — use FM-only + dataset-only legends.
    fm_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c,
                   markeredgecolor="black", markersize=9, label=fm)
        for fm, c in fm_colors.items()
    ]
    ds_handles = [
        plt.Line2D([0], [0], marker=mk, color="w", markerfacecolor="gray",
                   markeredgecolor="black", markersize=9, label=ds)
        for ds, mk in ds_markers.items()
    ]
    axes[0, 1].legend(
        handles=fm_handles, loc="lower right", fontsize=8, title="FM",
    )
    axes[1, 1].legend(
        handles=ds_handles, loc="lower right", fontsize=8, title="Dataset",
    )
    fig.suptitle(
        "exp_30 — SDL as benchmark critique: within vs between arm", fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    pdf_path = FIGURES / "fig_sdl_benchmark_critique.pdf"
    png_path = FIGURES / "fig_sdl_benchmark_critique.png"
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    _log("I", f"saved {pdf_path.name} + .png")


# ----------------------------------------------------------------------
# Stage J — REPORT.md
# ----------------------------------------------------------------------
def _verdict(rho: float | None, ci_low: float | None, ci_high: float | None,
             direction: str) -> str:
    """direction: 'positive', 'zero', 'negative' — what we expected.

    Returns 'PASS' / 'FAIL' / 'UNDETERMINED'.
    """
    if rho is None or not np.isfinite(rho):
        return "UNDETERMINED"
    if not (np.isfinite(ci_low) and np.isfinite(ci_high)):
        return "UNDETERMINED"
    if direction == "positive":
        return "PASS" if ci_low > 0 else "FAIL"
    if direction == "negative":
        return "PASS" if ci_high < 0 else "FAIL"
    if direction == "zero":
        # Passes if CI crosses zero OR if |rho| < some threshold and CI wide.
        return "PASS" if (ci_low <= 0 <= ci_high) else "FAIL"
    return "UNDETERMINED"


def write_report(master: pd.DataFrame, corr: pd.DataFrame) -> None:
    _log("J", "writing REPORT.md")

    def _corr_row(predictor: str, arm: str) -> dict | None:
        sub = corr[(corr["predictor"] == predictor) & (corr["arm"] == arm)]
        if len(sub) != 1:
            return None
        r = sub.iloc[0].to_dict()
        return r

    lines: list[str] = []
    lines.append("# exp_30 — SDL vs Between REPORT")
    lines.append("")
    lines.append("Paper strategy tested: `docs/paper_strategy_sdl_critique.md` (v2).")
    lines.append("Data sources: `results/features_cache/frozen_*.npz`, `results/studies/exp_newdata/`, "
                 "`paper/figures/source_tables/`, `results/hhsa/`.")
    lines.append("")

    # -- Executive summary --
    lines.append("## 1. Executive summary")
    lines.append("")
    # Summarise means by arm.
    within = master[master["arm"] == "within"]
    between = master[master["arm"] == "between"]
    lines.append(
        "This pipeline assembles a 6-dataset x 3-FM master table (18 cells) and asks "
        "whether frozen-feature task variance predicts ΔBA (FT−LP) **differently** "
        "in within-subject vs between-subject benchmarks — the falsifiable core of v2."
    )
    lines.append("")
    f1 = _corr_row("frozen_label_frac", "within")
    f1b = _corr_row("frozen_label_frac", "between")
    if f1 and f1b:
        lines.append(
            f"Headline F1 (within arm, predictor = frozen_label_frac): ρ="
            f"{f1['rho']:.2f} [95% CI {f1['ci_low']:.2f}, {f1['ci_high']:.2f}] "
            f"(n={int(f1['n'])}). F2 (between arm, same predictor): ρ="
            f"{f1b['rho']:.2f} [95% CI {f1b['ci_low']:.2f}, {f1b['ci_high']:.2f}] "
            f"(n={int(f1b['n'])})."
        )
    else:
        lines.append("F1/F2 correlations not computable from the master table.")
    lines.append("")

    # -- Master table preview --
    lines.append("## 2. Master table preview (top rows)")
    lines.append("")
    lines.append("```")
    lines.append(master.head(18).round(3).to_string(index=False))
    lines.append("```")
    lines.append("")

    # -- Per-prediction verdicts --
    lines.append("## 3. Per-prediction verdicts (F1–F4)")
    lines.append("")

    f1_row = _corr_row("frozen_label_frac", "within")
    if f1_row:
        v = _verdict(f1_row["rho"], f1_row["ci_low"], f1_row["ci_high"], "positive")
        lines.append(
            f"**F1** (within-arm ρ(frozen_label_frac, ΔBA) > 0): "
            f"ρ={f1_row['rho']:.2f}, 95% CI [{f1_row['ci_low']:.2f}, "
            f"{f1_row['ci_high']:.2f}], n={int(f1_row['n'])}. **{v}**"
        )
        if v == "FAIL":
            lines.append(
                "- Possible causes: (a) n=9 is under-powered for Spearman; "
                "(b) label_frac is not the right predictor — HHSA contrast or RSA_label "
                "may carry more signal; (c) the two small within-subject datasets "
                "(Stress 14 pure subjects, SleepDep 36) have their own signal ceilings "
                "that dominate ΔBA; (d) FT ceiling for within-subject tasks may be "
                "hit regardless of frozen-feature quality."
            )
    lines.append("")

    f2_row = _corr_row("frozen_label_frac", "between")
    if f2_row:
        v = _verdict(f2_row["rho"], f2_row["ci_low"], f2_row["ci_high"], "zero")
        lines.append(
            f"**F2** (between-arm ρ(frozen_label_frac, ΔBA) ≈ 0): "
            f"ρ={f2_row['rho']:.2f}, 95% CI [{f2_row['ci_low']:.2f}, "
            f"{f2_row['ci_high']:.2f}], n={int(f2_row['n'])}. **{v}**"
        )
    lines.append("")

    f3_within = _corr_row("hhsa_median_g", "within")
    f3_between = _corr_row("hhsa_median_g", "between")
    if f3_within is None or not np.isfinite(f3_within.get("rho", np.nan)):
        lines.append(
            "**F3** (HHSA contrast × ΔBA differs between arms): "
            "UNTESTABLE — HHSA contrast data is only available for 4 of 6 datasets "
            "(ADFTD and TDBRAIN lack HHSA condition-contrast outputs). Follow-up: "
            "run the HHSA pipeline on ADFTD and TDBRAIN before citing F3."
        )
    else:
        v_within = _verdict(
            f3_within["rho"], f3_within["ci_low"], f3_within["ci_high"], "positive"
        )
        v_between_text = (
            f"ρ={f3_between['rho']:.2f}, 95% CI [{f3_between['ci_low']:.2f}, "
            f"{f3_between['ci_high']:.2f}]"
            if f3_between and np.isfinite(f3_between.get("rho", np.nan))
            else "between-arm not computable"
        )
        lines.append(
            f"**F3** (HHSA contrast correlates with ΔBA in within but not between): "
            f"within ρ={f3_within['rho']:.2f} [{f3_within['ci_low']:.2f}, "
            f"{f3_within['ci_high']:.2f}] ({v_within}); between {v_between_text}."
        )
    lines.append("")

    # F4: subject-id BA per arm (mean comparison, not a correlation).
    mean_subj_within = within["subject_id_ba"].mean()
    mean_subj_between = between["subject_id_ba"].mean()
    lines.append(
        f"**F4** (between-arm subject_id_ba > 0.7): "
        f"mean subject_id_ba = {mean_subj_within:.3f} (within) vs "
        f"{mean_subj_between:.3f} (between)."
    )
    if np.isfinite(mean_subj_between):
        if mean_subj_between > 0.7:
            lines.append("- Between-arm subject decodability is high, consistent with C2.")
        else:
            lines.append(
                "- Between-arm subject decodability is below the 0.7 threshold; "
                "C2 (subject-feature extractor claim) is weakened."
            )
    lines.append("")

    # -- Unexpected findings (anything where pooled sign disagrees with arm signs) --
    lines.append("## 4. Unexpected findings")
    lines.append("")
    flagged = []
    for pred in ("frozen_label_frac", "frozen_subject_frac", "rsa_label",
                 "subject_id_ba", "permanova_pseudo_F"):
        w = _corr_row(pred, "within")
        b = _corr_row(pred, "between")
        if w is None or b is None:
            continue
        if not (np.isfinite(w["rho"]) and np.isfinite(b["rho"])):
            continue
        if np.sign(w["rho"]) == np.sign(b["rho"]) and abs(w["rho"]) > 0.3 and abs(b["rho"]) > 0.3:
            flagged.append(
                f"- `{pred}`: ρ(within)={w['rho']:.2f}, ρ(between)={b['rho']:.2f} — "
                "same sign in both arms. If v2 predicted regime separation for this "
                "predictor, this is a counterexample."
            )
        elif np.sign(w["rho"]) != np.sign(b["rho"]) and abs(w["rho"]) > 0.3 and abs(b["rho"]) > 0.3:
            flagged.append(
                f"- `{pred}`: ρ(within)={w['rho']:.2f}, ρ(between)={b['rho']:.2f} — "
                "sign flip between arms, consistent with v2 thesis."
            )
    if not flagged:
        lines.append("(none flagged — all predictors within expected patterns or under-powered)")
    else:
        lines.extend(flagged)
    lines.append("")

    # -- Recommendation --
    lines.append("## 5. Recommendation")
    lines.append("")
    if f1_row and f2_row:
        f1_pass = _verdict(f1_row["rho"], f1_row["ci_low"], f1_row["ci_high"], "positive") == "PASS"
        f2_pass = _verdict(f2_row["rho"], f2_row["ci_low"], f2_row["ci_high"], "zero") == "PASS"
        if f1_pass and f2_pass:
            lines.append(
                "Both F1 and F2 pass. Data support proceeding with the v2 tex rewrite "
                "(SDL as benchmark critique). The two-arm separation is visible."
            )
        elif not f1_pass and not f2_pass:
            lines.append(
                "F1 and F2 both fail. Data do not support v2 in its current form. "
                "Recommend either (a) reverting to v1 framing with v2 machinery demoted "
                "to an appendix, or (b) iterating on the predictor (HHSA contrast, "
                "RSA_label) to see if another metric does split the arms cleanly."
            )
        elif f1_pass and not f2_pass:
            lines.append(
                "F1 passes but F2 fails — within-arm correlation is real, but the "
                "between-arm correlation is also non-zero. v2 critique is weakened: "
                "the predictor works in both regimes, blunting the 'regime-specific' "
                "claim. Consider keeping v1's positive framing with v2 as a subplot."
            )
        elif not f1_pass and f2_pass:
            lines.append(
                "F1 fails and F2 passes. Between-arm is clean but within-arm isn't — "
                "the critique lands but the positive diagnostic (C3/C4) doesn't. "
                "Reconsider which predictor best captures 'task-variance fraction' "
                "before rewriting the tex."
            )
    else:
        lines.append("Insufficient correlation data to recommend a framing.")
    lines.append("")

    # -- Limitations --
    lines.append("## 6. Limitations")
    lines.append("")
    lines.append(
        "- **N = 3 datasets × 3 FMs = 9 points per arm.** Spearman ρ at n=9 has very "
        "wide bootstrap CIs; most non-trivial effects are statistically indistinguishable "
        "from zero at 95%.\n"
        "- **Between-arm TDBRAIN has ~1200 subjects**, making subject-ID decodability "
        "mechanically different (more classes) from ADFTD (65) or Meditation (24). "
        "Comparing subject_id_ba across datasets in this arm is asymmetric.\n"
        "- **HHSA contrast data is missing for ADFTD and TDBRAIN.** F3 is thus untestable "
        "at full coverage; it is only evaluable on the 4-dataset subset (Stress, EEGMAT, "
        "SleepDep, Meditation).\n"
        "- **Stress/EEGMAT/ADFTD/TDBRAIN FM performance uses aggregated mean/std from "
        "`master_frozen_ft_table.json`, not per-seed values**, so the per-seed variance "
        "information is lost for 4 of 6 datasets.\n"
        "- **Variance decomposition for EEGMAT uses a crossed SS (not nested ω²).** "
        "Not directly comparable to ADFTD/TDBRAIN nested ω² values — interpret the "
        "label_frac / subject_frac columns as 'fraction of total SS attributable to "
        "factor X marginally' for crossed designs and as 'nested SS fraction' for "
        "nested designs. Paper-strategy_sdl_critique §3 acknowledges this asymmetry."
    )
    lines.append("")

    # -- Checkpoint inventory --
    lines.append("## 7. Checkpoint files")
    lines.append("")
    for name in [
        "fm_performance", "variance_decomposition", "rsa",
        "subject_decodability", "permanova", "hhsa_contrast",
        "master_table", "correlations",
    ]:
        p = TABLES / f"{name}.csv"
        lines.append(f"- `{p.relative_to(REPO_ROOT)}`")
    lines.append(f"- `{(FIGURES / 'fig_sdl_benchmark_critique.pdf').relative_to(REPO_ROOT)}`")
    lines.append("")

    report_path = STUDY_DIR / "REPORT.md"
    report_path.write_text("\n".join(lines))
    _log("J", f"wrote {report_path.relative_to(REPO_ROOT)}")


# ----------------------------------------------------------------------
# Orchestrator
# ----------------------------------------------------------------------
def main() -> None:
    _log("main", f"study dir: {STUDY_DIR}")
    fm_perf = collect_fm_performance()                 # A
    variance_df = load_variance_decomposition()        # B
    rsa_df = compute_rsa()                             # C
    subj_df = compute_subject_decodability()           # D
    perm_df = compute_permanova()                      # E
    hhsa_df = compute_hhsa_contrast()                  # F
    master = build_master_table(
        fm_perf, variance_df, rsa_df, subj_df, perm_df, hhsa_df,
    )                                                  # G
    corr = compute_correlations(master)                # H
    make_figure(master, corr)                          # I
    write_report(master, corr)                         # J
    _log("main", "pipeline complete")


if __name__ == "__main__":
    main()

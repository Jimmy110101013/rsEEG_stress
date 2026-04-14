"""Compute FM-task fitness metrics for all model × dataset combinations.

Metrics computed:
  1. Silhouette score (label vs subject)
  2. Fisher's criterion (between/within class variance ratio)
  3. kNN accuracy (k=5, LOO for small datasets)
  4. LogME (Bayesian transferability, You et al. ICML 2021)
  5. H-score (Bao et al. ICML 2019)
  6. RSA (Representational Similarity Analysis)
  7. CKA (Centered Kernel Alignment, linear)

All computed on frozen features only — no training required.
"""
import numpy as np
import os
import json
import glob
from itertools import product
from scipy.spatial.distance import pdist, squareform, cosine
from scipy.stats import spearmanr
from sklearn.metrics import silhouette_score, balanced_accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


# ──────────── Metric implementations ────────────


def compute_silhouette(features, labels, patient_ids):
    """Silhouette score for label clustering vs subject clustering."""
    X = StandardScaler().fit_transform(features)
    
    # Label silhouette (need >=2 classes with >=2 members)
    unique_labels = np.unique(labels)
    if len(unique_labels) >= 2:
        sil_label = silhouette_score(X, labels, metric="euclidean")
    else:
        sil_label = float("nan")
    
    # Subject silhouette (only for subjects with >=2 recordings)
    subj_counts = np.bincount(patient_ids.astype(int) if patient_ids.max() < 10000 
                               else np.unique(patient_ids, return_inverse=True)[1])
    # Need subjects with multiple recordings
    pid_mapped = np.unique(patient_ids, return_inverse=True)[1]
    mask = np.array([subj_counts[pid_mapped[i]] >= 2 for i in range(len(patient_ids))])
    if mask.sum() >= 4 and len(np.unique(pid_mapped[mask])) >= 2:
        sil_subject = silhouette_score(X[mask], pid_mapped[mask], metric="euclidean")
    else:
        sil_subject = float("nan")
    
    return {"sil_label": round(sil_label, 4), "sil_subject": round(sil_subject, 4)}


def compute_fisher(features, labels):
    """Fisher's criterion: ratio of between-class to within-class variance."""
    X = StandardScaler().fit_transform(features)
    classes = np.unique(labels)
    if len(classes) < 2:
        return {"fisher": float("nan")}
    
    overall_mean = X.mean(axis=0)
    
    # Between-class scatter
    sb = 0
    for c in classes:
        mask = labels == c
        nc = mask.sum()
        class_mean = X[mask].mean(axis=0)
        diff = class_mean - overall_mean
        sb += nc * np.dot(diff, diff)
    
    # Within-class scatter
    sw = 0
    for c in classes:
        mask = labels == c
        centered = X[mask] - X[mask].mean(axis=0)
        sw += np.sum(centered ** 2)
    
    fisher = sb / max(sw, 1e-10)
    return {"fisher": round(fisher, 6)}


def compute_knn(features, labels, patient_ids, k=5):
    """Subject-level stratified kNN accuracy."""
    X = StandardScaler().fit_transform(features)
    n = len(X)
    
    if n < 20:
        # LOO for small datasets
        loo = LeaveOneOut()
        preds = []
        for train_idx, test_idx in loo.split(X):
            knn = KNeighborsClassifier(n_neighbors=min(k, len(train_idx)))
            knn.fit(X[train_idx], labels[train_idx])
            preds.append(knn.predict(X[test_idx])[0])
        ba = balanced_accuracy_score(labels, preds)
    else:
        # Subject-level 5-fold CV
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        all_true, all_pred = [], []
        for train_idx, test_idx in cv.split(X, labels, groups=patient_ids):
            knn = KNeighborsClassifier(n_neighbors=min(k, len(train_idx)))
            knn.fit(X[train_idx], labels[train_idx])
            all_pred.extend(knn.predict(X[test_idx]))
            all_true.extend(labels[test_idx])
        ba = balanced_accuracy_score(all_true, all_pred)
    
    return {"knn_ba": round(ba, 4)}


def compute_logme(features, labels):
    """LogME: Log of Maximum Evidence (You et al., ICML 2021).
    
    Estimates Bayesian evidence for how well features predict labels.
    Higher = better transferability.
    """
    X = StandardScaler().fit_transform(features)
    n, d = X.shape
    classes = np.unique(labels)
    n_classes = len(classes)
    
    # One-hot encode labels
    Y = np.zeros((n, n_classes))
    for i, c in enumerate(classes):
        Y[labels == c, i] = 1.0
    
    # SVD of features
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    S2 = S ** 2
    
    # Evidence maximization (simplified)
    # Iterate to find optimal alpha and beta
    alpha = 1.0
    beta = 1.0
    
    for _ in range(50):
        # Effective number of parameters
        gamma = np.sum(S2 / (S2 + alpha / beta))
        
        # Update alpha
        # Mean of squared weights
        m = np.zeros((min(n, d), n_classes))
        for j in range(n_classes):
            proj = U.T @ Y[:, j]
            m[:, j] = beta * S * proj / (beta * S2 + alpha)
        
        alpha_new = gamma / (np.sum(m ** 2) / n_classes + 1e-10)
        
        # Update beta
        residual = 0
        for j in range(n_classes):
            pred = U @ (S[:, None] * m[:, j:j+1]).squeeze()
            residual += np.sum((Y[:, j] - pred) ** 2)
        beta_new = (n * n_classes - gamma) / (residual + 1e-10)
        
        if abs(alpha_new - alpha) / max(alpha, 1e-10) < 1e-4:
            break
        alpha = alpha_new
        beta = beta_new
    
    # Compute log evidence
    log_evidence = 0
    for j in range(n_classes):
        proj = U.T @ Y[:, j]
        # Log likelihood term
        pred = U @ (S[:, None] * m[:, j:j+1]).squeeze()
        log_evidence += -0.5 * beta * np.sum((Y[:, j] - pred) ** 2)
        # Log prior term
        log_evidence += -0.5 * alpha * np.sum(m[:, j] ** 2)
        # Log determinant terms
        log_evidence += 0.5 * np.sum(np.log(alpha / (beta * S2 + alpha)))
        log_evidence += 0.5 * n * np.log(beta / (2 * np.pi))
        log_evidence += 0.5 * min(n, d) * np.log(alpha / (2 * np.pi))
    
    logme = log_evidence / (n * n_classes)
    return {"logme": round(float(logme), 4)}


def compute_hscore(features, labels):
    """H-score (Bao et al., ICML 2019).
    
    Measures inter-class feature variance relative to overall covariance.
    Higher = better transferability.
    """
    X = StandardScaler().fit_transform(features)
    n, d = X.shape
    classes = np.unique(labels)
    
    # Overall covariance (regularized)
    cov_all = np.cov(X.T) + 1e-4 * np.eye(d)
    cov_all_inv = np.linalg.inv(cov_all)
    
    # Between-class covariance
    class_means = []
    class_priors = []
    for c in classes:
        mask = labels == c
        class_means.append(X[mask].mean(axis=0))
        class_priors.append(mask.sum() / n)
    
    overall_mean = X.mean(axis=0)
    cov_between = np.zeros((d, d))
    for mu, prior in zip(class_means, class_priors):
        diff = (mu - overall_mean).reshape(-1, 1)
        cov_between += prior * (diff @ diff.T)
    
    # H-score = tr(cov_all_inv @ cov_between)
    hscore = np.trace(cov_all_inv @ cov_between)
    return {"hscore": round(float(hscore), 4)}


def compute_rsa(features, labels, patient_ids):
    """RSA: correlation between feature RDM and label/subject RDM."""
    # Feature RDM (cosine distance)
    feat_rdm = squareform(pdist(features, metric="cosine"))
    
    n = len(features)
    
    # Label RDM: 0 if same label, 1 if different
    label_rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            label_rdm[i, j] = 0 if labels[i] == labels[j] else 1
    
    # Subject RDM: 0 if same subject, 1 if different
    subject_rdm = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            subject_rdm[i, j] = 0 if patient_ids[i] == patient_ids[j] else 1
    
    # Extract upper triangle (excluding diagonal)
    tri_idx = np.triu_indices(n, k=1)
    feat_vec = feat_rdm[tri_idx]
    label_vec = label_rdm[tri_idx]
    subject_vec = subject_rdm[tri_idx]
    
    r_label, p_label = spearmanr(feat_vec, label_vec)
    r_subject, p_subject = spearmanr(feat_vec, subject_vec)
    
    return {
        "rsa_label_r": round(r_label, 4),
        "rsa_label_p": round(p_label, 6),
        "rsa_subject_r": round(r_subject, 4),
        "rsa_subject_p": round(p_subject, 6),
    }


def compute_cka(features1, features2):
    """Linear CKA between two feature matrices."""
    X = features1 - features1.mean(0)
    Y = features2 - features2.mean(0)
    
    hsic_xy = np.linalg.norm(X.T @ Y, "fro") ** 2
    hsic_xx = np.linalg.norm(X.T @ X, "fro") ** 2
    hsic_yy = np.linalg.norm(Y.T @ Y, "fro") ** 2
    
    cka = hsic_xy / (np.sqrt(hsic_xx * hsic_yy) + 1e-10)
    return round(float(cka), 4)


# ──────────── Main ────────────


def main():
    cache_dir = "results/features_cache"
    models = ["labram", "cbramod", "reve"]
    datasets = ["stress", "adftd", "tdbrain", "eegmat"]
    
    # Detect channel suffix per dataset
    ch_tag = {"stress": "30ch", "adftd": "19ch", "tdbrain": "19ch", "eegmat": "19ch"}
    
    all_results = {}
    
    for model, dataset in product(models, datasets):
        tag = f"{model}_{dataset}"
        npz_path = os.path.join(cache_dir, f"frozen_{model}_{dataset}_{ch_tag[dataset]}.npz")
        
        if not os.path.exists(npz_path):
            print(f"SKIP {tag}: {npz_path} not found")
            continue
        
        d = np.load(npz_path, allow_pickle=True)
        features = d["features"]
        patient_ids = d["patient_ids"]
        labels = d["labels"]
        
        print(f"\n{'='*60}")
        print(f"  {model.upper()} × {dataset.upper()}  "
              f"({features.shape[0]} rec, {len(np.unique(patient_ids))} subj, "
              f"{features.shape[1]}-d)")
        print(f"{'='*60}")
        
        result = {"model": model, "dataset": dataset,
                  "n_rec": int(features.shape[0]),
                  "n_subj": int(len(np.unique(patient_ids))),
                  "embed_dim": int(features.shape[1])}
        
        # 1. Silhouette
        sil = compute_silhouette(features, labels, patient_ids)
        result.update(sil)
        print(f"  Silhouette — label: {sil['sil_label']:.4f}, subject: {sil['sil_subject']:.4f}")
        
        # 2. Fisher's criterion
        fisher = compute_fisher(features, labels)
        result.update(fisher)
        print(f"  Fisher's criterion: {fisher['fisher']:.6f}")
        
        # 3. kNN
        knn = compute_knn(features, labels, patient_ids)
        result.update(knn)
        print(f"  kNN BA (k=5, subject-CV): {knn['knn_ba']:.4f}")
        
        # 4. LogME
        logme = compute_logme(features, labels)
        result.update(logme)
        print(f"  LogME: {logme['logme']:.4f}")
        
        # 5. H-score
        hscore = compute_hscore(features, labels)
        result.update(hscore)
        print(f"  H-score: {hscore['hscore']:.4f}")
        
        # 6. RSA
        rsa = compute_rsa(features, labels, patient_ids)
        result.update(rsa)
        print(f"  RSA — label: r={rsa['rsa_label_r']:.4f} (p={rsa['rsa_label_p']:.4f}), "
              f"subject: r={rsa['rsa_subject_r']:.4f} (p={rsa['rsa_subject_p']:.4f})")
        
        all_results[tag] = result
    
    # 7. CKA: cross-model comparisons within each dataset
    print(f"\n{'='*60}")
    print("  CKA (cross-model similarity within dataset)")
    print(f"{'='*60}")
    
    cka_results = {}
    for dataset in datasets:
        model_feats = {}
        for model in models:
            npz_path = os.path.join(cache_dir, f"frozen_{model}_{dataset}_{ch_tag[dataset]}.npz")
            if os.path.exists(npz_path):
                model_feats[model] = np.load(npz_path)["features"]
        
        for m1, m2 in [("labram", "cbramod"), ("labram", "reve"), ("cbramod", "reve")]:
            if m1 in model_feats and m2 in model_feats:
                cka_val = compute_cka(model_feats[m1], model_feats[m2])
                key = f"cka_{m1}_vs_{m2}_{dataset}"
                cka_results[key] = cka_val
                print(f"  {dataset}: {m1} vs {m2} = {cka_val:.4f}")
    
    # Save everything
    output = {"per_model_dataset": all_results, "cka_cross_model": cka_results}
    out_path = "results/studies/exp06_fm_task_fitness/fitness_metrics.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n\nSaved to {out_path}")
    
    # Print summary table
    print(f"\n{'='*100}")
    print(f"{'Model':<10} {'Dataset':<10} {'Sil_L':>7} {'Sil_S':>7} {'Fisher':>8} "
          f"{'kNN_BA':>7} {'LogME':>7} {'H-score':>8} {'RSA_L':>7} {'RSA_S':>7}")
    print(f"{'='*100}")
    for key in sorted(all_results.keys()):
        r = all_results[key]
        print(f"{r['model']:<10} {r['dataset']:<10} "
              f"{r['sil_label']:>7.4f} {r['sil_subject']:>7.4f} {r['fisher']:>8.4f} "
              f"{r['knn_ba']:>7.4f} {r['logme']:>7.4f} {r.get('hscore', 0):>8.4f} "
              f"{r['rsa_label_r']:>7.4f} {r['rsa_subject_r']:>7.4f}")


if __name__ == "__main__":
    main()

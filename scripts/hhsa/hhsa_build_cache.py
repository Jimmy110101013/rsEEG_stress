"""One-shot HHSA cache builder: recording-level CEEMDAN + NHT.

Reads raw .set/.edf files → resample 200Hz → per-channel CEEMDAN on full
recording → NHT → save {imfs, IF, IA} per recording.

All downstream analyses (Dir 1-5, WSCI, any future direction) read from
this cache. CEEMDAN is never re-run.

Cache layout:
  results/hhsa/cache/
  ├── stress/{group}_p{pid}_rec{rid}.npz
  ├── eegmat/Subject{NN}_{K}.npz
  └── manifest.json  (metadata: recording list, durations, n_imf)

Each .npz: {imfs, IF, IA} as float32 (n_ch, n_samp, n_imf), plus metadata.

Usage:
    python scripts/hhsa_build_cache.py --dataset both --n-jobs 128
    python scripts/hhsa_build_cache.py --dataset stress --n-jobs 64
"""
import argparse
import glob
import json
import os
import sys
import time
import warnings

import multiprocessing as mp
import numpy as np

warnings.filterwarnings("ignore", message=".*boundary.*", category=RuntimeWarning)
warnings.filterwarnings("ignore", message=".*annotation.*", category=RuntimeWarning)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import pipeline.hhsa as hhsa
import emd

hhsa.N_ENSEMBLES_L1 = 24
hhsa.N_ENSEMBLES_L2 = 12

CACHE_DIR = "results/hhsa/cache"
FS_TARGET = 200.0
MAX_IMF = 8


def _fft_resample(data, src_sfreq, dst_sfreq):
    """Bandlimited resample (from eegmat_dataset.py)."""
    from scipy.fft import rfft, irfft
    n = data.shape[-1]
    new_n = int(round(n * dst_sfreq / src_sfreq))
    if new_n == n:
        return data.astype(np.float64, copy=False)
    X = rfft(data, axis=-1)
    new_keep = new_n // 2 + 1
    if new_keep <= X.shape[-1]:
        X_new = X[..., :new_keep]
    else:
        pad_shape = list(X.shape)
        pad_shape[-1] = new_keep - X.shape[-1]
        X_new = np.concatenate([X, np.zeros(pad_shape, dtype=X.dtype)], axis=-1)
    y = irfft(X_new, n=new_n, axis=-1)
    y = y * (new_n / n)
    return y.astype(np.float64)


def ceemdan_one_channel(x_1d, noise_seed):
    """CEEMDAN + NHT on a single channel full recording.

    Returns (imfs, IF, IA) each of shape (n_samp, n_imf), or None on failure.
    """
    if x_1d.std() < 1e-10 or not np.all(np.isfinite(x_1d)):
        return None
    imfs = hhsa._safe_ceemdan(x_1d, hhsa.N_ENSEMBLES_L1, noise_seed)
    if imfs is None:
        return None
    imfs = hhsa._truncate_imfs(imfs, MAX_IMF)
    _, IF, IA = emd.spectra.frequency_transform(imfs, FS_TARGET, 'nht')
    IF, IA = hhsa._sanitize_nht(IF, IA)
    return imfs.astype(np.float32), IF.astype(np.float32), IA.astype(np.float32)




# ===== Dataset-specific task collectors (lightweight — no data loading) =====

def collect_stress_tasks():
    """Return list of (dataset, rec_id, src_path, out_path, metadata) for Stress."""
    import pandas as pd
    out_dir = os.path.join(CACHE_DIR, "stress")
    os.makedirs(out_dir, exist_ok=True)

    df = pd.read_csv("data/comprehensive_labels.csv")
    tasks = []
    for _, row in df.iterrows():
        pid = int(row["Patient_ID"])
        rid = int(row["Recording_ID"])
        group = row["Group"]
        rec_id = f"{group}_p{pid:02d}_rec{rid}"
        set_files = glob.glob(f"data/{group}/{rid}.set")
        if not set_files:
            print(f"  [WARN] Missing: data/{group}/{rid}.set")
            continue
        out_path = os.path.join(out_dir, f"{rec_id}.npz")
        tasks.append(("stress", rec_id, set_files[0], out_path, {}))
    return tasks


def collect_eegmat_tasks():
    """Return list of (dataset, rec_id, src_path, out_path, metadata) for EEGMAT."""
    out_dir = os.path.join(CACHE_DIR, "eegmat")
    os.makedirs(out_dir, exist_ok=True)

    records_path = "data/eegmat/RECORDS"
    with open(records_path) as f:
        fnames = [l.strip() for l in f if l.strip().endswith(".edf")]

    tasks = []
    for fname in sorted(fnames):
        stem = fname[:-4]
        eeg_path = os.path.join("data/eegmat", fname)
        out_path = os.path.join(out_dir, f"{stem}.npz")
        tasks.append(("eegmat", stem, eeg_path, out_path, {}))
    return tasks


def collect_meditation_tasks():
    """Return list of tasks for meditation dataset (BioSemi .bdf, COMMON_19, middle 300s)."""
    out_dir = os.path.join(CACHE_DIR, "meditation")
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for bdf_path in sorted(glob.glob("data/meditation/sub-*/ses-*/eeg/*.bdf")):
        # e.g. sub-001_ses-01_task-meditation_eeg
        fname = os.path.basename(bdf_path).replace("_eeg.bdf", "")
        out_path = os.path.join(out_dir, f"{fname}.npz")
        tasks.append(("meditation", fname, bdf_path, out_path, {}))
    return tasks


def collect_sam40_tasks():
    """Return list of tasks for SAM40 dataset (filtered .mat files, COMMON_19 channels)."""
    out_dir = os.path.join(CACHE_DIR, "sam40")
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for mat_path in sorted(glob.glob("data/sam40/Data/filtered_data/*.mat")):
        stem = os.path.basename(mat_path).replace(".mat", "")
        out_path = os.path.join(out_dir, f"{stem}.npz")
        tasks.append(("sam40", stem, mat_path, out_path, {}))
    return tasks


def collect_sleepdep_tasks():
    """Return list of (dataset, rec_id, src_path, out_path, metadata) for sleep_deprivation.

    Uses COMMON_19 channels (matched to EEGMAT/Stress) with 10-10 → 10-20 mapping.
    """
    out_dir = os.path.join(CACHE_DIR, "sleep_deprivation")
    os.makedirs(out_dir, exist_ok=True)

    tasks = []
    for set_path in sorted(glob.glob("data/sleep_deprivation/sub-*/ses-*/eeg/*.set")):
        # e.g. sub-01_ses-1_task-eyesclosed_eeg
        fname = os.path.basename(set_path).replace("_eeg.set", "")
        out_path = os.path.join(out_dir, f"{fname}.npz")
        tasks.append(("sleepdep", fname, set_path, out_path, {}))
    return tasks


# ===== Worker: loads data from disk, runs CEEMDAN, saves cache =====

def _load_and_process(task):
    """Worker function: load raw EEG from disk → CEEMDAN → save .npz.

    Each worker is self-contained — no large data passed through the pipe.
    """
    dataset, rec_id, src_path, out_path, _ = task
    if os.path.exists(out_path):
        return {"rec_id": rec_id, "status": "cached"}

    try:
        return _load_and_process_inner(task)
    except Exception as e:
        return {"rec_id": rec_id, "status": f"error:{type(e).__name__}"}


def _load_and_process_inner(task):
    """Inner worker — may raise; caller catches and returns error status."""
    dataset, rec_id, src_path, out_path, _ = task

    # --- Load from disk inside the worker ---
    if dataset == "stress":
        import mne
        raw = mne.io.read_raw_eeglab(src_path, preload=True, verbose=False)
        sfreq = raw.info["sfreq"]
        data = raw.get_data() * 1e6  # V → µV
        ch_names = list(raw.ch_names)
        del raw
    elif dataset == "eegmat":
        import pyedflib
        from pipeline.common_channels import COMMON_19, normalize_channel_name
        reader = pyedflib.EdfReader(src_path)
        try:
            raw_labels = reader.getSignalLabels()
            sfreq = float(reader.getSampleFrequency(0))
            sig_idx = {}
            for i, lbl in enumerate(raw_labels):
                norm = normalize_channel_name(lbl)
                if norm not in sig_idx:
                    sig_idx[norm] = i
            picks, ch_names = [], []
            for tgt in COMMON_19:
                tn = normalize_channel_name(tgt)
                if tn in sig_idx:
                    picks.append(sig_idx[tn])
                    ch_names.append(tgt)
            data = np.stack([reader.readSignal(i).astype(np.float64) for i in picks])
        finally:
            reader.close()
    elif dataset == "meditation":
        import mne
        from pipeline.common_channels import COMMON_19, normalize_channel_name
        # BioSemi 64 → 10-20 mapping
        BIOSEMI_TO_1020 = {
            'A1':'Fp1','A5':'F3','A7':'F7','A13':'C3','A15':'T7','A21':'P3',
            'A23':'P7','A27':'O1','A31':'Pz',
            'B2':'Fp2','B6':'Fz','B8':'F4','B10':'F8','B16':'Cz','B18':'C4',
            'B20':'T8','B26':'P4','B28':'P8','B32':'O2',
        }
        MAP_1010_TO_1020 = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
        raw = mne.io.read_raw_bdf(src_path, preload=False, verbose=False)
        sfreq = raw.info["sfreq"]
        total_samp = raw.n_times
        # Crop middle 300s
        total_s = total_samp / sfreq
        crop_s = min(300.0, total_s)
        t_start = (total_s - crop_s) / 2.0
        raw.crop(tmin=t_start, tmax=t_start + crop_s)
        raw.load_data()
        all_data = raw.get_data() * 1e6  # V → µV
        raw_ch = raw.ch_names
        del raw
        # Map BioSemi → 10-20 → normalize
        ch_idx = {}
        for i, ch in enumerate(raw_ch):
            name_1020 = BIOSEMI_TO_1020.get(ch)
            if name_1020 is None:
                continue
            up = name_1020.upper()
            mapped = MAP_1010_TO_1020.get(up, up)
            norm = normalize_channel_name(mapped)
            if norm not in ch_idx:
                ch_idx[norm] = i
        picks, ch_names = [], []
        for tgt in COMMON_19:
            tn = normalize_channel_name(tgt)
            if tn in ch_idx:
                picks.append(ch_idx[tn])
                ch_names.append(tgt)
        if not picks:
            return {"rec_id": rec_id, "status": "failed"}
        data = all_data[picks, :]
        del all_data
    elif dataset == "sam40":
        import scipy.io as sio
        from pipeline.common_channels import COMMON_19, normalize_channel_name
        MAP_1010_TO_1020 = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
        # SAM40 channel order from Coordinates.locs
        SAM40_CH = ['Cz','Fz','Fp1','F7','F3','FC1','C3','FC5','FT9','T7',
                     'CP5','CP1','P3','P7','PO9','O1','Pz','Oz','O2','PO10',
                     'P8','P4','CP2','CP6','T8','FT10','FC6','C4','FC2','F4','F8','Fp2']
        sfreq = 128.0
        mat = sio.loadmat(src_path)
        all_data = mat["Clean_data"].astype(np.float64)  # (32, 3200)
        # Build name→index map
        ch_idx = {}
        for i, ch in enumerate(SAM40_CH):
            up = ch.upper()
            mapped = MAP_1010_TO_1020.get(up, up)
            norm = normalize_channel_name(mapped)
            if norm not in ch_idx:
                ch_idx[norm] = i
        picks, ch_names = [], []
        for tgt in COMMON_19:
            tn = normalize_channel_name(tgt)
            if tn in ch_idx:
                picks.append(ch_idx[tn])
                ch_names.append(tgt)
        data = all_data[picks, :]
        del all_data
    elif dataset == "sleepdep":
        import mne
        from pipeline.common_channels import COMMON_19, normalize_channel_name
        # 10-10 → 10-20 mapping for channel matching
        MAP_1010_TO_1020 = {"T7": "T3", "T8": "T4", "P7": "T5", "P8": "T6"}
        try:
            raw = mne.io.read_raw_eeglab(src_path, preload=True, verbose=False)
        except (RuntimeError, ValueError):
            # Some .set files have boundary events causing sample mismatch;
            # fall back to reading without preload then cropping to valid data
            raw = mne.io.read_raw_eeglab(src_path, preload=False, verbose=False)
            raw.load_data()
        sfreq = raw.info["sfreq"]
        all_data = raw.get_data() * 1e6  # V → µV
        raw_ch = raw.ch_names
        del raw
        # Build name→index map with 10-10→10-20 aliasing
        ch_idx = {}
        for i, ch in enumerate(raw_ch):
            up = ch.upper()
            mapped = MAP_1010_TO_1020.get(up, up)
            norm = normalize_channel_name(mapped)
            if norm not in ch_idx:
                ch_idx[norm] = i
        picks, ch_names = [], []
        for tgt in COMMON_19:
            tn = normalize_channel_name(tgt)
            if tn in ch_idx:
                picks.append(ch_idx[tn])
                ch_names.append(tgt)
        if not picks:
            return {"rec_id": rec_id, "status": "failed"}
        data = all_data[picks, :]
        del all_data
    else:
        return {"rec_id": rec_id, "status": "failed"}

    # Resample + z-score
    if sfreq != FS_TARGET:
        data = _fft_resample(data, sfreq, FS_TARGET)
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True) + 1e-6
    data = (data - mean) / std

    # --- CEEMDAN per channel ---
    n_ch, n_samp = data.shape
    ch_results = []
    for ci in range(n_ch):
        r = ceemdan_one_channel(data[ci, :], ci * 10000)
        ch_results.append(r)
    del data

    max_nimf = 0
    for r in ch_results:
        if r is not None:
            max_nimf = max(max_nimf, r[0].shape[1])
    if max_nimf == 0:
        return {"rec_id": rec_id, "status": "failed"}

    imfs_all = np.zeros((n_ch, n_samp, max_nimf), dtype=np.float32)
    IF_all = np.zeros_like(imfs_all)
    IA_all = np.zeros_like(imfs_all)
    n_imf_per_ch = np.zeros(n_ch, dtype=np.int32)

    for ci, r in enumerate(ch_results):
        if r is None:
            continue
        imf, IF, IA = r
        ni = imf.shape[1]
        imfs_all[ci, :, :ni] = imf
        IF_all[ci, :, :ni] = IF
        IA_all[ci, :, :ni] = IA
        n_imf_per_ch[ci] = ni
    del ch_results

    np.savez_compressed(
        out_path,
        imfs=imfs_all, IF=IF_all, IA=IA_all,
        fs=FS_TARGET, n_samp=n_samp, n_imf_per_ch=n_imf_per_ch,
        ch_names=np.array(ch_names),
        duration_s=n_samp / FS_TARGET,
    )
    size_mb = os.path.getsize(out_path) / 1e6
    del imfs_all, IF_all, IA_all
    return {"rec_id": rec_id, "status": "ok", "n_samp": n_samp,
            "n_ch": n_ch, "max_nimf": max_nimf, "size_mb": size_mb}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["stress", "eegmat", "sleepdep", "sam40", "meditation", "both", "all"], default="both")
    parser.add_argument("--n-workers", type=int, default=4,
                        help="Number of recordings to process in parallel (each does all channels sequentially)")
    args = parser.parse_args()

    # Each worker processes 1 recording (30ch sequential CEEMDAN).
    # Memory: ~1-2 GB per worker. 4 workers ≈ 4-8 GB, safe for shared pool.
    # Speed: 4 recordings × 30 channels running simultaneously = 120 CEEMDAN instances.

    os.makedirs(CACHE_DIR, exist_ok=True)
    t_start = time.time()

    # Collect lightweight task descriptors (no data loaded yet)
    all_tasks = []
    if args.dataset in ("stress", "both", "all"):
        all_tasks.extend(collect_stress_tasks())
    if args.dataset in ("eegmat", "both", "all"):
        all_tasks.extend(collect_eegmat_tasks())
    if args.dataset in ("sleepdep", "all"):
        all_tasks.extend(collect_sleepdep_tasks())
    if args.dataset in ("sam40", "all"):
        all_tasks.extend(collect_sam40_tasks())
    if args.dataset in ("meditation", "all"):
        all_tasks.extend(collect_meditation_tasks())

    # Filter out already-cached
    pending = [t for t in all_tasks if not os.path.exists(t[3])]
    n_cached = len(all_tasks) - len(pending)
    print(f"\n{n_cached} cached, {len(pending)} to process with {args.n_workers} parallel workers")

    if pending:
        ctx = mp.get_context("spawn")
        t_batch = time.time()
        with ctx.Pool(args.n_workers) as pool:
            for result in pool.imap_unordered(_load_and_process, pending):
                dt = time.time() - t_batch
                rid = result["rec_id"]
                if result["status"] == "ok":
                    print(f"  [{dt:6.0f}s] {rid}: {result['n_ch']}ch × {result['n_samp']} samp, "
                          f"{result['max_nimf']} IMFs, {result['size_mb']:.1f}MB")
                else:
                    print(f"  [{dt:6.0f}s] {rid}: {result['status']}")

    total = time.time() - t_start
    n_files = len(glob.glob(f"{CACHE_DIR}/**/*.npz", recursive=True))
    total_mb = sum(os.path.getsize(f) for f in glob.glob(f"{CACHE_DIR}/**/*.npz", recursive=True)) / 1e6
    print(f"\n{'='*60}")
    print(f"Cache complete: {n_files} files, {total_mb:.0f} MB, {total:.0f}s")
    print(f"Cache dir: {CACHE_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

# Environment setup — `FM_analysis`

This document is for the **server administrator / IT team**. It describes
how to provision a Python environment that runs the `rsEEG_stress`
research codebase on a single Linux server with one NVIDIA GPU.

A researcher will use this environment afterwards through JupyterLab; they
do **not** need terminal access. Once this setup is done, hand the
JupyterLab instance back to them.

---

## What you're installing

- A conda (or micromamba) environment named `FM_analysis` containing Python
  3.10, PyTorch with CUDA support, and ~15 standard scientific Python
  packages pinned to known-compatible versions.
- A clone of the research code at `~/rsEEG_stress` (or wherever the user
  prefers).
- Total disk footprint: ~7 GB (env + repo + cached pip wheels).

There is **no system-level installation**, no daemons, no port-listening
services. Everything lives inside the conda env directory.

---

## Prerequisites

- Linux x86_64 with kernel ≥ 5.0.
- NVIDIA GPU + driver supporting **CUDA 12.x** (we tested on RTX 4090
  with driver 550, CUDA 12.5). Verify with `nvidia-smi` — the "CUDA
  Version" field at the top right must be ≥ 12.4.
- One of the following package managers already installed:
  - **conda** (Anaconda or Miniconda), or
  - **micromamba** (preferred — smaller, faster, no `base` env conflicts).
- `git` ≥ 2.20.
- Internet access during install to:
  - `repo.anaconda.com` (for conda) **or** `conda-forge.org` (for micromamba)
  - `download.pytorch.org` (for the CUDA-enabled torch wheels)
  - `pypi.org` (for the rest of the dependencies)
  - `github.com` (for the code repository)
- ~7 GB free disk under the user's home directory.

---

## Step 1 — create the empty environment

Pick **one** of the two commands depending on which package manager is
available. Both create an empty Python 3.10 env named `FM_analysis`.

```bash
# Option A: conda
conda create -n FM_analysis python=3.10 -y

# Option B: micromamba (recommended)
micromamba create -n FM_analysis python=3.10 -c conda-forge -y
```

Activate it:

```bash
conda activate FM_analysis        # or:  micromamba activate FM_analysis
```

> **Why a fresh env, not the existing one?** The codebase pins specific
> NumPy / SciPy / pandas versions for binary compatibility. Mixing it
> into an existing shared env will likely break either this project or
> the existing tools.

---

## Step 2 — install PyTorch (CUDA build) via pip

PyTorch must come from the official PyTorch wheel index, not default PyPI,
or you will silently get the CPU-only build.

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
    torch==2.5.1 torchvision==0.20.1
```

The `cu124` build is forward-compatible with the CUDA 12.5 driver.

Verify CUDA is detected:

```bash
python -c "import torch; print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0))"
```

Expected output:

```
cuda available: True
device: NVIDIA GeForce RTX 4090
```

If `cuda available: False`, the wrong wheel was installed — `pip uninstall torch torchvision -y` and rerun the command above.

---

## Step 3 — clone the code repository

```bash
cd ~      # or wherever the researcher wants the repo
git clone https://github.com/Jimmy110101013/rsEEG_stress.git
cd rsEEG_stress
```

> **If the GitHub repo is private at the time of cloning**, the user will
> provide either a one-time Personal Access Token (PAT) or a deploy key.
> With a PAT, the clone command becomes:
>
> ```bash
> git clone https://<token>@github.com/Jimmy110101013/rsEEG_stress.git
> ```
>
> Treat the token like a password — do not commit it anywhere, and the
> user can revoke it from GitHub once the clone is complete.

---

## Step 4 — install the rest of the Python dependencies

From inside the cloned `rsEEG_stress/` directory:

```bash
pip install -r requirements.txt
```

This pulls in NumPy 1.26, SciPy 1.14, pandas 2.2, scikit-learn 1.5, MNE,
h5py, einops, timm, matplotlib, and tqdm. Should take 2–5 minutes.

> **Important compatibility note**: `requirements.txt` pins NumPy to
> `1.26.4` on purpose. **Do not** let pip upgrade NumPy past 1.x —
> several pinned packages (SciPy 1.14, pandas 2.2) are built against
> the NumPy 1.x ABI and will silently break on NumPy 2.x with errors
> like `TypeError: Cannot convert numpy.ndarray to numpy.ndarray` or
> `ValueError: All ufuncs must have type 'numpy.ufunc'`.
>
> If pip ever pulls in NumPy 2.x as a transitive dependency, recover with:
>
> ```bash
> pip install --force-reinstall --no-deps "numpy==1.26.4"
> ```

---

## Step 5 — verify the environment

Run this verification command from inside `rsEEG_stress/`:

```bash
python - <<'PY'
import torch, numpy, scipy, sklearn, mne, h5py, einops, timm, pandas, matplotlib
print("torch    ", torch.__version__, "| cuda:", torch.cuda.is_available())
print("numpy    ", numpy.__version__)
print("pandas   ", pandas.__version__)
print("scipy    ", scipy.__version__)
print("sklearn  ", sklearn.__version__)
print("mne      ", mne.__version__)
print("h5py     ", h5py.__version__)
print("einops   ", einops.__version__)
print("timm     ", timm.__version__)
print("matplotlib", matplotlib.__version__)
PY
```

Expected output (versions must match exactly):

```
torch     2.5.1+cu124 | cuda: True
numpy     1.26.4
pandas    2.2.3
scipy     1.14.1
sklearn   1.5.2
mne       1.8.0
h5py      3.12.1
einops    0.8.0
timm      1.0.11
matplotlib 3.9.2
```

---

## Step 6 — make the env visible inside JupyterLab

So the researcher can pick `FM_analysis` as the kernel for their notebooks:

```bash
pip install ipykernel
python -m ipykernel install --user --name FM_analysis --display-name "Python (FM_analysis)"
```

After this, refreshing JupyterLab will show "Python (FM_analysis)" in the
kernel picker.

---

## Hand-off

That's it. Tell the researcher:

- The env is named `FM_analysis` (kernel display name "Python (FM_analysis)").
- The code is at `~/rsEEG_stress` (or wherever you cloned it).
- They should activate it via the JupyterLab kernel picker.

The researcher will handle everything else (uploading model weights,
pointing the loader at the dataset files, running experiments). You do
not need to install any data or model files — those go in via the
researcher's JupyterLab session.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `cuda: False` after Step 2 | Wrong wheel. `pip uninstall torch torchvision -y` then rerun the `--index-url https://download.pytorch.org/whl/cu124 ...` install. |
| `ValueError: All ufuncs must have type 'numpy.ufunc'` | NumPy got upgraded to 2.x. `pip install --force-reinstall --no-deps "numpy==1.26.4"`. |
| `TypeError: Cannot convert numpy.ndarray to numpy.ndarray` | Same as above. |
| `git clone` fails with `Authentication failed` | Repo is private and no PAT was supplied. Ask the researcher for a token. |
| `nvidia-smi` not found | NVIDIA driver not installed — install the matching driver for the GPU first. |
| `OSError: libcudart.so.12: cannot open shared object file` | CUDA driver is older than 12.x — upgrade the NVIDIA driver to support CUDA 12.x. |

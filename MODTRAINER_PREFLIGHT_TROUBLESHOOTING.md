# Modtrainer Preflight Troubleshooting (bitsandbytes / CUDA)

This runbook addresses the preflight failure where `torch` sees a GPU, but `bitsandbytes` fails CUDA setup.

## Is the Python 3.13 warning critical?

Usually **no**. It is a compatibility warning, not an immediate blocker:

- If you only need `torch` + `transformers` + `bitsandbytes` and installs succeed, you can continue on Python 3.13.
- It becomes **critical** only when your required wheel set is unavailable for cp313 (most commonly `torchvision`/`torchaudio` pins, or fallback source builds that fail).

Use the decision flow below and switch to Python 3.12 only when needed.

## Symptoms

- `torch` reports a CUDA tag that does not match your intended runtime (for example `2.8.0+cu128`) and `CUDA available: True`.
- `bitsandbytes` fails with:
  - `Required library version not found: libbitsandbytes_cudaXXX.so`
  - fallback to `libbitsandbytes_cpu.so`
  - `CUDA Setup failed`
- Warnings like:
  - `The following directories listed in your path were found to be non-existent: meta-llama/Meta-Llama-3-8B-Instruct`

## Root causes

1. **CUDA/torch/bitsandbytes mismatch**
   - Your PyTorch CUDA tag (for example `cu128`) does not match the `bitsandbytes` wheel binaries available in your environment.
2. **`PATH` contamination**
   - A model name string appears in `PATH`, which causes directory warnings.

## Fix plan

### 0) Decision flow: stay on 3.13 vs switch to 3.12

1. Try the pinned install flow in your current environment.
2. If install and preflight pass, stay on Python 3.13.
3. If you hit any of these blockers, switch to Python 3.12:
   - Missing wheel for your pinned package set (especially `torchvision`/`torchaudio` CUDA combos).
   - Source build failures for `sentencepiece`/native packages.
   - Repeated resolver conflicts that force incompatible versions.

Quick check command:

```bash
python - <<'PY'
import sys
print(sys.version)
PY

pip index versions torchvision --index-url https://download.pytorch.org/whl/cu124
pip index versions torchaudio --index-url https://download.pytorch.org/whl/cu124
```

### 1) Clean shell variables

Ensure model identifiers are not inserted into `PATH`:

```bash
unset MODEL_NAME
export MODEL_NAME="meta-llama/Meta-Llama-3-8B-Instruct"

echo "$PATH" | tr ':' '\n' | rg 'meta-llama|govchain-model' || true
```

If those entries appear in `PATH`, remove the faulty export from `~/.bashrc`/`~/.zshrc` and restart shell.

### 2) Align dependency versions to your preflight baseline

Your preflight expects approximately:

- `torch==2.5.1+cu124`
- `transformers==4.39.3`
- `sentencepiece==0.2.0`
- `accelerate==0.28.0`
- `trl==0.8.1`
- `datasets==2.18.0`

Use a clean virtual environment and reinstall pinned versions.

### 3) Pick a supported CUDA path for bitsandbytes

For CUDA 12.4 systems, prefer a PyTorch `cu124` build plus a matching recent `bitsandbytes` wheel.

Example (CUDA 12.4 route):

```bash
pip uninstall -y torch torchvision torchaudio bitsandbytes
pip cache purge

pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124

pip install bitsandbytes==0.45.2 \
  transformers==4.39.3 peft==0.10.0 accelerate==0.28.0 \
  trl==0.8.1 datasets==2.18.0 sentencepiece==0.2.0 numpy==1.26.4
```

If you are on Python 3.13, use this variant to avoid source builds:

```bash
pip install bitsandbytes==0.45.2 \
  transformers==4.39.3 peft==0.10.0 accelerate==0.28.0 \
  trl==0.8.1 datasets==2.18.0 sentencepiece==0.2.1 numpy==1.26.4
```

Notes:

- `torchvision` and `torchaudio` are not required by this repository's training path.
- On Python 3.13, `torch==2.5.1+cu124` may install while matching `torchvision==0.20.1+cu124` is unavailable.
- On Python 3.13, `sentencepiece==0.2.0` may fall back to source build and fail if `cmake`/`pkg-config` are missing. Prefer `sentencepiece==0.2.1` on Python 3.13+.
- If you need `torchvision`/`torchaudio`, either:
  1) create a Python 3.12 virtual environment and use the strict `2.5.1` family, or
  2) choose versions that exist for your Python ABI from the cu124 index (`pip index versions torchvision --index-url https://download.pytorch.org/whl/cu124`).

### 3b) Runbook: switch to Python 3.12 only when necessary

If Step 0 shows your required wheels are unavailable on cp313, use this minimal downgrade path.

#### Option A: using `pyenv` (recommended)

```bash
pyenv install 3.12.9
pyenv local 3.12.9

python -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
```

#### Option B: system Python 3.12 available

```bash
python3.12 -m venv .venv312
source .venv312/bin/activate
python -m pip install --upgrade pip
```

If `python3.12` is not installed on your host (for example: `python3.12: command not found`), keep using your current interpreter:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python check_env.py --strict --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
```

Only switch to Python 3.12 after preflight identifies a real blocker (e.g., wheel/build incompatibility).

Reinstall baseline packages:

```bash
pip install --index-url https://download.pytorch.org/whl/cu124 \
  torch==2.5.1+cu124 torchvision==0.20.1+cu124 torchaudio==2.5.1+cu124

pip install -r requirements.txt
```

Then validate:

```bash
python check_env.py --strict --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
python -m bitsandbytes
```

### 4) Verify CUDA runtime library visibility

```bash
python - <<'PY'
import torch
print('torch:', torch.__version__)
print('cuda:', torch.version.cuda)
print('is_available:', torch.cuda.is_available())
PY

python -m bitsandbytes
```

If `python -m bitsandbytes` still cannot find `libcudart.so`, locate it and export `LD_LIBRARY_PATH`:

```bash
find /usr /opt -name 'libcudart.so*' 2>/dev/null
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:/path/to/cuda/lib64"
```

Persist that export only after confirming the correct directory.

## Expected outcome

After alignment, preflight should no longer fail on `bitsandbytes` import, and version warnings should be reduced to only intentionally unpinned packages.

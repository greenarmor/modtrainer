# Modtrainer Preflight Troubleshooting (bitsandbytes / CUDA)

This runbook addresses the preflight failure where `torch` sees a GPU, but `bitsandbytes` fails CUDA setup.

## Symptoms

- `torch` reports `2.8.0+cu128` and `CUDA available: True`.
- `bitsandbytes` fails with:
  - `Required library version not found: libbitsandbytes_cuda128.so`
  - fallback to `libbitsandbytes_cpu.so`
  - `CUDA Setup failed`
- Warnings like:
  - `The following directories listed in your path were found to be non-existent: meta-llama/Meta-Llama-3-8B-Instruct`

## Root causes

1. **CUDA/torch/bitsandbytes mismatch**
   - You are on a CUDA 12.8 PyTorch build (`cu128`), but your `bitsandbytes` wheel does not provide `libbitsandbytes_cuda128.so`.
2. **`PATH` contamination**
   - A model name string appears in `PATH`, which causes directory warnings.

## Fix plan

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

- `torch==2.2.2`
- `transformers==4.39.3`
- `sentencepiece==0.2.0`
- `accelerate==0.28.0`
- `trl==0.8.1`
- `datasets==2.18.0`

Use a clean virtual environment and reinstall pinned versions.

### 3) Pick a supported CUDA path for bitsandbytes

The most reliable option is using a PyTorch build with CUDA 12.1 (or 11.8) plus a matching `bitsandbytes` wheel.

Example (CUDA 12.1 route):

```bash
pip uninstall -y torch torchvision torchaudio bitsandbytes
pip cache purge

pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2

pip install bitsandbytes==0.43.1 \
  transformers==4.39.3 peft==0.10.0 accelerate==0.28.0 \
  trl==0.8.1 datasets==2.18.0 sentencepiece==0.2.0 numpy==1.26.4
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

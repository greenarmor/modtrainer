# ModTrainer Fine-Tuning How-To (Start to End)

This guide walks you through setting up the environment, preparing data, running LoRA fine-tuning, and validating outputs with the built-in policy checks.

## 1) What this repository does

This project fine-tunes an instruction model using LoRA adapters and then evaluates response quality with policy-oriented checks:

- **Training entrypoint:** `training/finetune_lora.py`
- **Environment preflight check:** `check_env.py`
- **Evaluation scripts:** `eval/run_policy_eval.py` and `eval/policy_checks.py`

The default base model is `meta-llama/Meta-Llama-3-8B-Instruct` (gated on Hugging Face), with automatic fallback to `TinyLlama/TinyLlama-1.1B-Chat-v1.0` if loading fails.

---

## 2) Prerequisites

- Python 3.10+ (Python 3.12 recommended; 3.13 can work with caveats)
- Optional but strongly recommended: NVIDIA GPU + CUDA for faster training
- A Hugging Face account and token if using gated models (for example, Meta Llama)

Install dependencies:

```bash
pip install -r requirements.txt
# requirements now target CUDA 12.4-compatible PyTorch wheels (cu124)
```

> Note: `requirements.txt` pins package versions. If you deviate, run the preflight checker to confirm compatibility.

---

## 3) Configure environment variables

Copy `.env.example` to `.env` and set real values:

```bash
cp .env.example .env
```

Update at least:

- `HF_TOKEN` (required for gated models)

Optional overrides available in `.env.example`:

- `MODEL_NAME`
- `TRAIN_FILE`
- `VAL_FILE`
- `OUTPUT_DIR`
- `SAVE_DIR`
- `SEED`
- `MAX_SEQ_LENGTH`

Preflight now auto-loads `.env` if present, but loading it in your shell is still useful for training commands (example):

```bash
set -a
source .env
set +a
```

---

## 4) Run preflight checks (recommended)

Run:

```bash
python check_env.py --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
```

This verifies:

- `.env` loading and HF token presence (or warns)
- CUDA availability
- PATH contamination for obvious model-path mistakes
- Installed package versions

If you want CI/automation to fail on warnings:

```bash
python check_env.py --strict --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"
```

---

## 5) Prepare training/validation data

Expected file format is JSONL with fields:

- `instruction`
- `context`
- `response`

Each line should be one valid JSON object. Example:

```json
{"instruction":"Explain custody controls","context":"Public treasury modernization","response":"Use lawful, sovereign controls with independent audit trails."}
```

By default, training uses:

- `data/train.jsonl`
- `data/val.jsonl`

You can override via CLI flags or environment variables.

---

## 6) Launch LoRA fine-tuning

Basic command:

```bash
python training/finetune_lora.py \
  --train-file "${TRAIN_FILE:-data/train.jsonl}" \
  --val-file "${VAL_FILE:-data/val.jsonl}" \
  --output-dir "${OUTPUT_DIR:-./govchain-model}" \
  --save-dir "${SAVE_DIR:-govchain-lora}" \
  --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --seed "${SEED:-42}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-1024}" \
  --hf-token "${HF_TOKEN}"
```

What happens during training:

1. Train/validation JSONL files are loaded.
2. Records are formatted into an instruction/context/response prompt template.
3. The script tries the selected base model; if that fails, it attempts `TinyLlama` fallback.
4. LoRA adapters are attached (`q_proj`, `v_proj`; rank 16; alpha 32).
5. SFT training runs and checkpoints are saved.
6. Final adapter model is saved in `--save-dir`.

Default training config in code:

- Per-device batch size: `2`
- Gradient accumulation: `8`
- Learning rate: `2e-4`
- Epochs: `6`
- Eval/save interval: every `50` steps

---

## 7) Evaluate policy compliance

Run policy evaluation over one or more JSONL files with `response` fields:

```bash
python eval/run_policy_eval.py \
  --inputs data/val.jsonl data/govchain_redteam_500.jsonl \
  --max-errors 0
```

Behavior:

- Prints totals and failure counts per file
- Displays up to 5 sample failures
- Exits non-zero if total failures exceed `--max-errors`

Current checks include:

- **Forbidden phrases** (e.g., DeFi/yield-farming style terms)
- **Required concept coverage** (`public funds`, `sovereign`, `legal`, `audit`)

---

## 8) End-to-end quick run (copy/paste)

```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Configure env
cp .env.example .env
# edit .env and set HF_TOKEN if needed
set -a && source .env && set +a

# 3) Preflight
python check_env.py --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}"

# 4) Train
python training/finetune_lora.py \
  --train-file "${TRAIN_FILE:-data/train.jsonl}" \
  --val-file "${VAL_FILE:-data/val.jsonl}" \
  --output-dir "${OUTPUT_DIR:-./govchain-model}" \
  --save-dir "${SAVE_DIR:-govchain-lora}" \
  --model-name "${MODEL_NAME:-meta-llama/Meta-Llama-3-8B-Instruct}" \
  --seed "${SEED:-42}" \
  --max-seq-length "${MAX_SEQ_LENGTH:-1024}" \
  --hf-token "${HF_TOKEN}"

# 5) Evaluate
python eval/run_policy_eval.py --inputs data/val.jsonl data/govchain_redteam_500.jsonl --max-errors 0
```

---

## 9) Optional: run with Docker

Build:

```bash
docker build -t modtrainer .
```

Run (mount local repo + pass token):

```bash
docker run --gpus all --rm -it \
  -v "$(pwd):/app" \
  -e HF_TOKEN="$HF_TOKEN" \
  modtrainer
```

The image default command launches `python3 training/finetune_lora.py`.

---

## 10) Troubleshooting

- **Model download/auth error:** ensure `HF_TOKEN` is set and accepted for the model repo.
- **Very slow training:** verify CUDA is available (`python check_env.py`).
- **OOM errors:** reduce `--max-seq-length` or tune batch/accumulation settings in `training/finetune_lora.py`.
- **Policy eval failing:** inspect printed violations/missing concepts and adjust generated responses or dataset targets.
- **`sentencepiece` build error on Python 3.13 (`cmake`/`pkg-config` missing):** ensure `sentencepiece==0.2.1` is installed (already pinned here), or switch to a Python 3.12 virtualenv if your platform lacks wheels.
- **Host has CUDA 12.8 installed but repo requires CUDA 12.4 torch wheels:** keep your NVIDIA driver, but make sure the Python wheel is `torch==2.5.1+cu124` from this repo. Use a clean venv and run `pip install --force-reinstall -r requirements.txt`, then verify `torch.version.cuda` reports `12.4` via `python check_env.py`.

---

## 11) Output artifacts you should expect

- Training outputs/checkpoints under `--output-dir` (default `./govchain-model`)
- Saved LoRA adapter under `--save-dir` (default `govchain-lora`)
- Policy evaluation summary in terminal output with non-zero exit for threshold breaches


import argparse
import json
import os
import warnings
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
DEFAULT_ADAPTER_DIR = "govchain-lora"
DEFAULT_INPUT = "data/val.jsonl"
DEFAULT_OUTPUT = "outputs/generated_outputs.jsonl"
DEFAULT_OFFLOAD_DIR = "./offload"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate JSONL responses from val prompts using a LoRA adapter.")
    parser.add_argument("--model-name", default=os.environ.get("MODEL_NAME", DEFAULT_MODEL))
    parser.add_argument("--adapter-dir", default=os.environ.get("ADAPTER_DIR", DEFAULT_ADAPTER_DIR))
    parser.add_argument("--input", default=os.environ.get("INPUT_PATH", DEFAULT_INPUT))
    parser.add_argument("--output", default=os.environ.get("OUTPUT_PATH", DEFAULT_OUTPUT))
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--max-new-tokens", type=int, default=220)
    parser.add_argument("--offload-dir", default=os.environ.get("OFFLOAD_DIR", DEFAULT_OFFLOAD_DIR))
    parser.add_argument(
        "--device-map",
        default=os.environ.get("DEVICE_MAP", "auto"),
        help="Transformers device_map. Use 'cpu' if your GPU cannot fit the model.",
    )
    return parser.parse_args()


def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`resume_download` is deprecated",
        category=FutureWarning,
        module=r"huggingface_hub\.file_download",
    )


def load_tokenizer_with_fallback(model_name: str, hf_token: Optional[str]):
    tok_kwargs = {"token": hf_token} if hf_token else {}

    try:
        return AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    except Exception as exc:  # noqa: BLE001
        print(f"Fast tokenizer failed for {model_name}: {exc}")
        print("Retrying with use_fast=False ...")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False, **tok_kwargs)


def build_prompt(instruction: str, context: str) -> str:
    return (
        "### Instruction\n"
        f"{instruction}\n\n"
        "### Context\n"
        f"{context}\n\n"
        "### Response\n"
    )


def load_prompt_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            instruction = str(record.get("instruction", "")).strip()
            context = str(record.get("context", "")).strip()
            if not instruction:
                print(f"Skipping line {idx}: missing instruction")
                continue
            yield {"instruction": instruction, "context": context}


def main() -> None:
    args = parse_args()
    configure_warnings()

    input_path = Path(args.input)
    output_path = Path(args.output)
    offload_dir = Path(args.offload_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not Path(args.adapter_dir).exists():
        raise FileNotFoundError(f"Adapter directory not found: {args.adapter_dir}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    offload_dir.mkdir(parents=True, exist_ok=True)

    model_kwargs: Dict[str, object] = {
        "device_map": args.device_map,
        "low_cpu_mem_usage": True,
        "offload_folder": str(offload_dir),
    }
    if args.hf_token:
        model_kwargs["token"] = args.hf_token

    tokenizer = load_tokenizer_with_fallback(args.model_name, args.hf_token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model = PeftModel.from_pretrained(
        base_model,
        args.adapter_dir,
        device_map=args.device_map,
        offload_folder=str(offload_dir),
    )
    model.eval()

    device = "cuda" if torch.cuda.is_available() and args.device_map != "cpu" else "cpu"

    written = 0
    with output_path.open("w", encoding="utf-8") as out:
        for row in load_prompt_rows(input_path):
            prompt = build_prompt(row["instruction"], row["context"])
            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=0.0,
                    pad_token_id=tokenizer.eos_token_id,
                )

            full_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            if full_text.startswith(prompt):
                response = full_text[len(prompt) :].strip()
            else:
                response = full_text.strip()

            out.write(
                json.dumps(
                    {
                        "instruction": row["instruction"],
                        "context": row["context"],
                        "response": response,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            written += 1

    print(f"Wrote {written} rows to {output_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

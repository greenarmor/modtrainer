import argparse
import os
import random
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

MODEL_ALIASES = {
    "mistral-7b-instruct-v0.2": DEFAULT_MODEL,
    "mistral/Mistral-7B-Instruct-v0.2": DEFAULT_MODEL,
    "minstral/Mistral-7B-Instruct-v0.2": DEFAULT_MODEL,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a policy model using LoRA.")
    parser.add_argument("--train-file", default="data/train.jsonl")
    parser.add_argument("--val-file", default="data/val.jsonl")
    parser.add_argument("--output-dir", default="./govchain-model")
    parser.add_argument("--save-dir", default="govchain-lora")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_example(example: Dict[str, str]) -> Dict[str, str]:
    prompt = (
        "### Instruction\n"
        f"{example['instruction']}\n\n"
        "### Context\n"
        f"{example['context']}\n\n"
        "### Response\n"
        f"{example['response']}"
    )
    return {"text": prompt}


def normalize_model_name(model_name: str) -> str:
    normalized = model_name.strip()

    if "minstral" in normalized.lower():
        normalized = normalized.replace("minstral", "mistral")
        normalized = normalized.replace("Minstral", "Mistral")

    return MODEL_ALIASES.get(normalized, normalized)


def load_tokenizer_with_fallback(model_name: str, hf_token: Optional[str]):
    tok_kwargs = {"token": hf_token} if hf_token else {}

    try:
        return AutoTokenizer.from_pretrained(model_name, **tok_kwargs)
    except Exception as fast_exc:  # noqa: BLE001
        print(f"Fast tokenizer load failed for {model_name}: {fast_exc}. Retrying with use_fast=False.")
        return AutoTokenizer.from_pretrained(model_name, use_fast=False, **tok_kwargs)


def is_missing_protobuf_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "requires the protobuf library" in message or "protobuf" in message and "not found" in message


def is_missing_sentencepiece_error(exc: Exception) -> bool:
    message = str(exc).lower()
    return "sentencepiece" in message and ("not found" in message or "install" in message)


def raise_dependency_error(candidate: str, exc: Exception) -> None:
    raise RuntimeError(
        f"Failed to load {candidate} due to a missing tokenizer dependency: {exc}. "
        "Install required packages in your venv and retry: `pip install protobuf sentencepiece`."
    ) from exc


def build_model_load_attempts(use_4bit: bool) -> List[Tuple[str, Dict[str, object]]]:
    attempts: List[Tuple[str, Dict[str, object]]] = []

    if use_4bit:
        attempts.append(("4-bit quantized", {"load_in_4bit": True}))

    attempts.append(("default", {}))
    return attempts




def model_has_fp16_parameters(model: torch.nn.Module) -> bool:
    return any(param.dtype == torch.float16 for param in model.parameters())


def load_model_with_fallback(model_name: str, hf_token: Optional[str] = None):
    requested_model_name = model_name
    model_name = normalize_model_name(model_name)
    if model_name != requested_model_name:
        print(f"Interpreting model name '{requested_model_name}' as '{model_name}'.")

    candidate_models = [model_name]
    if model_name != FALLBACK_MODEL:
        candidate_models.append(FALLBACK_MODEL)

    last_error = None
    for candidate in candidate_models:
        use_4bit = torch.cuda.is_available()
        base_kwargs: Dict[str, object] = {"device_map": "auto", "low_cpu_mem_usage": True}
        if hf_token:
            base_kwargs["token"] = hf_token

        for attempt_name, attempt_kwargs in build_model_load_attempts(use_4bit):
            try:
                model_kwargs = dict(base_kwargs)
                model_kwargs.update(attempt_kwargs)
                print(f"Trying model load for {candidate} with mode: {attempt_name}")
                model = AutoModelForCausalLM.from_pretrained(candidate, **model_kwargs)
                tokenizer = load_tokenizer_with_fallback(candidate, hf_token)
                return candidate, model, tokenizer
            except Exception as exc:  # noqa: BLE001
                print(f"Failed loading {candidate} ({attempt_name}): {exc}")
                if candidate == model_name and (is_missing_protobuf_error(exc) or is_missing_sentencepiece_error(exc)):
                    raise_dependency_error(candidate, exc)
                last_error = exc

    raise RuntimeError("Unable to load any configured model.") from last_error


def configure_warnings() -> None:
    warnings.filterwarnings(
        "ignore",
        message=r"`resume_download` is deprecated",
        category=FutureWarning,
        module=r"huggingface_hub\.file_download",
    )


def main() -> None:
    args = parse_args()
    configure_warnings()
    set_seed(args.seed)

    dataset = load_dataset("json", data_files={"train": args.train_file, "validation": args.val_file})
    dataset = dataset.map(format_example)

    selected_model_name, model, tokenizer = load_model_with_fallback(args.model_name, args.hf_token)
    print(f"Using base model: {selected_model_name}")

    tokenizer.pad_token = tokenizer.eos_token

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)

    use_fp16 = torch.cuda.is_available()
    if use_fp16 and model_has_fp16_parameters(model):
        print("Detected FP16 model parameters; disabling Trainer fp16 mixed precision to avoid GradScaler unscale errors.")
        use_fp16 = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=6,
        fp16=use_fp16,
        logging_steps=10,
        evaluation_strategy="steps",
        eval_steps=50,
        save_steps=50,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        args=training_args,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )

    trainer.train()
    trainer.save_model(args.save_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()

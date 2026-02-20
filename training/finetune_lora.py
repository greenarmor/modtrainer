import argparse
import os
import random
from typing import Dict, Optional

import numpy as np
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
FALLBACK_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


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


def load_model_with_fallback(model_name: str, hf_token: Optional[str] = None):
    candidate_models = [model_name]
    if model_name != FALLBACK_MODEL:
        candidate_models.append(FALLBACK_MODEL)

    last_error = None
    for candidate in candidate_models:
        try:
            use_4bit = torch.cuda.is_available()
            model_kwargs = {"device_map": "auto"}
            if hf_token:
                model_kwargs["token"] = hf_token

            if use_4bit:
                try:
                    model = AutoModelForCausalLM.from_pretrained(
                        candidate,
                        load_in_4bit=True,
                        **model_kwargs,
                    )
                except Exception as quant_exc:  # noqa: BLE001
                    print(f"4-bit load failed for {candidate}: {quant_exc}. Falling back to non-quantized load.")
                    model = AutoModelForCausalLM.from_pretrained(candidate, **model_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(candidate, **model_kwargs)

            tok_kwargs = {"token": hf_token} if hf_token else {}
            tokenizer = AutoTokenizer.from_pretrained(candidate, **tok_kwargs)
            return candidate, model, tokenizer
        except Exception as exc:  # noqa: BLE001
            print(f"Failed to load {candidate}: {exc}")
            last_error = exc

    raise RuntimeError("Unable to load any configured model.") from last_error


def main() -> None:
    args = parse_args()
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

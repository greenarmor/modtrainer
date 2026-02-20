import argparse
import importlib
import os
import sys
from importlib import metadata
from typing import Dict, List, Tuple

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

REQUIRED_PACKAGES: Dict[str, str] = {
    "torch": "2.2.2",
    "transformers": "4.39.3",
    "datasets": "2.18.0",
    "accelerate": "0.28.0",
    "peft": "0.10.0",
    "bitsandbytes": "0.43.0",
    "trl": "0.8.1",
    "sentencepiece": "0.2.0",
    "numpy": "1.26.4",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preflight checker for finetuning environment compatibility."
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit code when warnings are found.",
    )
    return parser.parse_args()


def is_gated_model(model_name: str) -> bool:
    return model_name.startswith("meta-llama/")


def check_token(model_name: str) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    infos: List[str] = []

    token = os.environ.get("HF_TOKEN")
    if token:
        infos.append("HF_TOKEN is set.")
    elif is_gated_model(model_name):
        warnings.append(
            "HF_TOKEN is not set for a likely gated model. Set HF_TOKEN before training."
        )
    else:
        infos.append("HF_TOKEN is not set (non-gated model may still work).")

    return infos, warnings


def check_cuda() -> Tuple[List[str], List[str]]:
    infos: List[str] = []
    warnings: List[str] = []

    try:
        import torch

        infos.append(f"torch version: {torch.__version__}")
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            infos.append(f"CUDA available: True ({device_name})")
            infos.append(f"CUDA runtime version (torch): {torch.version.cuda}")
        else:
            warnings.append(
                "CUDA is not available. Training will run without 4-bit GPU acceleration and may be slow."
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Unable to import torch/cuda info: {exc}")

    return infos, warnings


def check_packages() -> Tuple[List[str], List[str]]:
    infos: List[str] = []
    warnings: List[str] = []

    for pkg, expected in REQUIRED_PACKAGES.items():
        try:
            importlib.import_module(pkg)
            installed = metadata.version(pkg)
            if installed != expected:
                warnings.append(
                    f"{pkg} version mismatch: installed={installed}, expected={expected}"
                )
            else:
                infos.append(f"{pkg}=={installed}")
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"{pkg} not importable/installed as expected: {exc}")

    return infos, warnings


def main() -> None:
    args = parse_args()

    all_infos: List[str] = []
    all_warnings: List[str] = []

    token_infos, token_warnings = check_token(args.model_name)
    cuda_infos, cuda_warnings = check_cuda()
    pkg_infos, pkg_warnings = check_packages()

    all_infos.extend(token_infos + cuda_infos + pkg_infos)
    all_warnings.extend(token_warnings + cuda_warnings + pkg_warnings)

    print("== Preflight Environment Check ==")
    print(f"Model target: {args.model_name}")

    if all_infos:
        print("\n[INFO]")
        for item in all_infos:
            print(f"- {item}")

    if all_warnings:
        print("\n[WARNINGS]")
        for item in all_warnings:
            print(f"- {item}")
    else:
        print("\nNo warnings detected.")

    if args.strict and all_warnings:
        sys.exit(1)


if __name__ == "__main__":
    main()

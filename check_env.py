import argparse
import importlib
import os
import shlex
import sys
from importlib import metadata
from typing import Dict, List, Tuple

TARGET_TORCH_CUDA = "12.4"

DEFAULT_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"

BASE_REQUIRED_PACKAGES: Dict[str, str] = {
    "torch": "2.5.1+cu124",
    "transformers": "4.53.0",
    "datasets": "2.18.0",
    "accelerate": "0.28.0",
    "peft": "0.10.0",
    "bitsandbytes": "0.45.2",
    "trl": "0.8.1",
    "rich": "13.7.1",
    "numpy": "1.26.4",
}


def get_required_packages() -> Dict[str, str]:
    required = dict(BASE_REQUIRED_PACKAGES)
    # Use 0.2.1 across supported Python versions to avoid source-build pain.
    required["sentencepiece"] = "0.2.1"
    return required



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




def load_dotenv_if_present(dotenv_path: str = ".env") -> Tuple[List[str], List[str]]:
    infos: List[str] = []
    warnings: List[str] = []

    if not os.path.exists(dotenv_path):
        infos.append("No .env file detected (skipping dotenv load).")
        return infos, warnings

    loaded_keys: List[str] = []

    try:
        with open(dotenv_path, "r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue

                if line.startswith("export "):
                    line = line[len("export "):].strip()

                if "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                if not key:
                    continue

                value = value.strip()
                if value:
                    try:
                        value = shlex.split(value)[0] if value[0] in {'"', "'"} else value
                    except Exception:
                        pass

                if key not in os.environ:
                    os.environ[key] = value
                    loaded_keys.append(key)

        if loaded_keys:
            infos.append(
                f"Loaded {len(loaded_keys)} variable(s) from {dotenv_path}: "
                + ", ".join(sorted(loaded_keys))
            )
        else:
            infos.append(
                f"Found {dotenv_path}, but did not override existing shell variables."
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Failed to load {dotenv_path}: {exc}")

    return infos, warnings

def is_gated_model(model_name: str) -> bool:
    return model_name.startswith("meta-llama/")


def check_token(model_name: str) -> Tuple[List[str], List[str]]:
    warnings: List[str] = []
    infos: List[str] = []

    token = os.environ.get("HF_TOKEN")
    if token and token != "hf_your_huggingface_token_here":
        infos.append("HF_TOKEN is set.")
    elif token == "hf_your_huggingface_token_here":
        warnings.append(
            "HF_TOKEN still has the placeholder value from .env.example. Set your real token before training."
        )
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
            runtime_version = torch.version.cuda
            infos.append(f"CUDA available: True ({device_name})")
            infos.append(f"CUDA runtime version (torch): {runtime_version}")

            if runtime_version != TARGET_TORCH_CUDA:
                warnings.append(
                    "Torch CUDA runtime is "
                    f"{runtime_version}, but this repo requires {TARGET_TORCH_CUDA} (cu124) for stability on RTX 3090 setups. "
                    "Reinstall from requirements.txt in a clean venv to force cu124 wheels."
                )
            else:
                infos.append(f"Torch CUDA runtime matches required cu{TARGET_TORCH_CUDA.replace('.', '')}.")
        else:
            warnings.append(
                "CUDA is not available. Training will run without 4-bit GPU acceleration and may be slow."
            )
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"Unable to import torch/cuda info: {exc}")

    return infos, warnings


def check_python_version() -> Tuple[List[str], List[str]]:
    infos: List[str] = []
    warnings: List[str] = []

    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    infos.append(f"python version: {py_version}")

    if sys.version_info >= (3, 13):
        infos.append(
            "Python 3.13 detected. This repo can work on 3.13, but if you hit wheel/install issues "
            "use Python 3.12 for the most predictable environment."
        )

    return infos, warnings


def check_path_contamination() -> Tuple[List[str], List[str]]:
    infos: List[str] = []
    warnings: List[str] = []

    path_entries = [entry for entry in os.environ.get("PATH", "").split(":") if entry]
    suspicious_entries = [
        entry
        for entry in path_entries
        if "meta-llama" in entry or "govchain-model" in entry
    ]

    if suspicious_entries:
        warnings.append(
            "PATH appears to contain model identifiers/paths that are not binaries. "
            "Remove these entries from shell startup files: "
            + ", ".join(suspicious_entries)
        )
    else:
        infos.append("PATH does not contain obvious model-name contamination.")

    return infos, warnings

def check_packages() -> Tuple[List[str], List[str]]:
    infos: List[str] = []
    warnings: List[str] = []

    for pkg, expected in get_required_packages().items():
        try:
            module = importlib.import_module(pkg)
            installed = module.__version__ if pkg == "torch" else metadata.version(pkg)
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

    dotenv_infos, dotenv_warnings = load_dotenv_if_present()
    token_infos, token_warnings = check_token(args.model_name)
    py_infos, py_warnings = check_python_version()
    cuda_infos, cuda_warnings = check_cuda()
    path_infos, path_warnings = check_path_contamination()
    pkg_infos, pkg_warnings = check_packages()

    all_infos.extend(dotenv_infos + token_infos + py_infos + cuda_infos + path_infos + pkg_infos)
    all_warnings.extend(
        dotenv_warnings + token_warnings + py_warnings + cuda_warnings + path_warnings + pkg_warnings
    )

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

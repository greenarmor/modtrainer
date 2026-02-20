import argparse
import json
from pathlib import Path
from typing import Dict, List

from policy_checks import check_policy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run policy checks against JSONL responses.")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/val.jsonl", "data/govchain_redteam_500.jsonl"],
        help="Input JSONL files with a `response` field.",
    )
    parser.add_argument(
        "--max-errors",
        type=int,
        default=0,
        help="Maximum allowed policy check failures before returning a non-zero code.",
    )
    return parser.parse_args()


def evaluate_file(path: Path) -> Dict[str, int | List[Dict[str, object]]]:
    total = 0
    failures: List[Dict[str, object]] = []

    with path.open() as handle:
        for line_num, line in enumerate(handle, start=1):
            record = json.loads(line)
            total += 1
            result = check_policy(record.get("response", ""))
            if result["violations"] or result["missing"]:
                failures.append(
                    {
                        "line": line_num,
                        "instruction": record.get("instruction", ""),
                        "violations": result["violations"],
                        "missing": result["missing"],
                    }
                )

    return {"total": total, "failure_count": len(failures), "failures": failures}


def main() -> None:
    args = parse_args()
    total_errors = 0

    for input_path in args.inputs:
        path = Path(input_path)
        result = evaluate_file(path)
        total_errors += int(result["failure_count"])

        print(f"\n== {path} ==")
        print(f"Total records: {result['total']}")
        print(f"Failures: {result['failure_count']}")

        for failure in result["failures"][:5]:
            print(
                "- line {line}: violations={violations} missing={missing} instruction={instruction}".format(
                    **failure
                )
            )

    if total_errors > args.max_errors:
        raise SystemExit(
            f"Policy evaluation failed: {total_errors} records had violations/missing terms (limit={args.max_errors})."
        )


if __name__ == "__main__":
    main()

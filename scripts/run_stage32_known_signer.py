from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.known_signer_runner import run_next_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run exactly the next fold-major Stage-32 known-signer "
            "evaluation on the frozen NVIDIA T4 runtime."
        )
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help=(
            "External durable root (normally mounted Google Drive). "
            "It must be outside the Git checkout."
        ),
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=(
            PROJECT_ROOT
            / "cache"
            / "raw"
            / "stage17_raw_f32.bin"
        ),
        help="Path to the frozen Stage-17 binary cache.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()

    print("=" * 78)
    print("STAGE 32B - KNOWN-SIGNER GPU RUNNER")
    print("=" * 78)
    print()
    print("One invocation executes at most one outer evaluation.")
    print("The frozen registry determines the next fold/model automatically.")
    print()

    result = run_next_evaluation(
        project_root=PROJECT_ROOT,
        artifact_root=arguments.artifact_root,
        raw_cache_path=arguments.raw_cache,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    print()

    if result["status"] == "ALL_COMPLETE":
        print("FINAL STATUS: ALL 25 STAGE-32 OUTER EVALUATIONS COMPLETE")
    elif result["status"] in {"COMPLETE", "SKIP_COMPLETE"}:
        print("FINAL STATUS: ONE STAGE-32 OUTER EVALUATION COMPLETE")
    else:
        raise RuntimeError(f"Unexpected Stage-32 result: {result['status']}.")

    print("=" * 78)


if __name__ == "__main__":
    main()

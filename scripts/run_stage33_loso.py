from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.training.loso_runner import run_next_evaluation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly the next frozen Stage-33 LOSO evaluation."
    )
    parser.add_argument(
        "--artifact-root",
        type=Path,
        required=True,
        help="External durable output root, normally mounted Google Drive.",
    )
    parser.add_argument(
        "--raw-cache",
        type=Path,
        default=PROJECT_ROOT / "cache/raw/stage17_raw_f32.bin",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    print("=" * 78)
    print("STAGE 33B - LOSO UNSEEN-SIGNER GPU RUNNER")
    print("=" * 78)
    print("One invocation executes at most one LOSO model evaluation.\n")
    result = run_next_evaluation(
        project_root=PROJECT_ROOT,
        artifact_root=arguments.artifact_root,
        raw_cache_path=arguments.raw_cache,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True))
    if result["status"] == "ALL_COMPLETE":
        print("\nFINAL STATUS: ALL 15 STAGE-33 LOSO EVALUATIONS COMPLETE")
    elif result["status"] in {"COMPLETE", "SKIP_COMPLETE"}:
        print("\nFINAL STATUS: ONE STAGE-33 LOSO EVALUATION COMPLETE")
    else:
        raise RuntimeError(f"Unexpected Stage-33 status: {result['status']}.")
    print("=" * 78)


if __name__ == "__main__":
    main()

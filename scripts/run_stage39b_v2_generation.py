"""Run or resume the frozen Stage 39B V2 generation."""

import argparse
import json
from pathlib import Path

from src.evaluation.stage39b_v2_runner import DEFAULT_BATCH_SIZE, run_v2_generation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-requests", type=int)
    arguments = parser.parse_args()
    result = run_v2_generation(
        project_root=PROJECT_ROOT,
        output_root=arguments.output_root,
        batch_size=arguments.batch_size,
        max_requests=arguments.max_requests,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

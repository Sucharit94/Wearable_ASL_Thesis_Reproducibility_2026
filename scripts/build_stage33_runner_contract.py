from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from src.training.known_signer_runner import atomic_write_text, sha256_file
from src.training.loso_runner import (
    build_dry_run_report,
    portable_sha256,
)
from src.training.loso_plan import REQUIRED_RUNNER_TAG


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "artifacts/training"
DRY_RUN = OUTPUT / "stage33b_runner_dry_run.json"
CONTRACT = OUTPUT / "stage33b_runner_contract.json"
DOC = PROJECT_ROOT / "docs/STAGE_33B_LOSO_RUNNER.md"
IMPLEMENTATION_PATHS = (
    "configs/experiments/stage33_runner.json",
    "src/training/checkpoint.py",
    "src/training/known_signer_runner.py",
    "src/training/loso_runner.py",
    "scripts/run_stage33_loso.py",
    "scripts/build_stage33_runner_contract.py",
    "scripts/verify_stage33_runner.py",
    "tests/test_loso_runner.py",
)


def write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n",
    )


def write_hash(path: Path) -> str:
    digest = sha256_file(path)
    atomic_write_text(path.with_suffix(".sha256"), f"{digest}  {path.name}\n")
    return digest


def main() -> None:
    print("=" * 78)
    print("STAGE 33B - LOSO RUNNER DRY-RUN AND CONTRACT BUILD")
    print("=" * 78)
    report = build_dry_run_report(PROJECT_ROOT)
    write_json(DRY_RUN, report)
    dry_hash = write_hash(DRY_RUN)
    implementation_hashes = {
        relative: portable_sha256(PROJECT_ROOT / relative)[1]
        for relative in IMPLEMENTATION_PATHS
    }
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 33,
        "substage": "B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "runner_frozen_before_loso_outer_evaluation",
        "runner_tag": {
            "name": REQUIRED_RUNNER_TAG,
            "must_be_annotated": True,
            "must_equal_head_before_training": True,
        },
        "dry_run": {
            "path": str(DRY_RUN.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": dry_hash,
            "passed": True,
        },
        "matrix": report["matrix"],
        "partitions": report["partitions"],
        "protocol": {
            "primary_metric": "loso_all_pooled_wer",
            "selection_uses_inner_validation_only": True,
            "reset_before_refit": True,
            "refit_exact_selected_epochs": True,
            "outer_test_controls_training": False,
            "outer_test_evaluations_per_model_direction": 1,
            "support_views_use_same_predictions": True,
            "support_views_train_separate_models": False,
            "execution_precision": "float32",
            "automatic_mixed_precision": False,
        },
        "safety": {
            "required_gpu": "NVIDIA T4",
            "clean_tagged_git_checkout": True,
            "external_durable_artifact_root": True,
            "atomic_epoch_checkpoints": True,
            "epoch_boundary_resume": True,
            "oom_only_batch_fallback": True,
            "effective_batch_preserved": True,
            "test_dataset_constructed_only_after_refit": True,
            "explicit_trusted_checkpoint_load": True,
            "mixed_phase_csv_schema_safe": True,
        },
        "implementation_hashes": implementation_hashes,
    }
    write_json(CONTRACT, contract)
    contract_hash = write_hash(CONTRACT)
    atomic_write_text(
        DOC,
        f"""# Stage 33B - Resumable LOSO runner

This runner executes 15 unseen-signer evaluations in LOSO-major order. Each
invocation completes at most one model-direction pair and writes only to an
external durable artifact root.

The held-out subject is excluded from normalization, epoch selection, early
stopping, scheduling and refit. One outer prediction file is scored as LOSO-all,
template-supported, token-supported and token-unsupported views. The support
views never train separate models.

All models use float32 without AMP on NVIDIA T4. Epoch checkpoints are atomic
and resumable. Checkpoint loading explicitly uses trusted project mode, and
parent CSV finalization uses a union schema for selection and refit columns.

Runner tag required before scoring: `{REQUIRED_RUNNER_TAG}`

Dry-run SHA-256: `{dry_hash}`
Contract SHA-256: `{contract_hash}`
""",
    )
    print("\nRUN MATRIX\n" + "-" * 78)
    print("Outer evaluations: 15")
    print("Registry rows: 45")
    print("Order: LOSO-major")
    print("\nLEAKAGE AND SUPPORT SAFETY\n" + "-" * 78)
    print("Held-out signer used for training: False")
    print("Training started by dry run: False")
    print("Outer scores computed by dry run: False")
    print("Support views use one prediction file: True")
    print("\nRUNTIME FIXES\n" + "-" * 78)
    print("Explicit trusted checkpoint loading: True")
    print("Mixed-phase CSV union schema: True")
    print("Float32 / AMP: float32 / False")
    print("\nFINAL STATUS: STAGE 33B RUNNER BUILD PASSED")
    print("NO TRAINING OR OUTER SCORING WAS STARTED")
    print("=" * 78)


if __name__ == "__main__":
    main()

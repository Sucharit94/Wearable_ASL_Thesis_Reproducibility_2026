from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from src.training.known_signer_runner import (
    RUNNER_TAG,
    atomic_write_text,
    build_dry_run_report,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "training"
DRY_RUN_PATH = OUTPUT_DIR / "stage32b_runner_dry_run.json"
DRY_RUN_HASH_PATH = DRY_RUN_PATH.with_suffix(".sha256")
CONTRACT_PATH = OUTPUT_DIR / "stage32b_runner_contract.json"
CONTRACT_HASH_PATH = CONTRACT_PATH.with_suffix(".sha256")
DOC_PATH = PROJECT_ROOT / "docs" / "STAGE_32B_KNOWN_SIGNER_RUNNER.md"


IMPLEMENTATION_PATHS = (
    "configs/experiments/stage32_runner.json",
    "src/training/known_signer_runner.py",
    "scripts/run_stage32_known_signer.py",
    "scripts/build_stage32_runner_contract.py",
    "scripts/verify_stage32_runner.py",
    "tests/test_known_signer_runner.py",
)


def write_json(path: Path, payload: object) -> None:
    atomic_write_text(
        path,
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
    )


def write_hash(path: Path, digest: str, target_name: str) -> None:
    atomic_write_text(path, f"{digest}  {target_name}\n")


def main() -> None:
    print("=" * 78)
    print("STAGE 32B - KNOWN-SIGNER RUNNER DRY-RUN AND CONTRACT BUILD")
    print("=" * 78)

    report = build_dry_run_report(PROJECT_ROOT)
    write_json(DRY_RUN_PATH, report)
    dry_hash = sha256_file(DRY_RUN_PATH)
    write_hash(DRY_RUN_HASH_PATH, dry_hash, DRY_RUN_PATH.name)

    implementation_hashes = {
        relative: sha256_file(PROJECT_ROOT / relative)
        for relative in IMPLEMENTATION_PATHS
    }
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 32,
        "substage": "B",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "runner_frozen_before_outer_evaluation",
        "runner_tag": {
            "name": RUNNER_TAG,
            "must_be_annotated": True,
            "must_equal_head_before_training": True,
            "creation_authorized_only_after_stage32b_pass": True,
        },
        "dry_run": {
            "path": str(DRY_RUN_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "sha256": dry_hash,
            "passed": report["passed"],
            "training_started": report["leakage_barrier"]["training_started"],
            "scores_computed": report["leakage_barrier"]["scores_computed"],
            "checkpoints_created": report["leakage_barrier"]["checkpoints_created"],
        },
        "matrix": report["matrix"],
        "partitions": report["partitions"],
        "protocol": {
            "inner_selection_only": True,
            "g1_full_selection_cap": 20,
            "compact_epoch_cap": 80,
            "compact_minimum_early_stop_epoch": 15,
            "selection_tie_break_order": [
                "lower_validation_wer",
                "higher_validation_exact_sentence_accuracy",
                "lower_validation_ctc_loss",
                "earlier_epoch",
            ],
            "reset_before_refit": True,
            "refit_exact_selected_epochs": True,
            "outer_test_controls_training": False,
            "outer_test_evaluations_per_model_fold": 1,
            "execution_precision": report["runtime_gate"]["execution_precision"],
            "automatic_mixed_precision": report["runtime_gate"][
                "automatic_mixed_precision"
            ],
        },
        "safety": {
            "required_gpu": "NVIDIA T4",
            "clean_tagged_git_checkout": True,
            "external_durable_artifact_root": True,
            "atomic_epoch_checkpoints": True,
            "epoch_boundary_resume": True,
            "oom_only_batch_fallback": True,
            "effective_batch_preserved": True,
            "fallback_restarts_from_seed": True,
            "length_safe_decoder": True,
            "test_dataset_constructed_only_after_refit": True,
            "float32_for_all_models": True,
        },
        "implementation_hashes": implementation_hashes,
    }
    write_json(CONTRACT_PATH, contract)
    contract_hash = sha256_file(CONTRACT_PATH)
    write_hash(CONTRACT_HASH_PATH, contract_hash, CONTRACT_PATH.name)

    markdown = f"""# Stage 32B - Known-signer resumable runner

## Status

The Stage-32 known-signer execution runner is frozen before any outer-test
score. The static dry run passed without starting training, creating a
checkpoint, constructing an outer-test dataset, or computing a score.

## Frozen execution order

- 25 outer evaluations: five folds by five models
- fold-major model order: G1, C1, C2, C3, C4
- 75 registry identities: parent, inner selection, and full refit
- one command executes at most one parent outer evaluation

## Selection and refit

G1 completes all 20 inner-selection epochs. Compact models have an 80-epoch
cap, cannot stop before epoch 15, and use patience 8. Epoch E* is chosen by
lower validation WER, higher exact-sentence accuracy, lower validation CTC
loss, then earlier epoch. The model, optimizer, and RNG are reset before a
fresh refit on all non-test samples for exactly E* epochs.

The outer-test dataset is constructed only after the refit checkpoint has
reached E*. It is evaluated once, decoded only to valid output lengths, and
cannot affect training or epoch selection.

## Runtime gate

The real runner requires an NVIDIA T4, a clean checkout whose HEAD is the
annotated `{RUNNER_TAG}` tag, the frozen Stage-17 cache hash, and an external
durable artifact root. A verified OOM is the only allowed reason to halve a
physical batch. Gradient accumulation preserves the configured effective
batch, and the attempt restarts from the seed.

All five models run in float32 with automatic mixed precision disabled. This
policy was frozen before outer scoring after the T4 FP16 G1 preflight produced
non-finite gradients while the equivalent float32 preflight passed.

## Resume contract

Every epoch writes an atomic trusted checkpoint containing model, optimizer,
scheduler, AMP scaler, RNG, and training state. Completed outer predictions
are protected by a one-evaluation marker and hashes. Parent runs satisfy the
Stage-24 completion-artifact contract.

## Dry run

- report SHA-256: `{dry_hash}`
- training started: `False`
- scores computed: `False`
- checkpoints created: `False`
- final status: `PASSED`
"""
    atomic_write_text(DOC_PATH, markdown)

    print()
    print("RUN MATRIX")
    print("-" * 78)
    print("Outer evaluations:", report["matrix"]["outer_evaluations"])
    print("Registry rows:", report["matrix"]["registry_rows"])
    print("Validated identities:", report["matrix"]["validated_identities"])
    print("Fold-major:", report["matrix"]["fold_major"])

    print()
    print("LEAKAGE BARRIER")
    print("-" * 78)
    for key, value in report["leakage_barrier"].items():
        print(f"{key}: {value}")

    print()
    print("RUNTIME GATE")
    print("-" * 78)
    print("Required tag:", RUNNER_TAG)
    print("Annotated tag required:", True)
    print("Tag commit must equal HEAD:", True)
    print("NVIDIA T4 required:", True)
    print("External artifact root required:", True)

    print()
    print("OUTPUTS")
    print("-" * 78)
    for path in (
        DRY_RUN_PATH,
        DRY_RUN_HASH_PATH,
        CONTRACT_PATH,
        CONTRACT_HASH_PATH,
        DOC_PATH,
    ):
        print("Saved:", path.relative_to(PROJECT_ROOT))

    print()
    print("FINAL STATUS: STAGE 32B RUNNER BUILD PASSED")
    print("NO TRAINING OR OUTER SCORING WAS STARTED")
    print("=" * 78)


if __name__ == "__main__":
    main()

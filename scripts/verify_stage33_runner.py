from __future__ import annotations

import json
from pathlib import Path

from src.training.known_signer_runner import read_recorded_sha256, sha256_file
from src.training.loso_plan import REQUIRED_RUNNER_TAG
from src.training.loso_runner import (
    build_dry_run_report,
    portable_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DRY_RUN = PROJECT_ROOT / "artifacts/training/stage33b_runner_dry_run.json"
CONTRACT = PROJECT_ROOT / "artifacts/training/stage33b_runner_contract.json"
DOC = PROJECT_ROOT / "docs/STAGE_33B_LOSO_RUNNER.md"


def main() -> None:
    print("=" * 78)
    print("STAGE 33B - LOSO RUNNER VERIFIER")
    print("=" * 78)
    required = (
        PROJECT_ROOT / "configs/experiments/stage33_runner.json",
        PROJECT_ROOT / "src/training/loso_runner.py",
        PROJECT_ROOT / "scripts/run_stage33_loso.py",
        PROJECT_ROOT / "scripts/build_stage33_runner_contract.py",
        PROJECT_ROOT / "scripts/verify_stage33_runner.py",
        PROJECT_ROOT / "tests/test_loso_runner.py",
        DRY_RUN, DRY_RUN.with_suffix(".sha256"),
        CONTRACT, CONTRACT.with_suffix(".sha256"), DOC,
    )
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing Stage-33B files: " + ", ".join(missing))
    dry_match = read_recorded_sha256(DRY_RUN.with_suffix(".sha256")) == sha256_file(DRY_RUN)
    contract_match = read_recorded_sha256(CONTRACT.with_suffix(".sha256")) == sha256_file(CONTRACT)
    if not dry_match or not contract_match:
        raise RuntimeError("Generated Stage-33B hash mismatch.")
    recorded = json.loads(DRY_RUN.read_text(encoding="utf-8-sig"))
    current = build_dry_run_report(PROJECT_ROOT)
    comparable = tuple(key for key in recorded if key != "created_at_utc")
    if any(recorded[key] != current[key] for key in comparable):
        raise RuntimeError("Recorded Stage-33B dry run is stale.")
    contract = json.loads(CONTRACT.read_text(encoding="utf-8-sig"))
    for relative, expected in contract["implementation_hashes"].items():
        if portable_sha256(PROJECT_ROOT / relative)[1] != expected:
            raise RuntimeError(f"Implementation hash mismatch: {relative}.")
    checks = {
        "stage": contract["stage"] == 33,
        "substage": contract["substage"] == "B",
        "runner_tag": contract["runner_tag"]["name"] == REQUIRED_RUNNER_TAG,
        "outer_evaluations": recorded["matrix"]["outer_evaluations"] == 15,
        "registry_rows": recorded["matrix"]["registry_rows"] == 45,
        "loso_major": recorded["matrix"]["loso_major"] is True,
        "training_not_started": recorded["leakage_barrier"]["training_started"] is False,
        "scores_not_computed": recorded["leakage_barrier"]["scores_computed"] is False,
        "held_out_not_training": recorded["leakage_barrier"]["held_out_subject_used_for_training"] is False,
        "same_predictions": recorded["support_reporting"]["same_predictions_for_all_views"] is True,
        "no_separate_support_models": recorded["support_reporting"]["separate_models_trained"] is False,
        "float32": contract["protocol"]["execution_precision"] == "float32",
        "amp_disabled": contract["protocol"]["automatic_mixed_precision"] is False,
        "checkpoint_fix": contract["safety"]["explicit_trusted_checkpoint_load"] is True,
        "csv_fix": contract["safety"]["mixed_phase_csv_schema_safe"] is True,
        "dry_hash": dry_match,
        "contract_hash": contract_match,
    }
    failed = [name for name, passed in checks.items() if passed is not True]
    print("\nRUN MATRIX\n" + "-" * 78)
    print("Outer evaluations:", recorded["matrix"]["outer_evaluations"])
    print("Registry rows:", recorded["matrix"]["registry_rows"])
    print("LOSO-major:", recorded["matrix"]["loso_major"])
    print("First:", recorded["matrix"]["first"])
    print("Last:", recorded["matrix"]["last"])
    print("\nPARTITION SAFETY\n" + "-" * 78)
    for row in recorded["partitions"]:
        print(
            f"{row['loso_id']}: train={row['inner_train']} "
            f"validation={row['inner_validation']} test={row['outer_test']} leaks=0"
        )
    print("\nSUPPORT REPORTING\n" + "-" * 78)
    print("Views:", recorded["support_reporting"]["views"])
    print("Same predictions for every view: True")
    print("Separate support models trained: False")
    print("\nRUNTIME SAFETY\n" + "-" * 78)
    print("Precision: float32")
    print("Automatic mixed precision: False")
    print("Explicit trusted checkpoint load: True")
    print("Mixed-phase CSV union schema: True")
    print("\nFINAL CHECKS\n" + "-" * 78)
    print("Total checks:", len(checks))
    print("Failed checks:", len(failed))
    if failed:
        raise RuntimeError("Stage-33B checks failed: " + ", ".join(failed))
    print("\nFINAL STATUS: STAGE 33B LOSO RUNNER PASSED")
    print("SAFE TO COMMIT AND CREATE THE ANNOTATED RUNNER TAG")
    print("DO NOT START LOSO SCORING ON THE LOCAL CPU MACHINE")
    print("=" * 78)


if __name__ == "__main__":
    main()

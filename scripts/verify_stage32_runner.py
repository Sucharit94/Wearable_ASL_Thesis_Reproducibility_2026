from __future__ import annotations

import json
from pathlib import Path

from src.training.known_signer_runner import (
    MODEL_ORDER,
    OUTER_FOLDS,
    RUNNER_TAG,
    build_dry_run_report,
    read_recorded_sha256,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "training"
DRY_RUN_PATH = OUTPUT_DIR / "stage32b_runner_dry_run.json"
DRY_RUN_HASH_PATH = DRY_RUN_PATH.with_suffix(".sha256")
CONTRACT_PATH = OUTPUT_DIR / "stage32b_runner_contract.json"
CONTRACT_HASH_PATH = CONTRACT_PATH.with_suffix(".sha256")
DOC_PATH = PROJECT_ROOT / "docs" / "STAGE_32B_KNOWN_SIGNER_RUNNER.md"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def main() -> None:
    print("=" * 78)
    print("STAGE 32B - KNOWN-SIGNER RUNNER VERIFIER")
    print("=" * 78)

    required = (
        PROJECT_ROOT / "configs" / "experiments" / "stage32_runner.json",
        PROJECT_ROOT / "src" / "training" / "known_signer_runner.py",
        PROJECT_ROOT / "scripts" / "run_stage32_known_signer.py",
        PROJECT_ROOT / "scripts" / "build_stage32_runner_contract.py",
        PROJECT_ROOT / "scripts" / "verify_stage32_runner.py",
        PROJECT_ROOT / "tests" / "test_known_signer_runner.py",
        DRY_RUN_PATH,
        DRY_RUN_HASH_PATH,
        CONTRACT_PATH,
        CONTRACT_HASH_PATH,
        DOC_PATH,
    )
    missing = [str(path.relative_to(PROJECT_ROOT)) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError("Missing Stage-32B files: " + ", ".join(missing) + ".")

    dry_hash_match = read_recorded_sha256(DRY_RUN_HASH_PATH) == sha256_file(DRY_RUN_PATH)
    contract_hash_match = (
        read_recorded_sha256(CONTRACT_HASH_PATH) == sha256_file(CONTRACT_PATH)
    )
    if not dry_hash_match or not contract_hash_match:
        raise RuntimeError("A Stage-32B generated-artifact hash does not match.")

    recorded_dry = read_json(DRY_RUN_PATH)
    current_dry = build_dry_run_report(PROJECT_ROOT)
    comparable_keys = (
        "project_generation",
        "stage",
        "substage",
        "mode",
        "frozen_hashes",
        "runner_config_sha256",
        "model_config_hashes",
        "matrix",
        "partitions",
        "selection",
        "refit",
        "leakage_barrier",
        "runtime_gate",
        "passed",
    )
    if any(recorded_dry[key] != current_dry[key] for key in comparable_keys):
        raise RuntimeError("Recorded Stage-32B dry run is stale.")

    contract = read_json(CONTRACT_PATH)
    for relative, expected_hash in contract["implementation_hashes"].items():
        if sha256_file(PROJECT_ROOT / relative) != expected_hash:
            raise RuntimeError(f"Stage-32B implementation hash mismatch: {relative}.")

    expected_order = [
        (fold, model_id)
        for fold in OUTER_FOLDS
        for model_id in MODEL_ORDER
    ]
    plan = read_json(
        PROJECT_ROOT / "artifacts" / "experiments" / "stage32_known_run_plan.json"
    )
    observed_order = [
        (int(row["outer_fold"]), row["model_id"])
        for row in plan["evaluations"]
    ]
    if observed_order != expected_order:
        raise RuntimeError("Fold-major order changed.")

    checks = {
        "stage": contract["stage"] == 32,
        "substage": contract["substage"] == "B",
        "status_frozen_before_scoring": (
            contract["status"] == "runner_frozen_before_outer_evaluation"
        ),
        "runner_tag": contract["runner_tag"]["name"] == RUNNER_TAG,
        "runner_tag_annotated": contract["runner_tag"]["must_be_annotated"] is True,
        "dry_run_hash_match": dry_hash_match,
        "contract_hash_match": contract_hash_match,
        "dry_run_passed": recorded_dry["passed"] is True,
        "outer_evaluations": recorded_dry["matrix"]["outer_evaluations"] == 25,
        "registry_rows": recorded_dry["matrix"]["registry_rows"] == 75,
        "validated_identities": recorded_dry["matrix"]["validated_identities"] == 75,
        "fold_major": recorded_dry["matrix"]["fold_major"] is True,
        "all_fold_counts": all(
            partition["inner_train"] == 5328
            and partition["inner_validation"] == 592
            and partition["outer_training_pool"] == 5920
            and partition["outer_test"] == 1480
            and partition["all_unique"] is True
            for partition in recorded_dry["partitions"]
        ),
        "training_not_started": (
            recorded_dry["leakage_barrier"]["training_started"] is False
        ),
        "scores_not_computed": (
            recorded_dry["leakage_barrier"]["scores_computed"] is False
        ),
        "checkpoints_not_created": (
            recorded_dry["leakage_barrier"]["checkpoints_created"] is False
        ),
        "no_premature_artifacts": (
            recorded_dry["leakage_barrier"]["premature_repository_artifacts"] == 0
        ),
        "test_does_not_control_training": (
            contract["protocol"]["outer_test_controls_training"] is False
        ),
        "one_test_evaluation": (
            contract["protocol"]["outer_test_evaluations_per_model_fold"] == 1
        ),
        "external_artifact_root": (
            contract["safety"]["external_durable_artifact_root"] is True
        ),
        "t4_required": contract["safety"]["required_gpu"] == "NVIDIA T4",
        "oom_only_fallback": contract["safety"]["oom_only_batch_fallback"] is True,
        "effective_batch_preserved": (
            contract["safety"]["effective_batch_preserved"] is True
        ),
        "test_after_refit": (
            contract["safety"]["test_dataset_constructed_only_after_refit"] is True
        ),
        "float32_for_all_models": (
            contract["safety"]["float32_for_all_models"] is True
            and contract["protocol"]["execution_precision"] == "float32"
            and recorded_dry["runtime_gate"]["execution_precision"] == "float32"
        ),
        "automatic_mixed_precision_disabled": (
            contract["protocol"]["automatic_mixed_precision"] is False
            and recorded_dry["runtime_gate"]["automatic_mixed_precision"] is False
        ),
    }
    failed = [name for name, passed in checks.items() if passed is not True]

    print()
    print("RUN MATRIX")
    print("-" * 78)
    print("Outer evaluations:", recorded_dry["matrix"]["outer_evaluations"])
    print("Registry rows:", recorded_dry["matrix"]["registry_rows"])
    print("Fold-major:", recorded_dry["matrix"]["fold_major"])
    print("First:", recorded_dry["matrix"]["first"])
    print("Last:", recorded_dry["matrix"]["last"])

    print()
    print("SELECTION / REFIT")
    print("-" * 78)
    print("G1 full selection cap:", contract["protocol"]["g1_full_selection_cap"])
    print("Compact epoch cap:", contract["protocol"]["compact_epoch_cap"])
    print(
        "Compact minimum stop epoch:",
        contract["protocol"]["compact_minimum_early_stop_epoch"],
    )
    print("Fresh reset before refit:", contract["protocol"]["reset_before_refit"])
    print("Refit exactly E* epochs:", contract["protocol"]["refit_exact_selected_epochs"])
    print("Outer test controls training:", contract["protocol"]["outer_test_controls_training"])

    print()
    print("LEAKAGE BARRIER")
    print("-" * 78)
    print("Training started:", recorded_dry["leakage_barrier"]["training_started"])
    print("Scores computed:", recorded_dry["leakage_barrier"]["scores_computed"])
    print("Checkpoints created:", recorded_dry["leakage_barrier"]["checkpoints_created"])
    print(
        "Premature repository artifacts:",
        recorded_dry["leakage_barrier"]["premature_repository_artifacts"],
    )

    print()
    print("HASHES")
    print("-" * 78)
    print("Dry-run hash match:", dry_hash_match)
    print("Contract hash match:", contract_hash_match)
    print("Implementation hashes:", len(contract["implementation_hashes"]))

    print()
    print("NUMERICAL PRECISION")
    print("-" * 78)
    print("Execution precision:", contract["protocol"]["execution_precision"])
    print(
        "Automatic mixed precision:",
        contract["protocol"]["automatic_mixed_precision"],
    )
    print("Applies to all five models:", contract["safety"]["float32_for_all_models"])

    print()
    print("FINAL CHECKS")
    print("-" * 78)
    print("Total checks:", len(checks))
    print("Failed checks:", len(failed))
    if failed:
        print("Failures:", ", ".join(failed))
        raise RuntimeError("Stage-32B verifier failed.")

    print()
    print("FINAL STATUS: STAGE 32B RUNNER PASSED")
    print("SAFE TO COMMIT AND CREATE THE ANNOTATED RUNNER TAG")
    print("DO NOT START OUTER SCORING ON THIS LOCAL CPU MACHINE")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""Build the Stage 37 language-correction protocol freeze."""

from pathlib import Path

from src.evaluation.stage37_language_protocol import (
    STAGE37_ROOT,
    arm_rows,
    build_fold_plan,
    leakage_rows,
    metric_rows,
    portable_text_sha256,
    prompt_contract,
    read_csv,
    read_json,
    render_report,
    require_portable_recorded_hash,
    system_rows,
    validate_language_config,
    validate_targets,
    verify_prior_contract,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_stage37(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    stage35_hash = verify_prior_contract(
        root, 35, Path("artifacts/analysis/stage35")
    )
    stage36_hash = verify_prior_contract(
        root, 36, Path("artifacts/analysis/stage36")
    )
    stage35 = read_json(
        root / "artifacts/analysis/stage35/stage35_contract.json"
    )
    stage36 = read_json(
        root / "artifacts/analysis/stage36/stage36_contract.json"
    )
    decision = stage35.get("frozen_decision", {})
    if decision.get("accuracy_reference") != "g1" or decision.get("recommended_lightweight") != "c2":
        raise RuntimeError("Stage 35 model roles changed.")
    if stage36.get("findings", {}).get("recommended_lightweight") != "c2":
        raise RuntimeError("Stage 36 lightweight finding changed.")

    config_path = root / "configs/experiments/language.yaml"
    split_path = root / "artifacts/splits/stage32_known_inner_assignment.csv"
    split_hash_path = split_path.with_suffix(".sha256")
    target_path = root / "artifacts/labels/stage10_sentence_targets.csv"
    target_hash_path = target_path.with_suffix(".sha256")
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    config = read_json(config_path)
    validate_language_config(config)
    split_hash = require_portable_recorded_hash(split_path, split_hash_path)
    target_hash = require_portable_recorded_hash(target_path, target_hash_path)
    targets = read_csv(target_path)
    validate_targets(targets, expected_count=300)
    assignments = read_csv(split_path)
    folds = build_fold_plan(assignments, expected_folds=5)
    if len(assignments) != 37000:
        raise RuntimeError(f"Expected 37000 fold rows, found {len(assignments)}.")

    systems = system_rows()
    arms = arm_rows()
    metrics = metric_rows()
    leakage = leakage_rows()
    prompt = prompt_contract()
    output_root = root / STAGE37_ROOT
    outputs = {
        "systems": output_root / "stage37_system_matrix.csv",
        "arms": output_root / "stage37_evaluation_arms.csv",
        "folds": output_root / "stage37_fold_language_plan.csv",
        "metrics": output_root / "stage37_metric_matrix.csv",
        "leakage": output_root / "stage37_leakage_firewall.csv",
        "prompt": output_root / "stage37_constrained_prompt_contract.json",
        "report": root / "docs/STAGE_37_LANGUAGE_CORRECTION_PROTOCOL.md",
    }
    write_csv(outputs["systems"], systems)
    write_csv(outputs["arms"], arms)
    write_csv(outputs["folds"], folds)
    write_csv(outputs["metrics"], metrics)
    write_csv(outputs["leakage"], leakage)
    write_json(outputs["prompt"], prompt)
    outputs["report"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report"].write_text(render_report(folds), encoding="utf-8", newline="\n")

    implementation = (
        root / "src/evaluation/stage37_language_protocol.py",
        root / "scripts/build_stage37_language_protocol.py",
        root / "scripts/verify_stage37_language_protocol.py",
        root / "tests/test_stage37_language_protocol.py",
    )
    for path in implementation:
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 37,
        "name": "language_correction_protocol_and_leakage_firewall",
        "status": "passed",
        "training_started": False,
        "language_inference_started": False,
        "recognition_predictions_modified": False,
        "scope": "mixed_subject_duplicate_safe_five_fold",
        "input_contract_portable_sha256": {
            "stage35": stage35_hash,
            "stage36": stage36_hash,
        },
        "frozen_input_hashes": {
            "language_config": portable_text_sha256(config_path),
            "stage32_inner_assignment_recorded": split_hash,
            "sentence_targets_recorded": target_hash,
        },
        "model_roles": {
            "primary_lightweight": "c2",
            "recognition_quality_reference": "g1",
            "comparative_recognizers": ["c1", "c3", "c4"],
            "oracle_gloss": "language_ceiling_only",
        },
        "systems": [row["system_id"] for row in systems],
        "evaluation_arms": [row["arm_id"] for row in arms],
        "primary_metrics": [row["metric"] for row in metrics if row["primary"]],
        "rules": {
            "accuracy_used_alone": False,
            "outer_reference_visible_during_correction": False,
            "retrieval_and_prompt_examples": "current_fold_training_partition_only",
            "predicted_and_oracle_results_separate": True,
            "generator_must_be_frozen_before_execution": True,
            "human_post_editing": False,
        },
        "validation": {
            "sentence_targets": len(targets),
            "assignment_rows": len(assignments),
            "folds": len(folds),
            "systems": len(systems),
            "evaluation_arms": len(arms),
            "metrics": len(metrics),
            "leakage_checks": len(leakage),
        },
        "implementation_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in implementation
        },
        "generated_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in outputs.values()
        },
    }
    contract_path = output_root / "stage37_contract.json"
    contract_hash_path = output_root / "stage37_contract.sha256"
    write_json(contract_path, contract)
    contract_hash_path.write_text(
        portable_text_sha256(contract_path) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_hash_path": contract_hash_path,
        "outputs": outputs,
        "folds": folds,
    }


def main() -> None:
    result = build_stage37(PROJECT_ROOT)
    print("=" * 78)
    print("STAGE 37 — LANGUAGE-CORRECTION PROTOCOL FREEZE")
    print("=" * 78)
    print()
    print("MODEL ROLES")
    print("-" * 78)
    print("Primary lightweight recognizer: C2")
    print("Recognition-quality reference: G1")
    print("Oracle gloss: language-component ceiling only")
    print()
    print("SYSTEMS")
    print("-" * 78)
    print("L0: raw normalized gloss")
    print("L1: deterministic training-only retrieval")
    print("L2: constrained generator")
    print("L3: template-holdout challenge protocol")
    print()
    print("LEAKAGE FIREWALL")
    print("-" * 78)
    for row in result["folds"]:
        print(
            f"Fold {row['outer_fold']}: selection={row['selection_retrieval_bank_samples']} "
            f"validation={row['selection_validation_samples']} "
            f"final_bank={row['final_retrieval_bank_samples']} "
            f"outer_test={row['outer_evaluation_samples']} leaks=0"
        )
    print("Outer references visible during correction: False")
    print("Predicted/oracle results mixed: False")
    print()
    print("EVALUATION")
    print("-" * 78)
    print("Primary metrics: English WER, exact accuracy, content F1,")
    print("unsupported-content rate and correction-harm rate")
    print("Accuracy used alone: False")
    print("Training started: False")
    print("Language inference started: False")
    print()
    print("OUTPUTS")
    print("-" * 78)
    for path in result["outputs"].values():
        print(path)
    print(result["contract_path"])
    print(result["contract_hash_path"])
    print()
    print("FINAL STATUS: STAGE 37 BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

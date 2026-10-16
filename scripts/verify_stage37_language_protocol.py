"""Verify Stage 37 without running correction, training or scoring."""

from pathlib import Path

from src.evaluation.stage37_language_protocol import (
    STAGE37_ROOT,
    build_fold_plan,
    portable_text_sha256,
    read_csv,
    read_json,
    require_portable_recorded_hash,
    validate_language_config,
    validate_targets,
    verify_prior_contract,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / STAGE37_ROOT
    contract_path = output_root / "stage37_contract.json"
    hash_path = output_root / "stage37_contract.sha256"
    require(contract_path.is_file(), "Stage 37 contract is missing.")
    require(hash_path.is_file(), "Stage 37 contract hash is missing.")
    require(
        hash_path.read_text(encoding="utf-8-sig").strip()
        == portable_text_sha256(contract_path),
        "Stage 37 contract hash mismatch.",
    )
    contract = read_json(contract_path)
    require(contract.get("stage") == 37, "Incorrect stage identity.")
    require(contract.get("status") == "passed", "Stage 37 is not passed.")
    require(contract.get("training_started") is False, "Training was started.")
    require(contract.get("language_inference_started") is False, "Language inference was started.")
    require(contract.get("recognition_predictions_modified") is False, "Recognition predictions were modified.")
    require(contract["model_roles"]["primary_lightweight"] == "c2", "Primary lightweight role changed.")
    require(contract["model_roles"]["recognition_quality_reference"] == "g1", "Quality reference changed.")
    require(contract["model_roles"]["comparative_recognizers"] == ["c1", "c3", "c4"], "Comparative model set changed.")
    rules = contract["rules"]
    require(rules["accuracy_used_alone"] is False, "Accuracy-only judgement detected.")
    require(rules["outer_reference_visible_during_correction"] is False, "Outer references leaked.")
    require(rules["predicted_and_oracle_results_separate"] is True, "Oracle/predicted arms mixed.")

    observed_prior = {
        "stage35": verify_prior_contract(PROJECT_ROOT, 35, Path("artifacts/analysis/stage35")),
        "stage36": verify_prior_contract(PROJECT_ROOT, 36, Path("artifacts/analysis/stage36")),
    }
    require(observed_prior == contract["input_contract_portable_sha256"], "Prior-stage contracts changed.")
    config_path = PROJECT_ROOT / "configs/experiments/language.yaml"
    validate_language_config(read_json(config_path))
    require(portable_text_sha256(config_path) == contract["frozen_input_hashes"]["language_config"], "Language config changed.")
    split_path = PROJECT_ROOT / "artifacts/splits/stage32_known_inner_assignment.csv"
    target_path = PROJECT_ROOT / "artifacts/labels/stage10_sentence_targets.csv"
    require_portable_recorded_hash(split_path, split_path.with_suffix(".sha256"))
    require_portable_recorded_hash(target_path, target_path.with_suffix(".sha256"))
    targets = read_csv(target_path)
    validate_targets(targets, expected_count=300)
    assignments = read_csv(split_path)
    folds = build_fold_plan(assignments, expected_folds=5)
    require(len(assignments) == 37000, "Assignment row count changed.")
    require(all(int(row["outer_samples_in_retrieval_bank"]) == 0 for row in folds), "Fold leakage detected.")

    for relative, expected in contract["implementation_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing implementation: {relative}")
        require(portable_text_sha256(path) == expected, f"Implementation hash mismatch: {relative}")
    for relative, expected in contract["generated_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing output: {relative}")
        require(portable_text_sha256(path) == expected, f"Generated output hash mismatch: {relative}")

    systems = read_csv(output_root / "stage37_system_matrix.csv")
    arms = read_csv(output_root / "stage37_evaluation_arms.csv")
    metrics = read_csv(output_root / "stage37_metric_matrix.csv")
    leakage = read_csv(output_root / "stage37_leakage_firewall.csv")
    require(len(systems) == 4, "System matrix changed.")
    require(len(arms) == 6, "Evaluation arms changed.")
    require(len(metrics) == 8, "Metric matrix changed.")
    require(len(leakage) == 7, "Leakage firewall changed.")

    print("=" * 78)
    print("STAGE 37 — LANGUAGE-CORRECTION PROTOCOL VERIFIER")
    print("=" * 78)
    print()
    print("INTEGRITY")
    print("-" * 78)
    print("Stage 35 and 36 contracts unchanged: True")
    print("Stage 37 contract hash match: True")
    print("Implementation and generated hashes match: True")
    print("Sentence targets: 300")
    print("Five-fold assignment rows: 37000")
    print()
    print("LEAKAGE AND CLAIM CONTROL")
    print("-" * 78)
    print("Outer samples in retrieval banks: 0")
    print("Outer references visible during correction: False")
    print("Predicted and oracle arms separate: True")
    print("Accuracy used alone: False")
    print()
    print("EXECUTION SAFETY")
    print("-" * 78)
    print("Training started: False")
    print("Language inference started: False")
    print("Recognition predictions modified: False")
    print()
    print("FINAL STATUS: STAGE 37 PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

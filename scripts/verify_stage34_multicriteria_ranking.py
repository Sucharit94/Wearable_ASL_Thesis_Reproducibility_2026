"""Verify Stage 34E final multi-criteria model selection."""

from pathlib import Path

from src.evaluation.stage34_analysis import read_csv, read_json, sha256_file
from src.evaluation.stage34_ranking import MODEL_ORDER, WEIGHT_SCENARIOS


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / "artifacts" / "analysis" / "stage34"
    contract_path = output_root / "stage34e_contract.json"
    hash_path = output_root / "stage34e_contract.sha256"
    matrix_path = output_root / "stage34e_decision_matrix.csv"
    sensitivity_path = output_root / "stage34e_sensitivity_ranking.csv"
    document_path = PROJECT_ROOT / "docs" / "STAGE_34E_FINAL_MODEL_SELECTION.md"
    for path in (contract_path, hash_path, matrix_path, sensitivity_path, document_path):
        require(path.is_file(), f"Missing Stage 34E output: {path}")

    require(
        hash_path.read_text(encoding="utf-8").strip() == sha256_file(contract_path),
        "Stage 34E contract hash mismatch.",
    )
    contract = read_json(contract_path)
    require(contract["stage"] == 34 and contract["substage"] == "E", "Wrong stage.")
    require(contract["status"] == "passed", "Stage 34E is not passed.")
    require(contract["training_started"] is False, "Stage 34E claims training.")
    require(contract["validation"]["accuracy_not_used_alone"] is True, "Accuracy-only ranking detected.")
    require(contract["validation"]["uncertainty_used"] is True, "Uncertainty evidence missing.")
    require(contract["validation"]["efficiency_used"] is True, "Efficiency evidence missing.")
    require(contract["decision"]["c2_c4_wer_difference_resolved"] is False, "C2-C4 claim changed.")
    required_criteria = {
        "pooled_wer",
        "exact_sentence_accuracy",
        "trainable_parameters",
        "batch1_latency_median_ms",
        "batch16_throughput_samples_per_second",
    }
    require(
        required_criteria.issubset(set(contract["method"]["active_criteria"])),
        "A required quality or efficiency criterion is missing.",
    )

    for relative, expected in contract["implementation_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing implementation: {relative}")
        require(sha256_file(path) == expected, f"Implementation hash mismatch: {relative}")
    for relative, expected in contract["generated_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing generated output: {relative}")
        require(sha256_file(path) == expected, f"Generated hash mismatch: {relative}")

    matrix = read_csv(matrix_path)
    sensitivity = read_csv(sensitivity_path)
    require(len(matrix) == 5, "Expected five decision rows.")
    require(tuple(row["model_id"] for row in matrix) == MODEL_ORDER, "Model order changed.")
    eligible_count = sum(row["quality_eligible"].lower() == "true" for row in matrix)
    require(eligible_count >= 2, "Too few quality-eligible models.")
    require(len(sensitivity) == eligible_count * len(WEIGHT_SCENARIOS), "Sensitivity matrix incomplete.")
    for scenario in WEIGHT_SCENARIOS:
        ranks = sorted(
            int(row["rank"])
            for row in sensitivity
            if row["scenario"] == scenario
        )
        require(ranks == list(range(1, eligible_count + 1)), f"Invalid ranks: {scenario}")

    recommended = contract["decision"]["recommended_lightweight"]
    selected = next(row for row in matrix if row["model_id"] == recommended)
    require(selected["quality_eligible"].lower() == "true", "Recommendation fails quality threshold.")
    require(selected["lightweight_eligible"].lower() == "true", "Recommendation is not lightweight.")

    print("=" * 78)
    print("STAGE 34E — FINAL MODEL-SELECTION VERIFIER")
    print("=" * 78)
    print()
    print("EVIDENCE")
    print("-" * 78)
    print("Recognition metrics used: True")
    print("Confidence intervals and paired inference used: True")
    print("Parameters, size, latency and throughput used: True")
    print(
        "Peak RSS used in ranking: "
        + str(contract["method"]["memory_criterion_used"])
    )
    print("Three weight-sensitivity scenarios: True")
    print("Accuracy used alone: False")
    print()
    print("DECISION")
    print("-" * 78)
    print(f"Accuracy reference: {str(contract['decision']['accuracy_reference']).upper()}")
    print(f"Recommended lightweight model: {str(recommended).upper()}")
    print(f"Pareto frontier: {[value.upper() for value in contract['decision']['pareto_frontier']]}")
    print()
    print("FINAL STATUS: STAGE 34E PASSED")
    print("NO TRAINING WAS STARTED")
    print("=" * 78)


if __name__ == "__main__":
    main()

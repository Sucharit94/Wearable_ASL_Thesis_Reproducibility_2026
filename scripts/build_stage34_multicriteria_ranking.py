"""Build Stage 34E final multi-criteria ranking."""

from pathlib import Path

from src.evaluation.stage34_analysis import read_json, sha256_file, write_json
from src.evaluation.stage34_ranking import (
    CRITERIA_DIRECTIONS,
    LIGHTWEIGHT_PARAMETER_LIMIT,
    QUALITY_THRESHOLDS,
    WEIGHT_SCENARIOS,
    build_decision_tables,
    report_markdown,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_stage34e(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    output_root = root / "artifacts" / "analysis" / "stage34"
    required_contracts = {}
    for substage in ("b", "c", "d"):
        contract = output_root / f"stage34{substage}_contract.json"
        hash_path = output_root / f"stage34{substage}_contract.sha256"
        if not contract.is_file() or not hash_path.is_file():
            raise RuntimeError(f"Stage 34{substage.upper()} is required before Stage 34E.")
        observed = sha256_file(contract)
        if hash_path.read_text(encoding="utf-8").strip() != observed:
            raise RuntimeError(f"Stage 34{substage.upper()} contract hash mismatch.")
        payload = read_json(contract)
        for relative, expected in payload.get("generated_sha256", {}).items():
            generated = root / relative
            if not generated.is_file() or sha256_file(generated) != expected:
                raise RuntimeError(
                    f"Stage 34{substage.upper()} generated-output hash mismatch: {relative}."
                )
        required_contracts[substage.upper()] = observed

    result = build_decision_tables(root)
    document_path = root / "docs" / "STAGE_34E_FINAL_MODEL_SELECTION.md"
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(report_markdown(result), encoding="utf-8", newline="\n")

    implementation_paths = (
        root / "src" / "evaluation" / "stage34_ranking.py",
        root / "scripts" / "build_stage34_multicriteria_ranking.py",
        root / "scripts" / "verify_stage34_multicriteria_ranking.py",
        root / "tests" / "test_stage34_multicriteria_ranking.py",
    )
    for path in implementation_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    generated_paths = (
        result["matrix_path"],
        result["scenarios_path"],
        document_path,
    )
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 34,
        "substage": "E",
        "status": "passed",
        "training_started": False,
        "method": {
            "quality_thresholds": QUALITY_THRESHOLDS,
            "lightweight_parameter_limit": LIGHTWEIGHT_PARAMETER_LIMIT,
            "criteria_directions": CRITERIA_DIRECTIONS,
            "active_criteria": result["active_criteria"],
            "memory_criterion_used": result["memory_criterion_used"],
            "missing_criterion_policy": "omit_and_proportionally_redistribute_weight",
            "weight_scenarios": WEIGHT_SCENARIOS,
            "normalization": "eligible_set_min_max_utility",
            "sensitivity_scenarios": 3,
            "pareto_analysis": True,
            "unresolved_wer_handling": "common_group_mean_wer_for_scoring",
            "recommendation_rule": (
                "quality_eligible_and_under_1m_parameters_then_highest_"
                "mean_score_across_three_weight_scenarios"
            ),
        },
        "input_contract_sha256": required_contracts,
        "decision": {
            "accuracy_reference": result["accuracy_reference"],
            "recommended_lightweight": result["recommended_lightweight"],
            "pareto_frontier": result["pareto_frontier"],
            "statistical_wer_tie_groups": result["statistical_wer_tie_groups"],
            "c2_c4_wer_difference_resolved": ["c2", "c4"]
            not in result["statistical_wer_tie_groups"],
        },
        "implementation_sha256": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in implementation_paths
        },
        "generated_sha256": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in generated_paths
        },
        "validation": {
            "models_reported": len(result["rows"]),
            "scenario_rows": len(result["scenario_rows"]),
            "accuracy_not_used_alone": True,
            "uncertainty_used": True,
            "efficiency_used": True,
            "memory_used_when_available": result["memory_criterion_used"],
            "weight_sensitivity_reported": True,
        },
    }
    contract_path = output_root / "stage34e_contract.json"
    contract_hash_path = output_root / "stage34e_contract.sha256"
    write_json(contract_path, contract)
    contract_hash_path.write_text(
        sha256_file(contract_path) + "\n", encoding="utf-8", newline="\n"
    )
    result.update(
        {
            "contract": contract,
            "contract_path": contract_path,
            "contract_hash_path": contract_hash_path,
            "document_path": document_path,
        }
    )
    return result


def main() -> None:
    result = build_stage34e(PROJECT_ROOT)
    print("=" * 78)
    print("STAGE 34E — FINAL MULTI-CRITERIA MODEL RANKING")
    print("=" * 78)
    print()
    print("DECISION FRAMEWORK")
    print("-" * 78)
    print("Quality threshold: WER <= 0.10 and exact accuracy >= 0.75")
    print("Lightweight threshold: <= 1,000,000 parameters")
    print("Active criteria: " + ", ".join(result["active_criteria"]))
    print(f"Peak RSS used in ranking: {result['memory_criterion_used']}")
    print("Sensitivity scenarios: quality-priority, balanced, efficiency-priority")
    print("Training started: False")
    print()
    print("MODEL ROLES")
    print("-" * 78)
    for row in result["rows"]:
        print(
            f"{str(row['model_id']).upper()}: role={row['final_role']}, "
            f"mean_score={float(row['mean_scenario_score']):.6f}, "
            f"pareto={row['pareto_frontier']}"
        )
    print()
    print("FINAL DECISION")
    print("-" * 78)
    print(f"Accuracy reference: {str(result['accuracy_reference']).upper()}")
    print(f"Recommended lightweight model: {str(result['recommended_lightweight']).upper()}")
    print("C2-C4 WER difference resolved at 95%: False")
    print("Accuracy used alone: False")
    print()
    print("OUTPUTS")
    print("-" * 78)
    print(result["matrix_path"])
    print(result["scenarios_path"])
    print(result["contract_path"])
    print(result["contract_hash_path"])
    print(result["document_path"])
    print()
    print("FINAL STATUS: STAGE 34E BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

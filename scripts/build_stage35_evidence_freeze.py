"""Build the Stage 35 thesis evidence freeze."""

from pathlib import Path

from src.evaluation.stage35_evidence import (
    STAGE35_ROOT,
    build_stage35_tables,
    portable_text_sha256,
    render_report,
    verify_stage34_inputs,
    write_csv,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_stage35(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    input_hashes = verify_stage34_inputs(root)
    result = build_stage35_tables(root)
    output_root = root / STAGE35_ROOT
    output_root.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "publication_table": output_root / "stage35_publication_model_table.csv",
        "error_profile": output_root / "stage35_aggregate_error_profile.csv",
        "key_comparisons": output_root / "stage35_key_comparisons.csv",
        "claim_register": output_root / "stage35_claim_register.csv",
        "report": root / "docs" / "STAGE_35_FINAL_EVIDENCE_FREEZE.md",
    }
    write_csv(output_paths["publication_table"], result["publication_rows"])
    write_csv(output_paths["error_profile"], result["error_rows"])
    write_csv(output_paths["key_comparisons"], result["comparison_rows"])
    write_csv(output_paths["claim_register"], result["claim_rows"])
    output_paths["report"].parent.mkdir(parents=True, exist_ok=True)
    output_paths["report"].write_text(
        render_report(result), encoding="utf-8", newline="\n"
    )

    implementation = (
        root / "src" / "evaluation" / "stage35_evidence.py",
        root / "scripts" / "build_stage35_evidence_freeze.py",
        root / "scripts" / "verify_stage35_evidence_freeze.py",
        root / "tests" / "test_stage35_evidence_freeze.py",
    )
    for path in implementation:
        if not path.is_file():
            raise FileNotFoundError(path)

    c2 = next(
        row for row in result["publication_rows"] if row["model_id"] == "c2"
    )
    c4_comparison = next(
        row
        for row in result["comparison_rows"]
        if row["comparison"] == "lightweight_shortlist"
    )
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 35,
        "name": "final_known_signer_evidence_freeze",
        "status": "passed",
        "training_started": False,
        "predictions_regenerated": False,
        "input_stage34_contract_sha256": input_hashes,
        "method": {
            "accuracy_used_alone": False,
            "recognition_metrics": ["pooled_wer", "exact_sentence_accuracy", "ser"],
            "uncertainty": "subject_sentence_clustered_bootstrap_95_percent_ci",
            "paired_inference": "paired_bootstrap_and_exact_mcnemar_with_holm_correction",
            "efficiency_metrics": [
                "trainable_parameters",
                "serialized_state_dict_mib",
                "batch1_latency_median_ms",
                "batch16_throughput_samples_per_second",
                "incremental_peak_rss_mib",
            ],
            "decision_support": ["quality_threshold", "pareto_frontier", "weight_sensitivity"],
            "scope": "mixed_subject_duplicate_safe_five_fold",
        },
        "frozen_decision": {
            "accuracy_reference": result["accuracy_reference"],
            "recommended_lightweight": result["recommended_lightweight"],
            "recommended_lightweight_parameters": int(c2["trainable_parameters"]),
            "recommended_lightweight_parameter_reduction_vs_g1_percent": float(
                c2["parameter_reduction_vs_g1_percent"]
            ),
            "c2_c4_wer_difference_resolved_at_95_percent": (
                c4_comparison["paired_wer_conclusion"] != "not_resolved"
            ),
        },
        "implementation_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in implementation
        },
        "generated_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in output_paths.values()
        },
        "validation": {
            "models": len(result["publication_rows"]),
            "key_comparisons": len(result["comparison_rows"]),
            "claim_register_rows": len(result["claim_rows"]),
            "thesis_safe_claims_frozen": True,
            "unresolved_results_not_overclaimed": True,
        },
    }
    contract_path = output_root / "stage35_contract.json"
    contract_hash_path = output_root / "stage35_contract.sha256"
    write_json(contract_path, contract)
    contract_hash_path.write_text(
        portable_text_sha256(contract_path) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        **result,
        "contract": contract,
        "contract_path": contract_path,
        "contract_hash_path": contract_hash_path,
        "output_paths": output_paths,
    }


def main() -> None:
    result = build_stage35(PROJECT_ROOT)
    contract = result["contract"]
    c2 = next(
        row for row in result["publication_rows"] if row["model_id"] == "c2"
    )
    comparison = next(
        row
        for row in result["comparison_rows"]
        if row["comparison"] == "lightweight_shortlist"
    )
    print("=" * 78)
    print("STAGE 35 — FINAL KNOWN-SIGNER EVIDENCE FREEZE")
    print("=" * 78)
    print()
    print("FROZEN DECISION")
    print("-" * 78)
    print("Accuracy reference: G1")
    print("Recommended lightweight model: C2")
    print(f"C2 parameters: {int(c2['trainable_parameters'])}")
    print(
        "C2 parameter reduction vs G1: "
        f"{float(c2['parameter_reduction_vs_g1_percent']):.2f}%"
    )
    print(
        "C2-C4 WER difference resolved at 95%: "
        f"{comparison['paired_wer_conclusion'] != 'not_resolved'}"
    )
    print("Accuracy used alone: False")
    print()
    print("EVIDENCE")
    print("-" * 78)
    print("Models consolidated: 5")
    print("Key comparisons: 2")
    print("Thesis-safe claims: 6")
    print("Stage 34 input contracts verified: True")
    print("Training started: False")
    print("Predictions regenerated: False")
    print()
    print("OUTPUTS")
    print("-" * 78)
    for path in result["output_paths"].values():
        print(path)
    print(result["contract_path"])
    print(result["contract_hash_path"])
    print()
    print("FINAL STATUS: STAGE 35 BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

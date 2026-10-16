"""Verify Stage 35 without rebuilding outputs or training models."""

from pathlib import Path

from src.evaluation.stage35_evidence import (
    MODEL_ORDER,
    STAGE35_ROOT,
    portable_text_sha256,
    read_csv,
    read_json,
    verify_stage34_inputs,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / STAGE35_ROOT
    contract_path = output_root / "stage35_contract.json"
    hash_path = output_root / "stage35_contract.sha256"
    require(contract_path.is_file(), "Stage 35 contract is missing.")
    require(hash_path.is_file(), "Stage 35 contract hash is missing.")
    require(
        hash_path.read_text(encoding="utf-8").strip()
        == portable_text_sha256(contract_path),
        "Stage 35 contract hash mismatch.",
    )
    contract = read_json(contract_path)
    require(contract["stage"] == 35, "Incorrect stage identity.")
    require(contract["status"] == "passed", "Stage 35 is not passed.")
    require(contract["training_started"] is False, "Stage 35 claims training.")
    require(
        contract["predictions_regenerated"] is False,
        "Stage 35 regenerated predictions.",
    )
    require(
        contract["method"]["accuracy_used_alone"] is False,
        "Accuracy-only judgement detected.",
    )
    require(
        contract["frozen_decision"]["accuracy_reference"] == "g1",
        "Accuracy reference changed.",
    )
    require(
        contract["frozen_decision"]["recommended_lightweight"] == "c2",
        "Lightweight recommendation changed.",
    )
    require(
        contract["frozen_decision"]["c2_c4_wer_difference_resolved_at_95_percent"]
        is False,
        "C2-C4 uncertainty was overclaimed.",
    )

    observed_inputs = verify_stage34_inputs(PROJECT_ROOT)
    require(
        observed_inputs == contract["input_stage34_contract_sha256"],
        "Frozen Stage 34 inputs changed.",
    )
    for relative, expected in contract["implementation_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing implementation: {relative}")
        require(
            portable_text_sha256(path) == expected,
            f"Implementation hash mismatch: {relative}",
        )
    for relative, expected in contract["generated_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing generated output: {relative}")
        require(
            portable_text_sha256(path) == expected,
            f"Output hash mismatch: {relative}",
        )

    publication = read_csv(output_root / "stage35_publication_model_table.csv")
    errors = read_csv(output_root / "stage35_aggregate_error_profile.csv")
    comparisons = read_csv(output_root / "stage35_key_comparisons.csv")
    claims = read_csv(output_root / "stage35_claim_register.csv")
    require(tuple(row["model_id"] for row in publication) == MODEL_ORDER, "Model table changed.")
    require(tuple(row["model_id"] for row in errors) == MODEL_ORDER, "Error table changed.")
    require(len(comparisons) == 2, "Expected two key comparisons.")
    require(len(claims) == 6, "Expected six thesis claims.")
    require(
        any(row["status"] == "not_resolved" for row in claims),
        "Unresolved evidence is not recorded.",
    )

    print("=" * 78)
    print("STAGE 35 — FINAL EVIDENCE VERIFIER")
    print("=" * 78)
    print()
    print("INTEGRITY")
    print("-" * 78)
    print("Stage 34B-E contracts unchanged: True")
    print("Stage 35 contract hash match: True")
    print("Implementation hashes match: True")
    print("Generated output hashes match: True")
    print()
    print("SCIENTIFIC CLAIM CONTROL")
    print("-" * 78)
    print("Accuracy reference: G1")
    print("Recommended lightweight model: C2")
    print("C2-C4 WER statistically resolved: False")
    print("Accuracy used alone: False")
    print("Thesis-safe claim register complete: True")
    print()
    print("EXECUTION SAFETY")
    print("-" * 78)
    print("Training started: False")
    print("Predictions regenerated: False")
    print()
    print("FINAL STATUS: STAGE 35 PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

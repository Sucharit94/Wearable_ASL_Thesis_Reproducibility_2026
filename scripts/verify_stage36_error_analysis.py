"""Verify Stage 36 outputs without rebuilding figures or training."""

import xml.etree.ElementTree as ET
from pathlib import Path

from scripts.build_stage36_error_analysis import verify_stage35
from src.evaluation.stage35_evidence import (
    MODEL_ORDER,
    portable_text_sha256,
    read_csv,
    read_json,
    verify_stage34_inputs,
)
from src.evaluation.stage36_diagnostics import STAGE36_ROOT


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / STAGE36_ROOT
    contract_path = output_root / "stage36_contract.json"
    hash_path = output_root / "stage36_contract.sha256"
    require(contract_path.is_file(), "Stage 36 contract is missing.")
    require(hash_path.is_file(), "Stage 36 contract hash is missing.")
    require(
        hash_path.read_text(encoding="utf-8").strip()
        == portable_text_sha256(contract_path),
        "Stage 36 contract hash mismatch.",
    )
    contract = read_json(contract_path)
    require(contract["stage"] == 36, "Incorrect stage identity.")
    require(contract["status"] == "passed", "Stage 36 is not passed.")
    require(contract["training_started"] is False, "Stage 36 claims training.")
    require(
        contract["predictions_regenerated"] is False,
        "Stage 36 regenerated predictions.",
    )
    require(
        contract["method"]["accuracy_used_alone"] is False,
        "Accuracy-only analysis detected.",
    )
    require(
        contract["findings"]["accuracy_reference"] == "g1",
        "Accuracy reference changed.",
    )
    require(
        contract["findings"]["recommended_lightweight"] == "c2",
        "Lightweight recommendation changed.",
    )
    require(
        contract["findings"]["c2_c4_statistical_conclusion"]
        == "not_resolved_at_95_percent",
        "C2-C4 uncertainty was overclaimed.",
    )

    observed34 = verify_stage34_inputs(PROJECT_ROOT)
    observed35 = verify_stage35(PROJECT_ROOT)
    require(
        observed34 == contract["input_contract_sha256"]["stage34"],
        "Stage 34 inputs changed.",
    )
    require(
        observed35 == contract["input_contract_sha256"]["stage35"],
        "Stage 35 input changed.",
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

    folds = read_csv(output_root / "stage36_fold_diagnostics.csv")
    stability = read_csv(output_root / "stage36_model_stability.csv")
    errors = read_csv(output_root / "stage36_error_composition.csv")
    require(len(folds) == 25, "Expected 25 fold diagnostics.")
    require(len(stability) == 5, "Expected five stability rows.")
    require(len(errors) == 5, "Expected five error rows.")
    require(
        tuple(row["model_id"] for row in stability) == MODEL_ORDER,
        "Stability model order changed.",
    )
    require(
        set(int(row["wer_rank_in_fold"]) for row in folds) == {1, 2, 3, 4, 5},
        "Fold rankings are incomplete.",
    )
    require(
        int(next(row for row in stability if row["model_id"] == "g1")["fold_wins"])
        == 5,
        "G1 did not win every fold.",
    )
    require(
        next(row for row in errors if row["model_id"] == "c3")[
            "dominant_error_type"
        ]
        == "deletion",
        "C3 deletion failure profile changed.",
    )
    for name in ("stage36_fold_wer.svg", "stage36_quality_efficiency.svg"):
        ET.parse(output_root / name)

    print("=" * 78)
    print("STAGE 36 — ERROR ANALYSIS AND FIGURE VERIFIER")
    print("=" * 78)
    print()
    print("INTEGRITY")
    print("-" * 78)
    print("Stage 34 inputs unchanged: True")
    print("Stage 35 evidence unchanged: True")
    print("Stage 36 contract hash match: True")
    print("Implementation and output hashes match: True")
    print()
    print("DIAGNOSTICS")
    print("-" * 78)
    print("Outer fold rows: 25")
    print("Model stability rows: 5")
    print("Error decomposition rows: 5")
    print("Valid vector figures: 2")
    print("G1 won all five folds: True")
    print("C3 dominant error type: deletion")
    print("C2-C4 overall WER resolved: False")
    print()
    print("SAFETY")
    print("-" * 78)
    print("Training started: False")
    print("Predictions regenerated: False")
    print("Accuracy used alone: False")
    print()
    print("FINAL STATUS: STAGE 36 PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

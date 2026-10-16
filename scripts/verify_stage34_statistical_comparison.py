"""Verify Stage 34C statistical outputs."""

from pathlib import Path

from src.evaluation.stage34_analysis import read_csv, read_json, sha256_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / "artifacts" / "analysis" / "stage34"
    contract_path = output_root / "stage34c_contract.json"
    hash_path = output_root / "stage34c_contract.sha256"
    intervals_path = output_root / "stage34c_model_confidence_intervals.csv"
    pairs_path = output_root / "stage34c_pairwise_tests.csv"
    document_path = PROJECT_ROOT / "docs" / "STAGE_34C_STATISTICAL_COMPARISON.md"

    for path in (contract_path, hash_path, intervals_path, pairs_path, document_path):
        require(path.is_file(), f"Missing Stage 34C output: {path}")
    require(
        hash_path.read_text(encoding="utf-8").strip() == sha256_file(contract_path),
        "Stage 34C contract hash mismatch.",
    )
    contract = read_json(contract_path)
    require(contract["stage"] == 34 and contract["substage"] == "C", "Wrong stage.")
    require(contract["status"] == "passed", "Stage 34C contract is not passed.")
    require(contract["training_started"] is False, "Statistical stage claims training.")
    require(contract["method"]["bootstrap_replicates"] == 10000, "Wrong replicate count.")
    require(contract["method"]["clusters"] == 368, "Wrong cluster count.")
    require(contract["method"]["pairwise_comparisons"] == 10, "Wrong pair count.")
    require(contract["validation"]["paired_samples"] == 7400, "Wrong paired count.")
    require(contract["validation"]["paired_references_match"] is True, "Pairing failed.")
    require(contract["validation"]["holm_applied"] is True, "Holm was not applied.")

    for relative, expected in contract["implementation_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing implementation file: {relative}")
        require(sha256_file(path) == expected, f"Implementation hash mismatch: {relative}")

    for relative, expected in contract["generated_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing hashed output: {relative}")
        require(sha256_file(path) == expected, f"Hash mismatch: {relative}")

    intervals = read_csv(intervals_path)
    pairs = read_csv(pairs_path)
    require(len(intervals) == 5, "Expected five model intervals.")
    require(len(pairs) == 10, "Expected ten pairwise comparisons.")
    require(
        all(float(row["pooled_wer_ci95_lower"]) <= float(row["pooled_wer"]) <= float(row["pooled_wer_ci95_upper"]) for row in intervals),
        "A WER estimate lies outside its interval.",
    )
    require(
        all(0.0 <= float(row["bootstrap_p_value_holm"]) <= 1.0 for row in pairs),
        "Invalid corrected bootstrap p-value.",
    )
    require(
        all(0.0 <= float(row["mcnemar_p_value_holm"]) <= 1.0 for row in pairs),
        "Invalid corrected McNemar p-value.",
    )

    unresolved = [
        f"{row['model_a'].upper()}-{row['model_b'].upper()}"
        for row in pairs
        if row["wer_difference_direction"] == "not_resolved"
    ]
    print("=" * 78)
    print("STAGE 34C — STATISTICAL COMPARISON VERIFIER")
    print("=" * 78)
    print()
    print("INTEGRITY")
    print("-" * 78)
    print("Contract hash match: True")
    print("Implementation hashes match: True")
    print("Generated output hashes match: True")
    print("Paired samples: 7400")
    print("Clusters: 368")
    print("Bootstrap replicates: 10000")
    print("Pairwise comparisons: 10")
    print("Holm correction applied: True")
    print()
    print("WER DIFFERENCES NOT RESOLVED AT 95%")
    print("-" * 78)
    print(unresolved if unresolved else "None")
    print()
    print("FINAL STATUS: STAGE 34C PASSED")
    print("NO TRAINING WAS STARTED")
    print("=" * 78)


if __name__ == "__main__":
    main()

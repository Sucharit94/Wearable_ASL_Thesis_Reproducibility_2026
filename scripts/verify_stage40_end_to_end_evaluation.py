from __future__ import annotations

import json
from pathlib import Path

from src.evaluation.stage40_language_evaluation import (
    MODEL_ORDER,
    SYSTEM_ORDER,
    read_csv_rows,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "artifacts/language/stage40"
CONTRACT_PATH = OUTPUT_ROOT / "stage40_contract.json"
CONTRACT_HASH_PATH = OUTPUT_ROOT / "stage40_contract.sha256"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    print("=" * 78)
    print("STAGE 40 — END-TO-END LANGUAGE EVALUATION VERIFIER")
    print("=" * 78)
    require(CONTRACT_PATH.is_file(), "Stage 40 contract is missing")
    require(CONTRACT_HASH_PATH.is_file(), "Stage 40 contract hash is missing")
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8-sig"))
    recorded_hash = CONTRACT_HASH_PATH.read_text(encoding="utf-8-sig").strip().split()[0]
    require(sha256_file(CONTRACT_PATH) == recorded_hash, "Stage 40 contract hash mismatch")

    for relative, expected in contract["outputs"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Stage 40 output is missing: {relative}")
        require(sha256_file(path) == expected, f"Stage 40 output hash mismatch: {relative}")

    for name in ("stage38_l0_l1", "stage39a_sample_request_map", "stage13_sample_metadata"):
        entry = contract["inputs"][name]
        path = PROJECT_ROOT / entry["path"]
        require(path.is_file(), f"Frozen input is missing: {entry['path']}")
        require(sha256_file(path) == entry["sha256"], f"Frozen input changed: {entry['path']}")

    summary = read_csv_rows(OUTPUT_ROOT / "stage40_system_summary.csv")
    system_tests = read_csv_rows(OUTPUT_ROOT / "stage40_paired_system_tests.csv")
    l2_tests = read_csv_rows(OUTPUT_ROOT / "stage40_l2_model_tests.csv")
    ranking = read_csv_rows(OUTPUT_ROOT / "stage40_l2_model_ranking.csv")
    l2_outputs = read_csv_rows(OUTPUT_ROOT / "stage40_l2_sample_outputs.csv")

    expected_pairs = {(model, system) for model in MODEL_ORDER for system in SYSTEM_ORDER}
    observed_pairs = {(row["model_id"], row["system_id"]) for row in summary}
    require(len(summary) == 15 and observed_pairs == expected_pairs, "Summary coverage is incomplete")
    require(len(system_tests) == 15, "Expected 15 within-model system comparisons")
    require(len(l2_tests) == 10, "Expected 10 all-model L2 comparisons")
    require(len(ranking) == 5, "Expected five models in the L2 ranking")
    require(
        {row["model_id"] for row in ranking} == set(MODEL_ORDER),
        "L2 ranking does not cover all models",
    )

    if contract["coverage"]["paired_model_samples"] == 37_000:
        require(len(l2_outputs) == 37_000, "Expected 37,000 L2 sample outputs")
        require(
            all(int(row["samples"]) == 7_400 for row in summary),
            "Every production model/system summary must contain 7,400 samples",
        )

    for row in summary:
        for metric in (
            "pooled_wer", "exact_sentence_accuracy", "content_precision", "content_recall",
            "content_f1", "unsupported_content_rate",
        ):
            value = float(row[metric])
            require(value >= 0.0, f"Negative metric: {row['model_id']} {row['system_id']} {metric}")
        if row["system_id"] == "L0":
            require(row["correction_harm_rate"] == "", "L0 harm rate must be unavailable")
        else:
            require(0.0 <= float(row["correction_harm_rate"]) <= 1.0, "Invalid harm rate")

    ranking_wers = [float(row["pooled_wer"]) for row in ranking]
    require(ranking_wers == sorted(ranking_wers), "L2 ranking is not sorted by pooled WER")
    require(contract["ranking"]["winner"] == ranking[0]["model_id"], "Winner mismatch")
    require(contract["metrics"]["accuracy_used_alone"] is False, "Accuracy-only claim is forbidden")
    execution = contract["execution"]
    require(execution["recognizer_training_started"] is False, "Training was unexpectedly started")
    require(execution["language_generation_started"] is False, "Generation was unexpectedly started")
    require(execution["v1_results_scored"] is False, "Failed V1 output must not be scored")

    print("\nINTEGRITY")
    print("-" * 78)
    print("Contract hash match: True")
    print("Frozen Stage 38/39A inputs unchanged: True")
    print("Generated output hashes match: True")
    print(f"L2 sample outputs: {len(l2_outputs)}")

    print("\nEVALUATION COVERAGE")
    print("-" * 78)
    print("Models: G1, C1, C2, C3 and C4")
    print("Systems: L0, L1 and L2")
    print(f"Summary rows: {len(summary)}")
    print(f"Within-model paired tests: {len(system_tests)}")
    print(f"Between-model L2 paired tests: {len(l2_tests)}")
    print("Metrics: WER, exact accuracy, content P/R/F1, unsupported content and harm")

    print("\nFINAL L2 RANKING")
    print("-" * 78)
    for row in ranking:
        print(
            f"{row['l2_quality_rank']}. {row['model_id'].upper()}: "
            f"WER={float(row['pooled_wer']):.6f}, "
            f"exact={float(row['exact_sentence_accuracy']):.6f}, "
            f"content_F1={float(row['content_f1']):.4f}"
        )

    print("\nSAFETY")
    print("-" * 78)
    print("Recognizer training started: False")
    print("Language generation started: False")
    print("Failed V1 output scored: False")
    print("Accuracy used alone: False")
    print("\nFINAL STATUS: STAGE 40 PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

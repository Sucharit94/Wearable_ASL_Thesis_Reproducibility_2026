"""Build Stage 34C paired statistical comparison."""

from pathlib import Path

from src.evaluation.stage34_statistics import build_stage34c


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    result = build_stage34c(PROJECT_ROOT)
    print("=" * 78)
    print("STAGE 34C — PAIRED STATISTICAL MODEL COMPARISON")
    print("=" * 78)
    print()
    print("METHOD")
    print("-" * 78)
    print("Paired outer-fold samples: 7400")
    print("Subject-sentence clusters: 368")
    print("Bootstrap replicates: 10000")
    print("Confidence level: 95%")
    print("Pairwise model comparisons: 10")
    print("Sentence-correctness test: exact two-sided McNemar")
    print("Multiple-comparison correction: Holm")
    print("Training started: False")
    print()
    print("MODEL CONFIDENCE INTERVALS")
    print("-" * 78)
    for row in result["tables"]["intervals"]:
        print(
            f"{str(row['model_id']).upper()}: "
            f"WER={float(row['pooled_wer']):.6f} "
            f"[{float(row['pooled_wer_ci95_lower']):.6f}, "
            f"{float(row['pooled_wer_ci95_upper']):.6f}], "
            f"accuracy={float(row['exact_sentence_accuracy']):.6f} "
            f"[{float(row['exact_accuracy_ci95_lower']):.6f}, "
            f"{float(row['exact_accuracy_ci95_upper']):.6f}]"
        )
    print()
    print("PAIRWISE WER CONCLUSIONS")
    print("-" * 78)
    for row in result["tables"]["pairs"]:
        print(
            f"{str(row['model_a']).upper()} vs {str(row['model_b']).upper()}: "
            f"delta={float(row['delta_wer_a_minus_b']):.6f}, "
            f"95% CI=[{float(row['delta_wer_ci95_lower']):.6f}, "
            f"{float(row['delta_wer_ci95_upper']):.6f}], "
            f"Holm p={float(row['bootstrap_p_value_holm']):.6g}, "
            f"{row['wer_difference_direction']}"
        )
    print()
    print("OUTPUTS")
    print("-" * 78)
    print(result["contract_path"])
    print(result["contract_hash_path"])
    print(result["document_path"])
    for relative in result["contract"]["generated_sha256"]:
        print(PROJECT_ROOT / relative)
    print()
    print("FINAL STATUS: STAGE 34C BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

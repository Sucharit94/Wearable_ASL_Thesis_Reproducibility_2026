from __future__ import annotations

import argparse
from pathlib import Path

from src.evaluation.stage41_thesis_evidence import (
    MODEL_ORDER, SYSTEM_ORDER, build_claim_register, build_final_selection,
    build_recognition_table, comparison_winner, find_model_test, markdown_report, normalize_language_summary,
    read_csv_rows, relative_name, sha256_file, write_csv_rows, write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build the final thesis evidence package")
    p.add_argument("--stage35-publication", type=Path, default=PROJECT_ROOT / "artifacts/analysis/stage35/stage35_publication_model_table.csv")
    p.add_argument("--stage35-claims", type=Path, default=PROJECT_ROOT / "artifacts/analysis/stage35/stage35_claim_register.csv")
    p.add_argument("--efficiency", type=Path, default=PROJECT_ROOT / "artifacts/analysis/stage34/stage34d_efficiency_metrics.csv")
    p.add_argument("--recognition-tests", type=Path, default=PROJECT_ROOT / "artifacts/analysis/stage34/stage34c_pairwise_tests.csv")
    p.add_argument("--stability", type=Path, default=PROJECT_ROOT / "artifacts/analysis/stage36/stage36_model_stability.csv")
    p.add_argument("--errors", type=Path, default=PROJECT_ROOT / "artifacts/analysis/stage36/stage36_error_composition.csv")
    p.add_argument("--language-summary", type=Path, default=PROJECT_ROOT / "artifacts/language/stage40/stage40_system_summary.csv")
    p.add_argument("--system-tests", type=Path, default=PROJECT_ROOT / "artifacts/language/stage40/stage40_paired_system_tests.csv")
    p.add_argument("--l2-tests", type=Path, default=PROJECT_ROOT / "artifacts/language/stage40/stage40_l2_model_tests.csv")
    p.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "artifacts/analysis/stage41")
    p.add_argument("--report", type=Path, default=PROJECT_ROOT / "docs/STAGE_41_FINAL_THESIS_EVIDENCE.md")
    return p


def main() -> None:
    args = parser().parse_args()
    inputs = [args.stage35_publication, args.stage35_claims, args.efficiency, args.recognition_tests, args.stability, args.errors, args.language_summary, args.system_tests, args.l2_tests]
    missing = [str(path) for path in inputs if not path.is_file()]
    if missing:
        raise RuntimeError("Required frozen evidence is missing:\n" + "\n".join(missing))

    print("=" * 78)
    print("STAGE 41 — FINAL THESIS EVIDENCE PACKAGE")
    print("=" * 78)
    publication = read_csv_rows(args.stage35_publication)
    frozen_claims = read_csv_rows(args.stage35_claims)
    recognition = build_recognition_table(publication, read_csv_rows(args.efficiency), read_csv_rows(args.stability), read_csv_rows(args.errors))
    language = normalize_language_summary(read_csv_rows(args.language_summary))
    system_tests = read_csv_rows(args.system_tests)
    l2_tests = read_csv_rows(args.l2_tests)
    recognition_test = find_model_test(read_csv_rows(args.recognition_tests), "c2", "c4")
    if comparison_winner(recognition_test, "model_a", "model_b") != "not_resolved":
        raise RuntimeError("Frozen Stage 34C evidence no longer reports C2-C4 recognition WER as unresolved")
    selections = build_final_selection(language, system_tests, l2_tests)
    claims = build_claim_register(language, selections)

    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    recognition_path = output / "stage41_recognition_model_table.csv"
    language_path = output / "stage41_language_system_table.csv"
    selection_path = output / "stage41_final_selection.csv"
    claim_path = output / "stage41_claim_register.csv"
    contract_path = output / "stage41_contract.json"
    hash_path = output / "stage41_contract.sha256"

    write_csv_rows(recognition_path, recognition, list(recognition[0]))
    write_csv_rows(language_path, language, list(language[0]))
    write_csv_rows(selection_path, selections, list(selections[0]))
    write_csv_rows(claim_path, claims, list(claims[0]))
    args.report.write_text(markdown_report(recognition, language, selections, claims), encoding="utf-8")

    generated = [recognition_path, language_path, selection_path, claim_path, args.report]
    contract = {
        "stage": "41",
        "title": "Final thesis evidence package",
        "protocol": "known_signer_mixed_subject_stratified_five_fold",
        "inputs": {relative_name(path, PROJECT_ROOT): sha256_file(path) for path in inputs},
        "input_rows": {"stage35_publication": len(publication), "stage35_claims": len(frozen_claims), "stage40_summary": 15, "stage40_system_tests": len(system_tests), "stage40_l2_tests": len(l2_tests)},
        "coverage": {"models": list(MODEL_ORDER), "language_systems": list(SYSTEM_ORDER), "recognition_rows": len(recognition), "language_rows": len(language), "selection_rows": len(selections), "claim_rows": len(claims)},
        "decisions": {"recognition_quality_reference": "g1", "recommended_lightweight_recognizer": "c2", "recommended_lightweight_end_to_end_system": "c2+l1", "best_constrained_generation_quality": "g1+l2"},
        "claim_boundaries": {"accuracy_used_alone": False, "open_vocabulary_claim_allowed": False, "unseen_template_claim_allowed": False, "l3_executed": False},
        "execution": {"recognizer_training_started": False, "predictions_regenerated": False, "language_generation_started": False, "statistical_tests_regenerated": False},
        "outputs": {relative_name(path, PROJECT_ROOT): sha256_file(path) for path in generated},
        "status": "COMPLETE",
    }
    write_json(contract_path, contract)
    hash_path.write_text(sha256_file(contract_path) + "\n", encoding="utf-8")

    print("\nFINAL DECISIONS")
    print("-" * 78)
    print("Recognition-quality reference: G1")
    print("Recommended lightweight recognizer: C2")
    print("Recommended lightweight end-to-end system: C2 + L1")
    print("Best constrained-generation system: G1 + L2")
    c2 = next(row for row in language if row["model_id"] == "c2" and row["system_id"] == "L1")
    print(f"C2 + L1: WER={float(c2['pooled_wer']):.6f}, exact={float(c2['exact_sentence_accuracy']):.6f}, content F1={float(c2['content_f1']):.4f}")
    print("\nCLAIM CONTROL")
    print("-" * 78)
    print("Accuracy used alone: False")
    print("Open-vocabulary claim allowed: False")
    print("Unseen-template claim allowed: False")
    print("L3 executed: False")
    print("Training or generation started: False")
    print("\nOUTPUTS")
    print("-" * 78)
    for path in generated + [contract_path, hash_path]:
        print(path)
    print("\nFINAL STATUS: STAGE 41 BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

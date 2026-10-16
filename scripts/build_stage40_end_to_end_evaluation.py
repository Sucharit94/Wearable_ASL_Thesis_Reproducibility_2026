from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from src.evaluation.stage40_language_evaluation import (
    CONFIDENCE_LEVEL,
    CONTENT_STOPWORDS,
    DEFAULT_BOOTSTRAP_REPLICATES,
    DEFAULT_SEED,
    MODEL_ORDER,
    SYSTEM_ORDER,
    assemble_and_score,
    build_l2_ranking,
    build_markdown_report,
    build_summaries_and_tests,
    compact_l2_rows,
    load_request_map,
    load_sample_metadata,
    load_stage38_outputs,
    load_v2_result_archive,
    relative_name,
    sha256_file,
    write_csv_rows,
    write_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build Stage 40 end-to-end language evaluation")
    value.add_argument(
        "--stage38",
        type=Path,
        default=PROJECT_ROOT / "artifacts/language/stage38/stage38_l0_l1_outputs.csv",
    )
    value.add_argument(
        "--request-map",
        type=Path,
        default=PROJECT_ROOT / "artifacts/language/stage39/stage39a_sample_request_map.csv",
    )
    value.add_argument(
        "--metadata",
        type=Path,
        default=PROJECT_ROOT / "artifacts/splits/stage13_known_signer_5fold.csv",
    )
    value.add_argument("--v2-results", type=Path, default=None)
    value.add_argument(
        "--output-root",
        type=Path,
        default=PROJECT_ROOT / "artifacts/language/stage40",
    )
    value.add_argument(
        "--report",
        type=Path,
        default=PROJECT_ROOT / "docs/STAGE_40_END_TO_END_LANGUAGE_EVALUATION.md",
    )
    value.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAP_REPLICATES)
    value.add_argument("--seed", type=int, default=DEFAULT_SEED)
    value.add_argument("--allow-small-fixture", action="store_true")
    return value


def find_v2_archive(explicit: Path | None) -> Path:
    if explicit is not None:
        if not explicit.is_file():
            raise RuntimeError(f"Stage 39B V2 result ZIP was not found: {explicit}")
        return explicit
    candidates = (
        PROJECT_ROOT / "imports/stage39b_v2_results.zip",
        PROJECT_ROOT / "stage39b_v2_results.zip",
        PROJECT_ROOT / "imports/stage39b_v2_results/stage39b_v2_results.zip",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Stage 39B V2 result ZIP was not found. Copy stage39b_v2_results.zip "
        "to the project's imports folder."
    )


def require_inputs(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise RuntimeError("Required Stage 40 inputs are missing:\n" + "\n".join(missing))


def production_checks(
    stage38_rows: list[dict[str, object]],
    request_rows: list[dict[str, object]],
    scored_rows: list[dict[str, object]],
    response_count: int,
) -> None:
    if len(stage38_rows) != 74_000:
        raise RuntimeError(f"Expected 74,000 Stage 38 L0/L1 rows; found {len(stage38_rows)}")
    if len(request_rows) != 37_000:
        raise RuntimeError(f"Expected 37,000 Stage 39A sample mappings; found {len(request_rows)}")
    if len(scored_rows) != 111_000:
        raise RuntimeError(f"Expected 111,000 paired L0/L1/L2 rows; found {len(scored_rows)}")
    if response_count != 11_720:
        raise RuntimeError(f"Expected 11,720 unique V2 responses; found {response_count}")
    counts = Counter((str(row["model_id"]), str(row["system_id"])) for row in scored_rows)
    expected = {(model, system): 7_400 for model in MODEL_ORDER for system in SYSTEM_ORDER}
    if counts != expected:
        raise RuntimeError(f"All-model/system coverage mismatch: {dict(counts)}")
    clusters = {str(row["cluster_id"]) for row in scored_rows}
    if len(clusters) != 368:
        raise RuntimeError(f"Expected 368 subject-sentence clusters; found {len(clusters)}")


def main() -> None:
    args = parser().parse_args()
    if args.bootstrap_replicates < 100 and not args.allow_small_fixture:
        raise RuntimeError("Production Stage 40 requires at least 100 bootstrap replicates")

    v2_archive = find_v2_archive(args.v2_results)
    require_inputs([args.stage38, args.request_map, args.metadata, v2_archive])

    print("=" * 78)
    print("STAGE 40 — ALL-MODEL END-TO-END LANGUAGE EVALUATION")
    print("=" * 78)
    print("\nVERIFYING AND JOINING FROZEN INPUTS")
    print("-" * 78)

    stage38_rows = load_stage38_outputs(args.stage38)
    request_rows = load_request_map(args.request_map)
    metadata = load_sample_metadata(args.metadata)
    responses, response_evidence = load_v2_result_archive(v2_archive)
    scored_rows = assemble_and_score(stage38_rows, request_rows, responses, metadata)

    if not args.allow_small_fixture:
        production_checks(
            stage38_rows,
            request_rows,
            scored_rows,
            int(response_evidence["response_count"]),
        )

    print(f"Stage 38 L0/L1 rows: {len(stage38_rows)}")
    print(f"Mapped model/sample rows: {len(request_rows)}")
    print(f"Unique V2 responses: {response_evidence['response_count']}")
    print(f"Paired L0/L1/L2 rows: {len(scored_rows)}")
    print(f"Subject-sentence clusters: {len({row['cluster_id'] for row in scored_rows})}")

    print("\nRUNNING PAIRED STATISTICAL EVALUATION")
    print("-" * 78)
    print(f"Bootstrap replicates: {args.bootstrap_replicates}")
    print(f"Confidence level: {CONFIDENCE_LEVEL:.0%}")
    print("Exact-correctness test: exact two-sided McNemar")
    print("Multiple-comparison correction: Holm")

    summaries, system_tests, l2_tests = build_summaries_and_tests(
        scored_rows,
        bootstrap_replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    ranking = build_l2_ranking(summaries)
    l2_rows = compact_l2_rows(scored_rows)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)

    l2_output_path = output_root / "stage40_l2_sample_outputs.csv"
    summary_path = output_root / "stage40_system_summary.csv"
    system_test_path = output_root / "stage40_paired_system_tests.csv"
    l2_test_path = output_root / "stage40_l2_model_tests.csv"
    ranking_path = output_root / "stage40_l2_model_ranking.csv"
    contract_path = output_root / "stage40_contract.json"
    contract_hash_path = output_root / "stage40_contract.sha256"

    write_csv_rows(l2_output_path, l2_rows)
    write_csv_rows(summary_path, summaries, _summary_fields())
    write_csv_rows(system_test_path, system_tests, _system_test_fields())
    write_csv_rows(l2_test_path, l2_tests, _l2_test_fields())
    write_csv_rows(ranking_path, ranking)
    args.report.write_text(
        build_markdown_report(
            summaries,
            ranking,
            system_tests,
            l2_tests,
            bootstrap_replicates=args.bootstrap_replicates,
        ),
        encoding="utf-8",
    )

    output_paths = [
        l2_output_path,
        summary_path,
        system_test_path,
        l2_test_path,
        ranking_path,
        args.report,
    ]
    counts_by_model_system = Counter(
        f"{row['model_id']}-{row['system_id']}" for row in scored_rows
    )
    contract = {
        "stage": "40",
        "title": "All-model end-to-end language evaluation",
        "protocol": "known_signer_stratified_five_fold",
        "inputs": {
            "stage38_l0_l1": {
                "path": relative_name(args.stage38, PROJECT_ROOT),
                "sha256": sha256_file(args.stage38),
                "rows": len(stage38_rows),
            },
            "stage39a_sample_request_map": {
                "path": relative_name(args.request_map, PROJECT_ROOT),
                "sha256": sha256_file(args.request_map),
                "rows": len(request_rows),
            },
            "stage13_sample_metadata": {
                "path": relative_name(args.metadata, PROJECT_ROOT),
                "sha256": sha256_file(args.metadata),
                "rows": len(metadata),
            },
            "stage39b_v2_results": {
                "archive_name": v2_archive.name,
                "archive_sha256": response_evidence["archive_sha256"],
                "response_sha256": response_evidence["response_sha256"],
                "unique_responses": response_evidence["response_count"],
                "outer_references_exposed": False,
                "training_started": False,
            },
        },
        "coverage": {
            "models": list(MODEL_ORDER),
            "systems": list(SYSTEM_ORDER),
            "paired_model_samples": len(request_rows),
            "scored_system_rows": len(scored_rows),
            "subject_sentence_clusters": len({str(row["cluster_id"]) for row in scored_rows}),
            "counts_by_model_system": dict(sorted(counts_by_model_system.items())),
        },
        "metrics": {
            "primary": "pooled_english_wer",
            "reported": [
                "pooled_wer", "macro_sample_wer", "exact_sentence_accuracy",
                "content_precision", "content_recall", "content_f1",
                "unsupported_content_rate", "correction_harm_rate",
            ],
            "content_stopwords": sorted(CONTENT_STOPWORDS),
            "accuracy_used_alone": False,
        },
        "statistics": {
            "bootstrap_unit": "subject_sentence_cluster",
            "bootstrap_replicates": args.bootstrap_replicates,
            "confidence_level": CONFIDENCE_LEVEL,
            "seed": args.seed,
            "sentence_correctness_test": "exact_two_sided_mcnemar",
            "multiple_comparison_correction": "holm",
            "holm_families": {
                "within_model_system_comparisons": len(system_tests),
                "l2_between_model_comparisons": len(l2_tests),
            },
        },
        "ranking": {
            "scope": "L2_end_to_end_language_quality_only",
            "primary_order": "lower_pooled_wer",
            "tie_breakers": ["higher_content_f1", "higher_exact_sentence_accuracy"],
            "winner": ranking[0]["model_id"],
            "recognition_quality_reference": "g1",
            "primary_lightweight_recognizer": "c2",
        },
        "execution": {
            "recognizer_training_started": False,
            "language_generation_started": False,
            "predictions_regenerated": False,
            "v1_results_scored": False,
        },
        "outputs": {
            relative_name(path, PROJECT_ROOT): sha256_file(path) for path in output_paths
        },
        "status": "COMPLETE",
    }
    write_json(contract_path, contract)
    contract_hash_path.write_text(sha256_file(contract_path) + "\n", encoding="utf-8")

    print("\nALL-MODEL L0/L1/L2 SUMMARY")
    print("-" * 78)
    for row in summaries:
        harm = "n.a." if row["correction_harm_rate"] is None else f"{float(row['correction_harm_rate']):.4f}"
        print(
            f"{str(row['model_id']).upper():<3} {row['system_id']}: "
            f"WER={float(row['pooled_wer']):.6f}, "
            f"exact={float(row['exact_sentence_accuracy']):.6f}, "
            f"content P/R/F1={float(row['content_precision']):.4f}/"
            f"{float(row['content_recall']):.4f}/{float(row['content_f1']):.4f}, "
            f"unsupported={float(row['unsupported_content_rate']):.4f}, harm={harm}"
        )

    print("\nL2 END-TO-END LANGUAGE-QUALITY RANKING")
    print("-" * 78)
    for row in ranking:
        print(
            f"{row['l2_quality_rank']}. {str(row['model_id']).upper()}: "
            f"WER={float(row['pooled_wer']):.6f}, "
            f"exact={float(row['exact_sentence_accuracy']):.6f}, "
            f"content_F1={float(row['content_f1']):.4f}"
        )

    print("\nOUTPUTS")
    print("-" * 78)
    for path in output_paths + [contract_path, contract_hash_path]:
        print(path)
    print("\nFINAL STATUS: STAGE 40 BUILD PASSED")
    print("No recognizer training or language generation was started.")
    print("=" * 78)


def _summary_fields() -> list[str]:
    base = [
        "model_id", "system_id", "samples", "reference_words", "hypothesis_words",
        "matches", "substitutions", "deletions", "insertions", "total_edits",
        "pooled_wer", "pooled_wer_ci_lower", "pooled_wer_ci_upper", "macro_sample_wer",
        "exact_correct", "exact_sentence_accuracy", "exact_sentence_accuracy_ci_lower",
        "exact_sentence_accuracy_ci_upper", "content_true_positive", "content_false_positive",
        "content_false_negative", "content_precision", "content_recall", "content_f1",
        "content_f1_ci_lower", "content_f1_ci_upper", "unsupported_content_tokens",
        "predicted_content_tokens", "unsupported_content_rate",
        "unsupported_content_rate_ci_lower", "unsupported_content_rate_ci_upper",
        "correction_harm_rate", "correction_harm_rate_ci_lower", "correction_harm_rate_ci_upper",
        "correction_improvement_rate", "correction_improvement_rate_ci_lower",
        "correction_improvement_rate_ci_upper", "correction_unchanged_rate",
    ]
    return base


def _comparison_fields(prefix: list[str]) -> list[str]:
    return prefix + [
        "paired_samples", "clusters", "wer_delta_a_minus_b", "wer_delta_ci_lower",
        "wer_delta_ci_upper", "wer_bootstrap_p_raw", "wer_bootstrap_p_holm",
        "mcnemar_a_only_correct", "mcnemar_b_only_correct", "mcnemar_p_raw",
        "mcnemar_p_holm", "wer_conclusion",
    ]


def _system_test_fields() -> list[str]:
    return _comparison_fields(["model_id", "system_a", "system_b"])


def _l2_test_fields() -> list[str]:
    return _comparison_fields(["model_a", "model_b"])


if __name__ == "__main__":
    main()

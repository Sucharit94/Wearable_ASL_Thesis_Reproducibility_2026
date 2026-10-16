"""Build Stage 36 diagnostics and thesis-ready SVG figures."""

from pathlib import Path

from src.evaluation.stage35_evidence import (
    portable_text_sha256,
    read_csv,
    read_json,
    verify_stage34_inputs,
    write_csv,
    write_json,
)
from src.evaluation.stage36_diagnostics import (
    STAGE34_ROOT,
    STAGE35_ROOT,
    STAGE36_ROOT,
    build_error_rows,
    build_stability_rows,
    c2_c4_fold_comparison,
    fold_wer_svg,
    normalize_fold_rows,
    quality_efficiency_svg,
    render_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_stage35(project_root: Path) -> str:
    contract_path = project_root / STAGE35_ROOT / "stage35_contract.json"
    hash_path = project_root / STAGE35_ROOT / "stage35_contract.sha256"
    if not contract_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError("Stage 35 must pass before Stage 36.")
    actual = portable_text_sha256(contract_path)
    if hash_path.read_text(encoding="utf-8").strip() != actual:
        raise RuntimeError("Stage 35 contract hash mismatch.")
    contract = read_json(contract_path)
    if contract.get("status") != "passed":
        raise RuntimeError("Stage 35 is not frozen as passed.")
    for relative, expected in contract.get("generated_sha256", {}).items():
        path = project_root / relative
        if not path.is_file() or portable_text_sha256(path) != expected:
            raise RuntimeError(f"Stage 35 output hash mismatch: {relative}")
    return actual


def build_stage36(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    stage34_hashes = verify_stage34_inputs(root)
    stage35_hash = verify_stage35(root)
    fold_source = root / STAGE34_ROOT / "stage34b_fold_metrics.csv"
    publication_source = (
        root / STAGE35_ROOT / "stage35_publication_model_table.csv"
    )
    error_fallback_source = (
        root / STAGE35_ROOT / "stage35_aggregate_error_profile.csv"
    )
    if not fold_source.is_file():
        raise FileNotFoundError(fold_source)
    if not publication_source.is_file():
        raise FileNotFoundError(publication_source)
    if not error_fallback_source.is_file():
        raise FileNotFoundError(error_fallback_source)

    fold_rows = normalize_fold_rows(read_csv(fold_source))
    stability_rows = build_stability_rows(fold_rows)
    error_rows = build_error_rows(
        fold_rows,
        read_csv(error_fallback_source),
    )
    c2_c4 = c2_c4_fold_comparison(fold_rows)
    publication_rows = read_csv(publication_source)

    output_root = root / STAGE36_ROOT
    output_root.mkdir(parents=True, exist_ok=True)
    outputs = {
        "fold_diagnostics": output_root / "stage36_fold_diagnostics.csv",
        "model_stability": output_root / "stage36_model_stability.csv",
        "error_composition": output_root / "stage36_error_composition.csv",
        "fold_figure": output_root / "stage36_fold_wer.svg",
        "quality_efficiency_figure": output_root
        / "stage36_quality_efficiency.svg",
        "report": root / "docs" / "STAGE_36_ERROR_ANALYSIS_AND_FIGURES.md",
    }
    write_csv(outputs["fold_diagnostics"], fold_rows)
    write_csv(outputs["model_stability"], stability_rows)
    write_csv(outputs["error_composition"], error_rows)
    outputs["fold_figure"].write_text(
        fold_wer_svg(fold_rows), encoding="utf-8", newline="\n"
    )
    outputs["quality_efficiency_figure"].write_text(
        quality_efficiency_svg(publication_rows),
        encoding="utf-8",
        newline="\n",
    )
    outputs["report"].parent.mkdir(parents=True, exist_ok=True)
    outputs["report"].write_text(
        render_report(stability_rows, error_rows, c2_c4),
        encoding="utf-8",
        newline="\n",
    )

    implementation = (
        root / "src" / "evaluation" / "stage36_diagnostics.py",
        root / "scripts" / "build_stage36_error_analysis.py",
        root / "scripts" / "verify_stage36_error_analysis.py",
        root / "tests" / "test_stage36_error_analysis.py",
    )
    for path in implementation:
        if not path.is_file():
            raise FileNotFoundError(path)

    stability = {row["model_id"]: row for row in stability_rows}
    errors = {row["model_id"]: row for row in error_rows}
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 36,
        "name": "fold_error_analysis_and_thesis_figures",
        "status": "passed",
        "training_started": False,
        "predictions_regenerated": False,
        "scope": "mixed_subject_duplicate_safe_five_fold",
        "input_contract_sha256": {
            "stage34": stage34_hashes,
            "stage35": stage35_hash,
        },
        "method": {
            "outer_evaluations": 25,
            "models": 5,
            "folds": 5,
            "fold_variation": [
                "sample_standard_deviation",
                "range",
                "coefficient_of_variation",
                "best_and_worst_fold",
                "within_fold_wer_rank",
            ],
            "error_decomposition": [
                "substitutions",
                "deletions",
                "insertions",
            ],
            "figures": "dependency_free_vector_svg",
            "accuracy_used_alone": False,
            "statistical_inference_reused_not_recomputed": True,
        },
        "findings": {
            "g1_fold_wins": int(stability["g1"]["fold_wins"]),
            "c2_lower_wer_folds_than_c4": c2_c4["c2_lower_wer_folds"],
            "c4_lower_wer_folds_than_c2": c2_c4["c4_lower_wer_folds"],
            "c2_c4_statistical_conclusion": c2_c4[
                "statistical_conclusion"
            ],
            "c3_dominant_error_type": errors["c3"]["dominant_error_type"],
            "accuracy_reference": "g1",
            "recommended_lightweight": "c2",
        },
        "implementation_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in implementation
        },
        "generated_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in outputs.values()
        },
        "validation": {
            "fold_rows": len(fold_rows),
            "stability_rows": len(stability_rows),
            "error_rows": len(error_rows),
            "figures": 2,
            "thesis_report_created": True,
            "unresolved_c2_c4_result_preserved": True,
        },
    }
    contract_path = output_root / "stage36_contract.json"
    hash_path = output_root / "stage36_contract.sha256"
    write_json(contract_path, contract)
    hash_path.write_text(
        portable_text_sha256(contract_path) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "hash_path": hash_path,
        "outputs": outputs,
        "stability_rows": stability_rows,
        "error_rows": error_rows,
        "c2_c4": c2_c4,
    }


def main() -> None:
    result = build_stage36(PROJECT_ROOT)
    stability = {
        row["model_id"]: row for row in result["stability_rows"]
    }
    errors = {row["model_id"]: row for row in result["error_rows"]}
    print("=" * 78)
    print("STAGE 36 — FOLD-LEVEL ERROR ANALYSIS AND THESIS FIGURES")
    print("=" * 78)
    print()
    print("FOLD STABILITY")
    print("-" * 78)
    for model_id in ("g1", "c1", "c2", "c3", "c4"):
        row = stability[model_id]
        print(
            f"{model_id.upper()}: WER={float(row['pooled_wer_mean']):.6f} "
            f"± {float(row['pooled_wer_sample_sd']):.6f}, "
            f"range={float(row['pooled_wer_range']):.6f}, "
            f"best/worst folds={row['best_wer_fold']}/{row['worst_wer_fold']}"
        )
    print()
    print("ERROR COMPOSITION")
    print("-" * 78)
    for model_id in ("g1", "c1", "c2", "c3", "c4"):
        row = errors[model_id]
        print(
            f"{model_id.upper()}: substitutions={row['substitutions']}, "
            f"deletions={row['deletions']}, insertions={row['insertions']}, "
            f"dominant={row['dominant_error_type']}"
        )
    print()
    print("C2-C4 FOLD CHECK")
    print("-" * 78)
    print(
        "C2 lower-WER folds: "
        f"{result['c2_c4']['c2_lower_wer_folds']}"
    )
    print(
        "C4 lower-WER folds: "
        f"{result['c2_c4']['c4_lower_wer_folds']}"
    )
    print("Overall paired conclusion: not resolved at 95%")
    print()
    print("SAFETY")
    print("-" * 78)
    print("Training started: False")
    print("Predictions regenerated: False")
    print("Accuracy used alone: False")
    print()
    print("OUTPUTS")
    print("-" * 78)
    for path in result["outputs"].values():
        print(path)
    print(result["contract_path"])
    print(result["hash_path"])
    print()
    print("FINAL STATUS: STAGE 36 BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

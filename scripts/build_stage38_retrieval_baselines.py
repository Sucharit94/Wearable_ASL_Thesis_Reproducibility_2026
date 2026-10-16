"""Build Stage 38 L0/L1 language baselines from frozen predictions."""

from pathlib import Path

from src.evaluation.stage37_language_protocol import (
    portable_text_sha256,
    read_csv,
    read_json,
    validate_targets,
    write_csv,
    write_json,
)
from src.evaluation.stage38_retrieval import (
    STAGE38_ROOT,
    execute_retrieval_baselines,
    prediction_sources,
    render_report,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def verify_stage37(root: Path) -> str:
    contract_path = root / "artifacts/language/stage37/stage37_contract.json"
    hash_path = root / "artifacts/language/stage37/stage37_contract.sha256"
    if not contract_path.is_file() or not hash_path.is_file():
        raise FileNotFoundError("Stage 37 must pass before Stage 38.")
    observed = portable_text_sha256(contract_path)
    if hash_path.read_text(encoding="utf-8-sig").strip() != observed:
        raise RuntimeError("Stage 37 contract hash mismatch.")
    contract = read_json(contract_path)
    if contract.get("status") != "passed" or contract.get("stage") != 37:
        raise RuntimeError("Stage 37 is not frozen as passed.")
    roles = contract["model_roles"]
    if (
        roles.get("primary_lightweight") != "c2"
        or roles.get("recognition_quality_reference") != "g1"
        or roles.get("comparative_recognizers") != ["c1", "c3", "c4"]
    ):
        raise RuntimeError("Stage 37 model roles changed.")
    return observed


def build_stage38(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    stage37_hash = verify_stage37(root)
    assignment_path = root / "artifacts/splits/stage32_known_inner_assignment.csv"
    target_path = root / "artifacts/labels/stage10_sentence_targets.csv"
    assignments = read_csv(assignment_path)
    targets = read_csv(target_path)
    validate_targets(targets, expected_count=300)
    if len(assignments) != 37000:
        raise RuntimeError("Stage 32 assignment must contain 37000 rows.")
    sources = prediction_sources(root)
    outputs, summaries, manifests = execute_retrieval_baselines(
        assignments, targets, sources
    )

    output_root = root / STAGE38_ROOT
    paths = {
        "outputs": output_root / "stage38_l0_l1_outputs.csv",
        "summary": output_root / "stage38_l0_l1_summary.csv",
        "manifest": output_root / "stage38_prediction_manifest.csv",
        "report": root / "docs/STAGE_38_RETRIEVAL_BASELINES.md",
    }
    write_csv(paths["outputs"], outputs)
    write_csv(paths["summary"], summaries)
    write_csv(paths["manifest"], manifests)
    paths["report"].parent.mkdir(parents=True, exist_ok=True)
    paths["report"].write_text(
        render_report(summaries), encoding="utf-8", newline="\n"
    )

    implementation = (
        root / "src/evaluation/stage38_retrieval.py",
        root / "scripts/build_stage38_retrieval_baselines.py",
        root / "scripts/verify_stage38_retrieval_baselines.py",
        root / "tests/test_stage38_retrieval_baselines.py",
    )
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 38,
        "name": "immutable_predictions_and_deterministic_retrieval_baselines",
        "status": "passed",
        "training_started": False,
        "recognizer_rescored": False,
        "generative_model_called": False,
        "human_post_editing": False,
        "input_stage37_contract_portable_sha256": stage37_hash,
        "prediction_sources": [
            {
                "model_id": source.model_id,
                "outer_fold": source.outer_fold,
                "sha256": source.sha256,
                "predictions": 1480,
            }
            for source in sources
        ],
        "protocol": {
            "primary_lightweight": "c2",
            "recognition_quality_reference": "g1",
            "systems": ["L0_raw_gloss", "L1_training_only_retrieval"],
            "retrieval_bank": "current_fold_inner_train_plus_inner_validation",
            "retrieval_templates_per_fold": 300,
            "outer_samples_in_retrieval_bank": 0,
            "accuracy_used_alone": False,
            "closed_set_template_limitation_declared": True,
        },
        "validation": {
            "models": 5,
            "folds_per_model": 5,
            "predictions_per_model": 7400,
            "prediction_files": 25,
            "language_output_rows": len(outputs),
            "summary_rows": len(summaries),
        },
        "implementation_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in implementation
        },
        "generated_portable_sha256": {
            path.relative_to(root).as_posix(): portable_text_sha256(path)
            for path in paths.values()
        },
    }
    contract_path = output_root / "stage38_contract.json"
    hash_path = output_root / "stage38_contract.sha256"
    write_json(contract_path, contract)
    hash_path.write_text(
        portable_text_sha256(contract_path) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "contract": contract,
        "contract_path": contract_path,
        "hash_path": hash_path,
        "paths": paths,
        "summaries": summaries,
        "manifests": manifests,
    }


def main() -> None:
    result = build_stage38(PROJECT_ROOT)
    print("=" * 78)
    print("STAGE 38 — IMMUTABLE PREDICTIONS AND L0/L1 RETRIEVAL BASELINES")
    print("=" * 78)
    print()
    print("PREDICTION INPUTS")
    print("-" * 78)
    print("Models: G1, C1, C2, C3 and C4")
    print("Outer prediction files: 25")
    print("Predictions per model: 7400")
    print("Recognizer training started: False")
    print("Recognizer rescored: False")
    print()
    print("RESULTS")
    print("-" * 78)
    for row in result["summaries"]:
        print(
            f"{str(row['model_id']).upper()} {row['system_id']}: "
            f"WER={float(row['english_word_error_rate']):.6f}, "
            f"accuracy={float(row['exact_sentence_accuracy']):.6f}, "
            f"content_F1={float(row['mean_content_token_f1']):.6f}, "
            f"unsupported={float(row['unsupported_content_rate']):.6f}"
        )
    print()
    print("SCIENTIFIC BOUNDARY")
    print("-" * 78)
    print("Retrieval examples from current-fold training data only: True")
    print("Outer samples in retrieval bank: 0")
    print("Closed-set 300-template limitation declared: True")
    print("Accuracy used alone: False")
    print("Generative model called: False")
    print()
    print("OUTPUTS")
    print("-" * 78)
    for path in result["paths"].values():
        print(path)
    print(result["contract_path"])
    print(result["hash_path"])
    print()
    print("FINAL STATUS: STAGE 38 BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

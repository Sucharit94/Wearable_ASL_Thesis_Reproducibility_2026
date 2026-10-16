"""Verify frozen Stage 34B outputs without training."""

from pathlib import Path

from src.evaluation.stage34_analysis import (
    EXPECTED_ARCHIVE_SHA256,
    MODEL_ORDER,
    read_csv,
    read_json,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / "artifacts" / "analysis" / "stage34"
    contract_path = output_root / "stage34b_contract.json"
    contract_hash_path = output_root / "stage34b_contract.sha256"
    model_path = output_root / "stage34b_model_metrics.csv"
    fold_path = output_root / "stage34b_fold_metrics.csv"
    subject_path = output_root / "stage34b_subject_metrics.csv"
    length_path = output_root / "stage34b_length_metrics.csv"
    document_path = PROJECT_ROOT / "docs" / "STAGE_34B_MODEL_EVALUATION.md"

    required = (
        contract_path,
        contract_hash_path,
        model_path,
        fold_path,
        subject_path,
        length_path,
        document_path,
    )
    for path in required:
        require(path.is_file(), f"Missing required output: {path}")

    contract = read_json(contract_path)
    recorded_contract_hash = contract_hash_path.read_text(encoding="utf-8").strip()
    require(recorded_contract_hash == sha256_file(contract_path), "Contract hash mismatch.")
    require(contract["stage"] == 34 and contract["substage"] == "B", "Wrong contract stage.")
    require(contract["status"] == "passed", "Contract is not passed.")
    require(contract["training_started"] is False, "Analysis claims training occurred.")
    require(contract["input"]["archive_sha256"] == EXPECTED_ARCHIVE_SHA256, "Archive hash mismatch.")
    require(contract["input"]["prediction_files"] == 25, "Prediction-file count mismatch.")
    require(contract["input"]["prediction_rows"] == 37000, "Prediction-row count mismatch.")
    require(contract["input"]["paired_samples"] == 7400, "Paired sample count mismatch.")
    require(contract["validation"]["paired_references_match"] is True, "References did not pair.")
    require(contract["validation"]["wer_ranking"] == ["g1", "c2", "c4", "c1", "c3"], "WER ranking changed.")

    for relative, recorded_hash in contract["generated_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing hashed output: {path}")
        require(sha256_file(path) == recorded_hash, f"Output hash mismatch: {relative}")

    models = read_csv(model_path)
    folds = read_csv(fold_path)
    subjects = read_csv(subject_path)
    lengths = read_csv(length_path)
    require(len(models) == 5, "Model table must contain five rows.")
    require(len(folds) == 25, "Fold table must contain 25 rows.")
    require(len(subjects) == 15, "Subject table must contain 15 rows.")
    require(len(lengths) > 0, "Length table is empty.")
    require(tuple(row["model_id"] for row in models) == MODEL_ORDER, "Model order changed.")

    ranking = sorted(models, key=lambda row: int(row["wer_rank"]))
    print("=" * 78)
    print("STAGE 34B — MODEL EVALUATION VERIFIER")
    print("=" * 78)
    print()
    print("INTEGRITY")
    print("-" * 78)
    print("Archive hash match: True")
    print("Contract hash match: True")
    print("Generated output hashes match: True")
    print("Paired samples: 7400")
    print("Prediction rows: 37000")
    print()
    print("TABLES")
    print("-" * 78)
    print("Models:", len(models))
    print("Model-fold rows:", len(folds))
    print("Model-subject rows:", len(subjects))
    print("Model-length rows:", len(lengths))
    print()
    print("WER RANKING")
    print("-" * 78)
    for row in ranking:
        print(
            f"{row['wer_rank']}. {row['model_id'].upper()}: "
            f"{float(row['pooled_wer']):.6f}"
        )
    print()
    print("FINAL STATUS: STAGE 34B PASSED")
    print("NO TRAINING WAS STARTED")
    print("=" * 78)


if __name__ == "__main__":
    main()

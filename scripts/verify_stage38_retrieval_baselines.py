"""Verify Stage 38 without rerunning retrieval or any model."""

from pathlib import Path

from scripts.build_stage38_retrieval_baselines import verify_stage37
from src.evaluation.stage37_language_protocol import portable_text_sha256, read_csv, read_json
from src.evaluation.stage38_retrieval import STAGE38_ROOT, prediction_sources


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / STAGE38_ROOT
    contract_path = output_root / "stage38_contract.json"
    hash_path = output_root / "stage38_contract.sha256"
    require(contract_path.is_file(), "Stage 38 contract is missing.")
    require(hash_path.is_file(), "Stage 38 contract hash is missing.")
    require(hash_path.read_text(encoding="utf-8-sig").strip() == portable_text_sha256(contract_path), "Stage 38 contract hash mismatch.")
    contract = read_json(contract_path)
    require(contract.get("stage") == 38 and contract.get("status") == "passed", "Stage 38 identity/status failed.")
    require(contract.get("training_started") is False, "Training was started.")
    require(contract.get("recognizer_rescored") is False, "Recognizer was rescored.")
    require(contract.get("generative_model_called") is False, "Generator was called.")
    require(contract["protocol"]["outer_samples_in_retrieval_bank"] == 0, "Retrieval leakage detected.")
    require(contract["protocol"]["accuracy_used_alone"] is False, "Accuracy-only judgement detected.")
    require(verify_stage37(PROJECT_ROOT) == contract["input_stage37_contract_portable_sha256"], "Stage 37 changed.")

    current_sources = prediction_sources(PROJECT_ROOT)
    current_hashes = {
        (source.model_id, source.outer_fold): source.sha256 for source in current_sources
    }
    frozen_hashes = {
        (row["model_id"], int(row["outer_fold"])): row["sha256"]
        for row in contract["prediction_sources"]
    }
    require(current_hashes == frozen_hashes, "Frozen prediction inputs changed.")
    for relative, expected in contract["implementation_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file() and portable_text_sha256(path) == expected, f"Implementation hash mismatch: {relative}")
    for relative, expected in contract["generated_portable_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file() and portable_text_sha256(path) == expected, f"Output hash mismatch: {relative}")

    outputs = read_csv(output_root / "stage38_l0_l1_outputs.csv")
    summary = read_csv(output_root / "stage38_l0_l1_summary.csv")
    manifest = read_csv(output_root / "stage38_prediction_manifest.csv")
    require(len(outputs) == 74000, "Expected 74000 L0/L1 output rows.")
    require(len(summary) == 10, "Expected ten summary rows.")
    require(len(manifest) == 25, "Expected 25 prediction manifest rows.")
    require(all(row["outer_samples_in_retrieval_bank"] == "0" for row in manifest), "Manifest leakage detected.")

    print("=" * 78)
    print("STAGE 38 — RETRIEVAL BASELINE VERIFIER")
    print("=" * 78)
    print()
    print("INTEGRITY")
    print("-" * 78)
    print("Stage 37 unchanged: True")
    print("Frozen prediction hashes match: True")
    print("Predictions per model: 7400")
    print("Models verified: G1, C1, C2, C3 and C4")
    print("Generated output rows: 74000")
    print()
    print("LEAKAGE AND EXECUTION SAFETY")
    print("-" * 78)
    print("Outer samples in retrieval banks: 0")
    print("Recognizer training started: False")
    print("Recognizer rescored: False")
    print("Generative model called: False")
    print("Accuracy used alone: False")
    print()
    print("FINAL STATUS: STAGE 38 PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

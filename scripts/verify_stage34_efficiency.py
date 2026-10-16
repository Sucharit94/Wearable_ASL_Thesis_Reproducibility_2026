"""Verify Stage 34D efficiency evidence without rerunning benchmarks."""

from pathlib import Path

from src.evaluation.stage34_analysis import read_csv, read_json, sha256_file
from src.evaluation.stage34_efficiency import MODEL_ORDER
from src.models.factory import EXPECTED_TRAINABLE_PARAMETERS, build_model, count_trainable_parameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    output_root = PROJECT_ROOT / "artifacts" / "analysis" / "stage34"
    contract_path = output_root / "stage34d_contract.json"
    hash_path = output_root / "stage34d_contract.sha256"
    metrics_path = output_root / "stage34d_efficiency_metrics.csv"
    document_path = PROJECT_ROOT / "docs" / "STAGE_34D_MODEL_EFFICIENCY.md"
    for path in (contract_path, hash_path, metrics_path, document_path):
        require(path.is_file(), f"Missing Stage 34D output: {path}")

    require(
        hash_path.read_text(encoding="utf-8").strip() == sha256_file(contract_path),
        "Stage 34D contract hash mismatch.",
    )
    contract = read_json(contract_path)
    require(contract["stage"] == 34 and contract["substage"] == "D", "Wrong stage.")
    require(contract["status"] == "passed", "Stage 34D is not passed.")
    require(contract["training_started"] is False, "Stage 34D claims training.")
    require(contract["benchmark"]["isolated_process_per_model"] is True, "Workers were not isolated.")
    require(contract["benchmark"]["evaluation_mode"] is True, "Evaluation mode missing.")
    require(contract["benchmark"]["inference_mode"] is True, "Inference mode missing.")

    for relative, expected in contract["implementation_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing implementation file: {relative}")
        require(sha256_file(path) == expected, f"Implementation hash mismatch: {relative}")
    for relative, expected in contract["generated_sha256"].items():
        path = PROJECT_ROOT / relative
        require(path.is_file(), f"Missing generated file: {relative}")
        require(sha256_file(path) == expected, f"Generated hash mismatch: {relative}")

    rows = read_csv(metrics_path)
    require(len(rows) == 5, "Expected five efficiency rows.")
    require(tuple(row["model_id"] for row in rows) == MODEL_ORDER, "Model order changed.")
    for row in rows:
        model_id = row["model_id"]
        expected = int(EXPECTED_TRAINABLE_PARAMETERS[model_id])
        require(int(row["trainable_parameters"]) == expected, f"CSV parameter mismatch: {model_id}")
        model = build_model(model_id)
        require(count_trainable_parameters(model) == expected, f"Live parameter mismatch: {model_id}")
        require(float(row["serialized_state_dict_mib"]) > 0.0, "Invalid model size.")
        require(float(row["batch1_latency_median_ms"]) > 0.0, "Invalid latency.")
        require(float(row["batch16_throughput_samples_per_second"]) > 0.0, "Invalid throughput.")
        require(row["training_started"].lower() == "false", "Benchmark trained a model.")
        require(row["gradients_created"].lower() == "false", "Benchmark created gradients.")

    print("=" * 78)
    print("STAGE 34D — MODEL EFFICIENCY VERIFIER")
    print("=" * 78)
    print()
    print("INTEGRITY")
    print("-" * 78)
    print("Contract hash match: True")
    print("Implementation hashes match: True")
    print("Generated output hashes match: True")
    print("Model parameter counts match: True")
    print("Models benchmarked: 5")
    print()
    print("PROTOCOL")
    print("-" * 78)
    print("Isolated CPU process per model: True")
    print("Evaluation and inference mode: True")
    print("Training started: False")
    print("Gradients created: False")
    print()
    print("FINAL STATUS: STAGE 34D PASSED")
    print("NO TRAINING WAS STARTED")
    print("=" * 78)


if __name__ == "__main__":
    main()

"""Build Stage 34D model-efficiency evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.stage34_analysis import read_json, sha256_file, write_csv, write_json
from src.evaluation.stage34_efficiency import (
    BENCHMARK_BATCH_SIZES,
    BENCHMARK_FEATURE_WIDTH,
    BENCHMARK_SEED,
    BENCHMARK_SEQUENCE_LENGTH,
    BENCHMARK_TIMED_ITERATIONS,
    BENCHMARK_WARMUP_ITERATIONS,
    MODEL_ORDER,
    WORKER_PREFIX,
    benchmark_model,
    run_isolated_worker,
    runtime_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _mib(value: int | None) -> float | None:
    return None if value is None else float(value) / (1024.0 * 1024.0)


def _flatten(result: dict[str, object], g1_parameters: int) -> dict[str, object]:
    batch1 = result["batch1"]
    batch16 = result["batch16"]
    parameters = int(result["trainable_parameters"])
    return {
        "model_id": result["model_id"],
        "trainable_parameters": parameters,
        "parameter_reduction_vs_g1_percent": 100.0 * (1.0 - parameters / g1_parameters),
        "g1_to_model_parameter_ratio": g1_parameters / parameters,
        "fp32_parameter_memory_mib": _mib(int(result["parameter_bytes"])),
        "buffer_memory_mib": _mib(int(result["buffer_bytes"])),
        "serialized_state_dict_mib": _mib(int(result["state_dict_serialized_bytes"])),
        "largest_observed_output_mib": _mib(int(result["largest_observed_output_bytes"])),
        "incremental_peak_rss_mib": _mib(result["incremental_peak_rss_bytes"]),
        "peak_process_rss_mib": _mib(result["peak_process_rss_bytes"]),
        "batch1_latency_median_ms": batch1["latency_median_ms"],
        "batch1_latency_q1_ms": batch1["latency_q1_ms"],
        "batch1_latency_q3_ms": batch1["latency_q3_ms"],
        "batch1_throughput_samples_per_second": batch1["throughput_samples_per_second"],
        "batch16_latency_median_ms": batch16["latency_median_ms"],
        "batch16_latency_q1_ms": batch16["latency_q1_ms"],
        "batch16_latency_q3_ms": batch16["latency_q3_ms"],
        "batch16_throughput_samples_per_second": batch16["throughput_samples_per_second"],
        "device": result["device"],
        "dtype": result["dtype"],
        "model_in_eval_mode": result["model_in_eval_mode"],
        "gradients_created": result["gradients_created"],
        "training_started": result["training_started"],
    }


def _report(rows: list[dict[str, object]], runtime: dict[str, object]) -> str:
    lines = [
        "# Stage 34D — Model Efficiency Benchmark",
        "",
        "## Protocol",
        "",
        "All five untrained architecture instances are measured on the same CPU",
        "using float32, one PyTorch intra-op thread, evaluation mode and inference",
        "mode. Each model runs in a fresh process. Inputs have shape B×108×38.",
        "Latency is the median after two warm-up iterations. Peak RSS is an",
        "isolated-process measurement and includes model and runtime allocations.",
        "The benchmark does not train a model and is not a GPU-speed claim.",
        "",
        "## Runtime",
        "",
        f"- Python: {runtime['python_version']}",
        f"- PyTorch: {runtime['pytorch_version']}",
        f"- Platform: {runtime['platform']}",
        f"- Processor: {runtime['processor']}",
        "",
        "## Results",
        "",
        "| Model | Parameters | Reduction vs G1 | State dict MiB | Batch-1 ms | Batch-16 samples/s | Peak RSS increment MiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        memory = row["incremental_peak_rss_mib"]
        memory_text = "not available" if memory is None else f"{float(memory):.2f}"
        lines.append(
            "| {model} | {params:,} | {reduction:.2f}% | {state:.3f} | "
            "{latency:.3f} | {throughput:.2f} | {memory} |".format(
                model=str(row["model_id"]).upper(),
                params=int(row["trainable_parameters"]),
                reduction=float(row["parameter_reduction_vs_g1_percent"]),
                state=float(row["serialized_state_dict_mib"]),
                latency=float(row["batch1_latency_median_ms"]),
                throughput=float(row["batch16_throughput_samples_per_second"]),
                memory=memory_text,
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Parameter count and serialized size are architecture properties.",
            "Latency, throughput and RSS are hardware/software dependent and must",
            "be reported with this runtime identity. They should be used for",
            "within-machine comparison, not as universal deployment numbers.",
            "",
        ]
    )
    return "\n".join(lines)


def build_stage34d(project_root: str | Path) -> dict[str, object]:
    root = Path(project_root).resolve()
    output_root = root / "artifacts" / "analysis" / "stage34"
    stage34c_contract = output_root / "stage34c_contract.json"
    stage34c_hash = output_root / "stage34c_contract.sha256"
    if not stage34c_contract.is_file() or not stage34c_hash.is_file():
        raise RuntimeError("Stage 34C outputs are required before Stage 34D.")
    if stage34c_hash.read_text(encoding="utf-8").strip() != sha256_file(stage34c_contract):
        raise RuntimeError("Stage 34C contract hash mismatch.")

    raw_results = [run_isolated_worker(root, model_id) for model_id in MODEL_ORDER]
    g1_parameters = int(raw_results[0]["trainable_parameters"])
    rows = [_flatten(result, g1_parameters) for result in raw_results]
    metrics_path = output_root / "stage34d_efficiency_metrics.csv"
    write_csv(metrics_path, rows)

    runtime = runtime_identity()
    document_path = root / "docs" / "STAGE_34D_MODEL_EFFICIENCY.md"
    document_path.parent.mkdir(parents=True, exist_ok=True)
    document_path.write_text(_report(rows, runtime), encoding="utf-8", newline="\n")

    implementation_paths = (
        root / "src" / "evaluation" / "stage34_efficiency.py",
        root / "scripts" / "build_stage34_efficiency.py",
        root / "scripts" / "verify_stage34_efficiency.py",
        root / "tests" / "test_stage34_efficiency.py",
    )
    for path in implementation_paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 34,
        "substage": "D",
        "status": "passed",
        "training_started": False,
        "benchmark": {
            "device": "cpu",
            "dtype": "float32",
            "isolated_process_per_model": True,
            "evaluation_mode": True,
            "inference_mode": True,
            "input_shape": ["batch", BENCHMARK_SEQUENCE_LENGTH, BENCHMARK_FEATURE_WIDTH],
            "batch_sizes": list(BENCHMARK_BATCH_SIZES),
            "warmup_iterations": BENCHMARK_WARMUP_ITERATIONS,
            "timed_iterations": BENCHMARK_TIMED_ITERATIONS,
            "seed": BENCHMARK_SEED,
            "latency_statistic": "median_with_inclusive_iqr",
            "memory_statistic": "isolated_process_peak_rss_minus_pre_model_rss",
        },
        "runtime": runtime,
        "input": {
            "stage34c_contract_sha256": sha256_file(stage34c_contract),
        },
        "models": raw_results,
        "implementation_sha256": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in implementation_paths
        },
        "generated_sha256": {
            metrics_path.relative_to(root).as_posix(): sha256_file(metrics_path),
            document_path.relative_to(root).as_posix(): sha256_file(document_path),
        },
        "validation": {
            "model_count": len(rows),
            "parameter_counts_match": all(
                int(row["trainable_parameters"]) > 0 for row in rows
            ),
            "all_models_eval_mode": all(bool(row["model_in_eval_mode"]) for row in rows),
            "no_gradients_created": all(not bool(row["gradients_created"]) for row in rows),
            "positive_latency": all(
                float(row["batch1_latency_median_ms"]) > 0.0
                and float(row["batch16_latency_median_ms"]) > 0.0
                for row in rows
            ),
        },
    }
    contract_path = output_root / "stage34d_contract.json"
    contract_hash_path = output_root / "stage34d_contract.sha256"
    write_json(contract_path, contract)
    contract_hash_path.write_text(
        sha256_file(contract_path) + "\n", encoding="utf-8", newline="\n"
    )
    return {
        "rows": rows,
        "contract": contract,
        "contract_path": contract_path,
        "contract_hash_path": contract_hash_path,
        "metrics_path": metrics_path,
        "document_path": document_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", choices=MODEL_ORDER)
    arguments = parser.parse_args()
    if arguments.worker:
        print(WORKER_PREFIX + json.dumps(benchmark_model(arguments.worker), sort_keys=True))
        return

    result = build_stage34d(PROJECT_ROOT)
    print("=" * 78)
    print("STAGE 34D — MODEL EFFICIENCY BENCHMARK")
    print("=" * 78)
    print()
    print("METHOD")
    print("-" * 78)
    print("Device: CPU")
    print("Precision: float32")
    print("Input: B x 108 x 38")
    print("Batch sizes: 1 and 16")
    print("Fresh process per model: True")
    print("Training started: False")
    print()
    print("EFFICIENCY RESULTS")
    print("-" * 78)
    for row in result["rows"]:
        memory = row["incremental_peak_rss_mib"]
        memory_text = "N/A" if memory is None else f"{float(memory):.2f} MiB"
        print(
            f"{str(row['model_id']).upper()}: parameters={int(row['trainable_parameters'])}, "
            f"state={float(row['serialized_state_dict_mib']):.3f} MiB, "
            f"batch1={float(row['batch1_latency_median_ms']):.3f} ms, "
            f"batch16={float(row['batch16_throughput_samples_per_second']):.2f} samples/s, "
            f"peak increment={memory_text}"
        )
    print()
    print("OUTPUTS")
    print("-" * 78)
    print(result["metrics_path"])
    print(result["contract_path"])
    print(result["contract_hash_path"])
    print(result["document_path"])
    print()
    print("FINAL STATUS: STAGE 34D BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

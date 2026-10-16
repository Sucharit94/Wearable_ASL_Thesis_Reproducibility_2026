"""Paired clustered inference for Stage 34C."""

from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from src.evaluation.stage34_analysis import (
    MODEL_ORDER,
    load_predictions,
    read_csv,
    sha256_file,
    write_csv,
    write_json,
)


BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260912
ALPHA = 0.05
PAIR_ORDER = tuple(
    (MODEL_ORDER[left], MODEL_ORDER[right])
    for left in range(len(MODEL_ORDER))
    for right in range(left + 1, len(MODEL_ORDER))
)


def percentile_interval(values: np.ndarray, alpha: float = ALPHA) -> tuple[float, float]:
    lower, upper = np.quantile(
        np.asarray(values, dtype=np.float64),
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method="linear",
    )
    return float(lower), float(upper)


def holm_adjust(p_values: Sequence[float]) -> list[float]:
    """Holm step-down family-wise error correction."""
    if not p_values:
        return []
    values = [float(value) for value in p_values]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError("P-values must be in [0, 1].")
    order = sorted(range(len(values)), key=values.__getitem__)
    adjusted = [0.0] * len(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, values[index] * (total - rank))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def exact_mcnemar_pvalue(a_correct_b_wrong: int, a_wrong_b_correct: int) -> float:
    """Two-sided exact McNemar/binomial p-value using stable log arithmetic."""
    b = int(a_correct_b_wrong)
    c = int(a_wrong_b_correct)
    if b < 0 or c < 0:
        raise ValueError("Discordant counts cannot be negative.")
    total = b + c
    if total == 0:
        return 1.0
    tail_end = min(b, c)
    logs = [
        math.lgamma(total + 1)
        - math.lgamma(index + 1)
        - math.lgamma(total - index + 1)
        - total * math.log(2.0)
        for index in range(tail_end + 1)
    ]
    maximum = max(logs)
    tail = math.exp(maximum) * sum(math.exp(value - maximum) for value in logs)
    return max(sys.float_info.min, min(1.0, 2.0 * tail))


def bootstrap_two_sided_pvalue(differences: np.ndarray) -> float:
    values = np.asarray(differences, dtype=np.float64)
    count = len(values)
    if count <= 0:
        raise ValueError("Bootstrap differences cannot be empty.")
    nonpositive = int(np.count_nonzero(values <= 0.0))
    nonnegative = int(np.count_nonzero(values >= 0.0))
    return min(1.0, 2.0 * (min(nonpositive, nonnegative) + 1) / (count + 1))


def _paired_sample_table(records: Sequence[dict]) -> tuple[list[str], dict[str, dict[str, dict]]]:
    by_sample: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in records:
        sample_id = str(row["sample_id"])
        model_id = str(row["model_id"])
        if model_id in by_sample[sample_id]:
            raise RuntimeError(f"Duplicate paired prediction for {sample_id}, {model_id}.")
        by_sample[sample_id][model_id] = row
    sample_ids = sorted(by_sample)
    if len(sample_ids) != 7400:
        raise RuntimeError(f"Expected 7400 paired samples; found {len(sample_ids)}.")
    for sample_id in sample_ids:
        if set(by_sample[sample_id]) != set(MODEL_ORDER):
            raise RuntimeError(f"Incomplete model pairing for {sample_id}.")
    return sample_ids, by_sample


def clustered_bootstrap(
    records: Sequence[dict],
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Bootstrap subject-by-sentence clusters while retaining model pairing."""
    if int(replicates) <= 0:
        raise ValueError("Bootstrap replicate count must be positive.")
    sample_ids, paired = _paired_sample_table(records)
    cluster_members: dict[tuple[str, int], list[str]] = defaultdict(list)
    for sample_id in sample_ids:
        reference = paired[sample_id][MODEL_ORDER[0]]
        cluster = (str(reference["subject"]), int(reference["sentence_id"]))
        cluster_members[cluster].append(sample_id)
    clusters = sorted(cluster_members)
    if len(clusters) != 368:
        raise RuntimeError(f"Expected 368 subject-sentence clusters; found {len(clusters)}.")

    model_count = len(MODEL_ORDER)
    cluster_count = len(clusters)
    edits = np.zeros((model_count, cluster_count), dtype=np.int64)
    correct = np.zeros((model_count, cluster_count), dtype=np.int64)
    references = np.zeros(cluster_count, dtype=np.int64)
    samples = np.zeros(cluster_count, dtype=np.int64)

    for cluster_index, cluster in enumerate(clusters):
        members = cluster_members[cluster]
        samples[cluster_index] = len(members)
        references[cluster_index] = sum(
            len(paired[sample_id][MODEL_ORDER[0]]["reference_token_ids"])
            for sample_id in members
        )
        for model_index, model_id in enumerate(MODEL_ORDER):
            edits[model_index, cluster_index] = sum(
                int(paired[sample_id][model_id]["edit_distance"])
                for sample_id in members
            )
            correct[model_index, cluster_index] = sum(
                bool(paired[sample_id][model_id]["exact_match"])
                for sample_id in members
            )

    rng = np.random.default_rng(int(seed))
    wer_bootstrap = np.empty((replicates, model_count), dtype=np.float64)
    accuracy_bootstrap = np.empty((replicates, model_count), dtype=np.float64)
    batch_size = 500
    for start in range(0, replicates, batch_size):
        stop = min(start + batch_size, replicates)
        indices = rng.integers(0, cluster_count, size=(stop - start, cluster_count))
        reference_denominator = references[indices].sum(axis=1)
        sample_denominator = samples[indices].sum(axis=1)
        for model_index in range(model_count):
            wer_bootstrap[start:stop, model_index] = (
                edits[model_index][indices].sum(axis=1) / reference_denominator
            )
            accuracy_bootstrap[start:stop, model_index] = (
                correct[model_index][indices].sum(axis=1) / sample_denominator
            )

    observed_wer = edits.sum(axis=1) / references.sum()
    observed_accuracy = correct.sum(axis=1) / samples.sum()
    return {
        "cluster_count": cluster_count,
        "cluster_definition": "subject_x_sentence_id",
        "cluster_size_minimum": int(samples.min()),
        "cluster_size_maximum": int(samples.max()),
        "observed_wer": observed_wer,
        "observed_accuracy": observed_accuracy,
        "wer_bootstrap": wer_bootstrap,
        "accuracy_bootstrap": accuracy_bootstrap,
        "paired": paired,
        "sample_ids": sample_ids,
    }


def calculate_inference_tables(records: Sequence[dict]) -> dict[str, list[dict]]:
    result = clustered_bootstrap(records)
    observed_wer = result["observed_wer"]
    observed_accuracy = result["observed_accuracy"]
    wer_bootstrap = result["wer_bootstrap"]
    accuracy_bootstrap = result["accuracy_bootstrap"]
    paired = result["paired"]
    sample_ids = result["sample_ids"]

    interval_rows: list[dict] = []
    for model_index, model_id in enumerate(MODEL_ORDER):
        wer_low, wer_high = percentile_interval(wer_bootstrap[:, model_index])
        accuracy_low, accuracy_high = percentile_interval(
            accuracy_bootstrap[:, model_index]
        )
        interval_rows.append(
            {
                "model_id": model_id,
                "pooled_wer": float(observed_wer[model_index]),
                "pooled_wer_ci95_lower": wer_low,
                "pooled_wer_ci95_upper": wer_high,
                "exact_sentence_accuracy": float(observed_accuracy[model_index]),
                "exact_accuracy_ci95_lower": accuracy_low,
                "exact_accuracy_ci95_upper": accuracy_high,
                "ser": 1.0 - float(observed_accuracy[model_index]),
                "ser_ci95_lower": 1.0 - accuracy_high,
                "ser_ci95_upper": 1.0 - accuracy_low,
                "bootstrap_replicates": BOOTSTRAP_REPLICATES,
                "bootstrap_clusters": int(result["cluster_count"]),
            }
        )

    pair_rows: list[dict] = []
    for model_a, model_b in PAIR_ORDER:
        index_a = MODEL_ORDER.index(model_a)
        index_b = MODEL_ORDER.index(model_b)
        differences = wer_bootstrap[:, index_a] - wer_bootstrap[:, index_b]
        difference_low, difference_high = percentile_interval(differences)
        observed_difference = float(observed_wer[index_a] - observed_wer[index_b])
        relative_reduction = (
            float(observed_wer[index_b] - observed_wer[index_a])
            / float(observed_wer[index_b])
        )
        a_correct_b_wrong = sum(
            bool(paired[sample_id][model_a]["exact_match"])
            and not bool(paired[sample_id][model_b]["exact_match"])
            for sample_id in sample_ids
        )
        a_wrong_b_correct = sum(
            not bool(paired[sample_id][model_a]["exact_match"])
            and bool(paired[sample_id][model_b]["exact_match"])
            for sample_id in sample_ids
        )
        accuracy_difference = float(observed_accuracy[index_a] - observed_accuracy[index_b])
        pair_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "wer_a": float(observed_wer[index_a]),
                "wer_b": float(observed_wer[index_b]),
                "delta_wer_a_minus_b": observed_difference,
                "delta_wer_ci95_lower": difference_low,
                "delta_wer_ci95_upper": difference_high,
                "relative_wer_reduction_a_vs_b": relative_reduction,
                "bootstrap_p_value_two_sided": bootstrap_two_sided_pvalue(differences),
                "accuracy_a": float(observed_accuracy[index_a]),
                "accuracy_b": float(observed_accuracy[index_b]),
                "delta_accuracy_a_minus_b": accuracy_difference,
                "mcnemar_a_correct_b_wrong": a_correct_b_wrong,
                "mcnemar_a_wrong_b_correct": a_wrong_b_correct,
                "mcnemar_p_value_exact_two_sided": exact_mcnemar_pvalue(
                    a_correct_b_wrong,
                    a_wrong_b_correct,
                ),
            }
        )

    bootstrap_adjusted = holm_adjust(
        [float(row["bootstrap_p_value_two_sided"]) for row in pair_rows]
    )
    mcnemar_adjusted = holm_adjust(
        [float(row["mcnemar_p_value_exact_two_sided"]) for row in pair_rows]
    )
    for row, bootstrap_p, mcnemar_p in zip(
        pair_rows,
        bootstrap_adjusted,
        mcnemar_adjusted,
        strict=True,
    ):
        row["bootstrap_p_value_holm"] = bootstrap_p
        row["bootstrap_significant_holm_0_05"] = bootstrap_p < ALPHA
        row["mcnemar_p_value_holm"] = mcnemar_p
        row["mcnemar_significant_holm_0_05"] = mcnemar_p < ALPHA
        lower = float(row["delta_wer_ci95_lower"])
        upper = float(row["delta_wer_ci95_upper"])
        row["wer_difference_direction"] = (
            "a_lower_wer" if upper < 0.0 else "b_lower_wer" if lower > 0.0 else "not_resolved"
        )

    return {"intervals": interval_rows, "pairs": pair_rows}


def _report(intervals: Sequence[Mapping[str, object]], pairs: Sequence[Mapping[str, object]]) -> str:
    lines = [
        "# Stage 34C — Paired Statistical Model Comparison",
        "",
        "## Method",
        "",
        "The analysis uses 10,000 paired bootstrap replicates. The resampling",
        "unit is the frozen subject-by-sentence cluster (368 clusters), so all",
        "repetitions and predictions from all five models remain paired.",
        "Pairwise WER differences use percentile confidence intervals. Exact",
        "sentence correctness uses two-sided exact McNemar tests. Holm correction",
        "controls family-wise error separately across the 10 WER comparisons and",
        "the 10 McNemar comparisons.",
        "",
        "## Model confidence intervals",
        "",
        "| Model | WER | WER 95% CI | Exact accuracy | Accuracy 95% CI |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in intervals:
        lines.append(
            "| {model} | {wer:.6f} | [{wl:.6f}, {wu:.6f}] | {acc:.6f} | "
            "[{al:.6f}, {au:.6f}] |".format(
                model=str(row["model_id"]).upper(),
                wer=float(row["pooled_wer"]),
                wl=float(row["pooled_wer_ci95_lower"]),
                wu=float(row["pooled_wer_ci95_upper"]),
                acc=float(row["exact_sentence_accuracy"]),
                al=float(row["exact_accuracy_ci95_lower"]),
                au=float(row["exact_accuracy_ci95_upper"]),
            )
        )
    lines.extend(
        [
            "",
            "## Pairwise conclusions",
            "",
            "| A | B | ΔWER A−B | 95% CI | Holm p | WER conclusion | Δ accuracy | McNemar Holm p |",
            "|---|---|---:|---:|---:|---|---:|---:|",
        ]
    )
    for row in pairs:
        lines.append(
            "| {a} | {b} | {delta:.6f} | [{low:.6f}, {high:.6f}] | {p:.6g} | "
            "{direction} | {accuracy:.6f} | {mp:.6g} |".format(
                a=str(row["model_a"]).upper(),
                b=str(row["model_b"]).upper(),
                delta=float(row["delta_wer_a_minus_b"]),
                low=float(row["delta_wer_ci95_lower"]),
                high=float(row["delta_wer_ci95_upper"]),
                p=float(row["bootstrap_p_value_holm"]),
                direction=str(row["wer_difference_direction"]),
                accuracy=float(row["delta_accuracy_a_minus_b"]),
                mp=float(row["mcnemar_p_value_holm"]),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A negative ΔWER means model A has lower WER. Statistical significance",
            "does not measure practical importance; absolute differences, relative",
            "WER reductions, model size and runtime evidence must also be considered.",
            "",
        ]
    )
    return "\n".join(lines)


def build_stage34c(project_root: str | Path) -> dict:
    root = Path(project_root).resolve()
    stage34_root = root / "artifacts" / "analysis" / "stage34"
    stage34b_contract = stage34_root / "stage34b_contract.json"
    stage34b_hash = stage34_root / "stage34b_contract.sha256"
    if not stage34b_contract.is_file() or not stage34b_hash.is_file():
        raise RuntimeError("Stage 34B outputs are required before Stage 34C.")
    if stage34b_hash.read_text(encoding="utf-8").strip() != sha256_file(stage34b_contract):
        raise RuntimeError("Stage 34B contract hash mismatch.")

    evidence_root = root / "imports" / "stage32_results" / "extracted"
    records, prediction_hashes = load_predictions(evidence_root)
    tables = calculate_inference_tables(records)

    interval_path = stage34_root / "stage34c_model_confidence_intervals.csv"
    pair_path = stage34_root / "stage34c_pairwise_tests.csv"
    write_csv(interval_path, tables["intervals"])
    write_csv(pair_path, tables["pairs"])

    document_path = root / "docs" / "STAGE_34C_STATISTICAL_COMPARISON.md"
    document_path.parent.mkdir(parents=True, exist_ok=True)
    with document_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(_report(tables["intervals"], tables["pairs"]))

    generated_hashes = {
        path.relative_to(root).as_posix(): sha256_file(path)
        for path in (interval_path, pair_path, document_path)
    }
    implementation_paths = (
        root / "src" / "evaluation" / "stage34_analysis.py",
        root / "src" / "evaluation" / "stage34_statistics.py",
        root / "scripts" / "build_stage34_statistical_comparison.py",
        root / "scripts" / "verify_stage34_statistical_comparison.py",
        root / "tests" / "test_stage34_statistics.py",
    )
    for path in implementation_paths:
        if not path.is_file():
            raise FileNotFoundError(path)
    contract = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 34,
        "substage": "C",
        "status": "passed",
        "training_started": False,
        "method": {
            "paired_predictions": True,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "cluster_definition": "subject_x_sentence_id",
            "clusters": 368,
            "confidence_level": 0.95,
            "pairwise_wer_test": "paired_cluster_bootstrap_two_sided",
            "sentence_correctness_test": "exact_mcnemar_two_sided",
            "multiple_comparison_correction": "holm_separate_test_families",
            "pairwise_comparisons": 10,
        },
        "input": {
            "stage34b_contract_sha256": sha256_file(stage34b_contract),
            "prediction_files": 25,
            "prediction_rows": len(records),
            "prediction_sha256": prediction_hashes,
        },
        "implementation_sha256": {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in implementation_paths
        },
        "validation": {
            "paired_samples": 7400,
            "paired_references_match": True,
            "all_model_intervals_created": len(tables["intervals"]) == 5,
            "all_pairwise_tests_created": len(tables["pairs"]) == 10,
            "holm_applied": True,
        },
        "generated_sha256": generated_hashes,
    }
    contract_path = stage34_root / "stage34c_contract.json"
    contract_hash_path = stage34_root / "stage34c_contract.sha256"
    write_json(contract_path, contract)
    with contract_hash_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(sha256_file(contract_path) + "\n")

    return {
        "contract": contract,
        "contract_path": contract_path,
        "contract_hash_path": contract_hash_path,
        "document_path": document_path,
        "tables": tables,
    }

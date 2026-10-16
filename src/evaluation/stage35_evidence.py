"""Stage 35 publication evidence and claim-freeze helpers.

This module performs no training.  It consolidates the already frozen Stage 34
quality, uncertainty, efficiency and decision outputs into thesis-ready tables.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any


MODEL_ORDER = ("g1", "c1", "c2", "c3", "c4")
MODEL_NAMES = {
    "g1": "Near-Gu Conv2D–2×BiLSTM–CTC",
    "c1": "Compact Conv1D–BiLSTM–CTC",
    "c2": "Compact Conv1D–BiLSTM–Self-Attention–CTC",
    "c3": "Compact Conv1D–Transformer Encoder–CTC",
    "c4": "Custom Tiny-Swin–BiLSTM–CTC",
}
STAGE34_ROOT = Path("artifacts/analysis/stage34")
STAGE35_ROOT = Path("artifacts/analysis/stage35")


_TEXT_HASH_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".svg", ".txt", ".yaml", ".yml"}


def sha256_file(path: str | Path) -> str:
    """Hash text canonically so Git line-ending conversion is harmless."""
    target = Path(path)
    digest = hashlib.sha256()
    if target.suffix.lower() in _TEXT_HASH_SUFFIXES:
        data = target.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        digest.update(data)
        return digest.hexdigest()
    with target.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def portable_text_sha256(path: str | Path) -> str:
    raw = Path(path).read_bytes()
    return hashlib.sha256(raw.replace(b"\r\n", b"\n").replace(b"\r", b"\n")).hexdigest()


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def write_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Cannot write an empty CSV: {path}")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _first(row: Mapping[str, Any], names: Iterable[str], *, required: bool = True) -> Any:
    for name in names:
        if name in row and row[name] not in (None, ""):
            return row[name]
    if required:
        raise KeyError(f"None of these fields is available: {tuple(names)}")
    return None


def _number(row: Mapping[str, Any], *names: str, required: bool = True) -> float | None:
    value = _first(row, names, required=required)
    if value is None:
        return None
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"Non-finite numeric value for {names}: {value}")
    return number


def _integer(row: Mapping[str, Any], *names: str, required: bool = True) -> int | None:
    value = _number(row, *names, required=required)
    return None if value is None else int(value)


def _boolean(row: Mapping[str, Any], *names: str) -> bool:
    value = str(_first(row, names)).strip().lower()
    if value not in {"true", "false"}:
        raise ValueError(f"Expected Boolean field {names}, received {value!r}")
    return value == "true"


def percent_reduction(reference: float, candidate: float) -> float:
    if reference <= 0.0 or candidate < 0.0:
        raise ValueError("Sizes must be non-negative and reference must be positive.")
    return 100.0 * (reference - candidate) / reference


def ratio(reference: float, candidate: float) -> float:
    if reference <= 0.0 or candidate <= 0.0:
        raise ValueError("Ratio inputs must be positive.")
    return reference / candidate


def verify_stage34_inputs(project_root: str | Path) -> dict[str, str]:
    root = Path(project_root).resolve()
    hashes: dict[str, str] = {}
    for substage in ("b", "c", "d", "e"):
        contract_path = root / STAGE34_ROOT / f"stage34{substage}_contract.json"
        hash_path = root / STAGE34_ROOT / f"stage34{substage}_contract.sha256"
        if not contract_path.is_file() or not hash_path.is_file():
            raise FileNotFoundError(f"Stage 34{substage.upper()} contract is required.")
        actual = sha256_file(contract_path)
        recorded = hash_path.read_text(encoding="utf-8").strip()
        if recorded != actual:
            raise RuntimeError(f"Stage 34{substage.upper()} contract hash mismatch.")
        contract = read_json(contract_path)
        if int(contract.get("stage", -1)) != 34:
            raise RuntimeError(f"Incorrect Stage 34{substage.upper()} contract identity.")
        for relative, expected in contract.get("generated_sha256", {}).items():
            generated = root / relative
            if not generated.is_file() or sha256_file(generated) != expected:
                raise RuntimeError(
                    f"Stage 34{substage.upper()} output hash mismatch: {relative}"
                )
        hashes[substage.upper()] = actual
    return hashes


def _by_model(rows: Sequence[Mapping[str, Any]], source: str) -> dict[str, Mapping[str, Any]]:
    indexed = {str(row["model_id"]).lower(): row for row in rows}
    if set(indexed) != set(MODEL_ORDER):
        raise RuntimeError(f"{source} model set is incomplete: {sorted(indexed)}")
    return indexed


def _optional_count(row: Mapping[str, Any], *names: str) -> int | None:
    return _integer(row, *names, required=False)


def _error_profile(model_id: str, row: Mapping[str, Any]) -> dict[str, Any]:
    substitutions = _optional_count(row, "substitutions", "substitutions_total")
    deletions = _optional_count(row, "deletions", "deletions_total")
    insertions = _optional_count(row, "insertions", "insertions_total")
    available = all(value is not None for value in (substitutions, deletions, insertions))
    total = (
        int(substitutions) + int(deletions) + int(insertions)
        if available
        else None
    )
    counts = {
        "substitution": substitutions,
        "deletion": deletions,
        "insertion": insertions,
    }
    dominant = (
        max(counts, key=lambda name: int(counts[name]))
        if available and total and total > 0
        else "not_available"
    )
    return {
        "model_id": model_id,
        "substitutions": substitutions if substitutions is not None else "",
        "deletions": deletions if deletions is not None else "",
        "insertions": insertions if insertions is not None else "",
        "total_edits": total if total is not None else "",
        "substitution_share": substitutions / total if total else "",
        "deletion_share": deletions / total if total else "",
        "insertion_share": insertions / total if total else "",
        "dominant_error_type": dominant,
        "aggregate_counts_available": available,
    }


def _pair_row(
    rows: Sequence[Mapping[str, Any]], model_a: str, model_b: str
) -> tuple[Mapping[str, Any] | None, bool]:
    for row in rows:
        left = str(_first(row, ("model_a", "model_1", "left_model"))).lower()
        right = str(_first(row, ("model_b", "model_2", "right_model"))).lower()
        if (left, right) == (model_a, model_b):
            return row, False
        if (left, right) == (model_b, model_a):
            return row, True
    return None, False


def _pair_text(row: Mapping[str, Any] | None, *names: str) -> str:
    if row is None:
        return ""
    value = _first(row, names, required=False)
    return "" if value is None else str(value)


def _comparison(
    label: str,
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    pairwise_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left_id = str(left["model_id"])
    right_id = str(right["model_id"])
    pair, reversed_pair = _pair_row(pairwise_rows, left_id, right_id)
    direction = _pair_text(
        pair,
        "wer_difference_direction",
        "wer_conclusion",
        "conclusion",
    )
    if reversed_pair and direction == "a_lower_wer":
        direction = "b_lower_wer"
    elif reversed_pair and direction == "b_lower_wer":
        direction = "a_lower_wer"
    left_memory = left.get("incremental_peak_rss_mib")
    right_memory = right.get("incremental_peak_rss_mib")
    return {
        "comparison": label,
        "model_a": left_id,
        "model_b": right_id,
        "wer_a": left["pooled_wer"],
        "wer_b": right["pooled_wer"],
        "wer_delta_b_minus_a": float(right["pooled_wer"]) - float(left["pooled_wer"]),
        "exact_accuracy_a": left["exact_sentence_accuracy"],
        "exact_accuracy_b": right["exact_sentence_accuracy"],
        "accuracy_delta_b_minus_a": float(right["exact_sentence_accuracy"])
        - float(left["exact_sentence_accuracy"]),
        "parameter_reduction_a_to_b_percent": percent_reduction(
            float(left["trainable_parameters"]), float(right["trainable_parameters"])
        ),
        "parameter_ratio_a_over_b": ratio(
            float(left["trainable_parameters"]), float(right["trainable_parameters"])
        ),
        "latency_reduction_a_to_b_percent": percent_reduction(
            float(left["batch1_latency_median_ms"]),
            float(right["batch1_latency_median_ms"]),
        ),
        "throughput_ratio_b_over_a": ratio(
            float(right["batch16_throughput_samples_per_second"]),
            float(left["batch16_throughput_samples_per_second"]),
        ),
        "peak_rss_reduction_a_to_b_percent": (
            percent_reduction(float(left_memory), float(right_memory))
            if left_memory is not None and right_memory is not None
            else ""
        ),
        "paired_wer_conclusion": direction,
        "holm_adjusted_p_value": _pair_text(
            pair,
            "holm_adjusted_p_value",
            "holm_p_value",
            "holm_p",
        ),
        "paired_test_available": pair is not None,
    }


def build_stage35_tables(project_root: str | Path) -> dict[str, Any]:
    root = Path(project_root).resolve()
    stage34 = root / STAGE34_ROOT
    b_rows = read_csv(stage34 / "stage34b_model_metrics.csv")
    c_rows = read_csv(stage34 / "stage34c_model_confidence_intervals.csv")
    pairwise = read_csv(stage34 / "stage34c_pairwise_tests.csv")
    d_rows = read_csv(stage34 / "stage34d_efficiency_metrics.csv")
    e_rows = read_csv(stage34 / "stage34e_decision_matrix.csv")
    b = _by_model(b_rows, "Stage 34B")
    c = _by_model(c_rows, "Stage 34C")
    d = _by_model(d_rows, "Stage 34D")
    e = _by_model(e_rows, "Stage 34E")

    publication: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for model_id in MODEL_ORDER:
        performance = b[model_id]
        interval = c[model_id]
        efficiency = d[model_id]
        decision = e[model_id]
        memory = _number(
            efficiency, "incremental_peak_rss_mib", required=False
        )
        row = {
            "model_id": model_id,
            "model_name": MODEL_NAMES[model_id],
            "pooled_wer": _number(performance, "pooled_wer", "pooled_wer_mean"),
            "pooled_wer_ci95_lower": _number(interval, "pooled_wer_ci95_lower"),
            "pooled_wer_ci95_upper": _number(interval, "pooled_wer_ci95_upper"),
            "exact_sentence_accuracy": _number(
                performance,
                "exact_sentence_accuracy",
                "exact_sentence_accuracy_mean",
            ),
            "exact_accuracy_ci95_lower": _number(
                interval,
                "exact_accuracy_ci95_lower",
                "exact_sentence_accuracy_ci95_lower",
            ),
            "exact_accuracy_ci95_upper": _number(
                interval,
                "exact_accuracy_ci95_upper",
                "exact_sentence_accuracy_ci95_upper",
            ),
            "ser": _number(performance, "ser", "ser_mean"),
            "trainable_parameters": _integer(efficiency, "trainable_parameters"),
            "parameter_reduction_vs_g1_percent": _number(
                efficiency, "parameter_reduction_vs_g1_percent"
            ),
            "g1_to_model_parameter_ratio": _number(
                efficiency, "g1_to_model_parameter_ratio"
            ),
            "serialized_state_dict_mib": _number(
                efficiency, "serialized_state_dict_mib"
            ),
            "batch1_latency_median_ms": _number(
                efficiency, "batch1_latency_median_ms"
            ),
            "batch16_throughput_samples_per_second": _number(
                efficiency, "batch16_throughput_samples_per_second"
            ),
            "incremental_peak_rss_mib": memory if memory is not None else "",
            "quality_eligible": _boolean(decision, "quality_eligible"),
            "lightweight_eligible": _boolean(decision, "lightweight_eligible"),
            "pareto_frontier": _boolean(decision, "pareto_frontier"),
            "mean_scenario_score": _number(decision, "mean_scenario_score"),
            "final_role": str(_first(decision, ("final_role",))),
        }
        publication.append(row)
        errors.append(_error_profile(model_id, performance))

    by_model = {str(row["model_id"]): row for row in publication}
    comparisons = [
        _comparison("accuracy_reference_vs_lightweight", by_model["g1"], by_model["c2"], pairwise),
        _comparison("lightweight_shortlist", by_model["c2"], by_model["c4"], pairwise),
    ]
    if by_model["g1"]["final_role"] != "accuracy_reference":
        raise RuntimeError("Stage 34E no longer identifies G1 as the accuracy reference.")
    if by_model["c2"]["final_role"] != "recommended_lightweight":
        raise RuntimeError("Stage 34E no longer recommends C2 as the lightweight model.")

    claims = [
        {
            "claim_id": "C35-01",
            "status": "supported",
            "thesis_safe_wording": "G1 achieved the lowest five-fold pooled WER and serves as the accuracy reference.",
            "evidence": "stage34b_model_metrics.csv; stage34c_model_confidence_intervals.csv",
            "prohibited_overclaim": "G1 is universally best for unseen signers or every deployment setting.",
        },
        {
            "claim_id": "C35-02",
            "status": "supported",
            "thesis_safe_wording": "C2 is the recommended lightweight recognizer under the predeclared quality and multi-criteria rules.",
            "evidence": "stage34e_decision_matrix.csv; stage34e_sensitivity_ranking.csv",
            "prohibited_overclaim": "C2 has the highest accuracy or lowest WER overall.",
        },
        {
            "claim_id": "C35-03",
            "status": "not_resolved",
            "thesis_safe_wording": "The five-fold WER difference between C2 and C4 was not resolved at 95% confidence after Holm correction.",
            "evidence": "stage34c_pairwise_tests.csv",
            "prohibited_overclaim": "C2 has significantly lower WER than C4.",
        },
        {
            "claim_id": "C35-04",
            "status": "supported",
            "thesis_safe_wording": "On the measured CPU protocol, C2 was operationally more efficient than C4 in latency, throughput and peak RSS.",
            "evidence": "stage34d_efficiency_metrics.csv",
            "prohibited_overclaim": "The measured CPU timings are identical on every device.",
        },
        {
            "claim_id": "C35-05",
            "status": "supported",
            "thesis_safe_wording": "C3 was the smallest model but did not meet the frozen recognition-quality threshold.",
            "evidence": "stage34d_efficiency_metrics.csv; stage34e_decision_matrix.csv",
            "prohibited_overclaim": "The smallest parameter count alone makes C3 the best lightweight recognizer.",
        },
        {
            "claim_id": "C35-06",
            "status": "methodological_rule",
            "thesis_safe_wording": "Model selection used recognition quality, uncertainty, paired inference and measured efficiency rather than accuracy alone.",
            "evidence": "stage34c_contract.json; stage34d_contract.json; stage34e_contract.json",
            "prohibited_overclaim": "One accuracy value is sufficient to establish model superiority.",
        },
    ]
    return {
        "publication_rows": publication,
        "error_rows": errors,
        "comparison_rows": comparisons,
        "claim_rows": claims,
        "accuracy_reference": "g1",
        "recommended_lightweight": "c2",
    }


def render_report(result: Mapping[str, Any]) -> str:
    rows = result["publication_rows"]
    by_model = {row["model_id"]: row for row in rows}
    g1 = by_model["g1"]
    c2 = by_model["c2"]
    c4 = by_model["c4"]
    error_by_model = {row["model_id"]: row for row in result["error_rows"]}

    lines = [
        "# Stage 35 — Final Known-Signer Evidence Freeze",
        "",
        "## Frozen conclusion",
        "",
        "G1 is the accuracy reference. C2 is the recommended lightweight model.",
        "This conclusion is based on quality, uncertainty, paired statistics and",
        "measured efficiency; accuracy is not used alone.",
        "",
        "## Publication-ready model table",
        "",
        "| Model | WER (95% CI) | Exact accuracy (95% CI) | SER | Parameters | CPU latency | Throughput | Role |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {str(row['model_id']).upper()} | "
            f"{float(row['pooled_wer']):.6f} "
            f"[{float(row['pooled_wer_ci95_lower']):.6f}, {float(row['pooled_wer_ci95_upper']):.6f}] | "
            f"{float(row['exact_sentence_accuracy']):.6f} "
            f"[{float(row['exact_accuracy_ci95_lower']):.6f}, {float(row['exact_accuracy_ci95_upper']):.6f}] | "
            f"{float(row['ser']):.6f} | {int(row['trainable_parameters']):,} | "
            f"{float(row['batch1_latency_median_ms']):.3f} ms | "
            f"{float(row['batch16_throughput_samples_per_second']):.2f}/s | "
            f"{row['final_role']} |"
        )
    lines.extend(
        [
            "",
            "## Main comparison",
            "",
            f"C2 uses {float(c2['parameter_reduction_vs_g1_percent']):.2f}% fewer parameters "
            f"than G1 ({int(c2['trainable_parameters']):,} versus "
            f"{int(g1['trainable_parameters']):,}). Its five-fold WER is "
            f"{float(c2['pooled_wer']):.6f}, compared with {float(g1['pooled_wer']):.6f} "
            "for the accuracy reference.",
            "",
            f"C2 and C4 have statistically unresolved WER at 95%, but C4's measured "
            "batch-1 latency was "
            f"{float(c4['batch1_latency_median_ms']) / float(c2['batch1_latency_median_ms']):.2f}× "
            "C2's latency, while C2 provided "
            f"{float(c2['batch16_throughput_samples_per_second']) / float(c4['batch16_throughput_samples_per_second']):.2f}× "
            "higher batch-16 throughput on the frozen CPU benchmark.",
            "",
            "## Aggregate error profile",
            "",
            "| Model | Substitutions | Deletions | Insertions | Dominant error |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for model_id in MODEL_ORDER:
        row = error_by_model[model_id]
        lines.append(
            f"| {model_id.upper()} | {row['substitutions']} | {row['deletions']} | "
            f"{row['insertions']} | {row['dominant_error_type']} |"
        )
    lines.extend(
        [
            "",
            "Blank aggregate counts mean that the upstream summary did not retain",
            "that count; no value is inferred or fabricated.",
            "",
            "## Thesis-safe interpretation",
            "",
            "- G1 is the recognition-quality reference, not a lightweight model.",
            "- C2 is the selected lightweight quality–efficiency compromise.",
            "- C4 remains a competitive quality comparator, but its measured runtime cost is high.",
            "- C3 demonstrates that a low parameter count does not guarantee adequate recognition.",
            "- These findings apply to the frozen mixed-subject five-fold protocol and measured hardware.",
            "",
            "## Methodological boundary",
            "",
            "No model was retrained and no test prediction was regenerated in Stage 35.",
            "The stage only consolidates previously frozen evidence.",
            "",
        ]
    )
    return "\n".join(lines)

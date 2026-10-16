from __future__ import annotations

import csv
from pathlib import Path

import pytest

from src.evaluation.stage41_thesis_evidence import (
    MODEL_ORDER, SYSTEM_ORDER, as_float, build_claim_register,
    build_final_selection, build_recognition_table, comparison_winner,
    normalize_language_summary, norm, sha256_file, write_csv_rows,
)


def language_rows() -> list[dict[str, object]]:
    rows = []
    for mi, model in enumerate(MODEL_ORDER):
        for si, system in enumerate(SYSTEM_ORDER):
            rows.append({"model_id": model, "system_id": system, "samples": 7400, "pooled_wer": 0.01 + mi * .02 + si * .01, "pooled_wer_ci_lower": 0.0, "pooled_wer_ci_upper": .2, "exact_sentence_accuracy": .9 - mi * .05, "content_precision": .9, "content_recall": .8, "content_f1": .85, "unsupported_content_rate": .1, "correction_harm_rate": "" if system == "L0" else .02})
    return rows


def system_tests() -> list[dict[str, object]]:
    return [{"model_id": model, "system_a": "L1", "system_b": "L2", "wer_conclusion": "a_lower_wer"} for model in MODEL_ORDER]


def test_normalization_helpers() -> None:
    assert norm("Exact Sentence Accuracy") == "exact_sentence_accuracy"
    assert as_float("98.5%") == pytest.approx(.985)


def test_text_hash_ignores_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.json"
    crlf = tmp_path / "crlf.json"
    lf.write_bytes(b"{\n  \"ok\": true\n}\n")
    crlf.write_bytes(b"{\r\n  \"ok\": true\r\n}\r\n")
    assert sha256_file(lf) == sha256_file(crlf)


def test_language_matrix_is_complete_and_ranked() -> None:
    rows = normalize_language_summary(language_rows())
    assert len(rows) == 15
    assert {row["quality_rank_within_system"] for row in rows if row["system_id"] == "L1"} == set(range(1, 6))


def test_language_matrix_rejects_missing_row() -> None:
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        normalize_language_summary(language_rows()[:-1])


def test_comparison_winner() -> None:
    row = {"system_a": "L1", "system_b": "L2", "wer_conclusion": "a_lower_wer"}
    assert comparison_winner(row, "system_a", "system_b") == "l1"


def test_final_selection_includes_c2_l1() -> None:
    language = normalize_language_summary(language_rows())
    l2 = [{"model_a": "c2", "model_b": "c4", "wer_conclusion": "a_lower_wer"}]
    rows = build_final_selection(language, system_tests(), l2)
    assert any(row["selected_model"] == "C2" and row["selected_language_system"] == "L1" for row in rows)


def test_claims_include_l3_boundary() -> None:
    language = normalize_language_summary(language_rows())
    claims = build_claim_register(language, [])
    assert any(row["status"] == "not_executed" and "L3" in row["thesis_safe_claim"] for row in claims)


def test_recognition_merge_and_parameter_reduction() -> None:
    publication = [{"model_id": model, "accuracy": .8, "ser": .2} for model in MODEL_ORDER]
    efficiency = [{"model_id": model, "parameters": 1000 if model == "g1" else 200, "model_size_mb": 1, "latency_ms": 1, "throughput_sps": 2, "peak_memory_mb": 3} for model in MODEL_ORDER]
    stability = [{"model_id": model, "mean_wer": .1, "std_wer": .01, "wer_range": .02} for model in MODEL_ORDER]
    errors = [{"model_id": model, "dominant_error": "deletion"} for model in MODEL_ORDER]
    result = build_recognition_table(publication, efficiency, stability, errors)
    c2 = next(row for row in result if row["model_id"] == "c2")
    assert c2["parameter_reduction_vs_g1_percent"] == pytest.approx(80.0)
    assert c2["role"] == "recommended_lightweight_recognizer"


def test_write_csv_uses_stable_header(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    write_csv_rows(path, [{"b": 2, "a": 1}], ["a", "b"])
    with path.open(newline="", encoding="utf-8") as handle:
        assert next(csv.reader(handle)) == ["a", "b"]

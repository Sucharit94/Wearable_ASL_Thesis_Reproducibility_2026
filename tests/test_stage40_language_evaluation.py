from __future__ import annotations

import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import pytest

from src.evaluation.stage40_language_evaluation import (
    MODEL_ORDER,
    aggregate_records,
    assemble_and_score,
    build_l2_ranking,
    build_summaries_and_tests,
    content_tokens,
    exact_mcnemar_p,
    holm_adjust,
    levenshtein_counts,
    load_request_map,
    load_stage38_outputs,
    load_v2_result_archive,
    normalized_text,
    score_sentence,
    sha256_file,
)


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_normalization_and_content_tokens() -> None:
    assert normalized_text("Happy Valentine’s_Day!") == "happy valentine's day"
    assert content_tokens("the rabbit is in the garden") == ("rabbit", "garden")


def test_text_hash_ignores_line_endings(tmp_path: Path) -> None:
    lf = tmp_path / "lf.csv"
    crlf = tmp_path / "crlf.csv"
    lf.write_bytes(b"a,b\n1,2\n")
    crlf.write_bytes(b"a,b\r\n1,2\r\n")
    assert sha256_file(lf) == sha256_file(crlf)


def test_deterministic_word_error_counts() -> None:
    counts = levenshtein_counts(
        ("are", "you", "ready", "for", "lunch"),
        ("you", "ready", "lunch"),
    )
    assert counts["total_edits"] == 2
    assert counts["deletions"] == 2


def test_content_precision_recall_and_unsupported_rate() -> None:
    row = score_sentence(
        "we see a cute rabbit",
        "we see cute rabbits outside",
        "we see cute rabbit",
    )
    assert row["content_true_positive"] == 3
    assert row["content_false_positive"] == 2
    assert row["content_false_negative"] == 1
    assert row["unsupported_content_tokens"] == 2


def test_exact_mcnemar_and_holm() -> None:
    a_only, b_only, p_value = exact_mcnemar_p([1, 1, 0, 0], [1, 0, 1, 1])
    assert (a_only, b_only) == (1, 2)
    assert 0.0 <= p_value <= 1.0
    adjusted = holm_adjust([0.01, 0.04, 0.2])
    assert adjusted == pytest.approx([0.03, 0.08, 0.2])


def make_v2_archive(path: Path, responses: list[dict[str, object]]) -> None:
    response_bytes = "".join(
        json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in responses
    ).encode("utf-8")
    digest = hashlib.sha256(response_bytes).hexdigest()
    manifest = {
        "status": "COMPLETE",
        "unique_responses": len(responses),
        "response_sha256": digest,
        "outer_references_exposed": False,
        "training_started": False,
    }
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("stage39b_v2_results/stage39b_v2_responses.jsonl", response_bytes)
        archive.writestr("stage39b_v2_results/stage39b_v2_responses.sha256", digest + "\n")
        archive.writestr(
            "stage39b_v2_results/stage39b_v2_execution_manifest.json",
            json.dumps(manifest),
        )


def test_archive_hash_and_reference_firewall(tmp_path: Path) -> None:
    archive = tmp_path / "stage39b_v2_results.zip"
    make_v2_archive(
        archive,
        [
            {
                "request_id": "r1",
                "model_id": "g1",
                "outer_fold": 1,
                "source_gloss": "hello",
                "sentence": "Hello",
                "status": "COMPLETE",
            }
        ],
    )
    responses, evidence = load_v2_result_archive(archive)
    assert list(responses) == ["r1"]
    assert evidence["response_count"] == 1


def test_flexible_stage38_loader(tmp_path: Path) -> None:
    path = tmp_path / "stage38.csv"
    write_csv(
        path,
        [
            {
                "recognizer": "G1",
                "fold": 1,
                "sample_id": "s1",
                "subject_id": "subject1",
                "target_id": 1,
                "predicted_gloss": "good morning",
                "target_english": "Good morning.",
                "language_system": "l0_raw_gloss",
                "predicted_english": "good morning",
            },
            {
                "recognizer": "G1",
                "fold": 1,
                "sample_id": "s1",
                "subject_id": "subject1",
                "target_id": 1,
                "predicted_gloss": "good morning",
                "target_english": "Good morning.",
                "language_system": "l1_retrieval",
                "predicted_english": "Good morning",
            },
        ],
    )
    rows = load_stage38_outputs(path)
    assert [row["system_id"] for row in rows] == ["L0", "L1"]


def synthetic_scored_rows() -> list[dict[str, object]]:
    stage38: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    responses: dict[str, dict[str, object]] = {}
    metadata = {
        "s1": {"subject": "subject1", "sentence_id": "1"},
        "s2": {"subject": "subject2", "sentence_id": "2"},
        "s3": {"subject": "subject3", "sentence_id": "3"},
    }
    for model_index, model in enumerate(MODEL_ORDER):
        for sample_index, sample_id in enumerate(("s1", "s2", "s3"), start=1):
            fold = sample_index if sample_index <= 3 else 1
            reference = ("are you ready", "good morning", "we see rabbit")[sample_index - 1]
            gloss = ("you ready", "good morning", "we see rabbit")[sample_index - 1]
            for system, output in (("L0", gloss), ("L1", reference)):
                stage38.append(
                    {
                        "model_id": model,
                        "outer_fold": fold,
                        "sample_id": sample_id,
                        "subject": "",
                        "sentence_id": "",
                        "source_gloss": gloss,
                        "reference_english": reference,
                        "system_id": system,
                        "output_english": output,
                    }
                )
            request_id = f"{model}-{sample_id}"
            request_rows.append(
                {
                    "request_id": request_id,
                    "model_id": model,
                    "outer_fold": fold,
                    "sample_id": sample_id,
                }
            )
            l2_output = reference if model_index < 3 else gloss
            responses[request_id] = {
                "request_id": request_id,
                "model_id": model,
                "outer_fold": fold,
                "source_gloss": gloss,
                "sentence": l2_output,
                "status": "COMPLETE",
            }
    return assemble_and_score(stage38, request_rows, responses, metadata)


def test_all_model_summary_statistics_and_ranking() -> None:
    scored = synthetic_scored_rows()
    assert len(scored) == 5 * 3 * 3
    summaries, system_tests, l2_tests = build_summaries_and_tests(
        scored, bootstrap_replicates=100, seed=123
    )
    assert len(summaries) == 15
    assert len(system_tests) == 15
    assert len(l2_tests) == 10
    ranking = build_l2_ranking(summaries)
    assert len(ranking) == 5
    assert ranking[0]["model_id"] == "g1"
    c2_l1 = next(row for row in summaries if row["model_id"] == "c2" and row["system_id"] == "L1")
    assert c2_l1["exact_sentence_accuracy"] == 1.0


def test_harm_rate_is_paired_to_l0() -> None:
    scored = synthetic_scored_rows()
    c4_l2 = [row for row in scored if row["model_id"] == "c4" and row["system_id"] == "L2"]
    aggregate = aggregate_records(c4_l2)
    assert aggregate["correction_harm_rate"] == 0.0


def test_complete_small_fixture_build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from scripts import build_stage40_end_to_end_evaluation as builder
    from scripts import verify_stage40_end_to_end_evaluation as verifier

    stage38_path = tmp_path / "stage38.csv"
    map_path = tmp_path / "map.csv"
    metadata_path = tmp_path / "metadata.csv"
    archive_path = tmp_path / "stage39b_v2_results.zip"
    output_root = tmp_path / "stage40"
    report_path = tmp_path / "STAGE_40.md"

    stage38_rows: list[dict[str, object]] = []
    request_rows: list[dict[str, object]] = []
    responses: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        for sample_number in range(1, 4):
            sample_id = f"s{sample_number}"
            reference = ("are you ready", "good morning", "we see rabbit")[sample_number - 1]
            gloss = ("you ready", "good morning", "we see rabbit")[sample_number - 1]
            for system, output in (("L0", gloss), ("L1", reference)):
                stage38_rows.append(
                    {
                        "model_id": model,
                        "outer_fold": sample_number,
                        "sample_id": sample_id,
                        "system_id": system,
                        "source_gloss": gloss,
                        "reference_english": reference,
                        "output_english": output,
                    }
                )
            request_id = f"{model}-{sample_id}"
            request_rows.append(
                {
                    "request_id": request_id,
                    "model_id": model,
                    "outer_fold": sample_number,
                    "sample_id": sample_id,
                }
            )
            responses.append(
                {
                    "request_id": request_id,
                    "model_id": model,
                    "outer_fold": sample_number,
                    "source_gloss": gloss,
                    "sentence": reference,
                    "status": "COMPLETE",
                }
            )
    write_csv(stage38_path, stage38_rows)
    write_csv(map_path, request_rows)
    write_csv(
        metadata_path,
        [
            {"sample_id": f"s{i}", "subject": f"subject{i}", "sentence_id": i}
            for i in range(1, 4)
        ],
    )
    make_v2_archive(archive_path, responses)

    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "build_stage40_end_to_end_evaluation",
            "--stage38", str(stage38_path),
            "--request-map", str(map_path),
            "--metadata", str(metadata_path),
            "--v2-results", str(archive_path),
            "--output-root", str(output_root),
            "--report", str(report_path),
            "--bootstrap-replicates", "100",
            "--allow-small-fixture",
        ],
    )
    builder.main()
    assert (output_root / "stage40_contract.json").is_file()
    assert len(load_request_map(map_path)) == 15
    contract = json.loads((output_root / "stage40_contract.json").read_text(encoding="utf-8"))
    assert contract["coverage"]["scored_system_rows"] == 45
    assert contract["execution"]["language_generation_started"] is False

    monkeypatch.setattr(verifier, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(verifier, "OUTPUT_ROOT", output_root)
    monkeypatch.setattr(verifier, "CONTRACT_PATH", output_root / "stage40_contract.json")
    monkeypatch.setattr(verifier, "CONTRACT_HASH_PATH", output_root / "stage40_contract.sha256")
    verifier.main()

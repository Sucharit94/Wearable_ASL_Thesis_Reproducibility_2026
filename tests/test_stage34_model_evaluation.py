from pathlib import Path

import pytest

from src.evaluation.stage34_analysis import (
    aggregate_records,
    prediction_identity,
    reference_length_bin,
)


def record(reference, prediction, substitutions, deletions, insertions):
    edits = substitutions + deletions + insertions
    return {
        "sample_id": "sample",
        "subject": "subject1",
        "sentence_id": 1,
        "reference_token_ids": reference,
        "predicted_token_ids": prediction,
        "substitutions": substitutions,
        "deletions": deletions,
        "insertions": insertions,
        "edit_distance": edits,
        "exact_match": edits == 0,
    }


def test_prediction_identity_is_read_from_portable_path():
    path = Path("root/artifacts/runs/known_signer/fold03/c4/predictions.jsonl")
    assert prediction_identity(path) == ("c4", 3)


@pytest.mark.parametrize(
    ("length", "expected"),
    [(1, "1"), (2, "2"), (3, "3"), (4, "4"), (5, "5+"), (8, "5+")],
)
def test_reference_length_bins(length, expected):
    assert reference_length_bin(length) == expected


def test_aggregate_metrics_include_error_and_length_diagnostics():
    rows = [
        record([1, 2], [1, 2], 0, 0, 0),
        record([3, 4], [3], 0, 1, 0),
        record([5], [], 0, 1, 0),
    ]
    result = aggregate_records(rows)
    assert result["samples"] == 3
    assert result["reference_tokens"] == 5
    assert result["hypothesis_tokens"] == 3
    assert result["deletions"] == 2
    assert result["total_edits"] == 2
    assert result["pooled_wer"] == pytest.approx(0.4)
    assert result["macro_sample_wer"] == pytest.approx(0.5)
    assert result["exact_sentence_accuracy"] == pytest.approx(1 / 3)
    assert result["blank_outputs"] == 1
    assert result["blank_output_rate"] == pytest.approx(1 / 3)
    assert result["hypothesis_reference_length_ratio"] == pytest.approx(0.6)


def test_empty_aggregate_is_rejected():
    with pytest.raises(ValueError):
        aggregate_records([])

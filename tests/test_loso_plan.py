from __future__ import annotations

from collections import Counter

import pytest

from src.training.loso_plan import (
    HELD_OUT_SUBJECTS,
    LOSO_IDS,
    MODEL_IDS,
    build_loso_assignment,
    build_registry_plan,
    build_run_matrix,
    canonical_json_sha256,
    validate_loso_assignment,
    validate_registry_plan,
    validate_run_matrix,
    validate_stage14_rows,
)


def synthetic_stage14() -> list[dict]:
    samples = []
    for subject in ("subject1", "subject2", "subject3"):
        for sentence in ("001", "002"):
            for repetition in range(1, 6):
                sample_id = f"{subject}_{sentence}_{repetition}"
                samples.append((sample_id, subject, sentence))

    rows = []
    for loso_id in LOSO_IDS:
        held_out = HELD_OUT_SUBJECTS[loso_id]
        for sample_id, subject, sentence in samples:
            rows.append(
                {
                    "loso_id": loso_id,
                    "held_out_subject": held_out,
                    "sample_id": sample_id,
                    "subject": subject,
                    "sentence_id": sentence,
                    "split_group_id": sample_id,
                    "role": "test" if subject == held_out else "train",
                }
            )
    return rows


def matrix() -> list[dict]:
    return build_run_matrix(
        model_config_hashes={model: model * 16 for model in MODEL_IDS},
        data_config_sha256="d" * 64,
        experiment_config_sha256="e" * 64,
        stage14_split_sha256="a" * 64,
        stage15_support_sha256="s" * 64,
        stage33_inner_split_sha256="i" * 64,
        data_manifest_sha256="m" * 64,
        label_vocab_sha256="v" * 64,
    )


def test_stage14_validation_covers_three_directions():
    summary = validate_stage14_rows(synthetic_stage14(), expected_recordings=30)
    assert summary["assignment_rows"] == 90
    assert summary["each_sample_outer_test_once"] is True


def test_stage14_rejects_held_out_leak():
    rows = synthetic_stage14()
    rows[0]["role"] = "train"
    with pytest.raises(ValueError, match="leaks"):
        validate_stage14_rows(rows, expected_recordings=30)


@pytest.mark.parametrize("loso_id", LOSO_IDS)
def test_inner_assignment_is_subject_disjoint(loso_id):
    assignments, summary = build_loso_assignment(synthetic_stage14(), loso_id)
    held_out = HELD_OUT_SUBJECTS[loso_id]
    assert all(
        row["subject"] != held_out
        for row in assignments
        if row["partition"] != "outer_test"
    )
    assert summary["held_out_subject_training_leaks"] == 0


def test_inner_assignment_is_deterministic():
    first, _ = build_loso_assignment(synthetic_stage14(), LOSO_IDS[0])
    second, _ = build_loso_assignment(synthetic_stage14(), LOSO_IDS[0])
    assert first == second


def test_atomic_group_crossing_is_rejected():
    assignments, _ = build_loso_assignment(synthetic_stage14(), LOSO_IDS[0])
    test_row = next(row for row in assignments if row["partition"] == "outer_test")
    train_row = next(row for row in assignments if row["partition"] == "inner_train")
    train_row["split_group_id"] = test_row["split_group_id"]
    with pytest.raises(ValueError, match="leakage"):
        validate_loso_assignment(assignments)


def test_run_matrix_has_frozen_order():
    runs = matrix()
    assert len(runs) == 15
    assert (runs[0]["loso_id"], runs[0]["model_id"]) == (LOSO_IDS[0], "g1")
    assert (runs[-1]["loso_id"], runs[-1]["model_id"]) == (LOSO_IDS[-1], "c4")
    assert validate_run_matrix(runs)["loso_major_order"] is True


def test_run_matrix_has_no_outer_scores():
    runs = matrix()
    assert Counter(run["status"] for run in runs) == {"PLANNED": 15}
    assert not any(run["outer_score_present"] for run in runs)


def test_registry_has_three_phases_per_evaluation():
    rows = build_registry_plan(matrix())
    assert len(rows) == 45
    assert validate_registry_plan(rows)["all_status_planned"] is True
    assert [row["phase"] for row in rows[:3]] == [
        "outer_evaluation", "inner_selection", "full_refit"
    ]


def test_registry_rejects_completed_status():
    rows = build_registry_plan(matrix())
    rows[0]["status"] = "COMPLETE"
    with pytest.raises(ValueError, match="not PLANNED"):
        validate_registry_plan(rows)


def test_canonical_hash_ignores_mapping_order():
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )

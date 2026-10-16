from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from src.training.known_signer_plan import (
    MODEL_IDS,
    OUTER_FOLDS,
    build_outer_fold_assignment,
    build_registry_plan,
    build_run_matrix,
    canonical_json_sha256,
    normalize_stage13_rows,
    validate_outer_fold_assignment,
    validate_registry_plan,
    validate_run_matrix,
    validate_stage13_rows,
)


def synthetic_stage13_rows() -> list[dict]:
    rows = []
    sample_index = 0

    for outer_fold in OUTER_FOLDS:
        for subject_index in range(1, 5):
            for sentence_index in range(1, 7):
                for repeat in range(3):
                    sample_index += 1
                    rows.append(
                        {
                            "sample_id": f"sample_{sample_index:04d}",
                            "subject": f"subject_{subject_index}",
                            "sentence_id": f"sentence_{sentence_index:02d}",
                            "stage12_split_group_id": (
                                f"group_{outer_fold}_{subject_index}_"
                                f"{sentence_index}_{repeat}"
                            ),
                            "outer_fold": outer_fold,
                        }
                    )

    # One exact-duplicate pair remains atomic inside Fold 1.
    duplicate_source = dict(rows[0])
    sample_index += 1
    duplicate_source["sample_id"] = f"sample_{sample_index:04d}"
    rows.append(duplicate_source)
    return rows


def matrix() -> list[dict]:
    return build_run_matrix(
        model_config_hashes={model_id: model_id * 64 for model_id in MODEL_IDS},
        data_config_sha256="a" * 64,
        experiment_config_sha256="b" * 64,
        stage13_split_sha256="c" * 64,
        stage32_inner_split_sha256="d" * 64,
        data_manifest_sha256="e" * 64,
        label_vocab_sha256="f" * 64,
    )


def test_normalize_accepts_documented_aliases():
    source = [
        {
            "sample_id": "s1",
            "subject_id": "p1",
            "template_id": "t1",
            "split_group_id": "g1",
            "outer_fold": "1",
        }
    ]
    assert normalize_stage13_rows(source) == [
        {
            "sample_id": "s1",
            "subject": "p1",
            "sentence_id": "t1",
            "stage12_split_group_id": "g1",
            "outer_fold": 1,
        }
    ]


def test_normalize_rejects_missing_identity_column():
    with pytest.raises(ValueError, match="resolve"):
        normalize_stage13_rows(
            [
                {
                    "sample_id": "s1",
                    "subject": "p1",
                    "sentence_id": "t1",
                    "outer_fold": 1,
                }
            ]
        )


def test_stage13_validation_reports_expected_structure():
    rows = synthetic_stage13_rows()
    summary = validate_stage13_rows(rows, expected_recordings=None)
    assert summary["recordings"] == 361
    assert summary["subjects"] == 4
    assert summary["sentences"] == 6
    assert summary["subject_sentence_strata"] == 24
    assert summary["split_groups"] == 360
    assert summary["duplicate_groups_crossing_outer_folds"] == 0


def test_stage13_rejects_duplicate_sample_id():
    rows = synthetic_stage13_rows()
    rows[-1]["sample_id"] = rows[0]["sample_id"]
    with pytest.raises(ValueError, match="not unique"):
        validate_stage13_rows(rows, expected_recordings=None)


def test_stage13_rejects_group_crossing_outer_folds():
    rows = synthetic_stage13_rows()
    rows[-1]["outer_fold"] = 2
    with pytest.raises(ValueError, match="cross"):
        validate_stage13_rows(rows, expected_recordings=None)


@pytest.mark.parametrize("outer_fold", OUTER_FOLDS)
def test_each_outer_fold_assignment_is_valid(outer_fold: int):
    rows = synthetic_stage13_rows()
    assignments, summary = build_outer_fold_assignment(rows, outer_fold)
    assert len(assignments) == len(rows)
    assert summary["outer_fold"] == outer_fold
    assert summary["duplicate_groups_crossing_partitions"] == 0
    assert summary["outer_test_assignment_mismatches"] == 0
    assert summary["subjects_missing_inner_partition"] == 0
    assert summary["strata_missing_inner_partition"] == 0
    assert summary["absolute_target_difference"] <= summary[
        "maximum_atomic_group_size"
    ]


def test_inner_assignment_is_deterministic():
    rows = synthetic_stage13_rows()
    first, first_summary = build_outer_fold_assignment(rows, 3)
    second, second_summary = build_outer_fold_assignment(rows, 3)
    assert first == second
    assert first_summary == second_summary


def test_duplicate_group_is_atomic():
    rows = synthetic_stage13_rows()
    assignments, _ = build_outer_fold_assignment(rows, 2)
    by_group = defaultdict(set)
    for row in assignments:
        by_group[row["stage12_split_group_id"]].add(row["partition"])
    assert all(len(partitions) == 1 for partitions in by_group.values())


def test_outer_test_is_exact_stage13_fold():
    rows = synthetic_stage13_rows()
    assignments, _ = build_outer_fold_assignment(rows, 4)
    expected = {row["sample_id"] for row in rows if row["outer_fold"] == 4}
    observed = {
        row["sample_id"]
        for row in assignments
        if row["partition"] == "outer_test"
    }
    assert observed == expected


def test_assignment_validator_rejects_group_crossing():
    rows = synthetic_stage13_rows()
    assignments, _ = build_outer_fold_assignment(rows, 2)
    group_counts = Counter(row["stage12_split_group_id"] for row in assignments)
    duplicate_group = next(key for key, count in group_counts.items() if count == 2)
    members = [
        row for row in assignments if row["stage12_split_group_id"] == duplicate_group
    ]
    members[0]["partition"] = "inner_validation"
    members[1]["partition"] = "inner_train"
    with pytest.raises(ValueError, match="cross"):
        validate_outer_fold_assignment(assignments)


def test_canonical_json_hash_ignores_mapping_order():
    assert canonical_json_sha256({"a": 1, "b": 2}) == canonical_json_sha256(
        {"b": 2, "a": 1}
    )


def test_run_matrix_has_frozen_fold_major_order():
    runs = matrix()
    summary = validate_run_matrix(runs)
    assert summary["runs"] == 25
    assert [(row["outer_fold"], row["model_id"]) for row in runs] == [
        (fold, model_id)
        for fold in OUTER_FOLDS
        for model_id in MODEL_IDS
    ]


def test_run_matrix_identities_and_directories_are_unique():
    runs = matrix()
    assert len({row["experiment_id"] for row in runs}) == 25
    assert len({row["run_directory"] for row in runs}) == 25
    assert all(row["required_code_tag"] == "stage32-known-runner-v1" for row in runs)


def test_run_matrix_rejects_changed_order():
    runs = matrix()
    runs[0], runs[1] = runs[1], runs[0]
    with pytest.raises(ValueError, match="fold-major"):
        validate_run_matrix(runs)


def test_registry_plan_has_three_rows_per_evaluation():
    registry_rows = build_registry_plan(matrix())
    summary = validate_registry_plan(registry_rows)
    assert summary == {
        "registry_rows": 75,
        "outer_evaluation_rows": 25,
        "inner_selection_rows": 25,
        "full_refit_rows": 25,
        "unique_registry_ids": 75,
        "all_status_planned": True,
    }


def test_registry_plan_rejects_phase_order_change():
    registry_rows = build_registry_plan(matrix())
    registry_rows[1], registry_rows[2] = registry_rows[2], registry_rows[1]
    registry_rows[1]["registry_order"] = 2
    registry_rows[2]["registry_order"] = 3
    with pytest.raises(ValueError, match="phase order"):
        validate_registry_plan(registry_rows)

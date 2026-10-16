from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.training.loso_plan import LOSO_IDS
from src.training.loso_runner import (
    EXPECTED_COUNTS,
    REGISTRY_PHASES,
    build_dry_run_report,
    initialize_or_validate_live_registry,
    load_loso_partitions,
    load_support_map,
    next_evaluation_row,
    score_prediction_subset,
    validate_runner_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = PROJECT_ROOT / "artifacts/splits/stage33_loso_inner_assignment.csv"
REGISTRY = PROJECT_ROOT / "artifacts/registry/stage33_loso_run_registry.csv"


@pytest.mark.parametrize("loso_id", LOSO_IDS)
def test_frozen_loso_partitions(loso_id):
    partition = load_loso_partitions(ASSIGNMENT, loso_id)
    expected = EXPECTED_COUNTS[loso_id]
    assert len(partition.inner_train) == expected["inner_train"]
    assert len(partition.inner_validation) == expected["inner_validation"]
    assert len(partition.outer_test) == expected["outer_test"]
    assert partition.to_summary()["held_out_training_leaks"] == 0


def test_support_map_exactly_covers_subject1_test():
    partition = load_loso_partitions(ASSIGNMENT, "loso_subject1")
    mapping = load_support_map(
        PROJECT_ROOT / "artifacts/splits/stage15_loso_support_assignment.csv",
        partition,
    )
    assert len(mapping) == 6000


def test_empty_support_slice_has_explicit_null_rates():
    metrics = score_prediction_subset([])
    assert metrics["samples"] == 0
    assert metrics["pooled_wer"] is None
    assert metrics["exact_sentence_accuracy"] is None


def test_support_subset_scoring_uses_token_edits():
    rows = [
        {"reference_token_ids": [1, 2], "predicted_token_ids": [1, 2]},
        {"reference_token_ids": [3], "predicted_token_ids": []},
    ]
    metrics = score_prediction_subset(rows)
    assert metrics["samples"] == 2
    assert metrics["total_edits"] == 1
    assert metrics["pooled_wer"] == pytest.approx(1 / 3)


def test_runner_config_freezes_float32_and_support_views():
    payload = json.loads(
        (PROJECT_ROOT / "configs/experiments/stage33_runner.json").read_text(
            encoding="utf-8-sig"
        )
    )
    validated = validate_runner_configuration(payload)
    assert validated["precision"]["automatic_mixed_precision"] is False
    assert validated["reporting"]["primary_metric"] == "loso_all_pooled_wer"


def test_live_registry_initialization_and_order(tmp_path: Path):
    live = tmp_path / "registry.csv"
    rows = initialize_or_validate_live_registry(
        REGISTRY, live, code_commit="010256a"
    )
    assert len(rows) == 45
    assert next_evaluation_row(rows)["experiment_id"] == (
        "g1_loso_subject1_seed20260826_v1"
    )
    assert [row["phase"] for row in rows[:3]] == list(REGISTRY_PHASES)


def test_completed_registry_has_no_next_evaluation(tmp_path: Path):
    live = tmp_path / "registry.csv"
    rows = initialize_or_validate_live_registry(
        REGISTRY, live, code_commit="010256a"
    )
    for row in rows:
        row["status"] = "COMPLETE"
    assert next_evaluation_row(rows) is None


def test_dry_run_never_trains_scores_or_loads_outer_test():
    report = build_dry_run_report(PROJECT_ROOT)
    assert report["passed"] is True
    assert report["matrix"]["outer_evaluations"] == 15
    assert report["matrix"]["registry_rows"] == 45
    assert report["leakage_barrier"]["training_started"] is False
    assert report["leakage_barrier"]["scores_computed"] is False
    assert report["leakage_barrier"]["held_out_subject_used_for_training"] is False


def test_checkpoint_load_explicitly_sets_trusted_mode():
    source = (PROJECT_ROOT / "src/training/checkpoint.py").read_text(
        encoding="utf-8-sig"
    )
    assert "weights_only=False" in source


def test_parent_csv_finalization_uses_union_schema():
    source = (PROJECT_ROOT / "src/training/known_signer_runner.py").read_text(
        encoding="utf-8-sig"
    )
    assert "combined_fields = tuple(" in source
    assert "for row in combined_log" in source


@pytest.mark.parametrize(
    ("data_newline", "recorded_newline"),
    (("\n", "\r\n"), ("\r\n", "\n")),
)
def test_portable_hash_accepts_lf_and_crlf_equivalence(
    tmp_path: Path,
    data_newline: str,
    recorded_newline: str,
):
    import hashlib

    from src.training.loso_runner import (
        require_portable_recorded_hash,
    )

    logical_lines = (
        "header",
        "value",
    )

    data_path = tmp_path / "frozen.csv"
    hash_path = tmp_path / "frozen.sha256"

    data_bytes = (
        data_newline
        .join(logical_lines)
        .encode("utf-8")
        + data_newline.encode("utf-8")
    )

    recorded_bytes = (
        recorded_newline
        .join(logical_lines)
        .encode("utf-8")
        + recorded_newline.encode("utf-8")
    )

    data_path.write_bytes(
        data_bytes
    )

    expected = hashlib.sha256(
        recorded_bytes
    ).hexdigest()

    hash_path.write_text(
        f"{expected}  frozen.csv\n",
        encoding="utf-8",
    )

    assert require_portable_recorded_hash(
        data_path,
        hash_path,
    ) == expected


def test_stage33_contract_uses_canonical_implementation_hashes():
    builder = (
        PROJECT_ROOT
        / "scripts"
        / "build_stage33_runner_contract.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    verifier = (
        PROJECT_ROOT
        / "scripts"
        / "verify_stage33_runner.py"
    ).read_text(
        encoding="utf-8-sig"
    )

    canonical_expression = (
        "portable_sha256(PROJECT_ROOT / relative)[1]"
    )

    assert canonical_expression in builder
    assert canonical_expression in verifier

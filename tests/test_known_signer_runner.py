from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from src.training.known_signer_runner import (
    STAGE31_COMMIT,
    build_dry_run_report,
    build_experiment_identity,
    compact_early_stopping_due,
    compact_warmup_epochs,
    deterministic_epoch_indices,
    ensure_external_artifact_root,
    gradient_accumulation_steps,
    load_known_signer_partitions,
    next_evaluation_row,
    physical_batch_candidates,
    refit_learning_rate_multiplier,
    select_best_epoch,
    set_live_registry_status,
    initialize_or_validate_live_registry,
    validate_runner_configuration,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage32_known_inner_assignment.csv"
)
REGISTRY_PATH = PROJECT_ROOT / "artifacts" / "registry" / "run_registry.csv"


def read_registry() -> list[dict[str, str]]:
    with REGISTRY_PATH.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def record(epoch: int, wer: float, accuracy: float, loss: float) -> dict:
    return {
        "epoch": epoch,
        "validation_wer": wer,
        "validation_exact_sentence_accuracy": accuracy,
        "validation_ctc_loss": loss,
    }


@pytest.mark.parametrize("outer_fold", [1, 2, 3, 4, 5])
def test_frozen_partition_counts_and_disjointness(outer_fold: int):
    partitions = load_known_signer_partitions(ASSIGNMENT_PATH, outer_fold)
    assert len(partitions.inner_train) == 5328
    assert len(partitions.inner_validation) == 592
    assert len(partitions.outer_training_pool) == 5920
    assert len(partitions.outer_test) == 1480
    assert not set(partitions.inner_train) & set(partitions.inner_validation)
    assert not set(partitions.outer_training_pool) & set(partitions.outer_test)


def test_epoch_selection_uses_wer_first():
    selected = select_best_epoch(
        [record(1, 0.4, 0.9, 1.0), record(2, 0.3, 0.1, 9.0)]
    )
    assert selected["epoch"] == 2


def test_epoch_selection_uses_accuracy_second():
    selected = select_best_epoch(
        [record(1, 0.3, 0.2, 1.0), record(2, 0.3, 0.4, 2.0)]
    )
    assert selected["epoch"] == 2


def test_epoch_selection_uses_loss_third():
    selected = select_best_epoch(
        [record(1, 0.3, 0.4, 2.0), record(2, 0.3, 0.4, 1.0)]
    )
    assert selected["epoch"] == 2


def test_epoch_selection_uses_earlier_epoch_last():
    selected = select_best_epoch(
        [record(2, 0.3, 0.4, 1.0), record(1, 0.3, 0.4, 1.0)]
    )
    assert selected["epoch"] == 1


def test_compact_early_stop_respects_minimum_epoch_and_patience():
    history = [record(1, 0.2, 0.5, 1.0)]
    history.extend(record(epoch, 0.3, 0.4, 1.1) for epoch in range(2, 16))
    assert compact_early_stopping_due(
        history,
        current_epoch=14,
        minimum_epoch=15,
        patience=8,
    ) is False
    assert compact_early_stopping_due(
        history,
        current_epoch=15,
        minimum_epoch=15,
        patience=8,
    ) is True


@pytest.mark.parametrize(
    ("selected_epochs", "expected"),
    [(1, 1), (10, 1), (11, 2), (40, 4), (50, 5), (80, 5)],
)
def test_compact_warmup_formula(selected_epochs: int, expected: int):
    assert compact_warmup_epochs(selected_epochs) == expected


def test_refit_schedule_has_frozen_endpoints():
    assert refit_learning_rate_multiplier(1, 40) == pytest.approx(0.25)
    assert refit_learning_rate_multiplier(4, 40) == pytest.approx(1.0)
    assert refit_learning_rate_multiplier(40, 40) == pytest.approx(0.01)


def test_oom_batch_candidates_and_accumulation_preserve_effective_batch():
    assert physical_batch_candidates(16) == (16, 8, 4, 2, 1)
    assert [gradient_accumulation_steps(16, value) for value in (16, 8, 4, 2, 1)] == [
        1,
        2,
        4,
        8,
        16,
    ]


def test_deterministic_epoch_shuffle_is_reproducible():
    first = deterministic_epoch_indices(50, seed=20260826, epoch=1)
    second = deterministic_epoch_indices(50, seed=20260826, epoch=1)
    different = deterministic_epoch_indices(50, seed=20260826, epoch=2)
    assert first == second
    assert first != different
    assert sorted(first) == list(range(50))


def test_external_artifact_root_cannot_be_inside_project(tmp_path: Path):
    with pytest.raises(ValueError, match="outside"):
        ensure_external_artifact_root(tmp_path, tmp_path / "artifacts")
    external = tmp_path.parent / f"{tmp_path.name}_external"
    assert ensure_external_artifact_root(tmp_path, external) == external.resolve()


def test_all_registry_rows_form_valid_frozen_identities():
    identities = [
        build_experiment_identity(row, STAGE31_COMMIT)
        for row in read_registry()
    ]
    assert len(identities) == 75
    assert len({identity.fingerprint for identity in identities}) == 75


def test_next_evaluation_is_first_fold_g1():
    row = next_evaluation_row(read_registry())
    assert row is not None
    assert row["experiment_id"] == "g1_known_fold01_seed20260826_v1"


def test_live_registry_is_external_and_transition_checked(tmp_path: Path):
    live_path = tmp_path / "run_registry.csv"
    rows = initialize_or_validate_live_registry(
        REGISTRY_PATH,
        live_path,
        code_commit=STAGE31_COMMIT,
    )
    assert len(rows) == 75
    first_id = rows[0]["registry_id"]
    updated = set_live_registry_status(live_path, first_id, "RUNNING")
    assert updated["status"] == "RUNNING"
    with pytest.raises(ValueError, match="Invalid run-status transition"):
        set_live_registry_status(live_path, first_id, "PLANNED")


def test_static_dry_run_never_trains_or_scores():
    report = build_dry_run_report(PROJECT_ROOT)
    assert report["passed"] is True
    assert report["matrix"]["outer_evaluations"] == 25
    assert report["matrix"]["registry_rows"] == 75
    assert report["leakage_barrier"]["training_started"] is False
    assert report["leakage_barrier"]["scores_computed"] is False
    assert report["leakage_barrier"]["checkpoints_created"] is False


def test_all_models_use_frozen_float32_execution_policy():
    path = PROJECT_ROOT / "configs" / "experiments" / "stage32_runner.json"
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    validated = validate_runner_configuration(payload)
    assert validated["precision"]["mode"] == "float32"
    assert validated["precision"]["automatic_mixed_precision"] is False
    assert tuple(validated["precision"]["applies_to_models"]) == (
        "g1",
        "c1",
        "c2",
        "c3",
        "c4",
    )

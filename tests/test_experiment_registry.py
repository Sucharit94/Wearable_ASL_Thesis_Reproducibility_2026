from __future__ import annotations

from dataclasses import replace

import pytest

from src.training.checkpoint import (
    build_checkpoint_envelope,
    checkpoint_metadata,
    validate_checkpoint_envelope,
    validate_checkpoint_metadata,
    write_checkpoint_metadata,
)
from src.training.registry import (
    REQUIRED_COMPLETE_ARTIFACTS,
    ExperimentIdentity,
    append_registry_event,
    canonical_json_sha256,
    read_registry_events,
    resume_decision,
    validate_identity_match,
    validate_registry_events,
    validate_status_transition,
    write_completion_marker,
    write_or_validate_identity,
)


def make_identity(
    *,
    model_id: str = "c3",
    config_hash: str = "c" * 64,
) -> ExperimentIdentity:

    return ExperimentIdentity(
        experiment_id=(
            f"{model_id}_known_fold02_"
            "seed20260826_v1"
        ),
        project_generation=(
            "fresh_start_2026_v1"
        ),
        model_id=model_id,
        phase="outer_evaluation",
        split_id="known_fold02_v1",
        data_manifest_sha256=(
            "a" * 64
        ),
        label_vocab_sha256=(
            "b" * 64
        ),
        config_sha256=config_hash,
        code_commit="abcdef1",
        seed=20260826,
        normalization=(
            "train_partition_feature_zscore"
        ),
        augmentation="none",
        hardware_required=(
            "NVIDIA_Tesla_T4"
        ),
    )


def make_envelope(
    identity: ExperimentIdentity,
):

    return build_checkpoint_envelope(
        identity,
        epoch=7,
        selected_epochs=None,
        model_state={
            "weight": [
                1,
                2,
            ]
        },
        optimizer_state={
            "step": 7
        },
        scheduler_state={
            "step": 7
        },
        amp_scaler_state=None,
        rng_state={
            "python": "python-state",
            "numpy": "numpy-state",
            "torch_cpu": "torch-state",
        },
    )


def append_complete_event_sequence(
    run_directory,
    identity,
):

    path = (
        run_directory
        / "events.jsonl"
    )

    append_registry_event(
        path,
        identity,
        event="PLAN",
        status="PLANNED",
    )

    append_registry_event(
        path,
        identity,
        event="START",
        status="RUNNING",
    )

    append_registry_event(
        path,
        identity,
        event="COMPLETE",
        status="COMPLETE",
    )


def populate_complete_artifacts(
    run_directory,
    identity,
    envelope,
):

    write_or_validate_identity(
        run_directory,
        identity,
    )

    write_checkpoint_metadata(
        run_directory
        / "last_checkpoint.json",
        envelope,
        identity,
    )

    for name in (
        REQUIRED_COMPLETE_ARTIFACTS
    ):

        path = (
            run_directory
            / name
        )

        if path.exists():
            continue

        path.write_text(
            "verified\n",
            encoding="utf-8",
        )


def test_canonical_config_hash_ignores_key_order():

    first = {
        "model": {
            "id": "c3",
            "layers": 2,
        },
        "seed": 20260826,
    }

    second = {
        "seed": 20260826,
        "model": {
            "layers": 2,
            "id": "c3",
        },
    }

    assert canonical_json_sha256(
        first
    ) == canonical_json_sha256(
        second
    )


def test_invalid_sha256_rejected():

    with pytest.raises(
        ValueError,
        match="SHA-256",
    ):
        make_identity(
            config_hash="invalid"
        )


def test_identity_write_and_roundtrip(
    tmp_path,
):

    identity = make_identity()

    path = write_or_validate_identity(
        tmp_path,
        identity,
    )

    assert path.is_file()

    write_or_validate_identity(
        tmp_path,
        identity,
    )


def test_identity_mismatch_rejected(
    tmp_path,
):

    identity = make_identity()

    write_or_validate_identity(
        tmp_path,
        identity,
    )

    changed = replace(
        identity,
        model_id="c2",
    )

    with pytest.raises(
        ValueError,
        match="mismatch",
    ):
        write_or_validate_identity(
            tmp_path,
            changed,
        )


def test_valid_status_transitions():

    validate_status_transition(
        "PLANNED",
        "RUNNING",
    )

    validate_status_transition(
        "RUNNING",
        "INTERRUPTED",
    )

    validate_status_transition(
        "INTERRUPTED",
        "RUNNING",
    )

    validate_status_transition(
        "RUNNING",
        "COMPLETE",
    )


def test_terminal_status_cannot_resume():

    with pytest.raises(
        ValueError,
        match="Invalid",
    ):
        validate_status_transition(
            "COMPLETE",
            "RUNNING",
        )


def test_append_and_validate_event_log(
    tmp_path,
):

    identity = make_identity()

    path = (
        tmp_path
        / "events.jsonl"
    )

    append_registry_event(
        path,
        identity,
        event="PLAN",
        status="PLANNED",
    )

    append_registry_event(
        path,
        identity,
        event="START",
        status="RUNNING",
    )

    append_registry_event(
        path,
        identity,
        event="EPOCH_END",
        status="RUNNING",
        details={
            "epoch": 1
        },
    )

    append_registry_event(
        path,
        identity,
        event="INTERRUPT",
        status="INTERRUPTED",
    )

    records = validate_registry_events(
        path,
        identity,
    )

    assert len(records) == 4
    assert records[-1]["status"] == (
        "INTERRUPTED"
    )


def test_checkpoint_envelope_validates():

    identity = make_identity()
    envelope = make_envelope(
        identity
    )

    assert validate_checkpoint_envelope(
        envelope,
        identity,
    ) == 7


def test_checkpoint_identity_mismatch_rejected():

    identity = make_identity()
    envelope = make_envelope(
        identity
    )

    changed = replace(
        identity,
        config_sha256="d" * 64,
    )

    with pytest.raises(
        ValueError,
        match="mismatch",
    ):
        validate_checkpoint_envelope(
            envelope,
            changed,
        )


def test_checkpoint_missing_rng_rejected():

    identity = make_identity()

    with pytest.raises(
        ValueError,
        match="rng_state",
    ):
        build_checkpoint_envelope(
            identity,
            epoch=1,
            selected_epochs=None,
            model_state={
                "weight": 1
            },
            optimizer_state={
                "step": 1
            },
            rng_state={
                "python": "state",
            },
        )


def test_completion_refuses_missing_artifacts(
    tmp_path,
):

    identity = make_identity()

    write_or_validate_identity(
        tmp_path,
        identity,
    )

    append_complete_event_sequence(
        tmp_path,
        identity,
    )

    with pytest.raises(
        ValueError,
        match="missing",
    ):
        write_completion_marker(
            tmp_path,
            identity,
        )


def test_completion_written_after_artifacts(
    tmp_path,
):

    identity = make_identity()
    envelope = make_envelope(
        identity
    )

    append_complete_event_sequence(
        tmp_path,
        identity,
    )

    populate_complete_artifacts(
        tmp_path,
        identity,
        envelope,
    )

    path = write_completion_marker(
        tmp_path,
        identity,
    )

    assert path.is_file()


def test_new_run_resume_decision_is_start(
    tmp_path,
):

    identity = make_identity()

    assert resume_decision(
        tmp_path,
        identity,
    ) == "START"


def test_resume_then_skip_complete(
    tmp_path,
):

    identity = make_identity()
    envelope = make_envelope(
        identity
    )

    write_or_validate_identity(
        tmp_path,
        identity,
    )

    (
        tmp_path
        / "last.pt"
    ).write_bytes(
        b"trusted-checkpoint"
    )

    write_checkpoint_metadata(
        tmp_path
        / "last_checkpoint.json",
        envelope,
        identity,
    )

    assert resume_decision(
        tmp_path,
        identity,
    ) == "RESUME"

    append_complete_event_sequence(
        tmp_path,
        identity,
    )

    populate_complete_artifacts(
        tmp_path,
        identity,
        envelope,
    )

    write_completion_marker(
        tmp_path,
        identity,
    )

    assert resume_decision(
        tmp_path,
        identity,
    ) == "SKIP_COMPLETE"

    assert len(
        read_registry_events(
            tmp_path
            / "events.jsonl"
        )
    ) == 3

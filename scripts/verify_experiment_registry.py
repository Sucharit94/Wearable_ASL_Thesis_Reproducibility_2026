from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

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
    validate_registry_events,
    validate_status_transition,
    write_completion_marker,
    write_or_validate_identity,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE23_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
    / "stage23_recognition_metrics_contract.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "registry"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage24_experiment_registry_contract.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_24_EXPERIMENT_REGISTRY.md"
)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 24 — EXPERIMENT IDENTITY, "
        "REGISTRY AND RESUME SAFETY"
    )
    print("=" * 78)

    stage23 = json.loads(
        STAGE23_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage23_dependency = all(
        [
            int(
                stage23[
                    "stage"
                ]
            ) == 23,
            int(
                stage23[
                    "perfect_control"
                ][
                    "total_edits"
                ]
            ) == 0,
            stage23[
                "dependencies"
            ][
                "stage21_decoder_ignores_padding"
            ]
            is True,
            stage23[
                "dependencies"
            ][
                "stage22_same_length_for_loss_decoder"
            ]
            is True,
        ]
    )

    if not stage23_dependency:
        raise RuntimeError(
            "Stage-23 dependency failed."
        )

    config_a = {
        "model": {
            "id": "c3",
            "layers": 2,
            "d_model": 96,
        },
        "seed": 20260826,
    }

    config_b = {
        "seed": 20260826,
        "model": {
            "d_model": 96,
            "layers": 2,
            "id": "c3",
        },
    }

    config_hash_a = (
        canonical_json_sha256(
            config_a
        )
    )

    config_hash_b = (
        canonical_json_sha256(
            config_b
        )
    )

    canonical_hash_stable = (
        config_hash_a
        == config_hash_b
    )

    if not canonical_hash_stable:
        raise RuntimeError(
            "Canonical config hashing failed."
        )

    identity = ExperimentIdentity(
        experiment_id=(
            "c3_known_fold02_"
            "seed20260826_v1"
        ),
        project_generation=(
            "fresh_start_2026_v1"
        ),
        model_id="c3",
        phase="outer_evaluation",
        split_id="known_fold02_v1",
        data_manifest_sha256=(
            "a" * 64
        ),
        label_vocab_sha256=(
            "b" * 64
        ),
        config_sha256=(
            config_hash_a
        ),
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

    envelope = build_checkpoint_envelope(
        identity,
        epoch=7,
        selected_epochs=None,
        model_state={
            "weight": [
                1,
                2,
                3,
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

    checkpoint_epoch = (
        validate_checkpoint_envelope(
            envelope,
            identity,
        )
    )

    metadata = checkpoint_metadata(
        envelope,
        identity,
    )

    metadata_epoch = (
        validate_checkpoint_metadata(
            metadata,
            identity,
        )
    )

    mismatch_rejected = False

    changed_identity = replace(
        identity,
        config_sha256="d" * 64,
    )

    try:

        validate_checkpoint_envelope(
            envelope,
            changed_identity,
        )

    except ValueError:

        mismatch_rejected = True

    if not mismatch_rejected:
        raise RuntimeError(
            "Changed configuration identity "
            "was not rejected."
        )

    valid_transitions = [
        (
            "PLANNED",
            "RUNNING",
        ),
        (
            "RUNNING",
            "INTERRUPTED",
        ),
        (
            "INTERRUPTED",
            "RUNNING",
        ),
        (
            "RUNNING",
            "COMPLETE",
        ),
    ]

    for current, next_status in (
        valid_transitions
    ):

        validate_status_transition(
            current,
            next_status,
        )

    terminal_resume_rejected = False

    try:

        validate_status_transition(
            "COMPLETE",
            "RUNNING",
        )

    except ValueError:

        terminal_resume_rejected = True

    if not terminal_resume_rejected:
        raise RuntimeError(
            "Terminal run was allowed to resume."
        )

    with tempfile.TemporaryDirectory(
        prefix="stage24_registry_"
    ) as temporary_directory:

        run_directory = (
            Path(
                temporary_directory
            )
            / identity.experiment_id
        )

        start_decision = resume_decision(
            run_directory,
            identity,
        )

        (
            run_directory
            / "last.pt"
        ).write_bytes(
            b"trusted-stage24-checkpoint"
        )

        write_checkpoint_metadata(
            run_directory
            / "last_checkpoint.json",
            envelope,
            identity,
        )

        resume_result = resume_decision(
            run_directory,
            identity,
        )

        events_path = (
            run_directory
            / "events.jsonl"
        )

        event_sequence = [
            (
                "PLAN",
                "PLANNED",
            ),
            (
                "START",
                "RUNNING",
            ),
            (
                "EPOCH_END",
                "RUNNING",
            ),
            (
                "INTERRUPT",
                "INTERRUPTED",
            ),
            (
                "RESUME",
                "RUNNING",
            ),
            (
                "COMPLETE",
                "COMPLETE",
            ),
        ]

        for index, (
            event,
            status,
        ) in enumerate(
            event_sequence
        ):

            append_registry_event(
                events_path,
                identity,
                event=event,
                status=status,
                details={
                    "sequence": index
                },
            )

        events = validate_registry_events(
            events_path,
            identity,
        )

        completion_guard_rejected_missing = (
            False
        )

        try:

            write_completion_marker(
                run_directory,
                identity,
            )

        except ValueError:

            completion_guard_rejected_missing = (
                True
            )

        if not completion_guard_rejected_missing:
            raise RuntimeError(
                "Completion marker ignored "
                "missing artifacts."
            )

        write_or_validate_identity(
            run_directory,
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

        completion_path = (
            write_completion_marker(
                run_directory,
                identity,
            )
        )

        complete_decision = (
            resume_decision(
                run_directory,
                identity,
            )
        )

        event_count = len(
            read_registry_events(
                events_path
            )
        )

        all_required_present = all(
            (
                run_directory
                / name
            ).is_file()
            and (
                run_directory
                / name
            ).stat().st_size > 0
            for name in (
                REQUIRED_COMPLETE_ARTIFACTS
            )
        )

        completion_written = (
            completion_path.is_file()
        )

    implementation_hashes = {
        "registry": sha256_file(
            PROJECT_ROOT
            / "src"
            / "training"
            / "registry.py"
        ),
        "checkpoint": sha256_file(
            PROJECT_ROOT
            / "src"
            / "training"
            / "checkpoint.py"
        ),
        "tests": sha256_file(
            PROJECT_ROOT
            / "tests"
            / "test_experiment_registry.py"
        ),
        "verifier": sha256_file(
            PROJECT_ROOT
            / "scripts"
            / "verify_experiment_registry.py"
        ),
    }

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 24,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "identity": {
            "experiment_id": (
                identity.experiment_id
            ),
            "model_id": (
                identity.model_id
            ),
            "phase": (
                identity.phase
            ),
            "split_id": (
                identity.split_id
            ),
            "fingerprint": (
                identity.fingerprint
            ),
            "canonical_config_hash_stable": (
                canonical_hash_stable
            ),
            "changed_config_rejected": (
                mismatch_rejected
            ),
        },
        "checkpoint": {
            "schema_version": 1,
            "verified_epoch": (
                checkpoint_epoch
            ),
            "metadata_epoch": (
                metadata_epoch
            ),
            "model_state_required": True,
            "optimizer_state_required": True,
            "rng_states_required": [
                "python",
                "numpy",
                "torch_cpu",
            ],
            "atomic_last_checkpoint_rule": True,
        },
        "registry": {
            "allowed_statuses": [
                "PLANNED",
                "RUNNING",
                "INTERRUPTED",
                "COMPLETE",
                "FAILED_SAFE",
                "INVALID",
            ],
            "event_count": (
                event_count
            ),
            "terminal_resume_rejected": (
                terminal_resume_rejected
            ),
            "append_only_events": True,
        },
        "resume": {
            "new_run_decision": (
                start_decision
            ),
            "checkpoint_run_decision": (
                resume_result
            ),
            "complete_run_decision": (
                complete_decision
            ),
            "identity_match_required": True,
        },
        "completion": {
            "required_artifact_count": len(
                REQUIRED_COMPLETE_ARTIFACTS
            ),
            "missing_artifact_guard_passed": (
                completion_guard_rejected_missing
            ),
            "all_required_present": (
                all_required_present
            ),
            "completion_written": (
                completion_written
            ),
            "completion_only_after_metrics": True,
        },
        "dependencies": {
            "stage23_passed": (
                stage23_dependency
            ),
            "stage21_padding_safe": (
                stage23[
                    "dependencies"
                ][
                    "stage21_decoder_ignores_padding"
                ]
            ),
            "stage22_length_safe": (
                stage23[
                    "dependencies"
                ][
                    "stage22_same_length_for_loss_decoder"
                ]
            ),
            "stage23_perfect_control_edits": (
                stage23[
                    "perfect_control"
                ][
                    "total_edits"
                ]
            ),
        },
        "hashes": (
            implementation_hashes
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 24 — Experiment Registry and Resume Safety

## Experiment identity

Every run is bound to an immutable identity containing model, phase, split,
manifest hash, vocabulary hash, configuration hash, Git commit, seed,
normalization, augmentation and required hardware.

Configuration hash stable across JSON key order:

{canonical_hash_stable}

Changed configuration rejected:

{mismatch_rejected}

## Checkpoint contract

Checkpoint schema version:

1

Verified epoch:

{checkpoint_epoch}

Required RNG states:

- Python
- NumPy
- PyTorch CPU
- PyTorch CUDA when execution occurs on the reported T4

The checkpoint contains model, optimizer, optional scheduler/AMP scaler,
epoch-selection information and RNG states.

## Resume decisions

New run:

{start_decision}

Interrupted run with matching checkpoint:

{resume_result}

Completed run:

{complete_decision}

## Completion safety

Required run artifacts:

{len(REQUIRED_COMPLETE_ARTIFACTS)}

Missing-artifact guard passed:

{completion_guard_rejected_missing}

Completion marker written:

{completion_written}

## Frozen dependency chain

Stage 21 padding-safe decoder:

{stage23["dependencies"]["stage21_decoder_ignores_padding"]}

Stage 22 length-safe processing:

{stage23["dependencies"]["stage22_same_length_for_loss_decoder"]}

Stage 23 perfect-control metric errors:

{stage23["perfect_control"]["total_edits"]}
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("EXPERIMENT IDENTITY")
    print("-" * 78)
    print(
        "Experiment ID:",
        identity.experiment_id,
    )
    print(
        "Identity fingerprint:",
        identity.fingerprint,
    )
    print(
        "Canonical config hash stable:",
        canonical_hash_stable,
    )
    print(
        "Changed config rejected:",
        mismatch_rejected,
    )

    print()
    print("CHECKPOINT CONTRACT")
    print("-" * 78)
    print(
        "Schema version:",
        1,
    )
    print(
        "Verified epoch:",
        checkpoint_epoch,
    )
    print(
        "Metadata epoch:",
        metadata_epoch,
    )
    print(
        "Required RNG states:",
        [
            "python",
            "numpy",
            "torch_cpu",
        ],
    )

    print()
    print("REGISTRY CONTRACT")
    print("-" * 78)
    print(
        "Events verified:",
        event_count,
    )
    print(
        "Terminal resume rejected:",
        terminal_resume_rejected,
    )
    print(
        "Append-only events:",
        True,
    )

    print()
    print("RESUME DECISIONS")
    print("-" * 78)
    print(
        "New run:",
        start_decision,
    )
    print(
        "Matching checkpoint:",
        resume_result,
    )
    print(
        "Completed run:",
        complete_decision,
    )

    print()
    print("COMPLETION CONTRACT")
    print("-" * 78)
    print(
        "Required artifacts:",
        len(
            REQUIRED_COMPLETE_ARTIFACTS
        ),
    )
    print(
        "Missing-artifact guard:",
        completion_guard_rejected_missing,
    )
    print(
        "All required present:",
        all_required_present,
    )
    print(
        "Completion written:",
        completion_written,
    )

    print()
    print("DEPENDENCIES")
    print("-" * 78)
    print(
        "Stage-23 passed:",
        stage23_dependency,
    )
    print(
        "Stage-21 padding safe:",
        contract[
            "dependencies"
        ][
            "stage21_padding_safe"
        ],
    )
    print(
        "Stage-22 length safe:",
        contract[
            "dependencies"
        ][
            "stage22_length_safe"
        ],
    )
    print(
        "Stage-23 perfect errors:",
        contract[
            "dependencies"
        ][
            "stage23_perfect_control_edits"
        ],
    )

    print()
    print("OUTPUTS")
    print("-" * 78)
    print(
        "Saved:",
        CONTRACT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Saved:",
        DOC_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    stage24_pass = all(
        [
            canonical_hash_stable,
            mismatch_rejected,
            checkpoint_epoch == 7,
            metadata_epoch == 7,
            terminal_resume_rejected,
            start_decision == "START",
            resume_result == "RESUME",
            complete_decision
            == "SKIP_COMPLETE",
            event_count == 6,
            completion_guard_rejected_missing,
            all_required_present,
            completion_written,
            stage23_dependency,
        ]
    )

    print()

    if stage24_pass:
        print(
            "FINAL STATUS: "
            "STAGE 24 BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 24 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

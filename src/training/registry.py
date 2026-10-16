from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from datetime import datetime, timezone
from pathlib import Path


SHA256_PATTERN = re.compile(
    r"^[0-9a-f]{64}$"
)

COMMIT_PATTERN = re.compile(
    r"^[0-9a-f]{7,64}$"
)

ID_PATTERN = re.compile(
    r"^[a-z0-9][a-z0-9_.-]*$"
)

RUN_STATUSES = {
    "PLANNED",
    "RUNNING",
    "INTERRUPTED",
    "COMPLETE",
    "FAILED_SAFE",
    "INVALID",
}

TERMINAL_STATUSES = {
    "COMPLETE",
    "FAILED_SAFE",
    "INVALID",
}

EVENT_TYPES = {
    "PLAN",
    "START",
    "EPOCH_END",
    "CHECKPOINT_SYNC",
    "INTERRUPT",
    "RESUME",
    "EVAL",
    "COMPLETE",
    "FAIL",
}

ALLOWED_TRANSITIONS = {
    "PLANNED": {
        "RUNNING",
        "INVALID",
    },
    "RUNNING": {
        "INTERRUPTED",
        "COMPLETE",
        "FAILED_SAFE",
        "INVALID",
    },
    "INTERRUPTED": {
        "RUNNING",
        "FAILED_SAFE",
        "INVALID",
    },
    "COMPLETE": set(),
    "FAILED_SAFE": set(),
    "INVALID": set(),
}

REQUIRED_COMPLETE_ARTIFACTS = (
    "identity.json",
    "config.yaml",
    "data_manifest_sha256.txt",
    "label_vocab_sha256.txt",
    "split_sha256.txt",
    "code_commit.txt",
    "train_log.csv",
    "last.pt",
    "best.pt",
    "last_checkpoint.json",
    "predictions.jsonl",
    "metrics.json",
    "events.jsonl",
)


def canonical_json_sha256(
    value,
) -> str:

    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(
            ",",
            ":",
        ),
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        encoded
    ).hexdigest()


def _require_nonempty(
    value: str,
    *,
    name: str,
) -> str:

    if (
        not isinstance(value, str)
        or not value.strip()
    ):
        raise ValueError(
            f"{name} must be a non-empty string."
        )

    return value


def _require_identifier(
    value: str,
    *,
    name: str,
) -> str:

    _require_nonempty(
        value,
        name=name,
    )

    if not ID_PATTERN.fullmatch(
        value
    ):
        raise ValueError(
            f"{name} contains invalid characters."
        )

    return value


def _require_sha256(
    value: str,
    *,
    name: str,
) -> str:

    if (
        not isinstance(value, str)
        or not SHA256_PATTERN.fullmatch(
            value
        )
    ):
        raise ValueError(
            f"{name} must be a valid lowercase SHA-256."
        )

    return value


@dataclass(
    frozen=True
)
class ExperimentIdentity:

    experiment_id: str
    project_generation: str
    model_id: str
    phase: str
    split_id: str
    data_manifest_sha256: str
    label_vocab_sha256: str
    config_sha256: str
    code_commit: str
    seed: int
    normalization: str
    augmentation: str
    hardware_required: str

    def __post_init__(
        self,
    ) -> None:

        for name in (
            "experiment_id",
            "project_generation",
            "model_id",
            "phase",
            "split_id",
        ):
            _require_identifier(
                getattr(
                    self,
                    name,
                ),
                name=name,
            )

        for name in (
            "data_manifest_sha256",
            "label_vocab_sha256",
            "config_sha256",
        ):
            _require_sha256(
                getattr(
                    self,
                    name,
                ),
                name=name,
            )

        if (
            not isinstance(
                self.code_commit,
                str,
            )
            or not COMMIT_PATTERN.fullmatch(
                self.code_commit
            )
        ):
            raise ValueError(
                "code_commit must contain "
                "7-64 lowercase hexadecimal characters."
            )

        if (
            isinstance(
                self.seed,
                bool,
            )
            or not isinstance(
                self.seed,
                int,
            )
            or self.seed < 0
        ):
            raise ValueError(
                "seed must be a non-negative integer."
            )

        for name in (
            "normalization",
            "augmentation",
            "hardware_required",
        ):
            _require_nonempty(
                getattr(
                    self,
                    name,
                ),
                name=name,
            )

    def to_dict(
        self,
    ) -> dict:

        return asdict(
            self
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping,
    ) -> "ExperimentIdentity":

        required = {
            field.name
            for field in fields(
                cls
            )
        }

        missing = (
            required
            - set(
                payload
            )
        )

        if missing:
            raise ValueError(
                "Identity is missing fields: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        values = {
            name: payload[
                name
            ]
            for name in required
        }

        return cls(
            **values
        )

    @property
    def fingerprint(
        self,
    ) -> str:

        return canonical_json_sha256(
            self.to_dict()
        )


def atomic_write_json(
    path: Path,
    payload: Mapping,
) -> None:

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_name(
        path.name
        + ".tmp"
    )

    with temporary.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        json.dump(
            payload,
            handle,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )

        handle.write(
            "\n"
        )

        handle.flush()
        os.fsync(
            handle.fileno()
        )

    os.replace(
        temporary,
        path,
    )


def read_json(
    path: Path,
) -> dict:

    return json.loads(
        Path(
            path
        ).read_text(
            encoding="utf-8-sig"
        )
    )


def validate_identity_match(
    expected: ExperimentIdentity,
    observed: ExperimentIdentity,
) -> None:

    expected_values = (
        expected.to_dict()
    )

    observed_values = (
        observed.to_dict()
    )

    mismatches = [
        name
        for name in expected_values
        if expected_values[
            name
        ] != observed_values[
            name
        ]
    ]

    if mismatches:
        raise ValueError(
            "Experiment identity mismatch: "
            + ", ".join(
                mismatches
            )
            + "."
        )


def write_or_validate_identity(
    run_directory: Path,
    identity: ExperimentIdentity,
) -> Path:

    run_directory = Path(
        run_directory
    )

    run_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = (
        run_directory
        / "identity.json"
    )

    if path.exists():

        observed = (
            ExperimentIdentity.from_dict(
                read_json(
                    path
                )
            )
        )

        validate_identity_match(
            identity,
            observed,
        )

        return path

    atomic_write_json(
        path,
        identity.to_dict(),
    )

    return path


def validate_status_transition(
    current_status: str,
    next_status: str,
) -> None:

    if current_status not in RUN_STATUSES:
        raise ValueError(
            "Unknown current run status."
        )

    if next_status not in RUN_STATUSES:
        raise ValueError(
            "Unknown next run status."
        )

    if current_status == next_status:
        return

    if next_status not in (
        ALLOWED_TRANSITIONS[
            current_status
        ]
    ):
        raise ValueError(
            "Invalid run-status transition: "
            f"{current_status} -> {next_status}."
        )


def append_registry_event(
    path: Path,
    identity: ExperimentIdentity,
    *,
    event: str,
    status: str,
    details: Mapping | None = None,
    created_at_utc: str | None = None,
) -> dict:

    if event not in EVENT_TYPES:
        raise ValueError(
            "Unknown registry event."
        )

    if status not in RUN_STATUSES:
        raise ValueError(
            "Unknown registry status."
        )

    if created_at_utc is None:
        created_at_utc = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

    record = {
        "created_at_utc": (
            created_at_utc
        ),
        "experiment_id": (
            identity.experiment_id
        ),
        "identity_fingerprint": (
            identity.fingerprint
        ),
        "event": event,
        "status": status,
        "details": dict(
            details
            or {}
        ),
    }

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with path.open(
        "a",
        encoding="utf-8",
        newline="\n",
    ) as handle:

        handle.write(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
            )
        )

        handle.write(
            "\n"
        )

        handle.flush()
        os.fsync(
            handle.fileno()
        )

    return record


def read_registry_events(
    path: Path,
) -> list[dict]:

    path = Path(
        path
    )

    if not path.exists():
        return []

    records = []

    with path.open(
        "r",
        encoding="utf-8-sig",
    ) as handle:

        for line_number, line in enumerate(
            handle,
            start=1,
        ):

            stripped = line.strip()

            if not stripped:
                continue

            try:
                record = json.loads(
                    stripped
                )

            except json.JSONDecodeError as exc:
                raise ValueError(
                    "Invalid registry JSONL at "
                    f"line {line_number}."
                ) from exc

            records.append(
                record
            )

    return records


def validate_registry_events(
    path: Path,
    identity: ExperimentIdentity,
) -> list[dict]:

    records = read_registry_events(
        path
    )

    if not records:
        raise ValueError(
            "Registry event log is empty."
        )

    previous_status = None

    for record in records:

        if record.get(
            "identity_fingerprint"
        ) != identity.fingerprint:

            raise ValueError(
                "Registry event identity mismatch."
            )

        event = record.get(
            "event"
        )

        status = record.get(
            "status"
        )

        if event not in EVENT_TYPES:
            raise ValueError(
                "Registry contains an unknown event."
            )

        if status not in RUN_STATUSES:
            raise ValueError(
                "Registry contains an unknown status."
            )

        if previous_status is not None:

            validate_status_transition(
                previous_status,
                status,
            )

        previous_status = status

    return records


def missing_or_empty_artifacts(
    run_directory: Path,
) -> list[str]:

    run_directory = Path(
        run_directory
    )

    missing = []

    for name in (
        REQUIRED_COMPLETE_ARTIFACTS
    ):

        path = (
            run_directory
            / name
        )

        if (
            not path.is_file()
            or path.stat().st_size <= 0
        ):
            missing.append(
                name
            )

    return missing


def _validate_checkpoint_metadata_identity(
    path: Path,
    identity: ExperimentIdentity,
) -> None:

    metadata = read_json(
        path
    )

    observed = (
        ExperimentIdentity.from_dict(
            metadata[
                "identity"
            ]
        )
    )

    validate_identity_match(
        identity,
        observed,
    )

    if metadata.get(
        "identity_fingerprint"
    ) != identity.fingerprint:

        raise ValueError(
            "Checkpoint identity fingerprint mismatch."
        )


def resume_decision(
    run_directory: Path,
    identity: ExperimentIdentity,
) -> str:

    run_directory = Path(
        run_directory
    )

    write_or_validate_identity(
        run_directory,
        identity,
    )

    completion_path = (
        run_directory
        / "completion.json"
    )

    if completion_path.exists():

        completion = read_json(
            completion_path
        )

        observed = (
            ExperimentIdentity.from_dict(
                completion[
                    "identity"
                ]
            )
        )

        validate_identity_match(
            identity,
            observed,
        )

        if completion.get(
            "status"
        ) != "COMPLETE":
            raise ValueError(
                "Completion marker does not "
                "contain COMPLETE status."
            )

        return "SKIP_COMPLETE"

    checkpoint_path = (
        run_directory
        / "last.pt"
    )

    metadata_path = (
        run_directory
        / "last_checkpoint.json"
    )

    if (
        checkpoint_path.exists()
        != metadata_path.exists()
    ):
        raise ValueError(
            "Incomplete resume checkpoint pair."
        )

    if checkpoint_path.exists():

        _validate_checkpoint_metadata_identity(
            metadata_path,
            identity,
        )

        return "RESUME"

    return "START"


def write_completion_marker(
    run_directory: Path,
    identity: ExperimentIdentity,
) -> Path:

    run_directory = Path(
        run_directory
    )

    write_or_validate_identity(
        run_directory,
        identity,
    )

    missing = missing_or_empty_artifacts(
        run_directory
    )

    if missing:
        raise ValueError(
            "Run cannot be completed; missing "
            "or empty artifacts: "
            + ", ".join(
                missing
            )
            + "."
        )

    _validate_checkpoint_metadata_identity(
        run_directory
        / "last_checkpoint.json",
        identity,
    )

    events = validate_registry_events(
        run_directory
        / "events.jsonl",
        identity,
    )

    if events[-1][
        "status"
    ] != "COMPLETE":

        raise ValueError(
            "Final registry status must be COMPLETE."
        )

    path = (
        run_directory
        / "completion.json"
    )

    payload = {
        "status": "COMPLETE",
        "completed_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "identity": (
            identity.to_dict()
        ),
        "identity_fingerprint": (
            identity.fingerprint
        ),
        "required_artifacts": list(
            REQUIRED_COMPLETE_ARTIFACTS
        ),
    }

    atomic_write_json(
        path,
        payload,
    )

    return path

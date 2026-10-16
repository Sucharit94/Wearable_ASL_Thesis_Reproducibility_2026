from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

from src.training.registry import (
    ExperimentIdentity,
    atomic_write_json,
    validate_identity_match,
)


CHECKPOINT_SCHEMA_VERSION = 1

REQUIRED_RNG_STATES = {
    "python",
    "numpy",
    "torch_cpu",
}


def build_checkpoint_envelope(
    identity: ExperimentIdentity,
    *,
    epoch: int,
    selected_epochs: int | None,
    model_state,
    optimizer_state,
    scheduler_state=None,
    amp_scaler_state=None,
    rng_state: Mapping,
) -> dict:

    envelope = {
        "checkpoint_schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "identity": (
            identity.to_dict()
        ),
        "identity_fingerprint": (
            identity.fingerprint
        ),
        "epoch": epoch,
        "selected_epochs": (
            selected_epochs
        ),
        "model_state": (
            model_state
        ),
        "optimizer_state": (
            optimizer_state
        ),
        "scheduler_state": (
            scheduler_state
        ),
        "amp_scaler_state": (
            amp_scaler_state
        ),
        "rng_state": dict(
            rng_state
        ),
    }

    validate_checkpoint_envelope(
        envelope,
        identity,
    )

    return envelope


def validate_checkpoint_envelope(
    envelope: Mapping,
    expected_identity: ExperimentIdentity,
) -> int:

    if envelope.get(
        "checkpoint_schema_version"
    ) != CHECKPOINT_SCHEMA_VERSION:

        raise ValueError(
            "Unsupported checkpoint schema version."
        )

    observed_identity = (
        ExperimentIdentity.from_dict(
            envelope[
                "identity"
            ]
        )
    )

    validate_identity_match(
        expected_identity,
        observed_identity,
    )

    if envelope.get(
        "identity_fingerprint"
    ) != expected_identity.fingerprint:

        raise ValueError(
            "Checkpoint identity fingerprint mismatch."
        )

    epoch = envelope.get(
        "epoch"
    )

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise ValueError(
            "Checkpoint epoch must be "
            "a non-negative integer."
        )

    selected_epochs = envelope.get(
        "selected_epochs"
    )

    if (
        selected_epochs is not None
        and (
            isinstance(
                selected_epochs,
                bool,
            )
            or not isinstance(
                selected_epochs,
                int,
            )
            or selected_epochs <= 0
        )
    ):
        raise ValueError(
            "selected_epochs must be positive "
            "or None."
        )

    if envelope.get(
        "model_state"
    ) is None:
        raise ValueError(
            "Checkpoint model_state is missing."
        )

    if envelope.get(
        "optimizer_state"
    ) is None:
        raise ValueError(
            "Checkpoint optimizer_state is missing."
        )

    rng_state = envelope.get(
        "rng_state"
    )

    if not isinstance(
        rng_state,
        Mapping,
    ):
        raise ValueError(
            "Checkpoint rng_state must be a mapping."
        )

    missing_rng = (
        REQUIRED_RNG_STATES
        - set(
            rng_state
        )
    )

    if missing_rng:
        raise ValueError(
            "Checkpoint rng_state is missing "
            "required entries: "
            + ", ".join(
                sorted(
                    missing_rng
                )
            )
            + "."
        )

    return epoch


def checkpoint_metadata(
    envelope: Mapping,
    expected_identity: ExperimentIdentity,
) -> dict:

    epoch = validate_checkpoint_envelope(
        envelope,
        expected_identity,
    )

    return {
        "checkpoint_schema_version": (
            CHECKPOINT_SCHEMA_VERSION
        ),
        "identity": (
            expected_identity.to_dict()
        ),
        "identity_fingerprint": (
            expected_identity.fingerprint
        ),
        "epoch": epoch,
        "selected_epochs": (
            envelope.get(
                "selected_epochs"
            )
        ),
        "model_state_present": True,
        "optimizer_state_present": True,
        "scheduler_state_present": (
            envelope.get(
                "scheduler_state"
            )
            is not None
        ),
        "amp_scaler_state_present": (
            envelope.get(
                "amp_scaler_state"
            )
            is not None
        ),
        "rng_state_names": sorted(
            envelope[
                "rng_state"
            ]
        ),
    }


def validate_checkpoint_metadata(
    metadata: Mapping,
    expected_identity: ExperimentIdentity,
) -> int:

    if metadata.get(
        "checkpoint_schema_version"
    ) != CHECKPOINT_SCHEMA_VERSION:

        raise ValueError(
            "Unsupported checkpoint metadata version."
        )

    observed_identity = (
        ExperimentIdentity.from_dict(
            metadata[
                "identity"
            ]
        )
    )

    validate_identity_match(
        expected_identity,
        observed_identity,
    )

    if metadata.get(
        "identity_fingerprint"
    ) != expected_identity.fingerprint:

        raise ValueError(
            "Checkpoint metadata identity mismatch."
        )

    if metadata.get(
        "model_state_present"
    ) is not True:

        raise ValueError(
            "Checkpoint metadata reports "
            "missing model state."
        )

    if metadata.get(
        "optimizer_state_present"
    ) is not True:

        raise ValueError(
            "Checkpoint metadata reports "
            "missing optimizer state."
        )

    rng_names = set(
        metadata.get(
            "rng_state_names",
            [],
        )
    )

    if not REQUIRED_RNG_STATES.issubset(
        rng_names
    ):
        raise ValueError(
            "Checkpoint metadata is missing RNG states."
        )

    epoch = metadata.get(
        "epoch"
    )

    if (
        isinstance(epoch, bool)
        or not isinstance(epoch, int)
        or epoch < 0
    ):
        raise ValueError(
            "Invalid checkpoint metadata epoch."
        )

    return epoch


def write_checkpoint_metadata(
    path: Path,
    envelope: Mapping,
    expected_identity: ExperimentIdentity,
) -> None:

    metadata = checkpoint_metadata(
        envelope,
        expected_identity,
    )

    atomic_write_json(
        Path(
            path
        ),
        metadata,
    )


def atomic_torch_save(
    envelope: Mapping,
    path: Path,
    expected_identity: ExperimentIdentity,
) -> None:
    """
    Atomically save a trusted project-owned PyTorch checkpoint.

    PyTorch is imported lazily because local infrastructure
    tests do not require it.
    """

    validate_checkpoint_envelope(
        envelope,
        expected_identity,
    )

    try:
        import torch

    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to save "
            "a training checkpoint."
        ) from exc

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

    torch.save(
        dict(
            envelope
        ),
        temporary,
    )

    os.replace(
        temporary,
        path,
    )


def load_torch_checkpoint(
    path: Path,
    expected_identity: ExperimentIdentity,
    *,
    map_location="cpu",
) -> dict:
    """
    Load only trusted checkpoints created by this project.
    """

    try:
        import torch

    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required to load "
            "a training checkpoint."
        ) from exc

    envelope = torch.load(
        Path(
            path
        ),
        map_location=map_location,
        weights_only=False,
    )

    validate_checkpoint_envelope(
        envelope,
        expected_identity,
    )

    return envelope

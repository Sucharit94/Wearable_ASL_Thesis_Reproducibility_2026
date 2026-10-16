from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from src.models.factory import (
    EXPECTED_TRAINABLE_PARAMETERS,
    MODEL_IDS,
    build_model,
    validate_model_parameter_count,
)
from src.training.smoke_overfit import (
    batch_slices,
    build_optimizer,
    is_out_of_memory_error,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "config"
    / "stage31_config_hashes.json"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(
        Path(path).read_bytes()
    )
    return digest.hexdigest()


def test_stage31_model_ids_are_exact_and_ordered():
    assert MODEL_IDS == (
        "g1",
        "c1",
        "c2",
        "c3",
        "c4",
    )


@pytest.mark.parametrize(
    "model_id",
    MODEL_IDS,
)
def test_factory_builds_frozen_parameter_count(
    model_id: str,
):
    model = build_model(
        model_id
    )

    assert validate_model_parameter_count(
        model_id,
        model,
    ) == EXPECTED_TRAINABLE_PARAMETERS[
        model_id
    ]


@pytest.mark.parametrize(
    (
        "model_id",
        "optimizer_name",
    ),
    [
        ("g1", "Adam"),
        ("c1", "AdamW"),
        ("c2", "AdamW"),
        ("c3", "AdamW"),
        ("c4", "AdamW"),
    ],
)
def test_optimizer_family_is_frozen(
    model_id: str,
    optimizer_name: str,
):
    del model_id

    model = nn.Linear(
        2,
        2,
    )

    optimizer = build_optimizer(
        model,
        {
            "optimizer": optimizer_name,
            "learning_rate": 5e-5,
            "weight_decay": 1e-4,
        },
    )

    expected_type = (
        torch.optim.Adam
        if optimizer_name == "Adam"
        else torch.optim.AdamW
    )

    assert type(optimizer) is expected_type


def test_batch_slices_preserve_fixed_order():
    items = [
        {
            "value": value
        }
        for value in range(7)
    ]

    observed = [
        item["value"]
        for item_batch in batch_slices(
            items,
            3,
        )
        for item in item_batch
    ]

    assert observed == list(
        range(7)
    )


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory.",
        (
            "DefaultCPUAllocator: "
            "can't allocate memory"
        ),
        (
            "Not enough memory resources "
            "are available."
        ),
    ],
)
def test_only_memory_errors_are_recognized_for_fallback(
    message: str,
):
    assert is_out_of_memory_error(
        RuntimeError(message)
    )


def test_scientific_or_shape_errors_are_not_memory_fallbacks():
    assert not is_out_of_memory_error(
        RuntimeError(
            "CTC-infeasible batch samples "
            "at indices [0]."
        )
    )


def test_all_frozen_configs_are_json_compatible_yaml():
    manifest = json.loads(
        CONFIG_MANIFEST_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    assert manifest["config_count"] == 11

    for entry in manifest["configs"]:
        payload = json.loads(
            (
                PROJECT_ROOT
                / entry["path"]
            ).read_text(
                encoding="utf-8-sig"
            )
        )

        assert payload[
            "project_generation"
        ] == "fresh_start_2026_v1"

        assert payload[
            "status"
        ] == "frozen_before_outer_evaluation"


def test_config_hash_manifest_matches_every_file():
    manifest = json.loads(
        CONFIG_MANIFEST_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    for entry in manifest["configs"]:
        assert sha256_file(
            PROJECT_ROOT
            / entry["path"]
        ) == entry["sha256"]

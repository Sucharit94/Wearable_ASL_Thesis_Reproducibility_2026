from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.models.contracts import (
    EXPECTED_MODEL_IDS,
    ModelOutput,
    ModelSpec,
    load_model_catalog,
    validate_model_output,
)
from src.training.registry import (
    canonical_json_sha256,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "models"
    / "stage25_model_catalog.json"
)


def test_valid_model_output():

    output = ModelOutput(
        logits_btc=np.zeros(
            (
                3,
                45,
                456,
            ),
            dtype=np.float32,
        ),
        output_lengths=np.array(
            [
                3,
                12,
                45,
            ],
            dtype=np.int64,
        ),
    )

    assert validate_model_output(
        output
    ) == (
        3,
        12,
        45,
    )


def test_model_output_rank_rejected():

    output = ModelOutput(
        logits_btc=np.zeros(
            (
                3,
                456,
            ),
            dtype=np.float32,
        ),
        output_lengths=[
            3,
        ],
    )

    with pytest.raises(
        ValueError,
        match="B x T x C",
    ):
        validate_model_output(
            output
        )


def test_wrong_class_count_rejected():

    output = ModelOutput(
        logits_btc=np.zeros(
            (
                2,
                10,
                455,
            ),
            dtype=np.float32,
        ),
        output_lengths=[
            10,
            10,
        ],
    )

    with pytest.raises(
        ValueError,
        match="456",
    ):
        validate_model_output(
            output
        )


def test_length_batch_mismatch_rejected():

    output = ModelOutput(
        logits_btc=np.zeros(
            (
                2,
                10,
                456,
            ),
            dtype=np.float32,
        ),
        output_lengths=[
            10,
        ],
    )

    with pytest.raises(
        ValueError,
        match="batch size",
    ):
        validate_model_output(
            output
        )


def test_nonpositive_length_rejected():

    output = ModelOutput(
        logits_btc=np.zeros(
            (
                2,
                10,
                456,
            ),
            dtype=np.float32,
        ),
        output_lengths=[
            10,
            0,
        ],
    )

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        validate_model_output(
            output
        )


def test_length_exceeding_time_rejected():

    output = ModelOutput(
        logits_btc=np.zeros(
            (
                1,
                10,
                456,
            ),
            dtype=np.float32,
        ),
        output_lengths=[
            11,
        ],
    )

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):
        validate_model_output(
            output
        )


def test_float_lengths_rejected():

    output = ModelOutput(
        logits_btc=np.zeros(
            (
                1,
                10,
                456,
            ),
            dtype=np.float32,
        ),
        output_lengths=np.array(
            [
                10.0,
            ],
            dtype=np.float32,
        ),
    )

    with pytest.raises(
        TypeError,
        match="integers",
    ):
        validate_model_output(
            output
        )


def test_nonfinite_logits_rejected():

    logits = np.zeros(
        (
            1,
            10,
            456,
        ),
        dtype=np.float32,
    )

    logits[
        0,
        4,
        12,
    ] = np.nan

    output = ModelOutput(
        logits_btc=logits,
        output_lengths=[
            10,
        ],
    )

    with pytest.raises(
        ValueError,
        match="NaN",
    ):
        validate_model_output(
            output
        )


def test_direct_spec_rejects_window_geometry():

    payload = json.loads(
        CATALOG_PATH.read_text(
            encoding="utf-8-sig"
        )
    )[
        "models"
    ][
        1
    ]

    payload[
        "window_length"
    ] = 36

    with pytest.raises(
        ValueError,
        match="Direct-sequence",
    ):
        ModelSpec.from_dict(
            payload
        )


def test_catalog_contains_exactly_five_models():

    catalog = load_model_catalog(
        CATALOG_PATH
    )

    assert set(
        catalog
    ) == EXPECTED_MODEL_IDS

    assert len(
        catalog
    ) == 5


def test_shared_ctc_schema_identical():

    catalog = load_model_catalog(
        CATALOG_PATH
    )

    for specification in (
        catalog.values()
    ):

        assert specification.feature_width == 38
        assert specification.output_classes == 456
        assert specification.blank_id == 0
        assert specification.target_padding_value == -1
        assert specification.returns_raw_logits is True
        assert (
            specification.valid_output_length_required
            is True
        )


def test_input_adapter_assignments():

    catalog = load_model_catalog(
        CATALOG_PATH
    )

    assert {
        model_id
        for model_id, specification
        in catalog.items()
        if specification.input_adapter
        == "direct_sequence"
    } == {
        "c1",
        "c2",
        "c3",
    }

    assert {
        model_id
        for model_id, specification
        in catalog.items()
        if specification.input_adapter
        == "windowed_2d"
    } == {
        "g1",
        "c4",
    }


def test_architecture_identity_flags():

    catalog = load_model_catalog(
        CATALOG_PATH
    )

    assert catalog[
        "g1"
    ].uses_bilstm is True

    assert catalog[
        "c1"
    ].uses_bilstm is True

    assert catalog[
        "c2"
    ].uses_self_attention is True

    assert catalog[
        "c3"
    ].uses_transformer_encoder is True

    assert catalog[
        "c4"
    ].uses_tiny_swin is True

    assert catalog[
        "c4"
    ].uses_bilstm is True


def test_primary_training_defaults():

    catalog = load_model_catalog(
        CATALOG_PATH
    )

    assert catalog[
        "g1"
    ].optimizer == "Adam"

    assert catalog[
        "g1"
    ].epoch_cap == 20

    assert catalog[
        "g1"
    ].physical_batch_size == 16

    for model_id in (
        "c1",
        "c2",
        "c3",
        "c4",
    ):

        specification = catalog[
            model_id
        ]

        assert specification.optimizer == "AdamW"

        assert specification.learning_rate == (
            pytest.approx(
                5e-5
            )
        )

        assert specification.weight_decay == (
            pytest.approx(
                1e-4
            )
        )

        assert specification.epoch_cap == 80
        assert specification.early_stopping_patience == 8
        assert specification.gradient_clip_norm == 5.0


def test_catalog_hash_is_canonical():

    payload = json.loads(
        CATALOG_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    reordered = {
        "models": payload[
            "models"
        ],
        "configuration_status": payload[
            "configuration_status"
        ],
        "project_generation": payload[
            "project_generation"
        ],
        "schema_version": payload[
            "schema_version"
        ],
    }

    assert canonical_json_sha256(
        payload
    ) == canonical_json_sha256(
        reordered
    )

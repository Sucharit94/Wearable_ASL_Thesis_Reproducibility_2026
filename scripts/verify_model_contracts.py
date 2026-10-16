from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.lengths import full_window_lengths
from src.models.contracts import (
    ModelOutput,
    load_model_catalog,
    validate_model_output,
)
from src.training.registry import (
    canonical_json_sha256,
)


ROOT = Path(__file__).resolve().parents[1]

CATALOG = (
    ROOT
    / "configs"
    / "models"
    / "stage25_model_catalog.json"
)

STAGE22 = (
    ROOT
    / "artifacts"
    / "ctc"
    / "stage22_sequence_length_contract.json"
)

STAGE24 = (
    ROOT
    / "artifacts"
    / "registry"
    / "stage24_experiment_registry_contract.json"
)

OUT = (
    ROOT
    / "artifacts"
    / "models"
)

CONTRACT = (
    OUT
    / "stage25_model_interface_contract.json"
)

HASH = (
    OUT
    / "stage25_model_catalog.sha256"
)

DOC = (
    ROOT
    / "docs"
    / "STAGE_25_MODEL_INTERFACE_CONFIG.md"
)


def load_json(
    path: Path,
) -> dict:

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
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
                return digest.hexdigest()

            digest.update(
                chunk
            )


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 25 — SHARED MODEL INTERFACE "
        "AND CONFIGURATION SCHEMA"
    )
    print("=" * 78)

    stage22 = load_json(
        STAGE22
    )

    stage24 = load_json(
        STAGE24
    )

    stage22_safe = (
        int(
            stage22[
                "stage"
            ]
        ) == 22

        and stage22[
            "principles"
        ][
            "valid_length_is_first_class"
        ] is True

        and stage22[
            "principles"
        ][
            "same_final_length_used_for_ctc_loss_and_decoder"
        ] is True

        and int(
            stage22[
                "g1_c4_window_adapter"
            ][
                "window_length"
            ]
        ) == 36

        and int(
            stage22[
                "g1_c4_window_adapter"
            ][
                "stride"
            ]
        ) == 18

        and int(
            stage22[
                "g1_c4_window_adapter"
            ][
                "stage20_length_mismatches"
            ]
        ) == 0

        and int(
            stage22[
                "g1_c4_window_adapter"
            ][
                "infeasible_recordings"
            ]
        ) == 0
    )

    stage24_passed = (
        int(
            stage24[
                "stage"
            ]
        ) == 24

        and stage24[
            "identity"
        ][
            "canonical_config_hash_stable"
        ] is True

        and stage24[
            "registry"
        ][
            "append_only_events"
        ] is True

        and stage24[
            "completion"
        ][
            "all_required_present"
        ] is True

        and stage24[
            "completion"
        ][
            "completion_written"
        ] is True

        and stage24[
            "dependencies"
        ][
            "stage23_passed"
        ] is True
    )

    identity_required = (
        stage24[
            "resume"
        ][
            "identity_match_required"
        ] is True
    )

    if not (
        stage22_safe
        and stage24_passed
        and identity_required
    ):

        raise RuntimeError(
            "A frozen Stage-22 or Stage-24 "
            "dependency failed."
        )

    payload = load_json(
        CATALOG
    )

    models = load_model_catalog(
        CATALOG
    )

    model_ids = sorted(
        models
    )

    raw_lengths = np.asarray(
        [
            72,
            234,
            828,
        ],
        dtype=np.int64,
    )

    window_lengths = (
        full_window_lengths(
            raw_lengths,
            window_length=36,
            stride=18,
        )
    )

    validated_lengths = (
        validate_model_output(
            ModelOutput(
                logits_btc=np.zeros(
                    (
                        3,
                        45,
                        456,
                    ),
                    dtype=np.float32,
                ),
                output_lengths=window_lengths,
            )
        )
    )

    direct = sorted(
        key
        for key, value in models.items()
        if value.input_adapter
        == "direct_sequence"
    )

    windowed = sorted(
        key
        for key, value in models.items()
        if value.input_adapter
        == "windowed_2d"
    )

    bilstm = sorted(
        key
        for key, value in models.items()
        if value.uses_bilstm
    )

    attention = sorted(
        key
        for key, value in models.items()
        if value.uses_self_attention
    )

    transformer = sorted(
        key
        for key, value in models.items()
        if value.uses_transformer_encoder
    )

    tiny_swin = sorted(
        key
        for key, value in models.items()
        if value.uses_tiny_swin
    )

    adam = sorted(
        key
        for key, value in models.items()
        if value.optimizer == "Adam"
    )

    adamw = sorted(
        key
        for key, value in models.items()
        if value.optimizer == "AdamW"
    )

    families = {
        key: models[
            key
        ].architecture_family
        for key in model_ids
    }

    validations = {
        "five_models": (
            model_ids
            == [
                "c1",
                "c2",
                "c3",
                "c4",
                "g1",
            ]
        ),

        "window_probe": (
            validated_lengths
            == (
                3,
                12,
                45,
            )
        ),

        "direct_models": (
            direct
            == [
                "c1",
                "c2",
                "c3",
            ]
        ),

        "windowed_models": (
            windowed
            == [
                "c4",
                "g1",
            ]
        ),

        "bilstm_models": (
            bilstm
            == [
                "c1",
                "c2",
                "c4",
                "g1",
            ]
        ),

        "attention_models": (
            attention
            == [
                "c2"
            ]
        ),

        "transformer_models": (
            transformer
            == [
                "c3"
            ]
        ),

        "tiny_swin_models": (
            tiny_swin
            == [
                "c4"
            ]
        ),

        "c4_bilstm": (
            models[
                "c4"
            ].uses_bilstm
            is True
        ),

        "adam_models": (
            adam
            == [
                "g1"
            ]
        ),

        "adamw_models": (
            adamw
            == [
                "c1",
                "c2",
                "c3",
                "c4",
            ]
        ),

        "shared_schema": all(
            value.feature_width == 38
            and value.output_classes == 456
            and value.blank_id == 0
            and value.target_padding_value == -1
            and value.returns_raw_logits is True
            and value.valid_output_length_required
            is True
            for value in models.values()
        ),
    }

    failed = [
        name
        for name, passed
        in validations.items()
        if not passed
    ]

    if failed:

        raise RuntimeError(
            "Stage-25 validations failed: "
            f"{failed}"
        )

    catalog_hash = sha256_file(
        CATALOG
    )

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 25,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "catalog": {
            "model_count": (
                len(
                    models
                )
            ),

            "model_ids": (
                model_ids
            ),

            "canonical_json_sha256": (
                canonical_json_sha256(
                    payload
                )
            ),

            "file_sha256": (
                catalog_hash
            ),
        },

        "shared_output": {
            "layout": (
                "B_T_C"
            ),

            "classes": 456,

            "blank_id": 0,

            "target_padding_value": -1,

            "raw_logits_required": True,

            "valid_output_lengths_required": (
                True
            ),

            "nonfinite_logits_allowed": (
                False
            ),
        },

        "input_adapters": {
            "direct_sequence": (
                direct
            ),

            "windowed_2d": (
                windowed
            ),

            "window_length": 36,

            "window_stride": 18,

            "probe_raw_lengths": (
                raw_lengths.tolist()
            ),

            "probe_window_lengths": (
                window_lengths.tolist()
            ),
        },

        "architectures": {
            "families": (
                families
            ),

            "bilstm_models": (
                bilstm
            ),

            "self_attention_models": (
                attention
            ),

            "transformer_models": (
                transformer
            ),

            "tiny_swin_models": (
                tiny_swin
            ),

            "c4_bilstm_backend": (
                models[
                    "c4"
                ].uses_bilstm
            ),
        },

        "training_defaults": {
            "adam_models": (
                adam
            ),

            "adamw_models": (
                adamw
            ),

            "gradient_clip_norm": 5.0,

            "g1_epoch_cap": (
                models[
                    "g1"
                ].epoch_cap
            ),

            "compact_epoch_cap": (
                models[
                    "c1"
                ].epoch_cap
            ),
        },

        "dependencies": {
            "stage22_window_length_safe": (
                stage22_safe
            ),

            "stage24_registry_passed": (
                stage24_passed
            ),

            "stage24_identity_required": (
                identity_required
            ),
        },

        "hashes": {
            "contracts": (
                sha256_file(
                    ROOT
                    / "src"
                    / "models"
                    / "contracts.py"
                )
            ),

            "catalog": (
                catalog_hash
            ),

            "tests": (
                sha256_file(
                    ROOT
                    / "tests"
                    / "test_model_contracts.py"
                )
            ),

            "verifier": (
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                )
            ),
        },
    }

    OUT.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOC.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT.write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    HASH.write_text(
        (
            f"{catalog_hash}  "
            "configs/models/"
            "stage25_model_catalog.json\n"
        ),
        encoding="utf-8",
    )

    documentation = f"""# Stage 25 — Shared Model Interface and Configuration Schema

## Model catalog

Five frozen recognisers:

{", ".join(model_ids)}

## Shared output contract

All models return raw `B x T x 456` logits and exact valid output lengths.

CTC blank ID:

0

Target padding:

-1

## Input adapters

Direct-sequence models:

{", ".join(direct)}

Windowed 2D models:

{", ".join(windowed)}

Window length:

36

Window stride:

18

## Architecture identities

G1:

{families["g1"]}

C1:

{families["c1"]}

C2:

{families["c2"]}

C3:

{families["c3"]}

C4:

{families["c4"]}

C4 retains its BiLSTM backend:

{models["c4"].uses_bilstm}

## Frozen dependencies

Stage-22 window-length safety:

{stage22_safe}

Stage-24 registry passed:

{stage24_passed}

Stage-24 identity matching required:

{identity_required}
"""

    DOC.write_text(
        documentation,
        encoding="utf-8",
    )

    print()
    print("MODEL CATALOG")
    print("-" * 78)
    print(
        "Model count:",
        len(
            models
        ),
    )
    print(
        "Model IDs:",
        model_ids,
    )

    print()
    print("SHARED OUTPUT CONTRACT")
    print("-" * 78)
    print(
        "Classes / blank / padding:",
        "456 / 0 / -1",
    )
    print(
        "Raw logits required:",
        True,
    )
    print(
        "Valid lengths required:",
        True,
    )
    print(
        "Non-finite logits allowed:",
        False,
    )

    print()
    print("INPUT ADAPTERS")
    print("-" * 78)
    print(
        "Direct:",
        direct,
    )
    print(
        "Windowed:",
        windowed,
    )
    print(
        "Window length / stride:",
        "36 / 18",
    )
    print(
        "Probe window lengths:",
        window_lengths.tolist(),
    )

    print()
    print("ARCHITECTURE IDENTITIES")
    print("-" * 78)
    print(
        "Families:",
        families,
    )
    print(
        "C4 BiLSTM backend:",
        models[
            "c4"
        ].uses_bilstm,
    )

    print()
    print("TRAINING DEFAULTS")
    print("-" * 78)
    print(
        "Adam:",
        adam,
    )
    print(
        "AdamW:",
        adamw,
    )
    print(
        "Gradient clip:",
        5.0,
    )
    print(
        "G1 / compact epoch caps:",
        "20 / 80",
    )

    print()
    print("FROZEN DEPENDENCIES")
    print("-" * 78)
    print(
        "Stage-22 window length safe:",
        stage22_safe,
    )
    print(
        "Stage-24 registry passed:",
        stage24_passed,
    )
    print(
        "Stage-24 identity required:",
        identity_required,
    )

    print()
    print("OUTPUTS")
    print("-" * 78)
    print(
        "Saved:",
        CONTRACT.relative_to(
            ROOT
        ),
    )
    print(
        "Saved:",
        HASH.relative_to(
            ROOT
        ),
    )
    print(
        "Saved:",
        DOC.relative_to(
            ROOT
        ),
    )

    print()
    print(
        "FINAL STATUS: "
        "STAGE 25 BUILD PASSED"
    )
    print("=" * 78)


if __name__ == "__main__":
    main()

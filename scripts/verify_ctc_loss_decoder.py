from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.evaluation.ctc_decode import (
    greedy_ctc_decode,
)
from src.training.ctc_loss import (
    pack_ctc_targets,
    unpack_ctc_targets,
    validate_ctc_batch_feasibility,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

TOKEN_TO_ID_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_token_to_id.json"
)

STAGE20_AUDIT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
    / "stage20_ctc_sample_audit.csv"
)

STAGE20_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
    / "stage20_ctc_sample_audit.sha256"
)

STAGE20_CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
    / "stage20_ctc_feasibility_contract.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage21_ctc_loss_decoder_contract.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_21_CTC_LOSS_DECODER.md"
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


def read_hash(
    path: Path,
) -> str:

    return (
        path.read_text(
            encoding="utf-8-sig"
        )
        .strip()
        .split()[0]
        .lower()
    )


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 21 — CTC LOSS + "
        "LENGTH-SAFE DECODER"
    )
    print("=" * 78)

    # ---------------------------------------------------------
    # Stage-20 frozen feasibility guard.
    # ---------------------------------------------------------

    recorded_stage20_hash = read_hash(
        STAGE20_HASH_PATH
    )

    actual_stage20_hash = sha256_file(
        STAGE20_AUDIT_PATH
    )

    if (
        recorded_stage20_hash
        != actual_stage20_hash
    ):

        raise RuntimeError(
            "Stage-20 audit hash mismatch."
        )

    stage20_contract = json.loads(
        STAGE20_CONTRACT_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage20_selected_infeasible = int(
        stage20_contract[
            "windowing"
        ][
            "selected_infeasible_recordings"
        ]
    )

    if stage20_selected_infeasible != 0:

        raise RuntimeError(
            "Stage-20 selected window rule "
            "is not fully CTC-feasible."
        )

    # ---------------------------------------------------------
    # Load all 300 frozen Stage-10 targets.
    # ---------------------------------------------------------

    with TARGET_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    if len(rows) != 300:

        raise RuntimeError(
            "Expected 300 sentence targets."
        )

    token_to_id = json.loads(
        TOKEN_TO_ID_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        len(
            token_to_id
        ) != 456
    ):

        raise RuntimeError(
            "Expected 456 CTC classes."
        )

    if (
        token_to_id.get(
            "<blank>"
        ) != 0
    ):

        raise RuntimeError(
            "CTC blank ID is not 0."
        )

    target_sequences = []

    for row in rows:

        ids = tuple(
            int(value)
            for value in json.loads(
                row[
                    "gloss_token_ids_json"
                ]
            )
        )

        if 0 in ids:

            raise RuntimeError(
                "Blank ID occurs inside "
                "a frozen reference target."
            )

        target_sequences.append(
            ids
        )

    max_target_length = max(
        len(
            target
        )
        for target in target_sequences
    )

    # ---------------------------------------------------------
    # Build one padded batch containing ALL 300 templates.
    # ---------------------------------------------------------

    padded = np.full(
        (
            len(
                target_sequences
            ),
            max_target_length,
        ),
        -1,
        dtype=np.int64,
    )

    lengths = np.asarray(
        [
            len(
                target
            )
            for target in target_sequences
        ],
        dtype=np.int64,
    )

    for index, target in enumerate(
        target_sequences
    ):

        padded[
            index,
            :len(
                target
            ),
        ] = target

    packed = pack_ctc_targets(
        padded,
        lengths,
    )

    unpacked = unpack_ctc_targets(
        packed,
        lengths,
    )

    roundtrip_failures = sum(
        observed
        != expected
        for observed, expected
        in zip(
            unpacked,
            target_sequences,
            strict=True,
        )
    )

    if roundtrip_failures:

        raise RuntimeError(
            "Target pack/unpack round-trip "
            "failure."
        )

    packed_blank_count = int(
        np.sum(
            packed == 0
        )
    )

    packed_padding_count = int(
        np.sum(
            packed == -1
        )
    )

    if packed_blank_count != 0:

        raise RuntimeError(
            "Packed reference target contains "
            "CTC blank."
        )

    if packed_padding_count != 0:

        raise RuntimeError(
            "Packed reference target still "
            "contains padding."
        )

    # ---------------------------------------------------------
    # Synthetic length-safe decoder proof.
    # ---------------------------------------------------------

    scores = np.zeros(
        (
            2,
            7,
            6,
        ),
        dtype=np.float32,
    )

    # Sample 0 valid path:
    # 1,1,0,2 -> decoded [1,2]
    scores[0, 0, 1] = 10
    scores[0, 1, 1] = 10
    scores[0, 2, 0] = 10
    scores[0, 3, 2] = 10

    # Invalid/padded region deliberately predicts class 5.
    scores[0, 4:, 5] = 100

    # Sample 1 valid path:
    # 3,0,3 -> decoded [3,3]
    scores[1, 0, 3] = 10
    scores[1, 1, 0] = 10
    scores[1, 2, 3] = 10

    # Again make padded area dangerous.
    scores[1, 3:, 4] = 100

    decoded = greedy_ctc_decode(
        scores,
        output_lengths=[
            4,
            3,
        ],
    )

    expected_decoded = [
        (
            1,
            2,
        ),
        (
            3,
            3,
        ),
    ]

    decoder_pass = (
        decoded
        == expected_decoded
    )

    if not decoder_pass:

        raise RuntimeError(
            "Length-safe decoder synthetic "
            "verification failed."
        )

    # ---------------------------------------------------------
    # Synthetic feasibility proof for repeated labels.
    # ---------------------------------------------------------

    repeated_target = np.array(
        [
            [
                8,
                8,
                9,
            ]
        ],
        dtype=np.int64,
    )

    repeated_margin = (
        validate_ctc_batch_feasibility(
            repeated_target,
            target_lengths=[
                3
            ],
            output_lengths=[
                4
            ],
        )
    )

    if int(
        repeated_margin[
            0
        ]
    ) != 0:

        raise RuntimeError(
            "Adjacent-repeat feasibility "
            "verification failed."
        )

    # ---------------------------------------------------------
    # PyTorch availability is informational locally.
    # ---------------------------------------------------------

    try:

        import torch

        torch_available = True

        torch_version = (
            torch.__version__
        )

    except ImportError:

        torch_available = False

        torch_version = None

    implementation_hashes = {
        "ctc_decode": (
            sha256_file(
                PROJECT_ROOT
                / "src"
                / "evaluation"
                / "ctc_decode.py"
            )
        ),

        "ctc_loss": (
            sha256_file(
                PROJECT_ROOT
                / "src"
                / "training"
                / "ctc_loss.py"
            )
        ),

        "tests": (
            sha256_file(
                PROJECT_ROOT
                / "tests"
                / "test_ctc_loss_decoder.py"
            )
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 21,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "ctc_schema": {
            "classes": (
                456
            ),

            "blank_id": (
                0
            ),

            "target_padding_value": (
                -1
            ),

            "zero_infinity": (
                False
            ),
        },

        "targets": {
            "sentence_templates": (
                len(
                    target_sequences
                )
            ),

            "packed_tokens": (
                int(
                    packed.shape[
                        0
                    ]
                )
            ),

            "pack_unpack_roundtrip_failures": (
                roundtrip_failures
            ),

            "packed_blank_count": (
                packed_blank_count
            ),

            "packed_padding_count": (
                packed_padding_count
            ),
        },

        "decoder": {
            "order": [
                "slice_to_valid_output_length",
                "argmax_over_classes",
                "collapse_consecutive_repeats",
                "remove_ctc_blank",
            ],

            "synthetic_length_safe_test": (
                decoder_pass
            ),

            "padded_timesteps_decoded": (
                False
            ),
        },

        "loss": {
            "input_logit_layout": (
                "B_T_C"
            ),

            "pytorch_ctc_layout": (
                "T_B_C"
            ),

            "log_softmax_before_loss": (
                True
            ),

            "targets_flattened_using": (
                "target_lengths"
            ),

            "feasibility_checked_before_loss": (
                True
            ),

            "zero_infinity": (
                False
            ),
        },

        "stage20": {
            "sample_audit_hash_match": (
                True
            ),

            "selected_stride": (
                int(
                    stage20_contract[
                        "windowing"
                    ][
                        "selected_stride"
                    ]
                )
            ),

            "selected_infeasible_recordings": (
                stage20_selected_infeasible
            ),
        },

        "local_runtime": {
            "pytorch_available": (
                torch_available
            ),

            "pytorch_version": (
                torch_version
            ),

            "note": (
                "Local PyTorch is optional. "
                "The loss wrapper imports PyTorch "
                "lazily; GPU runtime is validated "
                "during Colab model smoke tests."
            ),
        },

        "hashes": (
            implementation_hashes
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 21 — CTC Loss and Length-Safe Decoder

## CTC schema

Output classes:

456

Blank ID:

0

Temporary target padding:

-1

## Reference targets

All 300 Stage-10 sentence templates were packed into one flat CTC
target stream and reconstructed using their frozen target lengths.

Round-trip failures:

{roundtrip_failures}

Blank IDs in packed targets:

{packed_blank_count}

Padding IDs in packed targets:

{packed_padding_count}

## Decoder order

The frozen greedy decoder performs:

1. slice each sample to its valid model output length;
2. argmax over output classes;
3. collapse consecutive repeated predictions;
4. remove CTC blank ID 0.

Padding is therefore never decoded.

## CTC loss

Model logits are represented as:

`B x T x C`

The PyTorch loss wrapper applies log-softmax and converts this to:

`T x B x C`

Reference targets are packed using their true target lengths.

Before loss calculation, every sample is checked against the exact CTC
minimum alignment rule.

`zero_infinity=False` is intentional. Invalid CTC examples must be
detected rather than silently converted to zero loss.

## Local runtime

PyTorch available locally:

{torch_available}

This does not affect the frozen scientific contract. PyTorch is loaded
lazily by the loss wrapper because GPU model execution occurs in the
Colab environment.

## Stage-20 dependency

Selected Stage-20 stride:

{stage20_contract["windowing"]["selected_stride"]}

Selected-rule infeasible recordings:

{stage20_selected_infeasible}
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("CTC SCHEMA")
    print("-" * 78)

    print(
        "Output classes:",
        456,
    )

    print(
        "Blank ID:",
        0,
    )

    print(
        "Target padding:",
        -1,
    )

    print(
        "zero_infinity:",
        False,
    )

    print()
    print("TARGET PACKING")
    print("-" * 78)

    print(
        "Sentence templates:",
        len(
            target_sequences
        ),
    )

    print(
        "Packed target tokens:",
        packed.shape[
            0
        ],
    )

    print(
        "Pack/unpack failures:",
        roundtrip_failures,
    )

    print(
        "Blank IDs in packed targets:",
        packed_blank_count,
    )

    print(
        "Padding IDs in packed targets:",
        packed_padding_count,
    )

    print()
    print("DECODER CONTRACT")
    print("-" * 78)

    print(
        "Slice before argmax:",
        True,
    )

    print(
        "Collapse repeats:",
        True,
    )

    print(
        "Remove blank:",
        True,
    )

    print(
        "Decode padded timesteps:",
        False,
    )

    print(
        "Synthetic decoder test:",
        decoder_pass,
    )

    print()
    print("LOSS CONTRACT")
    print("-" * 78)

    print(
        "Input logits:",
        "B x T x C",
    )

    print(
        "CTCLoss log_probs:",
        "T x B x C",
    )

    print(
        "Pre-loss feasibility guard:",
        True,
    )

    print(
        "zero_infinity:",
        False,
    )

    print()
    print("STAGE-20 DEPENDENCY")
    print("-" * 78)

    print(
        "Audit hash match:",
        True,
    )

    print(
        "Selected stride:",
        stage20_contract[
            "windowing"
        ][
            "selected_stride"
        ],
    )

    print(
        "Selected infeasible:",
        stage20_selected_infeasible,
    )

    print()
    print("LOCAL PYTORCH")
    print("-" * 78)

    print(
        "Available:",
        torch_available,
    )

    print(
        "Version:",
        torch_version,
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    print(
        "Saved:",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        DOC_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    stage21_pass = all(
        [
            len(
                target_sequences
            ) == 300,
            len(
                token_to_id
            ) == 456,
            token_to_id[
                "<blank>"
            ] == 0,
            roundtrip_failures == 0,
            packed_blank_count == 0,
            packed_padding_count == 0,
            decoder_pass,
            stage20_selected_infeasible
            == 0,
        ]
    )

    print()

    if stage21_pass:

        print(
            "FINAL STATUS: "
            "STAGE 21 BUILD PASSED"
        )

    else:

        print(
            "FINAL STATUS: "
            "STAGE 21 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

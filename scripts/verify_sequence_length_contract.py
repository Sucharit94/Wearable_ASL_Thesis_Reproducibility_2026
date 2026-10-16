from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.lengths import (
    assert_mask_matches_lengths,
    full_window_lengths,
    identity_lengths,
    padding_mask_from_lengths,
    valid_mask_from_lengths,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.csv"
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

STAGE21_CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
    / "stage21_ctc_loss_decoder_contract.json"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
    / "stage22_sequence_length_contract.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_22_LENGTH_SAFE_SEQUENCE_PROCESSING.md"
)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

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


def load_csv(
    path: Path,
) -> list[dict]:

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 22 — LENGTH-SAFE "
        "SEQUENCE PROCESSING"
    )
    print("=" * 78)

    recorded_stage20_hash = (
        STAGE20_HASH_PATH
        .read_text(
            encoding="utf-8-sig"
        )
        .strip()
        .split()[0]
        .lower()
    )

    actual_stage20_hash = (
        sha256_file(
            STAGE20_AUDIT_PATH
        )
    )

    if (
        recorded_stage20_hash
        != actual_stage20_hash
    ):

        raise RuntimeError(
            "Stage-20 audit hash mismatch."
        )

    stage20 = json.loads(
        STAGE20_CONTRACT_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage21 = json.loads(
        STAGE21_CONTRACT_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    selected_stride = int(
        stage20[
            "windowing"
        ][
            "selected_stride"
        ]
    )

    window_length = int(
        stage20[
            "windowing"
        ][
            "window_length"
        ]
    )

    if (
        int(
            stage20[
                "windowing"
            ][
                "selected_infeasible_recordings"
            ]
        )
        != 0
    ):

        raise RuntimeError(
            "Stage-20 selected rule "
            "is not fully feasible."
        )

    if (
        stage21[
            "decoder"
        ][
            "padded_timesteps_decoded"
        ]
        is not False
    ):

        raise RuntimeError(
            "Stage-21 decoder contract changed."
        )

    index_rows = load_csv(
        CACHE_INDEX_PATH
    )

    audit_rows = load_csv(
        STAGE20_AUDIT_PATH
    )

    if len(index_rows) != 7400:

        raise RuntimeError(
            "Cache index must contain "
            "7,400 samples."
        )

    if len(audit_rows) != 7400:

        raise RuntimeError(
            "Stage-20 audit must contain "
            "7,400 samples."
        )

    index_by_sample = {
        row[
            "sample_id"
        ]: row
        for row in index_rows
    }

    audit_by_sample = {
        row[
            "sample_id"
        ]: row
        for row in audit_rows
    }

    if (
        set(
            index_by_sample
        )
        != set(
            audit_by_sample
        )
    ):

        raise RuntimeError(
            "Stage-17 and Stage-20 "
            "sample sets differ."
        )

    sample_ids = sorted(
        index_by_sample
    )

    raw_lengths = np.asarray(
        [
            int(
                index_by_sample[
                    sample_id
                ][
                    "timesteps"
                ]
            )
            for sample_id in sample_ids
        ],
        dtype=np.int64,
    )

    direct_lengths = identity_lengths(
        raw_lengths
    )

    direct_mismatches = int(
        np.sum(
            direct_lengths
            != raw_lengths
        )
    )

    selected_lengths = (
        full_window_lengths(
            raw_lengths,
            window_length=(
                window_length
            ),
            stride=(
                selected_stride
            ),
        )
    )

    stage20_selected_lengths = (
        np.asarray(
            [
                int(
                    audit_by_sample[
                        sample_id
                    ][
                        "selected_window_count"
                    ]
                )
                for sample_id in sample_ids
            ],
            dtype=np.int64,
        )
    )

    window_length_mismatches = int(
        np.sum(
            selected_lengths
            != stage20_selected_lengths
        )
    )

    if direct_mismatches != 0:

        raise RuntimeError(
            "Direct length propagation mismatch."
        )

    if window_length_mismatches != 0:

        raise RuntimeError(
            "Stage-22 window lengths disagree "
            "with Stage-20 audit."
        )

    selected_feasibility_failures = sum(
        row[
            "selected_feasible"
        ] != "True"
        for row in audit_rows
    )

    if selected_feasibility_failures:

        raise RuntimeError(
            "Selected Stage-20 temporal rule "
            "contains infeasible samples."
        )

    # ---------------------------------------------------------
    # Mask proof using shortest, median and longest samples.
    # ---------------------------------------------------------

    ordered = sorted(
        zip(
            raw_lengths.tolist(),
            sample_ids,
            strict=True,
        )
    )

    probe_ids = [
        ordered[0][1],
        ordered[
            len(
                ordered
            )
            // 2
        ][1],
        ordered[-1][1],
    ]

    probe_lengths = np.asarray(
        [
            int(
                index_by_sample[
                    sample_id
                ][
                    "timesteps"
                ]
            )
            for sample_id in probe_ids
        ],
        dtype=np.int64,
    )

    padded_time = int(
        probe_lengths.max()
    )

    valid_mask = (
        valid_mask_from_lengths(
            probe_lengths,
            padded_time=padded_time,
        )
    )

    padding_mask = (
        padding_mask_from_lengths(
            probe_lengths,
            padded_time=padded_time,
        )
    )

    assert_mask_matches_lengths(
        valid_mask,
        probe_lengths,
        padding_mask=False,
    )

    assert_mask_matches_lengths(
        padding_mask,
        probe_lengths,
        padding_mask=True,
    )

    mask_sum_failures = int(
        np.sum(
            valid_mask.sum(
                axis=1
            )
            != probe_lengths
        )
    )

    mask_inverse_pass = bool(
        np.array_equal(
            padding_mask,
            ~valid_mask,
        )
    )

    if mask_sum_failures:

        raise RuntimeError(
            "Valid mask counts do not match "
            "sequence lengths."
        )

    if not mask_inverse_pass:

        raise RuntimeError(
            "Padding mask is not exact inverse "
            "of valid mask."
        )

    implementation_hash = (
        sha256_file(
            PROJECT_ROOT
            / "src"
            / "data"
            / "lengths.py"
        )
    )

    test_hash = (
        sha256_file(
            PROJECT_ROOT
            / "tests"
            / "test_sequence_lengths.py"
        )
    )

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 22,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "principles": {
            "valid_length_is_first_class": (
                True
            ),

            "approximate_lengths_allowed_in_code": (
                False
            ),

            "padded_time_used_as_valid_length": (
                False
            ),

            "same_final_length_used_for_ctc_loss_and_decoder": (
                True
            ),
        },

        "direct_sequence": {
            "initial_rule": (
                "output_length = raw_valid_T"
            ),

            "recordings_checked": (
                7400
            ),

            "mismatches": (
                direct_mismatches
            ),
        },

        "g1_c4_window_adapter": {
            "window_length": (
                window_length
            ),

            "stride": (
                selected_stride
            ),

            "formula": (
                "1 + floor((T-W)/S)"
            ),

            "full_windows_only": (
                True
            ),

            "recordings_checked": (
                7400
            ),

            "stage20_length_mismatches": (
                window_length_mismatches
            ),

            "infeasible_recordings": (
                selected_feasibility_failures
            ),

            "minimum_window_length": (
                int(
                    selected_lengths.min()
                )
            ),

            "maximum_window_length": (
                int(
                    selected_lengths.max()
                )
            ),
        },

        "conv1d": {
            "formula": (
                "floor((L + 2P - "
                "D*(K-1) - 1)/S) + 1"
            ),

            "model_specific_parameters": (
                "frozen when each model "
                "architecture is implemented"
            ),
        },

        "recurrent_policy": {
            "bilstm_valid_length_change": (
                "none"
            ),

            "implementation_requirement": (
                "pack_padded_sequence or "
                "equivalent valid-length-safe processing"
            ),
        },

        "transformer_policy": {
            "valid_length_change": (
                "none unless an explicit "
                "temporal downsampling layer precedes it"
            ),

            "mask_requirement": (
                "exact padding mask derived "
                "from propagated valid lengths"
            ),
        },

        "swin_policy": {
            "source": (
                "Stage-20 full-window sequence"
            ),

            "time_global_pooling": (
                False
            ),

            "requirement": (
                "Every patch/merge operation "
                "that changes temporal resolution "
                "must propagate exact valid lengths"
            ),
        },

        "mask_verification": {
            "probe_sample_ids": (
                probe_ids
            ),

            "probe_lengths": (
                [
                    int(value)
                    for value
                    in probe_lengths
                ]
            ),

            "mask_sum_failures": (
                mask_sum_failures
            ),

            "padding_mask_exact_inverse": (
                mask_inverse_pass
            ),
        },

        "stage20_dependency": {
            "audit_hash_match": (
                True
            ),

            "selected_stride": (
                selected_stride
            ),

            "selected_infeasible_recordings": (
                selected_feasibility_failures
            ),
        },

        "stage21_dependency": {
            "decoder_slices_valid_length": (
                True
            ),

            "padded_timesteps_decoded": (
                False
            ),

            "pre_loss_feasibility_guard": (
                stage21[
                    "loss"
                ][
                    "feasibility_checked_before_loss"
                ]
            ),
        },

        "hashes": {
            "implementation_sha256": (
                implementation_hash
            ),

            "test_sha256": (
                test_hash
            ),
        },
    }

    OUTPUT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 22 — Length-Safe Sequence Processing

## Core rule

Every sample retains an exact valid temporal length from input through
the final CTC logits.

Padded tensor time is never substituted for valid sample length.

## Direct models

C1, C2 and C3 begin from the raw valid sequence length `T`.

Operations that preserve time retain the same valid length.

Any later convolution or downsampling operation must apply its exact
mathematical output-length formula.

## G1 and C4

Stage 20 froze:

- window length: {window_length}
- stride: {selected_stride}

The full-window temporal length is:

`1 + floor((T - W) / S)`

All 7,400 Stage-22 recomputations match the Stage-20 frozen audit.

## Conv1D

Exact formula:

`floor((L + 2P - D*(K-1) - 1) / S) + 1`

No code may substitute an approximate expression such as `T/2` or
`T/4`.

## BiLSTM

BiLSTM itself does not alter valid temporal length.

Variable-length recurrent processing must use packed sequences or an
equivalent method that prevents padded positions from influencing the
recurrent representation.

## Transformer

The Transformer receives an exact padding mask generated from the
propagated valid lengths.

## Tiny-Swin

The temporal-feature Swin path begins from the Stage-20 full-window
sequence.

No global temporal pooling is permitted before CTC.

Every temporal patch/merge operation must propagate exact lengths.

## CTC boundary

The final propagated output length is shared by:

- the Stage-21 pre-loss feasibility guard,
- CTCLoss,
- greedy CTC decoding.

Therefore loss and decoding cannot disagree about which temporal
positions are valid.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("RAW LENGTH CONTRACT")
    print("-" * 78)

    print(
        "Recordings checked:",
        len(
            raw_lengths
        ),
    )

    print(
        "Minimum raw T:",
        int(
            raw_lengths.min()
        ),
    )

    print(
        "Maximum raw T:",
        int(
            raw_lengths.max()
        ),
    )

    print(
        "Direct mismatches:",
        direct_mismatches,
    )

    print()
    print("G1/C4 WINDOW LENGTH CONTRACT")
    print("-" * 78)

    print(
        "Window length:",
        window_length,
    )

    print(
        "Stride:",
        selected_stride,
    )

    print(
        "Minimum output windows:",
        int(
            selected_lengths.min()
        ),
    )

    print(
        "Maximum output windows:",
        int(
            selected_lengths.max()
        ),
    )

    print(
        "Stage-20 mismatches:",
        window_length_mismatches,
    )

    print(
        "Infeasible recordings:",
        selected_feasibility_failures,
    )

    print()
    print("MASK CONTRACT")
    print("-" * 78)

    print(
        "Probe IDs:",
        probe_ids,
    )

    print(
        "Probe lengths:",
        probe_lengths.tolist(),
    )

    print(
        "Mask sum failures:",
        mask_sum_failures,
    )

    print(
        "Padding mask inverse:",
        mask_inverse_pass,
    )

    print()
    print("MODEL POLICIES")
    print("-" * 78)

    print(
        "BiLSTM length preserving:",
        True,
    )

    print(
        "BiLSTM packed/equivalent required:",
        True,
    )

    print(
        "Transformer exact padding mask:",
        True,
    )

    print(
        "Tiny-Swin global time pooling:",
        False,
    )

    print(
        "Exact Conv1D formula required:",
        True,
    )

    print()
    print("STAGE-20 / STAGE-21 DEPENDENCIES")
    print("-" * 78)

    print(
        "Stage-20 audit hash match:",
        True,
    )

    print(
        "Stage-20 selected stride:",
        selected_stride,
    )

    print(
        "Stage-20 infeasible:",
        selected_feasibility_failures,
    )

    print(
        "Stage-21 decoder ignores padding:",
        True,
    )

    print(
        "Stage-21 pre-loss feasibility:",
        stage21[
            "loss"
        ][
            "feasibility_checked_before_loss"
        ],
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    print(
        "Saved:",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        DOC_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    stage22_pass = all(
        [
            len(
                raw_lengths
            ) == 7400,

            direct_mismatches == 0,

            window_length_mismatches == 0,

            selected_feasibility_failures == 0,

            mask_sum_failures == 0,

            mask_inverse_pass,

            stage21[
                "loss"
            ][
                "feasibility_checked_before_loss"
            ]
            is True,
        ]
    )

    print()

    if stage22_pass:

        print(
            "FINAL STATUS: "
            "STAGE 22 BUILD PASSED"
        )

    else:

        print(
            "FINAL STATUS: "
            "STAGE 22 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

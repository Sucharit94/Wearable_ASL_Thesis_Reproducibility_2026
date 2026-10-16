from __future__ import annotations

from typing import Sequence

import numpy as np


INPUT_PADDING_VALUE = 0.0

# -1 is deliberately not CTC blank ID 0.
TARGET_PADDING_VALUE = -1


def collate_numpy(
    batch: Sequence[dict],
) -> dict:
    """
    Pad variable-length T x F sequences without changing
    their separately recorded valid lengths.

    Inputs are padded with 0 in normalized space.

    Targets are padded with -1, never with CTC blank 0.
    Stage 21 will construct the exact CTC-loss target form.
    """

    if not batch:
        raise ValueError(
            "Cannot collate an empty batch."
        )

    batch_size = len(
        batch
    )

    input_lengths = np.asarray(
        [
            int(
                item[
                    "input_length"
                ]
            )
            for item in batch
        ],
        dtype=np.int64,
    )

    target_lengths = np.asarray(
        [
            int(
                item[
                    "target_length"
                ]
            )
            for item in batch
        ],
        dtype=np.int64,
    )

    if np.any(
        input_lengths <= 0
    ):
        raise ValueError(
            "Batch contains non-positive "
            "input length."
        )

    if np.any(
        target_lengths <= 0
    ):
        raise ValueError(
            "Batch contains non-positive "
            "target length."
        )

    feature_widths = {
        int(
            item[
                "inputs"
            ].shape[1]
        )
        for item in batch
    }

    if len(feature_widths) != 1:
        raise ValueError(
            "Batch feature widths differ."
        )

    feature_width = next(
        iter(
            feature_widths
        )
    )

    max_input_length = int(
        input_lengths.max()
    )

    max_target_length = int(
        target_lengths.max()
    )

    inputs = np.full(
        (
            batch_size,
            max_input_length,
            feature_width,
        ),
        INPUT_PADDING_VALUE,
        dtype=np.float32,
    )

    valid_mask = np.zeros(
        (
            batch_size,
            max_input_length,
        ),
        dtype=bool,
    )

    targets_padded = np.full(
        (
            batch_size,
            max_target_length,
        ),
        TARGET_PADDING_VALUE,
        dtype=np.int64,
    )

    for batch_index, item in enumerate(
        batch
    ):

        sequence = np.asarray(
            item[
                "inputs"
            ],
            dtype=np.float32,
        )

        target = np.asarray(
            item[
                "target_ids"
            ],
            dtype=np.int64,
        )

        input_length = int(
            item[
                "input_length"
            ]
        )

        target_length = int(
            item[
                "target_length"
            ]
        )

        if sequence.ndim != 2:
            raise ValueError(
                "Input sequence is not T x F."
            )

        if sequence.shape[0] != (
            input_length
        ):
            raise ValueError(
                "Stored input length does not "
                "match sequence."
            )

        if target.ndim != 1:
            raise ValueError(
                "Target IDs must be 1D."
            )

        if target.shape[0] != (
            target_length
        ):
            raise ValueError(
                "Stored target length does not "
                "match target IDs."
            )

        if np.any(
            target == 0
        ):
            raise ValueError(
                "CTC blank ID 0 found inside "
                "reference target."
            )

        inputs[
            batch_index,
            :input_length,
            :,
        ] = sequence

        valid_mask[
            batch_index,
            :input_length,
        ] = True

        targets_padded[
            batch_index,
            :target_length,
        ] = target

    return {
        "sample_ids": tuple(
            item[
                "sample_id"
            ]
            for item in batch
        ),

        "subjects": tuple(
            item[
                "subject"
            ]
            for item in batch
        ),

        "sentence_ids": np.asarray(
            [
                int(
                    item[
                        "sentence_id"
                    ]
                )
                for item in batch
            ],
            dtype=np.int64,
        ),

        "inputs": (
            inputs
        ),

        "input_lengths": (
            input_lengths
        ),

        "valid_mask": (
            valid_mask
        ),

        "targets_padded": (
            targets_padded
        ),

        "target_lengths": (
            target_lengths
        ),

        "gloss_raw": tuple(
            item[
                "gloss_raw"
            ]
            for item in batch
        ),

        "english_raw": tuple(
            item[
                "english_raw"
            ]
            for item in batch
        ),
    }

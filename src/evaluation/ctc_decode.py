from __future__ import annotations

from collections.abc import Sequence

import numpy as np


CTC_BLANK_ID = 0


def collapse_ctc_path(
    path: Sequence[int],
    *,
    blank_id: int = CTC_BLANK_ID,
) -> tuple[int, ...]:
    """
    Standard greedy CTC collapse:

    1. collapse consecutive identical symbols;
    2. remove blank symbols.

    A blank between two identical symbols keeps them
    as two separate decoded target symbols.
    """

    blank_id = int(
        blank_id
    )

    if blank_id < 0:
        raise ValueError(
            "blank_id cannot be negative."
        )

    collapsed = []

    previous = None

    for value in path:

        token = int(
            value
        )

        if token < 0:
            raise ValueError(
                "CTC path contains "
                "a negative class ID."
            )

        if token != previous:

            if token != blank_id:

                collapsed.append(
                    token
                )

        previous = token

    return tuple(
        collapsed
    )


def validate_output_lengths(
    output_lengths: Sequence[int],
    *,
    batch_size: int,
    padded_time: int,
) -> np.ndarray:

    lengths = np.asarray(
        output_lengths,
        dtype=np.int64,
    )

    if lengths.ndim != 1:

        raise ValueError(
            "output_lengths must be 1D."
        )

    if lengths.shape[0] != batch_size:

        raise ValueError(
            "output_lengths size does not "
            "match batch size."
        )

    if np.any(
        lengths <= 0
    ):

        raise ValueError(
            "Every valid output length "
            "must be positive."
        )

    if np.any(
        lengths > padded_time
    ):

        raise ValueError(
            "A valid output length exceeds "
            "the padded temporal dimension."
        )

    return lengths


def greedy_ctc_decode(
    scores_btc: np.ndarray,
    output_lengths: Sequence[int],
    *,
    blank_id: int = CTC_BLANK_ID,
) -> list[tuple[int, ...]]:
    """
    Length-safe greedy CTC decoding.

    scores_btc:
        B x T x C logits, probabilities or log-probabilities.

    Critical order:
        slice to valid output length
        -> argmax
        -> collapse repeats
        -> remove blank

    Padded timesteps are therefore never decoded.
    """

    scores = np.asarray(
        scores_btc
    )

    if scores.ndim != 3:

        raise ValueError(
            "scores_btc must have shape "
            "B x T x C."
        )

    batch_size, padded_time, classes = (
        scores.shape
    )

    if classes <= 1:

        raise ValueError(
            "CTC requires more than "
            "one output class."
        )

    blank_id = int(
        blank_id
    )

    if not (
        0 <= blank_id < classes
    ):

        raise ValueError(
            "blank_id is outside "
            "the class range."
        )

    lengths = validate_output_lengths(
        output_lengths,
        batch_size=batch_size,
        padded_time=padded_time,
    )

    decoded = []

    for batch_index in range(
        batch_size
    ):

        valid_length = int(
            lengths[
                batch_index
            ]
        )

        # -----------------------------------------------------
        # CRITICAL:
        # Slice BEFORE argmax and before CTC collapse.
        # -----------------------------------------------------

        valid_scores = scores[
            batch_index,
            :valid_length,
            :,
        ]

        path = np.argmax(
            valid_scores,
            axis=-1,
        )

        decoded.append(
            collapse_ctc_path(
                path,
                blank_id=blank_id,
            )
        )

    return decoded

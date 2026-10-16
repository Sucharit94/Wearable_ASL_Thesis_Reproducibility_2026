from __future__ import annotations

from collections.abc import Sequence


CTC_BLANK_ID = 0


def validate_ctc_target(
    target_ids: Sequence[int],
) -> tuple[int, ...]:

    ids = tuple(
        int(value)
        for value in target_ids
    )

    if not ids:
        raise ValueError(
            "CTC target cannot be empty."
        )

    if CTC_BLANK_ID in ids:
        raise ValueError(
            "CTC blank ID 0 cannot occur "
            "inside a reference target."
        )

    if any(
        value < 1
        for value in ids
    ):
        raise ValueError(
            "CTC target IDs must be positive."
        )

    return ids


def adjacent_repeat_count(
    target_ids: Sequence[int],
) -> int:

    ids = validate_ctc_target(
        target_ids
    )

    return sum(
        current == previous
        for previous, current
        in zip(
            ids,
            ids[1:],
        )
    )


def minimum_ctc_timesteps(
    target_ids: Sequence[int],
) -> int:
    """
    Minimum number of temporal output positions needed
    for a valid CTC alignment.

    Each adjacent repeated target symbol requires at least
    one separating blank.
    """

    ids = validate_ctc_target(
        target_ids
    )

    repeats = adjacent_repeat_count(
        ids
    )

    return (
        len(ids)
        + repeats
    )


def full_window_count(
    input_length: int,
    *,
    window_length: int,
    stride: int,
) -> int:
    """
    Count full, non-padded sliding windows.

    N = 1 + floor((T - W) / S), when T >= W.
    """

    input_length = int(
        input_length
    )

    window_length = int(
        window_length
    )

    stride = int(
        stride
    )

    if input_length < 0:
        raise ValueError(
            "input_length cannot be negative."
        )

    if window_length <= 0:
        raise ValueError(
            "window_length must be positive."
        )

    if stride <= 0:
        raise ValueError(
            "stride must be positive."
        )

    if input_length < window_length:
        return 0

    return (
        1
        + (
            input_length
            - window_length
        )
        // stride
    )


def ctc_margin(
    output_length: int,
    target_ids: Sequence[int],
) -> int:

    return (
        int(
            output_length
        )
        - minimum_ctc_timesteps(
            target_ids
        )
    )


def is_ctc_feasible(
    output_length: int,
    target_ids: Sequence[int],
) -> bool:

    return (
        ctc_margin(
            output_length,
            target_ids,
        )
        >= 0
    )

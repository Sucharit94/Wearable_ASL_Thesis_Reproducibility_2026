from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _as_positive_lengths(
    lengths: Sequence[int] | np.ndarray,
) -> np.ndarray:

    values = np.asarray(
        lengths,
        dtype=np.int64,
    )

    if values.ndim != 1:
        raise ValueError(
            "Lengths must be a 1D array."
        )

    if values.size == 0:
        raise ValueError(
            "Lengths cannot be empty."
        )

    if np.any(values <= 0):
        raise ValueError(
            "All valid lengths must be positive."
        )

    return values


def identity_lengths(
    lengths: Sequence[int] | np.ndarray,
) -> np.ndarray:
    """
    Length-preserving temporal operation.

    Used conceptually for recurrent/attention operations
    that do not change temporal resolution.
    """

    return _as_positive_lengths(
        lengths
    ).copy()


def full_window_lengths(
    lengths: Sequence[int] | np.ndarray,
    *,
    window_length: int,
    stride: int,
) -> np.ndarray:
    """
    Number of complete, non-padded sliding windows:

    1 + floor((T - W) / S), for T >= W.
    """

    values = _as_positive_lengths(
        lengths
    )

    window_length = int(
        window_length
    )

    stride = int(
        stride
    )

    if window_length <= 0:
        raise ValueError(
            "window_length must be positive."
        )

    if stride <= 0:
        raise ValueError(
            "stride must be positive."
        )

    output = np.zeros_like(
        values
    )

    valid = (
        values >= window_length
    )

    output[valid] = (
        1
        + (
            values[valid]
            - window_length
        )
        // stride
    )

    if np.any(output <= 0):
        bad = np.flatnonzero(
            output <= 0
        ).tolist()

        raise ValueError(
            "Window operation produced "
            f"zero valid positions at indices {bad}."
        )

    return output


def conv1d_output_lengths(
    lengths: Sequence[int] | np.ndarray,
    *,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> np.ndarray:
    """
    Exact PyTorch-style Conv1D temporal output formula:

    floor(
        (L + 2P - D(K-1) - 1) / S
    ) + 1
    """

    values = _as_positive_lengths(
        lengths
    )

    kernel_size = int(
        kernel_size
    )

    stride = int(
        stride
    )

    padding = int(
        padding
    )

    dilation = int(
        dilation
    )

    if kernel_size <= 0:
        raise ValueError(
            "kernel_size must be positive."
        )

    if stride <= 0:
        raise ValueError(
            "stride must be positive."
        )

    if padding < 0:
        raise ValueError(
            "padding cannot be negative."
        )

    if dilation <= 0:
        raise ValueError(
            "dilation must be positive."
        )

    numerator = (
        values
        + 2 * padding
        - dilation
        * (
            kernel_size
            - 1
        )
        - 1
    )

    output = (
        np.floor_divide(
            numerator,
            stride,
        )
        + 1
    )

    if np.any(output <= 0):

        bad = np.flatnonzero(
            output <= 0
        ).tolist()

        raise ValueError(
            "Conv1D produced non-positive "
            f"temporal lengths at indices {bad}."
        )

    return np.asarray(
        output,
        dtype=np.int64,
    )


def propagate_conv1d_stack(
    lengths: Sequence[int] | np.ndarray,
    layers: Sequence[dict],
) -> np.ndarray:
    """
    Apply exact temporal length propagation through
    an ordered Conv1D stack.

    Architecture-specific layer parameters are supplied
    by the model configuration later.
    """

    output = _as_positive_lengths(
        lengths
    ).copy()

    for layer_index, layer in enumerate(
        layers
    ):

        try:

            output = conv1d_output_lengths(
                output,
                kernel_size=(
                    layer[
                        "kernel_size"
                    ]
                ),
                stride=(
                    layer.get(
                        "stride",
                        1,
                    )
                ),
                padding=(
                    layer.get(
                        "padding",
                        0,
                    )
                ),
                dilation=(
                    layer.get(
                        "dilation",
                        1,
                    )
                ),
            )

        except Exception as exc:

            raise ValueError(
                "Failed temporal propagation "
                f"at Conv1D layer {layer_index}."
            ) from exc

    return output


def validate_lengths_against_padded_time(
    lengths: Sequence[int] | np.ndarray,
    *,
    padded_time: int,
) -> np.ndarray:

    values = _as_positive_lengths(
        lengths
    )

    padded_time = int(
        padded_time
    )

    if padded_time <= 0:
        raise ValueError(
            "padded_time must be positive."
        )

    if np.any(
        values > padded_time
    ):

        raise ValueError(
            "A valid sequence length exceeds "
            "the padded temporal dimension."
        )

    return values


def valid_mask_from_lengths(
    lengths: Sequence[int] | np.ndarray,
    *,
    padded_time: int | None = None,
) -> np.ndarray:
    """
    True = valid temporal position.
    False = padding.
    """

    values = _as_positive_lengths(
        lengths
    )

    if padded_time is None:

        padded_time = int(
            values.max()
        )

    values = validate_lengths_against_padded_time(
        values,
        padded_time=padded_time,
    )

    positions = np.arange(
        padded_time,
        dtype=np.int64,
    )[
        None,
        :
    ]

    return (
        positions
        < values[
            :,
            None
        ]
    )


def padding_mask_from_lengths(
    lengths: Sequence[int] | np.ndarray,
    *,
    padded_time: int | None = None,
) -> np.ndarray:
    """
    True = padding position.

    This orientation matches the common Transformer
    key-padding-mask convention.
    """

    return ~valid_mask_from_lengths(
        lengths,
        padded_time=padded_time,
    )


def assert_mask_matches_lengths(
    mask: np.ndarray,
    lengths: Sequence[int] | np.ndarray,
    *,
    padding_mask: bool,
) -> None:

    values = _as_positive_lengths(
        lengths
    )

    observed = np.asarray(
        mask,
        dtype=bool,
    )

    if observed.ndim != 2:

        raise ValueError(
            "Mask must be B x T."
        )

    if observed.shape[0] != (
        values.shape[0]
    ):

        raise ValueError(
            "Mask batch dimension does not "
            "match lengths."
        )

    expected = (
        padding_mask_from_lengths(
            values,
            padded_time=(
                observed.shape[1]
            ),
        )
        if padding_mask
        else
        valid_mask_from_lengths(
            values,
            padded_time=(
                observed.shape[1]
            ),
        )
    )

    if not np.array_equal(
        observed,
        expected,
    ):

        raise ValueError(
            "Temporal mask does not match "
            "the supplied valid lengths."
        )

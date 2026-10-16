from __future__ import annotations

import numpy as np
import pytest

from src.data.lengths import (
    assert_mask_matches_lengths,
    conv1d_output_lengths,
    full_window_lengths,
    identity_lengths,
    padding_mask_from_lengths,
    propagate_conv1d_stack,
    valid_mask_from_lengths,
    validate_lengths_against_padded_time,
)


def test_identity_preserves_lengths():

    lengths = np.array(
        [
            81,
            241,
            831,
        ]
    )

    output = identity_lengths(
        lengths
    )

    np.testing.assert_array_equal(
        output,
        lengths,
    )


def test_stage20_window_reference_shortest():

    output = full_window_lengths(
        [
            81
        ],
        window_length=36,
        stride=18,
    )

    np.testing.assert_array_equal(
        output,
        [
            3
        ],
    )


def test_stage20_window_reference_longest():

    output = full_window_lengths(
        [
            831
        ],
        window_length=36,
        stride=18,
    )

    np.testing.assert_array_equal(
        output,
        [
            45
        ],
    )


def test_conv1d_exact_formula():

    output = conv1d_output_lengths(
        [
            81
        ],
        kernel_size=5,
        stride=2,
        padding=2,
    )

    np.testing.assert_array_equal(
        output,
        [
            41
        ],
    )


def test_two_conv_example_is_exact_not_approximate():

    output = propagate_conv1d_stack(
        [
            81
        ],
        layers=[
            {
                "kernel_size": 5,
                "stride": 2,
                "padding": 2,
            },
            {
                "kernel_size": 3,
                "stride": 2,
                "padding": 1,
            },
        ],
    )

    np.testing.assert_array_equal(
        output,
        [
            21
        ],
    )


def test_valid_mask():

    mask = valid_mask_from_lengths(
        [
            2,
            4,
        ],
        padded_time=5,
    )

    expected = np.array(
        [
            [
                True,
                True,
                False,
                False,
                False,
            ],
            [
                True,
                True,
                True,
                True,
                False,
            ],
        ],
        dtype=bool,
    )

    np.testing.assert_array_equal(
        mask,
        expected,
    )


def test_padding_mask_is_inverse():

    valid = valid_mask_from_lengths(
        [
            2,
            4,
        ],
        padded_time=5,
    )

    padding = padding_mask_from_lengths(
        [
            2,
            4,
        ],
        padded_time=5,
    )

    np.testing.assert_array_equal(
        padding,
        ~valid,
    )


def test_mask_validation_passes():

    lengths = [
        3,
        5,
    ]

    mask = valid_mask_from_lengths(
        lengths,
        padded_time=6,
    )

    assert_mask_matches_lengths(
        mask,
        lengths,
        padding_mask=False,
    )


def test_length_cannot_exceed_padded_time():

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):

        validate_lengths_against_padded_time(
            [
                5,
                7,
            ],
            padded_time=6,
        )


def test_non_positive_lengths_rejected():

    with pytest.raises(
        ValueError,
        match="positive",
    ):

        identity_lengths(
            [
                3,
                0,
            ]
        )

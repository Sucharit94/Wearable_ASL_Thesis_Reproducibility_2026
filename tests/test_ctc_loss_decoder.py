from __future__ import annotations

import numpy as np
import pytest

from src.evaluation.ctc_decode import (
    collapse_ctc_path,
    greedy_ctc_decode,
)
from src.training.ctc_loss import (
    pack_ctc_targets,
    unpack_ctc_targets,
    validate_ctc_batch_feasibility,
)


def test_basic_ctc_collapse():

    path = [
        1,
        1,
        0,
        2,
        2,
        0,
    ]

    assert collapse_ctc_path(
        path
    ) == (
        1,
        2,
    )


def test_blank_separates_identical_tokens():

    path = [
        5,
        5,
        0,
        5,
    ]

    assert collapse_ctc_path(
        path
    ) == (
        5,
        5,
    )


def test_decoder_slices_before_argmax_and_collapse():

    scores = np.zeros(
        (
            1,
            6,
            4,
        ),
        dtype=np.float32,
    )

    # Valid path = 1, 0, 2.
    scores[0, 0, 1] = 10
    scores[0, 1, 0] = 10
    scores[0, 2, 2] = 10

    # Padded region deliberately predicts class 3.
    scores[0, 3:, 3] = 100

    decoded = greedy_ctc_decode(
        scores,
        output_lengths=[
            3
        ],
    )

    assert decoded == [
        (
            1,
            2,
        )
    ]


def test_decoder_supports_different_valid_lengths():

    scores = np.zeros(
        (
            2,
            5,
            4,
        ),
        dtype=np.float32,
    )

    scores[0, 0, 1] = 5
    scores[0, 1, 1] = 5

    scores[1, 0, 2] = 5
    scores[1, 1, 0] = 5
    scores[1, 2, 3] = 5

    decoded = greedy_ctc_decode(
        scores,
        output_lengths=[
            2,
            3,
        ],
    )

    assert decoded == [
        (
            1,
        ),
        (
            2,
            3,
        ),
    ]


def test_pack_targets_removes_only_padding():

    targets = np.array(
        [
            [
                4,
                7,
                -1,
            ],
            [
                8,
                9,
                10,
            ],
        ],
        dtype=np.int64,
    )

    lengths = np.array(
        [
            2,
            3,
        ],
        dtype=np.int64,
    )

    packed = pack_ctc_targets(
        targets,
        lengths,
    )

    np.testing.assert_array_equal(
        packed,
        [
            4,
            7,
            8,
            9,
            10,
        ],
    )


def test_pack_unpack_roundtrip():

    targets = np.array(
        [
            [
                4,
                7,
                -1,
            ],
            [
                8,
                9,
                10,
            ],
        ],
        dtype=np.int64,
    )

    lengths = [
        2,
        3,
    ]

    packed = pack_ctc_targets(
        targets,
        lengths,
    )

    unpacked = unpack_ctc_targets(
        packed,
        lengths,
    )

    assert unpacked == [
        (
            4,
            7,
        ),
        (
            8,
            9,
            10,
        ),
    ]


def test_blank_inside_reference_target_rejected():

    targets = np.array(
        [
            [
                4,
                0,
                -1,
            ],
        ],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="blank",
    ):

        pack_ctc_targets(
            targets,
            [
                2
            ],
        )


def test_ctc_feasibility_includes_adjacent_repeat_requirement():

    targets = np.array(
        [
            [
                5,
                5,
                7,
            ],
        ],
        dtype=np.int64,
    )

    # Target length = 3.
    # One adjacent repeat => minimum CTC length = 4.

    margins = (
        validate_ctc_batch_feasibility(
            targets,
            target_lengths=[
                3
            ],
            output_lengths=[
                4
            ],
        )
    )

    np.testing.assert_array_equal(
        margins,
        [
            0
        ],
    )

    with pytest.raises(
        ValueError,
        match="CTC-infeasible",
    ):

        validate_ctc_batch_feasibility(
            targets,
            target_lengths=[
                3
            ],
            output_lengths=[
                3
            ],
        )

from __future__ import annotations

import pytest

from src.evaluation.ctc_feasibility import (
    adjacent_repeat_count,
    ctc_margin,
    full_window_count,
    is_ctc_feasible,
    minimum_ctc_timesteps,
)


def test_no_repeat_target():

    target = [
        1,
        2,
        3,
    ]

    assert (
        adjacent_repeat_count(
            target
        )
        == 0
    )

    assert (
        minimum_ctc_timesteps(
            target
        )
        == 3
    )


def test_one_adjacent_repeat():

    target = [
        1,
        1,
        2,
    ]

    assert (
        adjacent_repeat_count(
            target
        )
        == 1
    )

    assert (
        minimum_ctc_timesteps(
            target
        )
        == 4
    )


def test_three_same_tokens_need_extra_blanks():

    target = [
        5,
        5,
        5,
    ]

    assert (
        adjacent_repeat_count(
            target
        )
        == 2
    )

    assert (
        minimum_ctc_timesteps(
            target
        )
        == 5
    )


def test_reference_window_count():

    assert (
        full_window_count(
            81,
            window_length=36,
            stride=18,
        )
        == 3
    )


def test_smaller_stride_increases_windows():

    count18 = full_window_count(
        81,
        window_length=36,
        stride=18,
    )

    count6 = full_window_count(
        81,
        window_length=36,
        stride=6,
    )

    assert count6 > count18

    assert count6 == 8


def test_shorter_than_window_returns_zero():

    assert (
        full_window_count(
            35,
            window_length=36,
            stride=18,
        )
        == 0
    )


def test_margin_and_feasibility():

    target = [
        7,
        7,
        8,
    ]

    assert (
        minimum_ctc_timesteps(
            target
        )
        == 4
    )

    assert (
        ctc_margin(
            3,
            target,
        )
        == -1
    )

    assert not is_ctc_feasible(
        3,
        target,
    )

    assert is_ctc_feasible(
        4,
        target,
    )


def test_blank_inside_reference_rejected():

    with pytest.raises(
        ValueError,
        match="blank",
    ):

        minimum_ctc_timesteps(
            [
                4,
                0,
                5,
            ]
        )

from __future__ import annotations

import pytest

from src.evaluation.metrics import (
    levenshtein_alignment,
    score_batch,
    score_sequence,
)


def test_perfect_match():

    result = score_sequence(
        [
            1,
            2,
            3,
        ],
        [
            1,
            2,
            3,
        ],
    )

    assert result["edit_distance"] == 0
    assert result["wer"] == 0
    assert result["exact_match"] is True


def test_single_substitution():

    result = score_sequence(
        [
            1,
            2,
            3,
        ],
        [
            1,
            9,
            3,
        ],
    )

    assert result["substitutions"] == 1
    assert result["deletions"] == 0
    assert result["insertions"] == 0
    assert result["wer"] == pytest.approx(
        1 / 3
    )


def test_single_deletion():

    result = score_sequence(
        [
            1,
            2,
            3,
        ],
        [
            1,
            3,
        ],
    )

    assert result["substitutions"] == 0
    assert result["deletions"] == 1
    assert result["insertions"] == 0


def test_single_insertion():

    result = score_sequence(
        [
            1,
            2,
            3,
        ],
        [
            1,
            8,
            2,
            3,
        ],
    )

    assert result["substitutions"] == 0
    assert result["deletions"] == 0
    assert result["insertions"] == 1


def test_empty_hypothesis_is_all_deletions():

    result = score_sequence(
        [
            4,
            5,
            6,
        ],
        [],
    )

    assert result["deletions"] == 3
    assert result["wer"] == 1
    assert result["exact_match"] is False


def test_repeated_reference_tokens_are_supported():

    result = score_sequence(
        [
            5,
            5,
            7,
        ],
        [
            5,
            7,
        ],
    )

    assert result["deletions"] == 1
    assert result["edit_distance"] == 1


def test_alignment_tie_uses_substitution_precedence():

    result = score_sequence(
        [
            1,
            2,
        ],
        [
            2,
            1,
        ],
    )

    assert result["substitutions"] == 2
    assert result["deletions"] == 0
    assert result["insertions"] == 0


def test_pooled_wer_is_not_macro_sample_wer():

    result = score_batch(
        references=[
            [
                1,
            ],
            [
                1,
                2,
                3,
                4,
            ],
        ],
        hypotheses=[
            [],
            [
                1,
                2,
                3,
                9,
            ],
        ],
    )

    aggregate = result[
        "aggregate"
    ]

    assert aggregate["pooled_wer"] == (
        pytest.approx(
            2 / 5
        )
    )

    assert aggregate[
        "macro_sample_wer"
    ] == pytest.approx(
        0.625
    )


def test_aggregate_sdi_and_sentence_metrics():

    result = score_batch(
        references=[
            [
                1,
                2,
            ],
            [
                3,
                4,
                5,
            ],
            [
                6,
                7,
                8,
            ],
            [
                9,
                10,
                11,
            ],
        ],
        hypotheses=[
            [
                1,
                2,
            ],
            [
                3,
                99,
                5,
            ],
            [
                6,
                8,
            ],
            [
                9,
                12,
                10,
                11,
            ],
        ],
    )

    aggregate = result[
        "aggregate"
    ]

    assert aggregate["reference_tokens"] == 11
    assert aggregate["substitutions"] == 1
    assert aggregate["deletions"] == 1
    assert aggregate["insertions"] == 1
    assert aggregate["total_edits"] == 3
    assert aggregate["pooled_wer"] == pytest.approx(
        3 / 11
    )
    assert aggregate["ser"] == pytest.approx(
        0.75
    )
    assert aggregate[
        "exact_sentence_accuracy"
    ] == pytest.approx(
        0.25
    )


def test_blank_padding_and_negative_ids_rejected():

    with pytest.raises(
        ValueError,
        match="blank",
    ):
        score_sequence(
            [
                1,
                0,
            ],
            [
                1,
            ],
        )

    with pytest.raises(
        ValueError,
        match="padding",
    ):
        score_sequence(
            [
                1,
            ],
            [
                1,
                -1,
            ],
        )


def test_empty_reference_rejected():

    with pytest.raises(
        ValueError,
        match="cannot be empty",
    ):
        score_sequence(
            [],
            [],
        )


def test_batch_size_mismatch_rejected():

    with pytest.raises(
        ValueError,
        match="sizes do not match",
    ):
        score_batch(
            references=[
                [
                    1,
                ],
            ],
            hypotheses=[],
        )

from __future__ import annotations

import numpy as np
import pytest

from src.data.preprocessing import (
    CIRCULAR_FEATURE_WIDTH,
    PRIMARY_FEATURE_WIDTH,
    apply_feature_zscore,
    circular_encode_degrees,
    fit_feature_zscore,
    primary_raw_rotation,
)


def test_primary_representation_preserves_shape_and_values():

    x = np.arange(
        4 * PRIMARY_FEATURE_WIDTH,
        dtype=np.float64,
    ).reshape(
        4,
        PRIMARY_FEATURE_WIDTH,
    )

    y = primary_raw_rotation(
        x
    )

    assert y.shape == (
        4,
        PRIMARY_FEATURE_WIDTH,
    )

    assert y.dtype == np.float32

    np.testing.assert_array_equal(
        y,
        x.astype(
            np.float32
        ),
    )


def test_circular_encoding_shape_and_known_angles():

    x = np.zeros(
        (
            2,
            PRIMARY_FEATURE_WIDTH,
        ),
        dtype=np.float64,
    )

    x[0, 0] = 0.0
    x[1, 0] = 90.0

    y = circular_encode_degrees(
        x
    )

    assert y.shape == (
        2,
        CIRCULAR_FEATURE_WIDTH,
    )

    np.testing.assert_allclose(
        y[0, 0:2],
        [0.0, 1.0],
        atol=1e-6,
    )

    np.testing.assert_allclose(
        y[1, 0:2],
        [1.0, 0.0],
        atol=1e-6,
    )


def test_zscore_is_frame_weighted_population_statistics():

    train_a = np.array(
        [
            [0.0, 10.0],
            [2.0, 12.0],
        ],
        dtype=np.float64,
    )

    train_b = np.array(
        [
            [4.0, 14.0],
            [6.0, 16.0],
        ],
        dtype=np.float64,
    )

    stats = fit_feature_zscore(
        [
            train_a,
            train_b,
        ],
        sample_ids=[
            "a",
            "b",
        ],
        expected_width=2,
    )

    np.testing.assert_allclose(
        stats.mean,
        [3.0, 13.0],
        atol=1e-12,
    )

    np.testing.assert_allclose(
        stats.std,
        [
            np.sqrt(5.0),
            np.sqrt(5.0),
        ],
        atol=1e-12,
    )

    assert stats.frame_count == 4
    assert stats.recording_count == 2


def test_normalized_training_frames_have_zero_mean_unit_population_std():

    a = np.array(
        [
            [1.0, 4.0],
            [2.0, 8.0],
        ],
        dtype=np.float64,
    )

    b = np.array(
        [
            [3.0, 12.0],
            [5.0, 20.0],
            [8.0, 32.0],
        ],
        dtype=np.float64,
    )

    stats = fit_feature_zscore(
        [a, b],
        expected_width=2,
    )

    normalized = np.concatenate(
        [
            apply_feature_zscore(
                a,
                stats,
            ),
            apply_feature_zscore(
                b,
                stats,
            ),
        ],
        axis=0,
    )

    np.testing.assert_allclose(
        normalized.mean(
            axis=0
        ),
        np.zeros(2),
        atol=1e-6,
    )

    np.testing.assert_allclose(
        normalized.std(
            axis=0,
            ddof=0,
        ),
        np.ones(2),
        atol=1e-6,
    )


def test_test_data_cannot_influence_fitted_statistics():

    train = np.array(
        [
            [1.0, 2.0],
            [3.0, 6.0],
            [5.0, 10.0],
        ],
        dtype=np.float64,
    )

    test_a = np.array(
        [
            [100.0, 200.0],
        ],
        dtype=np.float64,
    )

    test_b = np.array(
        [
            [-100000.0, 999999.0],
        ],
        dtype=np.float64,
    )

    stats_a = fit_feature_zscore(
        [train],
        sample_ids=["train"],
        expected_width=2,
    )

    _ = apply_feature_zscore(
        test_a,
        stats_a,
    )

    stats_b = fit_feature_zscore(
        [train],
        sample_ids=["train"],
        expected_width=2,
    )

    _ = apply_feature_zscore(
        test_b,
        stats_b,
    )

    np.testing.assert_array_equal(
        stats_a.mean,
        stats_b.mean,
    )

    np.testing.assert_array_equal(
        stats_a.std,
        stats_b.std,
    )


def test_zero_variance_training_feature_is_rejected():

    x = np.array(
        [
            [1.0, 5.0],
            [2.0, 5.0],
            [3.0, 5.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="near-constant",
    ):

        fit_feature_zscore(
            [x],
            expected_width=2,
        )


def test_normalization_preserves_temporal_length():

    rng = np.random.default_rng(
        123
    )

    train = rng.normal(
        size=(7, 3)
    )

    test = rng.normal(
        size=(11, 3)
    )

    stats = fit_feature_zscore(
        [train],
        expected_width=3,
    )

    output = apply_feature_zscore(
        test,
        stats,
    )

    assert output.shape == (
        11,
        3,
    )


def test_duplicate_training_sample_ids_are_rejected():

    a = np.array(
        [
            [0.0, 1.0],
            [1.0, 2.0],
        ],
        dtype=np.float64,
    )

    b = np.array(
        [
            [2.0, 3.0],
            [3.0, 4.0],
        ],
        dtype=np.float64,
    )

    with pytest.raises(
        ValueError,
        match="not unique",
    ):

        fit_feature_zscore(
            [a, b],
            sample_ids=[
                "same",
                "same",
            ],
            expected_width=2,
        )

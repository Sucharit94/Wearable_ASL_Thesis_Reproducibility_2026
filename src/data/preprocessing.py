from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable, Sequence

import numpy as np


PRIMARY_FEATURE_WIDTH = 38
CIRCULAR_FEATURE_WIDTH = 76

DEFAULT_MIN_STD = 1.0e-8


@dataclass(frozen=True)
class FeatureZScoreStats:
    mean: np.ndarray
    std: np.ndarray
    frame_count: int
    recording_count: int
    feature_width: int
    sample_ids_sha256: str | None


def _validate_sequence(
    sequence: np.ndarray,
    expected_width: int | None = None,
) -> np.ndarray:

    array = np.asarray(
        sequence
    )

    if array.ndim != 2:
        raise ValueError(
            f"Expected a 2D T x F sequence, "
            f"got shape {array.shape}."
        )

    if array.shape[0] <= 0:
        raise ValueError(
            "Sequence contains zero time steps."
        )

    if (
        expected_width is not None
        and array.shape[1] != expected_width
    ):
        raise ValueError(
            f"Expected feature width "
            f"{expected_width}, "
            f"got {array.shape[1]}."
        )

    if not np.isfinite(
        array
    ).all():
        raise ValueError(
            "Sequence contains NaN or Inf."
        )

    return array


def primary_raw_rotation(
    sequence: np.ndarray,
) -> np.ndarray:

    array = _validate_sequence(
        sequence,
        expected_width=PRIMARY_FEATURE_WIDTH,
    )

    return np.asarray(
        array,
        dtype=np.float32,
        order="C",
    )


def circular_encode_degrees(
    sequence: np.ndarray,
) -> np.ndarray:

    array = _validate_sequence(
        sequence,
        expected_width=PRIMARY_FEATURE_WIDTH,
    ).astype(
        np.float64,
        copy=False,
    )

    radians = np.deg2rad(
        array
    )

    result = np.empty(
        (
            array.shape[0],
            CIRCULAR_FEATURE_WIDTH,
        ),
        dtype=np.float32,
    )

    result[:, 0::2] = np.sin(
        radians
    ).astype(
        np.float32
    )

    result[:, 1::2] = np.cos(
        radians
    ).astype(
        np.float32
    )

    return result


def canonical_sample_id_sha256(
    sample_ids: Sequence[str] | None,
) -> str | None:

    if sample_ids is None:
        return None

    ids = [
        str(value)
        for value in sample_ids
    ]

    if len(ids) != len(
        set(ids)
    ):
        raise ValueError(
            "Training sample IDs are not unique."
        )

    payload = "".join(
        f"{sample_id}\n"
        for sample_id in sorted(ids)
    ).encode(
        "utf-8"
    )

    return hashlib.sha256(
        payload
    ).hexdigest()


def fit_feature_zscore(
    sequences: Iterable[np.ndarray],
    *,
    sample_ids: Sequence[str] | None = None,
    expected_width: int | None = None,
    minimum_std: float = DEFAULT_MIN_STD,
) -> FeatureZScoreStats:

    if minimum_std <= 0:
        raise ValueError(
            "minimum_std must be positive."
        )

    sequence_list = list(
        sequences
    )

    if not sequence_list:
        raise ValueError(
            "Cannot fit normalization on "
            "an empty training collection."
        )

    if (
        sample_ids is not None
        and len(sample_ids)
        != len(sequence_list)
    ):
        raise ValueError(
            "sample_ids length does not match "
            "number of training recordings."
        )

    frame_count = 0
    recording_count = 0

    running_mean = None
    running_m2 = None
    feature_width = None

    for sequence in sequence_list:

        array = _validate_sequence(
            sequence,
            expected_width=expected_width,
        ).astype(
            np.float64,
            copy=False,
        )

        if feature_width is None:
            feature_width = int(
                array.shape[1]
            )

            running_mean = np.zeros(
                feature_width,
                dtype=np.float64,
            )

            running_m2 = np.zeros(
                feature_width,
                dtype=np.float64,
            )

        elif (
            array.shape[1]
            != feature_width
        ):
            raise ValueError(
                "Training sequences have "
                "inconsistent feature width."
            )

        batch_n = int(
            array.shape[0]
        )

        batch_mean = np.mean(
            array,
            axis=0,
            dtype=np.float64,
        )

        centered = (
            array
            - batch_mean
        )

        batch_m2 = np.sum(
            centered * centered,
            axis=0,
            dtype=np.float64,
        )

        if frame_count == 0:

            running_mean[:] = (
                batch_mean
            )

            running_m2[:] = (
                batch_m2
            )

            frame_count = (
                batch_n
            )

        else:

            total_n = (
                frame_count
                + batch_n
            )

            delta = (
                batch_mean
                - running_mean
            )

            running_mean[:] = (
                running_mean
                + delta
                * (
                    batch_n
                    / total_n
                )
            )

            running_m2[:] = (
                running_m2
                + batch_m2
                + delta
                * delta
                * (
                    frame_count
                    * batch_n
                    / total_n
                )
            )

            frame_count = (
                total_n
            )

        recording_count += 1

    assert running_mean is not None
    assert running_m2 is not None
    assert feature_width is not None

    variance = (
        running_m2
        / frame_count
    )

    variance = np.maximum(
        variance,
        0.0,
    )

    std = np.sqrt(
        variance
    )

    low_std_columns = np.flatnonzero(
        std <= minimum_std
    )

    if low_std_columns.size:

        raise ValueError(
            "Training partition contains "
            "near-constant feature columns "
            f"at indices "
            f"{low_std_columns.tolist()}."
        )

    return FeatureZScoreStats(
        mean=np.asarray(
            running_mean,
            dtype=np.float64,
        ),
        std=np.asarray(
            std,
            dtype=np.float64,
        ),
        frame_count=frame_count,
        recording_count=recording_count,
        feature_width=feature_width,
        sample_ids_sha256=(
            canonical_sample_id_sha256(
                sample_ids
            )
        ),
    )


def apply_feature_zscore(
    sequence: np.ndarray,
    stats: FeatureZScoreStats,
) -> np.ndarray:

    array = _validate_sequence(
        sequence,
        expected_width=stats.feature_width,
    ).astype(
        np.float64,
        copy=False,
    )

    normalized = (
        array
        - stats.mean
    ) / stats.std

    if not np.isfinite(
        normalized
    ).all():
        raise ValueError(
            "Normalization produced "
            "NaN or Inf."
        )

    return np.asarray(
        normalized,
        dtype=np.float32,
        order="C",
    )

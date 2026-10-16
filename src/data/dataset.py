from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from src.data.preprocessing import (
    FeatureZScoreStats,
    apply_feature_zscore,
    fit_feature_zscore,
    primary_raw_rotation,
)


PRIMARY_FEATURE_WIDTH = 38
CTC_BLANK_ID = 0


@dataclass(frozen=True)
class CacheEntry:
    sample_id: str
    subject: str
    sentence_id: int
    repetition_id: int
    timesteps: int
    feature_width: int
    cache_row_start: int
    cache_row_end_exclusive: int


@dataclass(frozen=True)
class TargetEntry:
    sentence_id: int
    gloss_raw: str
    english_raw: str
    gloss_tokens: tuple[str, ...]
    gloss_token_ids: tuple[int, ...]


class RawCacheStore:
    """
    Read-only Stage-17 contiguous float32 cache.

    The binary cache is an execution representation only.
    Original CSV files remain the scientific source.
    """

    def __init__(
        self,
        cache_path: str | Path,
        index_path: str | Path,
    ) -> None:

        self.cache_path = Path(
            cache_path
        )

        self.index_path = Path(
            index_path
        )

        if not self.cache_path.exists():
            raise FileNotFoundError(
                self.cache_path
            )

        if not self.index_path.exists():
            raise FileNotFoundError(
                self.index_path
            )

        with self.index_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            rows = list(
                csv.DictReader(handle)
            )

        if not rows:
            raise RuntimeError(
                "Stage-17 cache index is empty."
            )

        entries = []

        seen_sample_ids = set()

        expected_start = 0

        feature_widths = set()

        for row in rows:

            entry = CacheEntry(
                sample_id=row[
                    "sample_id"
                ],
                subject=row[
                    "subject"
                ],
                sentence_id=int(
                    row[
                        "sentence_id"
                    ]
                ),
                repetition_id=int(
                    row[
                        "repetition_id"
                    ]
                ),
                timesteps=int(
                    row[
                        "timesteps"
                    ]
                ),
                feature_width=int(
                    row[
                        "feature_width"
                    ]
                ),
                cache_row_start=int(
                    row[
                        "cache_row_start"
                    ]
                ),
                cache_row_end_exclusive=int(
                    row[
                        "cache_row_end_exclusive"
                    ]
                ),
            )

            if (
                entry.sample_id
                in seen_sample_ids
            ):
                raise RuntimeError(
                    f"Duplicate sample_id in cache index: "
                    f"{entry.sample_id}"
                )

            seen_sample_ids.add(
                entry.sample_id
            )

            if entry.timesteps <= 0:
                raise RuntimeError(
                    f"Invalid timestep count for "
                    f"{entry.sample_id}"
                )

            if (
                entry.cache_row_start
                != expected_start
            ):
                raise RuntimeError(
                    "Cache index is not contiguous at "
                    f"{entry.sample_id}: "
                    f"{entry.cache_row_start} != "
                    f"{expected_start}"
                )

            if (
                entry.cache_row_end_exclusive
                - entry.cache_row_start
                != entry.timesteps
            ):
                raise RuntimeError(
                    "Cache row range does not equal "
                    f"timesteps for {entry.sample_id}"
                )

            expected_start = (
                entry.cache_row_end_exclusive
            )

            feature_widths.add(
                entry.feature_width
            )

            entries.append(
                entry
            )

        if len(feature_widths) != 1:
            raise RuntimeError(
                "Cache contains inconsistent "
                "feature widths."
            )

        self.feature_width = next(
            iter(
                feature_widths
            )
        )

        if (
            self.feature_width
            != PRIMARY_FEATURE_WIDTH
        ):
            raise RuntimeError(
                f"Expected cache width 38, "
                f"found {self.feature_width}."
            )

        self.total_timesteps = (
            expected_start
        )

        expected_bytes = (
            self.total_timesteps
            * self.feature_width
            * np.dtype("<f4").itemsize
        )

        actual_bytes = (
            self.cache_path.stat().st_size
        )

        if actual_bytes != expected_bytes:
            raise RuntimeError(
                "Stage-17 binary file-size mismatch: "
                f"{actual_bytes} != "
                f"{expected_bytes}"
            )

        self.entries = tuple(
            entries
        )

        self.entry_by_sample_id = {
            entry.sample_id: entry
            for entry in self.entries
        }

        self.sample_ids = tuple(
            entry.sample_id
            for entry in self.entries
        )

        self._memmap = np.memmap(
            self.cache_path,
            dtype="<f4",
            mode="r",
            shape=(
                self.total_timesteps,
                self.feature_width,
            ),
            order="C",
        )

    def __len__(self) -> int:
        return len(
            self.entries
        )

    def get_entry(
        self,
        sample_id: str,
    ) -> CacheEntry:

        try:
            return self.entry_by_sample_id[
                sample_id
            ]

        except KeyError as exc:
            raise KeyError(
                f"Unknown sample_id: {sample_id}"
            ) from exc

    def get(
        self,
        sample_id: str,
        *,
        copy: bool = False,
    ) -> np.ndarray:

        entry = self.get_entry(
            sample_id
        )

        sequence = self._memmap[
            entry.cache_row_start:
            entry.cache_row_end_exclusive
        ]

        if sequence.shape != (
            entry.timesteps,
            entry.feature_width,
        ):
            raise RuntimeError(
                f"Cache slice shape mismatch for "
                f"{sample_id}: {sequence.shape}"
            )

        if copy:
            return np.array(
                sequence,
                dtype=np.float32,
                copy=True,
                order="C",
            )

        return np.asarray(
            sequence
        )


class TargetCatalog:
    """
    Frozen Stage-10 sentence/gloss target catalog.
    """

    def __init__(
        self,
        target_path: str | Path,
    ) -> None:

        self.target_path = Path(
            target_path
        )

        if not self.target_path.exists():
            raise FileNotFoundError(
                self.target_path
            )

        with self.target_path.open(
            "r",
            newline="",
            encoding="utf-8-sig",
        ) as handle:
            rows = list(
                csv.DictReader(handle)
            )

        if not rows:
            raise RuntimeError(
                "Target catalog is empty."
            )

        targets = {}

        for row in rows:

            sentence_id = int(
                row[
                    "sentence_id"
                ]
            )

            gloss_tokens = tuple(
                str(value)
                for value in json.loads(
                    row[
                        "gloss_tokens_json"
                    ]
                )
            )

            gloss_token_ids = tuple(
                int(value)
                for value in json.loads(
                    row[
                        "gloss_token_ids_json"
                    ]
                )
            )

            if not gloss_tokens:
                raise RuntimeError(
                    f"Sentence {sentence_id} "
                    "contains no gloss tokens."
                )

            if (
                len(gloss_tokens)
                != len(
                    gloss_token_ids
                )
            ):
                raise RuntimeError(
                    f"Token/ID length mismatch for "
                    f"sentence {sentence_id}."
                )

            if CTC_BLANK_ID in gloss_token_ids:
                raise RuntimeError(
                    f"CTC blank ID 0 occurs inside "
                    f"sentence {sentence_id} target."
                )

            if any(
                token_id < 1
                for token_id
                in gloss_token_ids
            ):
                raise RuntimeError(
                    f"Invalid target ID in "
                    f"sentence {sentence_id}."
                )

            target = TargetEntry(
                sentence_id=sentence_id,
                gloss_raw=row[
                    "gloss_raw"
                ],
                english_raw=row[
                    "english_raw"
                ],
                gloss_tokens=gloss_tokens,
                gloss_token_ids=(
                    gloss_token_ids
                ),
            )

            if sentence_id in targets:
                raise RuntimeError(
                    f"Duplicate sentence ID "
                    f"{sentence_id}."
                )

            targets[
                sentence_id
            ] = target

        self.targets = targets

        self.sentence_ids = tuple(
            sorted(
                targets
            )
        )

    def __len__(self) -> int:
        return len(
            self.targets
        )

    def get(
        self,
        sentence_id: int,
    ) -> TargetEntry:

        try:
            return self.targets[
                int(
                    sentence_id
                )
            ]

        except KeyError as exc:
            raise KeyError(
                f"Unknown sentence_id: "
                f"{sentence_id}"
            ) from exc


def fit_normalization_for_sample_ids(
    store: RawCacheStore,
    sample_ids: Sequence[str],
) -> FeatureZScoreStats:
    """
    Fit Stage-18 feature-wise z-score statistics using
    ONLY the supplied training sample IDs.
    """

    ids = [
        str(sample_id)
        for sample_id in sample_ids
    ]

    if not ids:
        raise ValueError(
            "Cannot fit normalization on "
            "an empty sample collection."
        )

    sequences = (
        store.get(
            sample_id,
            copy=False,
        )
        for sample_id in ids
    )

    return fit_feature_zscore(
        sequences,
        sample_ids=ids,
        expected_width=(
            PRIMARY_FEATURE_WIDTH
        ),
    )


class ASLSequenceDataset:
    """
    Framework-independent sequence dataset.

    It returns NumPy arrays so the same scientific
    data layer can be tested without model code.
    """

    def __init__(
        self,
        store: RawCacheStore,
        targets: TargetCatalog,
        sample_ids: Sequence[str],
        *,
        normalization_stats:
            FeatureZScoreStats
            | None = None,
    ) -> None:

        self.store = store
        self.targets = targets

        self.sample_ids = tuple(
            str(sample_id)
            for sample_id
            in sample_ids
        )

        if not self.sample_ids:
            raise ValueError(
                "Dataset sample list is empty."
            )

        if (
            len(
                set(
                    self.sample_ids
                )
            )
            != len(
                self.sample_ids
            )
        ):
            raise ValueError(
                "Dataset sample IDs are not unique."
            )

        for sample_id in (
            self.sample_ids
        ):

            entry = self.store.get_entry(
                sample_id
            )

            self.targets.get(
                entry.sentence_id
            )

        self.normalization_stats = (
            normalization_stats
        )

    def __len__(self) -> int:

        return len(
            self.sample_ids
        )

    def __getitem__(
        self,
        index: int,
    ) -> dict:

        sample_id = self.sample_ids[
            index
        ]

        entry = self.store.get_entry(
            sample_id
        )

        target = self.targets.get(
            entry.sentence_id
        )

        sequence = (
            primary_raw_rotation(
                self.store.get(
                    sample_id,
                    copy=False,
                )
            )
        )

        if (
            self.normalization_stats
            is not None
        ):

            sequence = apply_feature_zscore(
                sequence,
                self.normalization_stats,
            )

        if sequence.shape != (
            entry.timesteps,
            PRIMARY_FEATURE_WIDTH,
        ):
            raise RuntimeError(
                f"Sequence shape changed "
                f"unexpectedly for {sample_id}."
            )

        target_ids = np.asarray(
            target.gloss_token_ids,
            dtype=np.int64,
        )

        return {
            "sample_id": (
                sample_id
            ),
            "subject": (
                entry.subject
            ),
            "sentence_id": (
                entry.sentence_id
            ),
            "repetition_id": (
                entry.repetition_id
            ),
            "inputs": (
                sequence
            ),
            "input_length": (
                int(
                    entry.timesteps
                )
            ),
            "target_ids": (
                target_ids
            ),
            "target_length": (
                int(
                    target_ids.shape[0]
                )
            ),
            "gloss_raw": (
                target.gloss_raw
            ),
            "gloss_tokens": (
                target.gloss_tokens
            ),
            "english_raw": (
                target.english_raw
            ),
        }

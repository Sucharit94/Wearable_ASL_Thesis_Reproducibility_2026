from __future__ import annotations

import csv
from pathlib import Path


def _load_csv(
    path: str | Path,
) -> list[dict]:

    path = Path(
        path
    )

    if not path.exists():
        raise FileNotFoundError(
            path
        )

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def _validate_partition(
    train_ids: list[str],
    test_ids: list[str],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:

    train_set = set(
        train_ids
    )

    test_set = set(
        test_ids
    )

    if len(train_set) != len(
        train_ids
    ):
        raise RuntimeError(
            "Duplicate sample IDs inside "
            "training partition."
        )

    if len(test_set) != len(
        test_ids
    ):
        raise RuntimeError(
            "Duplicate sample IDs inside "
            "test partition."
        )

    overlap = (
        train_set
        & test_set
    )

    if overlap:
        raise RuntimeError(
            "Train/test sample overlap: "
            f"{sorted(overlap)[:10]}"
        )

    return (
        tuple(
            sorted(
                train_ids
            )
        ),
        tuple(
            sorted(
                test_ids
            )
        ),
    )


def load_known_signer_outer_fold(
    split_path: str | Path,
    outer_fold: int,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:

    if outer_fold not in {
        1,
        2,
        3,
        4,
        5,
    }:
        raise ValueError(
            "outer_fold must be 1..5."
        )

    rows = _load_csv(
        split_path
    )

    train_ids = []
    test_ids = []

    for row in rows:

        sample_id = row[
            "sample_id"
        ]

        fold = int(
            row[
                "outer_fold"
            ]
        )

        if fold == outer_fold:
            test_ids.append(
                sample_id
            )
        else:
            train_ids.append(
                sample_id
            )

    return _validate_partition(
        train_ids,
        test_ids,
    )


def load_loso_run(
    split_path: str | Path,
    loso_id: str,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:

    rows = _load_csv(
        split_path
    )

    run_rows = [
        row
        for row in rows
        if row[
            "loso_id"
        ] == loso_id
    ]

    if not run_rows:
        raise ValueError(
            f"Unknown LOSO run: {loso_id}"
        )

    train_ids = [
        row[
            "sample_id"
        ]
        for row in run_rows
        if row[
            "role"
        ] == "train"
    ]

    test_ids = [
        row[
            "sample_id"
        ]
        for row in run_rows
        if row[
            "role"
        ] == "test"
    ]

    return _validate_partition(
        train_ids,
        test_ids,
    )


def load_template_holdout(
    split_path: str | Path,
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
]:

    rows = _load_csv(
        split_path
    )

    train_ids = [
        row[
            "sample_id"
        ]
        for row in rows
        if row[
            "role"
        ] == "train"
    ]

    test_ids = [
        row[
            "sample_id"
        ]
        for row in rows
        if row[
            "role"
        ] == "test"
    ]

    return _validate_partition(
        train_ids,
        test_ids,
    )

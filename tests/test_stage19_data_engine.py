from __future__ import annotations

import csv
import json

import numpy as np

from src.data.collate import (
    TARGET_PADDING_VALUE,
    collate_numpy,
)
from src.data.dataset import (
    ASLSequenceDataset,
    RawCacheStore,
    TargetCatalog,
    fit_normalization_for_sample_ids,
)
from src.data.splits import (
    load_known_signer_outer_fold,
    load_loso_run,
    load_template_holdout,
)


def _make_small_cache(tmp_path):

    cache_path = tmp_path / "raw.bin"
    index_path = tmp_path / "index.csv"

    samples = {
        "s1": np.arange(
            3 * 38,
            dtype=np.float32,
        ).reshape(3, 38),

        "s2": (
            np.arange(
                5 * 38,
                dtype=np.float32,
            ).reshape(5, 38)
            + 10.0
        ),

        "s3": (
            np.arange(
                4 * 38,
                dtype=np.float32,
            ).reshape(4, 38)
            + 20.0
        ),
    }

    rows = []
    start = 0

    with cache_path.open("wb") as handle:

        for i, (sample_id, array) in enumerate(
            samples.items(),
            start=1,
        ):

            handle.write(
                array.astype("<f4").tobytes(
                    order="C"
                )
            )

            end = start + array.shape[0]

            rows.append(
                {
                    "cache_order": i,
                    "sample_id": sample_id,
                    "subject": (
                        "subject1"
                        if sample_id != "s3"
                        else "subject2"
                    ),
                    "sentence_id": (
                        1
                        if sample_id != "s2"
                        else 2
                    ),
                    "repetition_id": i,
                    "timesteps": array.shape[0],
                    "feature_width": 38,
                    "cache_row_start": start,
                    "cache_row_end_exclusive": end,
                    "raw_sha256": "unused",
                    "cache_slice_sha256": "unused",
                }
            )

            start = end

    with index_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return (
        cache_path,
        index_path,
        samples,
    )


def _make_targets(tmp_path):

    target_path = tmp_path / "targets.csv"

    rows = [
        {
            "sentence_id": 1,
            "gloss_raw": "hello",
            "english_raw": "hello",
            "gloss_tokens_json": json.dumps(
                ["hello"]
            ),
            "gloss_token_ids_json": json.dumps(
                [5]
            ),
        },
        {
            "sentence_id": 2,
            "gloss_raw": "you go",
            "english_raw": "you go",
            "gloss_tokens_json": json.dumps(
                ["you", "go"]
            ),
            "gloss_token_ids_json": json.dumps(
                [9, 4]
            ),
        },
    ]

    with target_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(rows)

    return target_path


def test_cache_store_recovers_exact_variable_lengths(
    tmp_path,
):

    cache_path, index_path, samples = (
        _make_small_cache(tmp_path)
    )

    store = RawCacheStore(
        cache_path,
        index_path,
    )

    assert len(store) == 3

    for sample_id, expected in samples.items():

        observed = store.get(
            sample_id,
            copy=True,
        )

        np.testing.assert_array_equal(
            observed,
            expected,
        )


def test_target_catalog_preserves_ctc_targets(
    tmp_path,
):

    target_path = _make_targets(
        tmp_path
    )

    catalog = TargetCatalog(
        target_path
    )

    assert len(catalog) == 2

    assert (
        catalog.get(2).gloss_token_ids
        == (9, 4)
    )


def test_dataset_returns_length_and_target_metadata(
    tmp_path,
):

    cache_path, index_path, _ = (
        _make_small_cache(tmp_path)
    )

    target_path = _make_targets(
        tmp_path
    )

    store = RawCacheStore(
        cache_path,
        index_path,
    )

    catalog = TargetCatalog(
        target_path
    )

    dataset = ASLSequenceDataset(
        store,
        catalog,
        ["s1", "s2"],
    )

    first = dataset[0]

    assert first["sample_id"] == "s1"
    assert first["input_length"] == 3
    assert first["target_length"] == 1

    assert first["inputs"].shape == (
        3,
        38,
    )


def test_train_fitted_normalization_applies_to_other_sample(
    tmp_path,
):

    cache_path, index_path, _ = (
        _make_small_cache(tmp_path)
    )

    target_path = _make_targets(
        tmp_path
    )

    store = RawCacheStore(
        cache_path,
        index_path,
    )

    catalog = TargetCatalog(
        target_path
    )

    stats = fit_normalization_for_sample_ids(
        store,
        ["s1", "s2"],
    )

    dataset = ASLSequenceDataset(
        store,
        catalog,
        ["s3"],
        normalization_stats=stats,
    )

    item = dataset[0]

    assert item["inputs"].dtype == np.float32

    assert np.isfinite(
        item["inputs"]
    ).all()


def test_collate_zero_pads_inputs_and_preserves_lengths(
    tmp_path,
):

    cache_path, index_path, _ = (
        _make_small_cache(tmp_path)
    )

    target_path = _make_targets(
        tmp_path
    )

    store = RawCacheStore(
        cache_path,
        index_path,
    )

    catalog = TargetCatalog(
        target_path
    )

    dataset = ASLSequenceDataset(
        store,
        catalog,
        ["s1", "s2"],
    )

    batch = collate_numpy(
        [
            dataset[0],
            dataset[1],
        ]
    )

    assert batch["inputs"].shape == (
        2,
        5,
        38,
    )

    np.testing.assert_array_equal(
        batch["input_lengths"],
        [3, 5],
    )

    assert np.all(
        batch["inputs"][0, 3:, :]
        == 0.0
    )

    assert np.all(
        batch["valid_mask"][0, :3]
    )

    assert not np.any(
        batch["valid_mask"][0, 3:]
    )


def test_target_padding_never_uses_ctc_blank(
    tmp_path,
):

    cache_path, index_path, _ = (
        _make_small_cache(tmp_path)
    )

    target_path = _make_targets(
        tmp_path
    )

    store = RawCacheStore(
        cache_path,
        index_path,
    )

    catalog = TargetCatalog(
        target_path
    )

    dataset = ASLSequenceDataset(
        store,
        catalog,
        ["s1", "s2"],
    )

    batch = collate_numpy(
        [
            dataset[0],
            dataset[1],
        ]
    )

    assert (
        batch["targets_padded"][0, 1]
        == TARGET_PADDING_VALUE
    )

    assert TARGET_PADDING_VALUE == -1


def test_known_signer_split_loader(
    tmp_path,
):

    split_path = tmp_path / "known.csv"

    with split_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "outer_fold",
            ],
        )

        writer.writeheader()

        writer.writerows(
            [
                {
                    "sample_id": "a",
                    "outer_fold": 1,
                },
                {
                    "sample_id": "b",
                    "outer_fold": 2,
                },
                {
                    "sample_id": "c",
                    "outer_fold": 1,
                },
            ]
        )

    train_ids, test_ids = (
        load_known_signer_outer_fold(
            split_path,
            1,
        )
    )

    assert set(train_ids) == {"b"}

    assert set(test_ids) == {
        "a",
        "c",
    }


def test_loso_and_template_loaders(
    tmp_path,
):

    loso_path = tmp_path / "loso.csv"

    with loso_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "loso_id",
                "sample_id",
                "role",
            ],
        )

        writer.writeheader()

        writer.writerows(
            [
                {
                    "loso_id": "loso_subject1",
                    "sample_id": "a",
                    "role": "test",
                },
                {
                    "loso_id": "loso_subject1",
                    "sample_id": "b",
                    "role": "train",
                },
            ]
        )

    train_ids, test_ids = load_loso_run(
        loso_path,
        "loso_subject1",
    )

    assert train_ids == ("b",)
    assert test_ids == ("a",)

    template_path = (
        tmp_path
        / "template.csv"
    )

    with template_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "role",
            ],
        )

        writer.writeheader()

        writer.writerows(
            [
                {
                    "sample_id": "a",
                    "role": "train",
                },
                {
                    "sample_id": "b",
                    "role": "test",
                },
            ]
        )

    train_ids, test_ids = (
        load_template_holdout(
            template_path
        )
    )

    assert train_ids == ("a",)
    assert test_ids == ("b",)

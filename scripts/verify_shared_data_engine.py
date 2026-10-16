from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

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


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_PATH = (
    PROJECT_ROOT
    / "cache"
    / "raw"
    / "stage17_raw_f32.bin"
)

CACHE_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.csv"
)

CACHE_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache.sha256"
)

CACHE_INDEX_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.sha256"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

TARGET_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.sha256"
)

TOKEN_TO_ID_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_token_to_id.json"
)

STAGE13_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage13_known_signer_5fold.csv"
)

STAGE13_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage13_known_signer_5fold.sha256"
)

STAGE14_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage14_loso_assignment.csv"
)

STAGE14_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage14_loso_assignment.sha256"
)

STAGE16_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage16_template_holdout_assignment.csv"
)

STAGE16_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage16_template_holdout_assignment.sha256"
)

STAGE18_CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
    / "stage18_preprocessing_contract.json"
)

PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "preprocessing.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "data"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage19_shared_data_engine_summary.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_19_SHARED_DATA_ENGINE.md"
)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_recorded_hash(
    path: Path,
) -> str:

    text = (
        path.read_text(
            encoding="utf-8-sig"
        )
        .strip()
    )

    value = (
        text.split()[0]
        .lower()
    )

    if (
        len(value) != 64
        or any(
            char not in
            "0123456789abcdef"
            for char in value
        )
    ):

        raise RuntimeError(
            f"Invalid hash record: "
            f"{path}"
        )

    return value


def verify_hash(
    file_path: Path,
    hash_path: Path,
) -> str:

    expected = (
        read_recorded_hash(
            hash_path
        )
    )

    actual = (
        sha256_file(
            file_path
        )
    )

    if expected != actual:

        raise RuntimeError(
            f"Hash mismatch: "
            f"{file_path}"
        )

    return actual


def load_csv(
    path: Path,
) -> list[dict]:

    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 19 — SHARED DATASET + "
        "COLLATE ENGINE"
    )
    print("=" * 78)

    # ---------------------------------------------------------
    # Frozen upstream evidence.
    # ---------------------------------------------------------

    upstream_hashes = {
        "stage17_binary": (
            verify_hash(
                CACHE_PATH,
                CACHE_HASH_PATH,
            )
        ),

        "stage17_index": (
            verify_hash(
                CACHE_INDEX_PATH,
                CACHE_INDEX_HASH_PATH,
            )
        ),

        "stage10_targets": (
            verify_hash(
                TARGET_PATH,
                TARGET_HASH_PATH,
            )
        ),

        "stage13_known_signer": (
            verify_hash(
                STAGE13_PATH,
                STAGE13_HASH_PATH,
            )
        ),

        "stage14_loso": (
            verify_hash(
                STAGE14_PATH,
                STAGE14_HASH_PATH,
            )
        ),

        "stage16_template_holdout": (
            verify_hash(
                STAGE16_PATH,
                STAGE16_HASH_PATH,
            )
        ),
    }

    stage18 = json.loads(
        STAGE18_CONTRACT_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    actual_preprocessing_hash = (
        sha256_file(
            PREPROCESSING_PATH
        )
    )

    expected_preprocessing_hash = (
        stage18[
            "hashes"
        ][
            "implementation_sha256"
        ]
    )

    if (
        actual_preprocessing_hash
        != expected_preprocessing_hash
    ):

        raise RuntimeError(
            "Stage-18 preprocessing "
            "implementation hash mismatch."
        )

    # ---------------------------------------------------------
    # Build shared components.
    # ---------------------------------------------------------

    store = RawCacheStore(
        CACHE_PATH,
        CACHE_INDEX_PATH,
    )

    targets = TargetCatalog(
        TARGET_PATH
    )

    if len(store) != 7400:

        raise RuntimeError(
            f"Expected 7,400 cache samples, "
            f"found {len(store)}."
        )

    if (
        store.total_timesteps
        != 1_829_320
    ):

        raise RuntimeError(
            "Unexpected total timestep count."
        )

    if store.feature_width != 38:

        raise RuntimeError(
            "Unexpected cache width."
        )

    if len(targets) != 300:

        raise RuntimeError(
            f"Expected 300 target sentences, "
            f"found {len(targets)}."
        )

    if (
        targets.sentence_ids
        != tuple(
            range(
                1,
                301,
            )
        )
    ):

        raise RuntimeError(
            "Sentence IDs are not 1..300."
        )

    # ---------------------------------------------------------
    # Vocabulary / target audit.
    # ---------------------------------------------------------

    token_to_id = json.loads(
        TOKEN_TO_ID_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        token_to_id.get(
            "<blank>"
        )
        != 0
    ):

        raise RuntimeError(
            "CTC blank ID is not 0."
        )

    if len(token_to_id) != 456:

        raise RuntimeError(
            f"Expected 456 CTC output classes, "
            f"found {len(token_to_id)}."
        )

    target_blank_violations = 0

    target_id_min = None
    target_id_max = None

    for sentence_id in (
        targets.sentence_ids
    ):

        entry = targets.get(
            sentence_id
        )

        ids = entry.gloss_token_ids

        if 0 in ids:
            target_blank_violations += 1

        local_min = min(
            ids
        )

        local_max = max(
            ids
        )

        target_id_min = (
            local_min
            if target_id_min is None
            else min(
                target_id_min,
                local_min,
            )
        )

        target_id_max = (
            local_max
            if target_id_max is None
            else max(
                target_id_max,
                local_max,
            )
        )

    if target_blank_violations:

        raise RuntimeError(
            "Blank ID occurs in reference targets."
        )

    if target_id_min != 1:

        raise RuntimeError(
            f"Unexpected minimum target ID: "
            f"{target_id_min}"
        )

    if target_id_max != 455:

        raise RuntimeError(
            f"Unexpected maximum target ID: "
            f"{target_id_max}"
        )

    # ---------------------------------------------------------
    # Split/sample-set consistency.
    # ---------------------------------------------------------

    cache_sample_set = set(
        store.sample_ids
    )

    stage13_rows = load_csv(
        STAGE13_PATH
    )

    stage13_sample_set = {
        row[
            "sample_id"
        ]
        for row in stage13_rows
    }

    if (
        stage13_sample_set
        != cache_sample_set
    ):

        raise RuntimeError(
            "Stage-13 and cache sample sets differ."
        )

    stage16_rows = load_csv(
        STAGE16_PATH
    )

    stage16_sample_set = {
        row[
            "sample_id"
        ]
        for row in stage16_rows
    }

    if (
        stage16_sample_set
        != cache_sample_set
    ):

        raise RuntimeError(
            "Stage-16 and cache sample sets differ."
        )

    for loso_id in [
        "loso_subject1",
        "loso_subject2",
        "loso_subject3",
    ]:

        loso_train, loso_test = (
            load_loso_run(
                STAGE14_PATH,
                loso_id,
            )
        )

        if (
            set(
                loso_train
            )
            | set(
                loso_test
            )
        ) != cache_sample_set:

            raise RuntimeError(
                f"{loso_id} does not cover "
                "the complete cache."
            )

    # ---------------------------------------------------------
    # Known-signer split engine.
    # ---------------------------------------------------------

    fold_counts = []

    for outer_fold in range(
        1,
        6,
    ):

        train_ids, test_ids = (
            load_known_signer_outer_fold(
                STAGE13_PATH,
                outer_fold,
            )
        )

        if len(
            train_ids
        ) != 5920:

            raise RuntimeError(
                f"Fold {outer_fold} training "
                "count mismatch."
            )

        if len(
            test_ids
        ) != 1480:

            raise RuntimeError(
                f"Fold {outer_fold} test "
                "count mismatch."
            )

        if (
            set(
                train_ids
            )
            | set(
                test_ids
            )
        ) != cache_sample_set:

            raise RuntimeError(
                f"Fold {outer_fold} does not "
                "cover all samples."
            )

        fold_counts.append(
            {
                "outer_fold": (
                    outer_fold
                ),
                "train": (
                    len(
                        train_ids
                    )
                ),
                "test": (
                    len(
                        test_ids
                    )
                ),
            }
        )

    # ---------------------------------------------------------
    # LOSO exact counts.
    # ---------------------------------------------------------

    expected_loso = {
        "loso_subject1": (
            1400,
            6000,
        ),

        "loso_subject2": (
            6400,
            1000,
        ),

        "loso_subject3": (
            7000,
            400,
        ),
    }

    loso_counts = {}

    for (
        loso_id,
        expected_counts,
    ) in expected_loso.items():

        train_ids, test_ids = (
            load_loso_run(
                STAGE14_PATH,
                loso_id,
            )
        )

        observed = (
            len(
                train_ids
            ),
            len(
                test_ids
            ),
        )

        if observed != expected_counts:

            raise RuntimeError(
                f"{loso_id} count mismatch: "
                f"{observed} != "
                f"{expected_counts}"
            )

        loso_counts[
            loso_id
        ] = {
            "train": (
                observed[0]
            ),
            "test": (
                observed[1]
            ),
        }

    # ---------------------------------------------------------
    # Template holdout.
    # ---------------------------------------------------------

    (
        template_train,
        template_test,
    ) = load_template_holdout(
        STAGE16_PATH
    )

    if (
        set(
            template_train
        )
        | set(
            template_test
        )
    ) != cache_sample_set:

        raise RuntimeError(
            "Template holdout does not cover "
            "all cache samples."
        )

    template_test_subjects = {
        store.get_entry(
            sample_id
        ).subject
        for sample_id in (
            template_test
        )
    }

    if template_test_subjects != {
        "subject1",
        "subject2",
        "subject3",
    }:

        raise RuntimeError(
            "Template holdout test does not "
            "contain all three subjects."
        )

    # ---------------------------------------------------------
    # Real Stage-18 normalization smoke test:
    # fit ONLY on Stage-13 Fold-1 training IDs.
    # ---------------------------------------------------------

    fold1_train, fold1_test = (
        load_known_signer_outer_fold(
            STAGE13_PATH,
            1,
        )
    )

    stats = (
        fit_normalization_for_sample_ids(
            store,
            fold1_train,
        )
    )

    expected_fold1_frames = sum(
        store.get_entry(
            sample_id
        ).timesteps
        for sample_id in fold1_train
    )

    if (
        stats.recording_count
        != len(
            fold1_train
        )
    ):

        raise RuntimeError(
            "Normalization recording count "
            "does not match Fold-1 training set."
        )

    if (
        stats.frame_count
        != expected_fold1_frames
    ):

        raise RuntimeError(
            "Normalization frame count "
            "does not match Fold-1 training frames."
        )

    if stats.feature_width != 38:

        raise RuntimeError(
            "Normalization width mismatch."
        )

    if not np.isfinite(
        stats.mean
    ).all():

        raise RuntimeError(
            "Normalization mean is non-finite."
        )

    if not np.isfinite(
        stats.std
    ).all():

        raise RuntimeError(
            "Normalization std is non-finite."
        )

    if np.any(
        stats.std <= 0.0
    ):

        raise RuntimeError(
            "Normalization contains "
            "non-positive std."
        )

    # ---------------------------------------------------------
    # Real heterogeneous batch.
    # ---------------------------------------------------------

    fold1_test_sorted = sorted(
        fold1_test,
        key=lambda sample_id: (
            store.get_entry(
                sample_id
            ).timesteps,
            sample_id,
        ),
    )

    batch_sample_ids = [
        fold1_test_sorted[0],
        fold1_test_sorted[
            len(
                fold1_test_sorted
            )
            // 2
        ],
        fold1_test_sorted[-1],
    ]

    batch_dataset = (
        ASLSequenceDataset(
            store,
            targets,
            batch_sample_ids,
            normalization_stats=stats,
        )
    )

    batch_items = [
        batch_dataset[
            index
        ]
        for index in range(
            len(
                batch_dataset
            )
        )
    ]

    batch = collate_numpy(
        batch_items
    )

    batch_lengths = [
        int(value)
        for value in batch[
            "input_lengths"
        ]
    ]

    max_batch_length = max(
        batch_lengths
    )

    if batch[
        "inputs"
    ].shape != (
        3,
        max_batch_length,
        38,
    ):

        raise RuntimeError(
            "Unexpected padded batch shape."
        )

    padding_violations = 0

    mask_violations = 0

    for batch_index, length in (
        enumerate(
            batch_lengths
        )
    ):

        if not np.all(
            batch[
                "valid_mask"
            ][
                batch_index,
                :length
            ]
        ):

            mask_violations += 1

        if np.any(
            batch[
                "valid_mask"
            ][
                batch_index,
                length:
            ]
        ):

            mask_violations += 1

        if not np.all(
            batch[
                "inputs"
            ][
                batch_index,
                length:,
                :
            ]
            == 0.0
        ):

            padding_violations += 1

    if padding_violations:
        raise RuntimeError(
            "Input padding is not zero."
        )

    if mask_violations:
        raise RuntimeError(
            "Valid-mask contract failed."
        )

    target_padding_blank_violations = int(
        np.any(
            (
                batch[
                    "targets_padded"
                ]
                == 0
            )
            & (
                np.arange(
                    batch[
                        "targets_padded"
                    ].shape[1]
                )[
                    None,
                    :
                ]
                >= batch[
                    "target_lengths"
                ][
                    :,
                    None
                ]
            )
        )
    )

    if target_padding_blank_violations:

        raise RuntimeError(
            "Target padding used CTC blank 0."
        )

    if TARGET_PADDING_VALUE != -1:

        raise RuntimeError(
            "Target padding value must be -1."
        )

    # ---------------------------------------------------------
    # Metadata consistency for all 7,400 cache samples.
    # ---------------------------------------------------------

    metadata_mismatches = 0

    missing_targets = 0

    for entry in store.entries:

        try:
            target = targets.get(
                entry.sentence_id
            )

        except KeyError:

            missing_targets += 1
            continue

        if (
            target.sentence_id
            != entry.sentence_id
        ):

            metadata_mismatches += 1

    if missing_targets:
        raise RuntimeError(
            f"Missing targets for "
            f"{missing_targets} samples."
        )

    if metadata_mismatches:
        raise RuntimeError(
            f"Metadata mismatches: "
            f"{metadata_mismatches}"
        )

    # ---------------------------------------------------------
    # Freeze implementation hashes.
    # ---------------------------------------------------------

    implementation_hashes = {
        "dataset_py": (
            sha256_file(
                PROJECT_ROOT
                / "src"
                / "data"
                / "dataset.py"
            )
        ),

        "splits_py": (
            sha256_file(
                PROJECT_ROOT
                / "src"
                / "data"
                / "splits.py"
            )
        ),

        "collate_py": (
            sha256_file(
                PROJECT_ROOT
                / "src"
                / "data"
                / "collate.py"
            )
        ),

        "stage19_tests": (
            sha256_file(
                PROJECT_ROOT
                / "tests"
                / "test_stage19_data_engine.py"
            )
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 19,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "engine": {
            "framework": (
                "numpy"
            ),

            "cache_access": (
                "read_only_memmap"
            ),

            "recordings": (
                len(
                    store
                )
            ),

            "total_timesteps": (
                store.total_timesteps
            ),

            "feature_width": (
                store.feature_width
            ),

            "target_sentences": (
                len(
                    targets
                )
            ),

            "ctc_output_classes": (
                len(
                    token_to_id
                )
            ),

            "ctc_blank_id": (
                0
            ),

            "target_padding_id": (
                TARGET_PADDING_VALUE
            ),
        },

        "target_checks": {
            "minimum_target_id": (
                target_id_min
            ),

            "maximum_target_id": (
                target_id_max
            ),

            "targets_containing_blank": (
                target_blank_violations
            ),
        },

        "known_signer_folds": (
            fold_counts
        ),

        "loso": (
            loso_counts
        ),

        "template_holdout": {
            "train_recordings": (
                len(
                    template_train
                )
            ),

            "test_recordings": (
                len(
                    template_test
                )
            ),

            "test_subjects": (
                sorted(
                    template_test_subjects
                )
            ),
        },

        "fold1_normalization_smoke": {
            "scientific_role": (
                "integration diagnostic only; "
                "not a globally reusable "
                "normalization vector"
            ),

            "fit_recordings": (
                stats.recording_count
            ),

            "fit_frames": (
                stats.frame_count
            ),

            "feature_width": (
                stats.feature_width
            ),

            "sample_ids_sha256": (
                stats.sample_ids_sha256
            ),

            "minimum_std": (
                float(
                    np.min(
                        stats.std
                    )
                )
            ),

            "maximum_std": (
                float(
                    np.max(
                        stats.std
                    )
                )
            ),
        },

        "collate_smoke": {
            "sample_ids": (
                batch_sample_ids
            ),

            "input_lengths": (
                batch_lengths
            ),

            "batch_shape": [
                int(value)
                for value in batch[
                    "inputs"
                ].shape
            ],

            "input_padding_value": (
                0.0
            ),

            "target_padding_value": (
                TARGET_PADDING_VALUE
            ),

            "padding_violations": (
                padding_violations
            ),

            "mask_violations": (
                mask_violations
            ),
        },

        "all_sample_checks": {
            "cache_vs_stage13_equal": (
                stage13_sample_set
                == cache_sample_set
            ),

            "cache_vs_stage16_equal": (
                stage16_sample_set
                == cache_sample_set
            ),

            "missing_targets": (
                missing_targets
            ),

            "metadata_mismatches": (
                metadata_mismatches
            ),
        },

        "deferred": [
            "CTC feasibility and output lengths — Stage 20",
            "CTC loss and decoder — Stage 21",
            "model output-length propagation — Stage 22",
            "PyTorch training loaders — later training engine",
        ],

        "hashes": {
            "upstream": (
                upstream_hashes
            ),

            "stage18_preprocessing_implementation": (
                actual_preprocessing_hash
            ),

            "stage19_implementation": (
                implementation_hashes
            ),
        },
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 19 — Shared Dataset and Collate Engine

## Purpose

Connect the frozen cache, labels, splits and train-only preprocessing
contract through one shared framework-independent data engine.

## Cache

Recordings:

{len(store):,}

Total valid time steps:

{store.total_timesteps:,}

Feature width:

{store.feature_width}

The Stage-17 binary is opened read-only through NumPy memmap.

## Targets

Sentence targets:

{len(targets)}

CTC output schema:

{len(token_to_id)} classes

CTC blank ID:

0

The blank ID is not permitted inside a reference gloss target.

## Dataset output

Each dataset item returns:

- sample ID,
- subject,
- sentence ID,
- repetition ID,
- variable-length `T x 38` input,
- original valid input length,
- gloss token IDs,
- target length,
- raw gloss text,
- English reference text.

## Normalization

Normalization statistics are supplied explicitly to a dataset.

Stage 19 does not fit a global normalization vector.

The integration smoke test fitted statistics only from known-signer
Fold-1 training recordings and then applied those fixed values to
held-out examples.

Those diagnostic values are not a reusable global normalization
artifact.

## Collation

A batch is padded to:

`B x T_max x 38`

Input padding value:

0.0

Original valid lengths are preserved separately.

A boolean valid mask is generated.

Reference targets use temporary padding value:

-1

CTC blank 0 is never overloaded as target padding.

## Split loaders

The same data engine can consume:

- five known-signer outer folds,
- three LOSO partitions,
- the frozen template-holdout partition.

## Deferred

Stage 19 does not calculate model output lengths.

CTC input feasibility is Stage 20.

CTC loss and decoding are Stage 21.

Length propagation through model architectures is Stage 22.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("SHARED ENGINE")
    print("-" * 78)

    print(
        "Cache recordings:",
        len(
            store
        ),
    )

    print(
        "Total timesteps:",
        store.total_timesteps,
    )

    print(
        "Feature width:",
        store.feature_width,
    )

    print(
        "Target sentences:",
        len(
            targets
        ),
    )

    print(
        "CTC output classes:",
        len(
            token_to_id
        ),
    )

    print(
        "CTC blank ID:",
        0,
    )

    print()
    print("TARGET CONTRACT")
    print("-" * 78)

    print(
        "Minimum target ID:",
        target_id_min,
    )

    print(
        "Maximum target ID:",
        target_id_max,
    )

    print(
        "Targets containing blank:",
        target_blank_violations,
    )

    print()
    print("KNOWN-SIGNER SPLITS")
    print("-" * 78)

    for row in fold_counts:

        print(
            f"Fold {row['outer_fold']}: "
            f"train={row['train']}, "
            f"test={row['test']}"
        )

    print()
    print("LOSO SPLITS")
    print("-" * 78)

    for loso_id in [
        "loso_subject1",
        "loso_subject2",
        "loso_subject3",
    ]:

        counts = loso_counts[
            loso_id
        ]

        print(
            f"{loso_id}: "
            f"train={counts['train']}, "
            f"test={counts['test']}"
        )

    print()
    print("TEMPLATE HOLDOUT")
    print("-" * 78)

    print(
        "Train:",
        len(
            template_train
        ),
    )

    print(
        "Test:",
        len(
            template_test
        ),
    )

    print(
        "Test subjects:",
        ", ".join(
            sorted(
                template_test_subjects
            )
        ),
    )

    print()
    print("FOLD-1 NORMALIZATION SMOKE")
    print("-" * 78)

    print(
        "Fit recordings:",
        stats.recording_count,
    )

    print(
        "Fit frames:",
        stats.frame_count,
    )

    print(
        "Feature width:",
        stats.feature_width,
    )

    print(
        "Minimum std:",
        float(
            np.min(
                stats.std
            )
        ),
    )

    print(
        "Maximum std:",
        float(
            np.max(
                stats.std
            )
        ),
    )

    print()
    print("COLLATE SMOKE")
    print("-" * 78)

    print(
        "Sample IDs:",
        batch_sample_ids,
    )

    print(
        "Input lengths:",
        batch_lengths,
    )

    print(
        "Batch shape:",
        batch[
            "inputs"
        ].shape,
    )

    print(
        "Padding violations:",
        padding_violations,
    )

    print(
        "Mask violations:",
        mask_violations,
    )

    print(
        "Input padding value:",
        0.0,
    )

    print(
        "Target padding value:",
        TARGET_PADDING_VALUE,
    )

    print()
    print("UPSTREAM HASHES")
    print("-" * 78)

    for name, value in (
        upstream_hashes.items()
    ):

        print(
            f"{name}: PASS | {value}"
        )

    print()
    print("OUTPUTS")
    print("-" * 78)

    print(
        "Saved:",
        SUMMARY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        DOC_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    stage19_pass = all(
        [
            len(store) == 7400,
            store.total_timesteps
            == 1_829_320,
            store.feature_width == 38,
            len(targets) == 300,
            len(token_to_id) == 456,
            token_to_id[
                "<blank>"
            ] == 0,
            target_blank_violations
            == 0,
            target_id_min == 1,
            target_id_max == 455,
            padding_violations == 0,
            mask_violations == 0,
            TARGET_PADDING_VALUE == -1,
            stage13_sample_set
            == cache_sample_set,
            stage16_sample_set
            == cache_sample_set,
            missing_targets == 0,
            metadata_mismatches == 0,
        ]
    )

    print()

    if stage19_pass:

        print(
            "FINAL STATUS: "
            "STAGE 19 BUILD PASSED"
        )

    else:

        print(
            "FINAL STATUS: "
            "STAGE 19 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

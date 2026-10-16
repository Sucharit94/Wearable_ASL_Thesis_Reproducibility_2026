from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "raw_manifest.csv"
)

MANIFEST_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "raw_manifest.sha256"
)

CACHE_DIR = (
    PROJECT_ROOT
    / "cache"
    / "raw"
)

CACHE_PATH = (
    CACHE_DIR
    / "stage17_raw_f32.bin"
)

TEMP_CACHE_PATH = (
    CACHE_DIR
    / "stage17_raw_f32.bin.part"
)

ARTIFACT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
)

INDEX_PATH = (
    ARTIFACT_DIR
    / "stage17_raw_cache_index.csv"
)

SUMMARY_PATH = (
    ARTIFACT_DIR
    / "stage17_raw_cache_summary.json"
)

CACHE_HASH_PATH = (
    ARTIFACT_DIR
    / "stage17_raw_cache.sha256"
)

INDEX_HASH_PATH = (
    ARTIFACT_DIR
    / "stage17_raw_cache_index.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_17_RAW_BINARY_CACHE.md"
)


EXPECTED_RECORDINGS = 7400
EXPECTED_FEATURE_WIDTH = 38
EXPECTED_TOTAL_TIMESTEPS = 1_829_320

CACHE_DTYPE = np.dtype("<f4")

EXPECTED_CACHE_BYTES = (
    EXPECTED_TOTAL_TIMESTEPS
    * EXPECTED_FEATURE_WIDTH
    * CACHE_DTYPE.itemsize
)


def sha256_file(path: Path) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def read_recorded_hash(path: Path) -> str:

    text = path.read_text(
        encoding="utf-8-sig"
    ).strip()

    token = text.split()[0].lower()

    if (
        len(token) != 64
        or any(
            char not in "0123456789abcdef"
            for char in token
        )
    ):

        raise RuntimeError(
            f"Invalid SHA-256 record: {path}"
        )

    return token


def load_manifest() -> list[dict]:

    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        return list(
            csv.DictReader(handle)
        )


def main() -> None:

    print("=" * 78)
    print("STAGE 17 — RAW BINARY CACHE")
    print("=" * 78)

    # ---------------------------------------------------------
    # Frozen manifest guard.
    # ---------------------------------------------------------

    recorded_manifest_hash = (
        read_recorded_hash(
            MANIFEST_HASH_PATH
        )
    )

    actual_manifest_hash = (
        sha256_file(
            MANIFEST_PATH
        )
    )

    if (
        recorded_manifest_hash
        != actual_manifest_hash
    ):

        raise RuntimeError(
            "Stage-4 manifest hash mismatch."
        )

    manifest = load_manifest()

    if len(manifest) != EXPECTED_RECORDINGS:

        raise RuntimeError(
            f"Expected {EXPECTED_RECORDINGS:,} "
            f"manifest rows, found {len(manifest):,}."
        )

    required_columns = {
        "sample_id",
        "subject",
        "sentence_id",
        "repetition_id",
        "relative_path",
        "bytes",
        "sha256",
    }

    missing_columns = (
        required_columns
        - set(
            manifest[0].keys()
        )
    )

    if missing_columns:

        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing_columns)}"
        )

    sample_ids = [
        row["sample_id"]
        for row in manifest
    ]

    if len(set(sample_ids)) != EXPECTED_RECORDINGS:

        raise RuntimeError(
            "Manifest sample IDs are not unique."
        )

    # Deterministic cache order.
    manifest = sorted(
        manifest,
        key=lambda row: (
            row["sample_id"]
        ),
    )

    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    ARTIFACT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if TEMP_CACHE_PATH.exists():

        TEMP_CACHE_PATH.unlink()

    # ---------------------------------------------------------
    # Build one contiguous little-endian float32 file.
    # ---------------------------------------------------------

    print()
    print("SOURCE CONTRACT")
    print("-" * 78)

    print(
        "Manifest rows:",
        len(manifest),
    )

    print(
        "Manifest hash match:",
        True,
    )

    print(
        "Expected feature width:",
        EXPECTED_FEATURE_WIDTH,
    )

    cache_digest = hashlib.sha256()

    index_rows = []

    current_row = 0

    verified_raw_hashes = 0

    global_max_cast_error = 0.0

    max_cast_error_sample = None

    print()
    print("CACHE BUILD")
    print("-" * 78)

    with TEMP_CACHE_PATH.open(
        "wb"
    ) as output:

        for index, row in enumerate(
            manifest,
            start=1,
        ):

            raw_path = (
                PROJECT_ROOT
                / Path(
                    row[
                        "relative_path"
                    ]
                )
            )

            if not raw_path.exists():

                raise FileNotFoundError(
                    raw_path
                )

            actual_size = (
                raw_path.stat().st_size
            )

            expected_size = int(
                row["bytes"]
            )

            if (
                actual_size
                != expected_size
            ):

                raise RuntimeError(
                    f"File-size mismatch for "
                    f"{row['sample_id']}: "
                    f"{actual_size} != "
                    f"{expected_size}"
                )

            actual_raw_hash = (
                sha256_file(
                    raw_path
                )
            )

            expected_raw_hash = (
                row[
                    "sha256"
                ].lower()
            )

            if (
                actual_raw_hash
                != expected_raw_hash
            ):

                raise RuntimeError(
                    f"Raw SHA-256 mismatch for "
                    f"{row['sample_id']}"
                )

            verified_raw_hashes += 1

            raw = np.loadtxt(
                raw_path,
                delimiter=",",
                dtype=np.float64,
                ndmin=2,
            )

            if (
                raw.ndim != 2
                or raw.shape[1]
                != EXPECTED_FEATURE_WIDTH
            ):

                raise RuntimeError(
                    f"Unexpected shape for "
                    f"{row['sample_id']}: "
                    f"{raw.shape}"
                )

            if not np.isfinite(
                raw
            ).all():

                raise RuntimeError(
                    f"Non-finite value detected in "
                    f"{row['sample_id']}"
                )

            cached = np.asarray(
                raw,
                dtype=CACHE_DTYPE,
                order="C",
            )

            restored = cached.astype(
                np.float64
            )

            cast_error = float(
                np.max(
                    np.abs(
                        raw
                        - restored
                    )
                )
            )

            if (
                cast_error
                > global_max_cast_error
            ):

                global_max_cast_error = (
                    cast_error
                )

                max_cast_error_sample = (
                    row[
                        "sample_id"
                    ]
                )

            cache_bytes = (
                cached.tobytes(
                    order="C"
                )
            )

            output.write(
                cache_bytes
            )

            cache_digest.update(
                cache_bytes
            )

            sequence_hash = (
                hashlib.sha256(
                    cache_bytes
                ).hexdigest()
            )

            timesteps = int(
                cached.shape[0]
            )

            start_row = (
                current_row
            )

            end_row = (
                start_row
                + timesteps
            )

            index_rows.append(
                {
                    "cache_order": (
                        index
                    ),

                    "sample_id": (
                        row[
                            "sample_id"
                        ]
                    ),

                    "subject": (
                        row[
                            "subject"
                        ]
                    ),

                    "sentence_id": (
                        int(
                            row[
                                "sentence_id"
                            ]
                        )
                    ),

                    "repetition_id": (
                        int(
                            row[
                                "repetition_id"
                            ]
                        )
                    ),

                    "timesteps": (
                        timesteps
                    ),

                    "feature_width": (
                        EXPECTED_FEATURE_WIDTH
                    ),

                    "cache_row_start": (
                        start_row
                    ),

                    "cache_row_end_exclusive": (
                        end_row
                    ),

                    "raw_sha256": (
                        expected_raw_hash
                    ),

                    "cache_slice_sha256": (
                        sequence_hash
                    ),
                }
            )

            current_row = (
                end_row
            )

            if (
                index == 1
                or index % 250 == 0
                or index
                == len(manifest)
            ):

                print(
                    f"Processed "
                    f"{index:>4,} / "
                    f"{len(manifest):,}"
                )

        output.flush()

        os.fsync(
            output.fileno()
        )

    # ---------------------------------------------------------
    # Cross-check Stage-6 total length.
    # ---------------------------------------------------------

    if (
        current_row
        != EXPECTED_TOTAL_TIMESTEPS
    ):

        raise RuntimeError(
            "Total timestep mismatch: "
            f"{current_row:,} != "
            f"{EXPECTED_TOTAL_TIMESTEPS:,}"
        )

    actual_temp_bytes = (
        TEMP_CACHE_PATH.stat().st_size
    )

    if (
        actual_temp_bytes
        != EXPECTED_CACHE_BYTES
    ):

        raise RuntimeError(
            "Cache byte-size mismatch: "
            f"{actual_temp_bytes:,} != "
            f"{EXPECTED_CACHE_BYTES:,}"
        )

    # Only publish final cache after the build itself passes.
    TEMP_CACHE_PATH.replace(
        CACHE_PATH
    )

    cache_hash = (
        cache_digest.hexdigest()
    )

    # ---------------------------------------------------------
    # Write deterministic index.
    # ---------------------------------------------------------

    with INDEX_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                index_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            index_rows
        )

    index_hash = (
        sha256_file(
            INDEX_PATH
        )
    )

    CACHE_HASH_PATH.write_text(
        cache_hash + "\n",
        encoding="utf-8",
    )

    INDEX_HASH_PATH.write_text(
        index_hash + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Full binary round-trip verification.
    #
    # Every raw CSV is parsed again and compared exactly
    # against its corresponding float32 binary slice.
    # ---------------------------------------------------------

    print()
    print("FULL CACHE VERIFICATION")
    print("-" * 78)

    cache = np.memmap(
        CACHE_PATH,
        dtype=CACHE_DTYPE,
        mode="r",
        shape=(
            EXPECTED_TOTAL_TIMESTEPS,
            EXPECTED_FEATURE_WIDTH,
        ),
        order="C",
    )

    verification_failures = []

    verified_slices = 0

    for index, (
        manifest_row,
        index_row,
    ) in enumerate(
        zip(
            manifest,
            index_rows,
            strict=True,
        ),
        start=1,
    ):

        if (
            manifest_row[
                "sample_id"
            ]
            != index_row[
                "sample_id"
            ]
        ):

            raise RuntimeError(
                "Manifest/index ordering mismatch."
            )

        raw_path = (
            PROJECT_ROOT
            / Path(
                manifest_row[
                    "relative_path"
                ]
            )
        )

        raw = np.loadtxt(
            raw_path,
            delimiter=",",
            dtype=np.float64,
            ndmin=2,
        )

        expected = np.asarray(
            raw,
            dtype=CACHE_DTYPE,
            order="C",
        )

        start = int(
            index_row[
                "cache_row_start"
            ]
        )

        end = int(
            index_row[
                "cache_row_end_exclusive"
            ]
        )

        observed = np.asarray(
            cache[
                start:end
            ]
        )

        if not np.array_equal(
            observed,
            expected,
        ):

            verification_failures.append(
                manifest_row[
                    "sample_id"
                ]
            )

        else:

            observed_hash = (
                hashlib.sha256(
                    observed.tobytes(
                        order="C"
                    )
                ).hexdigest()
            )

            if (
                observed_hash
                != index_row[
                    "cache_slice_sha256"
                ]
            ):

                verification_failures.append(
                    manifest_row[
                        "sample_id"
                    ]
                )

            else:

                verified_slices += 1

        if (
            index == 1
            or index % 500 == 0
            or index
            == len(manifest)
        ):

            print(
                f"Verified "
                f"{index:>4,} / "
                f"{len(manifest):,}"
            )

    del cache

    if verification_failures:

        raise RuntimeError(
            "Binary cache verification failed for "
            f"{len(verification_failures)} recordings. "
            f"First failures: "
            f"{verification_failures[:10]}"
        )

    # ---------------------------------------------------------
    # Index continuity verification.
    # ---------------------------------------------------------

    expected_start = 0

    continuity_failures = 0

    for row in index_rows:

        start = int(
            row[
                "cache_row_start"
            ]
        )

        end = int(
            row[
                "cache_row_end_exclusive"
            ]
        )

        timesteps = int(
            row[
                "timesteps"
            ]
        )

        if start != expected_start:

            continuity_failures += 1

        if (
            end - start
            != timesteps
        ):

            continuity_failures += 1

        expected_start = (
            end
        )

    if continuity_failures:

        raise RuntimeError(
            "Cache index continuity failure."
        )

    if (
        expected_start
        != EXPECTED_TOTAL_TIMESTEPS
    ):

        raise RuntimeError(
            "Final index endpoint mismatch."
        )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 17,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "cache_type": (
            "contiguous_raw_sensor_execution_cache"
        ),

        "source": {
            "recordings": (
                len(manifest)
            ),

            "feature_width": (
                EXPECTED_FEATURE_WIDTH
            ),

            "total_timesteps": (
                EXPECTED_TOTAL_TIMESTEPS
            ),

            "manifest_sha256": (
                actual_manifest_hash
            ),

            "manifest_hash_match": (
                True
            ),

            "raw_file_hashes_verified": (
                verified_raw_hashes
            ),
        },

        "binary": {
            "path": (
                "cache/raw/"
                "stage17_raw_f32.bin"
            ),

            "dtype": (
                "little_endian_float32"
            ),

            "numpy_dtype": (
                "<f4"
            ),

            "layout": (
                "row_major_C_contiguous"
            ),

            "shape": [
                EXPECTED_TOTAL_TIMESTEPS,
                EXPECTED_FEATURE_WIDTH,
            ],

            "bytes": (
                CACHE_PATH.stat().st_size
            ),

            "sha256": (
                cache_hash
            ),
        },

        "index": {
            "path": (
                "artifacts/cache/"
                "stage17_raw_cache_index.csv"
            ),

            "rows": (
                len(index_rows)
            ),

            "sha256": (
                index_hash
            ),

            "continuity_failures": (
                continuity_failures
            ),
        },

        "float32_conversion": {
            "performed": (
                True
            ),

            "maximum_absolute_cast_error": (
                global_max_cast_error
            ),

            "maximum_error_sample_id": (
                max_cast_error_sample
            ),

            "scientific_source_of_truth": (
                "Original author CSV files remain "
                "the immutable numerical source."
            ),
        },

        "verification": {
            "all_7400_slices_verified": (
                verified_slices
                == EXPECTED_RECORDINGS
            ),

            "verified_slices": (
                verified_slices
            ),

            "failed_slices": (
                len(
                    verification_failures
                )
            ),

            "exact_float32_roundtrip": (
                len(
                    verification_failures
                ) == 0
            ),
        },

        "scope": {
            "normalization": False,
            "filtering": False,
            "angle_unwrapping": False,
            "resampling": False,
            "truncation": False,
            "padding": False,
            "augmentation": False,
            "windowing": False,
            "label_processing": False,
            "split_specific_processing": False,
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

    # ---------------------------------------------------------
    # Documentation.
    # ---------------------------------------------------------

    markdown = f"""# Stage 17 — Raw Binary Cache

## Purpose

Convert the immutable author CSV recordings into one contiguous binary
execution cache for faster local and Colab loading.

The original CSV files remain the scientific source of truth.

## Source

Recordings:

{len(manifest):,}

Feature width:

{EXPECTED_FEATURE_WIDTH}

Total time steps:

{EXPECTED_TOTAL_TIMESTEPS:,}

All manifest raw-file SHA-256 values verified:

{verified_raw_hashes:,}

## Binary representation

File:

`cache/raw/stage17_raw_f32.bin`

Representation:

little-endian float32

Layout:

C-contiguous row-major

Logical shape:

`({EXPECTED_TOTAL_TIMESTEPS}, {EXPECTED_FEATURE_WIDTH})`

Size:

{CACHE_PATH.stat().st_size:,} bytes

SHA-256:

`{cache_hash}`

## Index

`artifacts/cache/stage17_raw_cache_index.csv`

Each recording stores its contiguous cache row range:

`cache_row_start : cache_row_end_exclusive`

The original variable sequence length is therefore preserved exactly.

## Float32 conversion

The author CSV values are parsed first as float64 and converted to
float32 for the execution cache.

Maximum observed absolute cast error:

{global_max_cast_error}

The float32 cache is not treated as a replacement for the immutable
CSV source.

## Full verification

Verified cache slices:

{verified_slices:,}

Failed cache slices:

{len(verification_failures):,}

Every cached slice was compared exactly against a fresh float32 cast of
its original CSV recording.

## No preprocessing

Stage 17 performs no:

- normalization,
- filtering,
- angle unwrapping,
- resampling,
- truncation,
- padding,
- augmentation,
- windowing,
- label processing,
- split-specific transformation.

Those decisions belong to later stages.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console report.
    # ---------------------------------------------------------

    print()
    print("CACHE CONTRACT")
    print("-" * 78)

    print(
        "Recordings:",
        len(manifest),
    )

    print(
        "Total timesteps:",
        EXPECTED_TOTAL_TIMESTEPS,
    )

    print(
        "Feature width:",
        EXPECTED_FEATURE_WIDTH,
    )

    print(
        "Binary dtype:",
        "<f4",
    )

    print(
        "Binary bytes:",
        CACHE_PATH.stat().st_size,
    )

    print()
    print("SOURCE VERIFICATION")
    print("-" * 78)

    print(
        "Raw SHA-256 files verified:",
        verified_raw_hashes,
    )

    print(
        "Manifest hash match:",
        True,
    )

    print()
    print("FLOAT32 CONVERSION")
    print("-" * 78)

    print(
        "Maximum absolute cast error:",
        global_max_cast_error,
    )

    print(
        "Maximum-error sample:",
        max_cast_error_sample,
    )

    print()
    print("CACHE VERIFICATION")
    print("-" * 78)

    print(
        "Verified slices:",
        verified_slices,
    )

    print(
        "Failed slices:",
        len(
            verification_failures
        ),
    )

    print(
        "Index continuity failures:",
        continuity_failures,
    )

    print()
    print("FROZEN HASHES")
    print("-" * 78)

    print(
        "Binary:",
        cache_hash,
    )

    print(
        "Index:",
        index_hash,
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    print(
        "Generated binary:",
        CACHE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    for path in [
        INDEX_PATH,
        SUMMARY_PATH,
        CACHE_HASH_PATH,
        INDEX_HASH_PATH,
        DOC_PATH,
    ]:

        print(
            "Saved:",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    stage17_pass = (
        len(manifest)
        == EXPECTED_RECORDINGS

        and current_row
        == EXPECTED_TOTAL_TIMESTEPS

        and CACHE_PATH.stat().st_size
        == EXPECTED_CACHE_BYTES

        and verified_raw_hashes
        == EXPECTED_RECORDINGS

        and verified_slices
        == EXPECTED_RECORDINGS

        and len(
            verification_failures
        ) == 0

        and continuity_failures
        == 0

        and len(index_rows)
        == EXPECTED_RECORDINGS
    )

    print()

    if stage17_pass:

        print(
            "FINAL STATUS: "
            "STAGE 17 BUILD PASSED"
        )

    else:

        print(
            "FINAL STATUS: "
            "STAGE 17 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter
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

SOURCE_CONTRACT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "source"
    / "SOURCE_CONTRACT.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
)

RECORDING_AUDIT_PATH = (
    OUTPUT_DIR
    / "stage5_recording_integrity.csv"
)

FEATURE_RANGE_PATH = (
    OUTPUT_DIR
    / "stage5_feature_ranges.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage5_integrity_summary.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_5_DATASET_INTEGRITY.md"
)


EXPECTED_FEATURES = 38


def sha256_file(
    path: Path,
    chunk_size: int = 1024 * 1024,
) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)

            if not chunk:
                break

            digest.update(chunk)

    return digest.hexdigest()


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(encoding="utf-8-sig")
    )


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def safe_float_string(value: float | None) -> str:
    if value is None:
        return ""

    if not math.isfinite(value):
        return ""

    return repr(float(value))


def main() -> None:
    print("=" * 78)
    print("STAGE 5 — DATASET INTEGRITY AUDIT")
    print("=" * 78)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing Stage 4 manifest: {MANIFEST_PATH}"
        )

    if not MANIFEST_HASH_PATH.exists():
        raise FileNotFoundError(
            f"Missing manifest hash file: "
            f"{MANIFEST_HASH_PATH}"
        )

    if not SOURCE_CONTRACT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Stage 3 source contract: "
            f"{SOURCE_CONTRACT_PATH}"
        )

    recorded_manifest_hash = (
        MANIFEST_HASH_PATH
        .read_text(encoding="utf-8")
        .strip()
        .lower()
    )

    actual_manifest_hash = sha256_file(
        MANIFEST_PATH
    )

    if actual_manifest_hash != recorded_manifest_hash:
        raise RuntimeError(
            "Stage 4 raw manifest hash mismatch. "
            "Do not continue."
        )

    print("Stage 4 manifest integrity: PASS")

    contract = load_json(
        SOURCE_CONTRACT_PATH
    )

    manifest = load_manifest()

    expected_recordings = int(
        contract[
            "project_raw_source"
        ][
            "csv_files"
        ]
    )

    print(
        f"Manifest recordings: "
        f"{len(manifest):,}"
    )

    if len(manifest) != expected_recordings:
        raise RuntimeError(
            f"Expected {expected_recordings:,} manifest rows "
            f"but found {len(manifest):,}."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    global_feature_min = np.full(
        EXPECTED_FEATURES,
        np.inf,
        dtype=np.float64,
    )

    global_feature_max = np.full(
        EXPECTED_FEATURES,
        -np.inf,
        dtype=np.float64,
    )

    global_feature_finite_count = np.zeros(
        EXPECTED_FEATURES,
        dtype=np.int64,
    )

    audit_rows: list[dict] = []

    total_numeric_rows = 0
    total_numeric_values = 0

    parse_failures = 0
    width_failures = 0
    empty_failures = 0
    nonfinite_files = 0
    size_failures = 0

    print()
    print("Parsing all 7,400 raw CSV recordings...")
    print()

    for index, manifest_row in enumerate(
        manifest,
        start=1,
    ):
        sample_id = manifest_row["sample_id"]

        relative_path = manifest_row[
            "relative_path"
        ]

        path = PROJECT_ROOT / Path(
            relative_path
        )

        manifest_bytes = int(
            manifest_row["bytes"]
        )

        result = {
            "sample_id": sample_id,
            "subject": manifest_row["subject"],
            "sentence_id": manifest_row[
                "sentence_id"
            ],
            "repetition_id": manifest_row[
                "repetition_id"
            ],
            "relative_path": relative_path,
            "manifest_bytes": manifest_bytes,
            "actual_bytes": "",
            "size_match": False,
            "rows": "",
            "columns": "",
            "numeric_values": "",
            "nan_count": "",
            "positive_inf_count": "",
            "negative_inf_count": "",
            "nonfinite_count": "",
            "finite_min": "",
            "finite_max": "",
            "zero_variance_columns_in_recording": "",
            "parse_status": "FAIL",
            "integrity_status": "FAIL",
            "error": "",
        }

        if not path.exists():
            result["error"] = "FILE_NOT_FOUND"
            parse_failures += 1
            audit_rows.append(result)
            continue

        actual_bytes = path.stat().st_size

        result["actual_bytes"] = actual_bytes
        result["size_match"] = (
            actual_bytes == manifest_bytes
        )

        if actual_bytes != manifest_bytes:
            size_failures += 1

        try:
            array = np.loadtxt(
                path,
                delimiter=",",
                dtype=np.float64,
                ndmin=2,
            )

            result["parse_status"] = "PASS"

        except Exception as exc:
            result["error"] = (
                f"{type(exc).__name__}: {exc}"
            )

            parse_failures += 1
            audit_rows.append(result)

            if (
                index == 1
                or index % 250 == 0
                or index == len(manifest)
            ):
                print(
                    f"Processed {index:>4,} / "
                    f"{len(manifest):,}"
                )

            continue

        if array.ndim != 2:
            result["error"] = (
                f"Unexpected ndim={array.ndim}"
            )
            parse_failures += 1
            audit_rows.append(result)
            continue

        rows, columns = array.shape

        result["rows"] = rows
        result["columns"] = columns
        result["numeric_values"] = int(
            array.size
        )

        if rows <= 0:
            empty_failures += 1

        if columns != EXPECTED_FEATURES:
            width_failures += 1

        nan_count = int(
            np.isnan(array).sum()
        )

        positive_inf_count = int(
            np.isposinf(array).sum()
        )

        negative_inf_count = int(
            np.isneginf(array).sum()
        )

        nonfinite_count = (
            nan_count
            + positive_inf_count
            + negative_inf_count
        )

        result["nan_count"] = nan_count
        result[
            "positive_inf_count"
        ] = positive_inf_count

        result[
            "negative_inf_count"
        ] = negative_inf_count

        result[
            "nonfinite_count"
        ] = nonfinite_count

        if nonfinite_count > 0:
            nonfinite_files += 1

        finite_mask = np.isfinite(
            array
        )

        finite_values = array[
            finite_mask
        ]

        if finite_values.size > 0:
            result["finite_min"] = (
                safe_float_string(
                    float(
                        finite_values.min()
                    )
                )
            )

            result["finite_max"] = (
                safe_float_string(
                    float(
                        finite_values.max()
                    )
                )
            )

        if (
            rows > 0
            and columns == EXPECTED_FEATURES
        ):
            finite_feature_counts = (
                np.isfinite(array).sum(
                    axis=0
                )
            )

            global_feature_finite_count += (
                finite_feature_counts
            )

            for feature_index in range(
                EXPECTED_FEATURES
            ):
                feature_values = array[
                    :,
                    feature_index,
                ]

                finite_feature_values = (
                    feature_values[
                        np.isfinite(
                            feature_values
                        )
                    ]
                )

                if (
                    finite_feature_values.size
                    > 0
                ):
                    feature_min = float(
                        finite_feature_values.min()
                    )

                    feature_max = float(
                        finite_feature_values.max()
                    )

                    global_feature_min[
                        feature_index
                    ] = min(
                        global_feature_min[
                            feature_index
                        ],
                        feature_min,
                    )

                    global_feature_max[
                        feature_index
                    ] = max(
                        global_feature_max[
                            feature_index
                        ],
                        feature_max,
                    )

            per_recording_min = np.min(
                array,
                axis=0,
            )

            per_recording_max = np.max(
                array,
                axis=0,
            )

            zero_variance_columns = int(
                np.sum(
                    per_recording_min
                    == per_recording_max
                )
            )

            result[
                "zero_variance_columns_in_recording"
            ] = zero_variance_columns

        structural_pass = (
            bool(
                result["size_match"]
            )
            and rows > 0
            and columns == EXPECTED_FEATURES
            and nonfinite_count == 0
        )

        result[
            "integrity_status"
        ] = (
            "PASS"
            if structural_pass
            else "FAIL"
        )

        if structural_pass:
            total_numeric_rows += rows
            total_numeric_values += int(
                array.size
            )

        audit_rows.append(result)

        if (
            index == 1
            or index % 250 == 0
            or index == len(manifest)
        ):
            print(
                f"Processed {index:>4,} / "
                f"{len(manifest):,}"
            )

    fieldnames = [
        "sample_id",
        "subject",
        "sentence_id",
        "repetition_id",
        "relative_path",
        "manifest_bytes",
        "actual_bytes",
        "size_match",
        "rows",
        "columns",
        "numeric_values",
        "nan_count",
        "positive_inf_count",
        "negative_inf_count",
        "nonfinite_count",
        "finite_min",
        "finite_max",
        "zero_variance_columns_in_recording",
        "parse_status",
        "integrity_status",
        "error",
    ]

    with RECORDING_AUDIT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(audit_rows)

    integrity_failures = [
        row
        for row in audit_rows
        if row[
            "integrity_status"
        ] != "PASS"
    ]

    feature_rows = []

    globally_constant_features = []

    for feature_index in range(
        EXPECTED_FEATURES
    ):
        minimum = float(
            global_feature_min[
                feature_index
            ]
        )

        maximum = float(
            global_feature_max[
                feature_index
            ]
        )

        finite_count = int(
            global_feature_finite_count[
                feature_index
            ]
        )

        if (
            not math.isfinite(minimum)
            or not math.isfinite(maximum)
        ):
            range_value = None
            is_constant = None

        else:
            range_value = (
                maximum - minimum
            )

            is_constant = bool(
                maximum == minimum
            )

        if is_constant:
            globally_constant_features.append(
                feature_index
            )

        feature_rows.append(
            {
                "feature_index_zero_based": (
                    feature_index
                ),
                "feature_number_one_based": (
                    feature_index + 1
                ),
                "finite_value_count": (
                    finite_count
                ),
                "global_min": (
                    safe_float_string(
                        minimum
                    )
                ),
                "global_max": (
                    safe_float_string(
                        maximum
                    )
                ),
                "global_range": (
                    ""
                    if range_value is None
                    else safe_float_string(
                        range_value
                    )
                ),
                "globally_constant": (
                    ""
                    if is_constant is None
                    else is_constant
                ),
            }
        )

    with FEATURE_RANGE_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "feature_index_zero_based",
                "feature_number_one_based",
                "finite_value_count",
                "global_min",
                "global_max",
                "global_range",
                "globally_constant",
            ],
        )

        writer.writeheader()
        writer.writerows(feature_rows)

    row_counts = [
        int(row["rows"])
        for row in audit_rows
        if (
            row["integrity_status"]
            == "PASS"
        )
    ]

    subject_counts = Counter(
        row["subject"]
        for row in audit_rows
        if (
            row["integrity_status"]
            == "PASS"
        )
    )

    summary = {
        "project_generation": (
            contract[
                "project_generation"
            ]
        ),
        "stage": 5,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "input_manifest": (
            "artifacts/manifests/"
            "raw_manifest.csv"
        ),
        "input_manifest_sha256": (
            actual_manifest_hash
        ),
        "expected_feature_count": (
            EXPECTED_FEATURES
        ),
        "recordings_expected": (
            expected_recordings
        ),
        "recordings_audited": (
            len(audit_rows)
        ),
        "recordings_passed": (
            len(audit_rows)
            - len(
                integrity_failures
            )
        ),
        "recordings_failed": (
            len(integrity_failures)
        ),
        "parse_failures": (
            parse_failures
        ),
        "size_failures": (
            size_failures
        ),
        "empty_recordings": (
            empty_failures
        ),
        "wrong_width_recordings": (
            width_failures
        ),
        "recordings_with_nonfinite_values": (
            nonfinite_files
        ),
        "total_valid_timesteps": (
            total_numeric_rows
        ),
        "total_valid_numeric_values": (
            total_numeric_values
        ),
        "subject_pass_counts": dict(
            sorted(
                subject_counts.items()
            )
        ),
        "globally_constant_feature_count": (
            len(
                globally_constant_features
            )
        ),
        "globally_constant_feature_indices_zero_based": (
            globally_constant_features
        ),
        "sequence_length_interpretation_deferred_to_stage": (
            6
        ),
        "feature_semantics_interpretation_deferred_to_stage": (
            7
        ),
        "raw_files_modified": False,
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 5 — Dataset Integrity Audit

## Purpose

Verify that every raw recording can be interpreted as a numerical
variable-length array with exactly 38 feature columns.

## Source

The audit consumes the frozen Stage 4 raw manifest.

Manifest SHA-256:

`{actual_manifest_hash}`

## Results

Recordings expected: {expected_recordings:,}

Recordings audited: {len(audit_rows):,}

Recordings passed: {len(audit_rows) - len(integrity_failures):,}

Recordings failed: {len(integrity_failures):,}

Parse failures: {parse_failures:,}

File-size mismatches: {size_failures:,}

Empty recordings: {empty_failures:,}

Wrong-width recordings: {width_failures:,}

Recordings containing NaN or Inf: {nonfinite_files:,}

Expected feature width: {EXPECTED_FEATURES}

Total valid timesteps: {total_numeric_rows:,}

Total valid numeric values: {total_numeric_values:,}

Globally constant features: {len(globally_constant_features)}

Zero-based constant-feature indices:

`{globally_constant_features}`

## Important scope boundary

This stage does not interpret sequence-length distributions.

Detailed length analysis belongs to Stage 6.

This stage does not assign anatomical meaning to the 38 columns.

Feature semantics belong to Stage 7.

This stage does not normalize, filter, augment, split, cache or train
the dataset.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("=" * 78)
    print("STAGE 5 AUDIT SUMMARY")
    print("=" * 78)

    print(
        f"Recordings audited:       "
        f"{len(audit_rows):,}"
    )

    print(
        f"Recordings passed:        "
        f"{len(audit_rows) - len(integrity_failures):,}"
    )

    print(
        f"Recordings failed:        "
        f"{len(integrity_failures):,}"
    )

    print(
        f"Parse failures:           "
        f"{parse_failures:,}"
    )

    print(
        f"Size mismatches:          "
        f"{size_failures:,}"
    )

    print(
        f"Empty recordings:         "
        f"{empty_failures:,}"
    )

    print(
        f"Wrong-width recordings:   "
        f"{width_failures:,}"
    )

    print(
        f"Non-finite recordings:    "
        f"{nonfinite_files:,}"
    )

    print(
        f"Total valid timesteps:    "
        f"{total_numeric_rows:,}"
    )

    print(
        f"Total numeric values:     "
        f"{total_numeric_values:,}"
    )

    print()
    print("Passed recordings by subject:")

    for subject, count in sorted(
        subject_counts.items()
    ):
        print(
            f"  {subject}: {count:,}"
        )

    print()
    print(
        "Globally constant features:",
        len(
            globally_constant_features
        ),
    )

    if globally_constant_features:
        print(
            "Zero-based indices:",
            globally_constant_features,
        )

    print()
    print(
        "Saved:",
        RECORDING_AUDIT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        FEATURE_RANGE_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

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

    print()

    structural_pass = (
        len(audit_rows)
        == expected_recordings
        and len(integrity_failures)
        == 0
        and parse_failures
        == 0
        and size_failures
        == 0
        and empty_failures
        == 0
        and width_failures
        == 0
        and nonfinite_files
        == 0
    )

    if structural_pass:
        print(
            "FINAL STATUS: "
            "STAGE 5 STRUCTURAL AUDIT PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 5 STRUCTURAL AUDIT FAILED"
        )

    print("=" * 78)

    if not structural_pass:
        sys.exit(1)


if __name__ == "__main__":
    main()

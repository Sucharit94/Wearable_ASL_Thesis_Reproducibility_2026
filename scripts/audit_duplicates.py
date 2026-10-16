from __future__ import annotations

import csv
import hashlib
import json
import struct
from collections import Counter, defaultdict
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

TARGETS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
)

BYTE_GROUPS_PATH = (
    OUTPUT_DIR
    / "stage12_byte_duplicate_groups.csv"
)

NUMERIC_GROUPS_PATH = (
    OUTPUT_DIR
    / "stage12_numeric_duplicate_groups.csv"
)

MEMBERS_PATH = (
    OUTPUT_DIR
    / "stage12_numeric_duplicate_members.csv"
)

ASSIGNMENT_PATH = (
    OUTPUT_DIR
    / "stage12_split_group_assignment.csv"
)

CONFLICTS_PATH = (
    OUTPUT_DIR
    / "stage12_target_conflicts.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage12_duplicate_summary.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_12_DUPLICATE_AUDIT.md"
)

FEATURE_WIDTH = 38


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


def read_recorded_hash(
    path: Path,
) -> str:
    text = path.read_text(
        encoding="utf-8-sig"
    ).strip()

    if not text:
        raise RuntimeError(
            f"Empty hash file: {path}"
        )

    # Works whether the file contains only the hash
    # or a conventional "hash filename" record.
    token = text.split()[0].strip().lower()

    if (
        len(token) != 64
        or any(
            char not in "0123456789abcdef"
            for char in token
        )
    ):
        raise RuntimeError(
            f"Could not parse SHA-256 from {path}"
        )

    return token


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


def canonical_numeric_hash(
    array: np.ndarray,
) -> str:
    """
    Hash exact parsed numerical content.

    Contract:
    - float64
    - little-endian
    - C contiguous
    - shape included
    - negative zero normalized to positive zero

    This detects equal numerical matrices even when
    source CSV formatting differs.
    """

    if array.ndim != 2:
        raise RuntimeError(
            f"Expected 2D array, got {array.shape}"
        )

    canonical = np.array(
        array,
        dtype="<f8",
        order="C",
        copy=True,
    )

    # Numerically, -0.0 and +0.0 are equal.
    # Normalize the sign bit so they hash identically.
    canonical[
        canonical == 0.0
    ] = 0.0

    digest = hashlib.sha256()

    digest.update(
        struct.pack(
            "<QQ",
            canonical.shape[0],
            canonical.shape[1],
        )
    )

    digest.update(
        canonical.tobytes(
            order="C"
        )
    )

    return digest.hexdigest()


def bool_text(value: bool) -> str:
    return "True" if value else "False"


def main() -> None:
    print("=" * 78)
    print("STAGE 12 — DUPLICATE AUDIT")
    print("=" * 78)

    # ---------------------------------------------------------
    # Manifest integrity guard
    # ---------------------------------------------------------

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing manifest: {MANIFEST_PATH}"
        )

    if not MANIFEST_HASH_PATH.exists():
        raise FileNotFoundError(
            f"Missing manifest hash: "
            f"{MANIFEST_HASH_PATH}"
        )

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
            "Stage 4 manifest SHA-256 mismatch."
        )

    manifest = load_csv(
        MANIFEST_PATH
    )

    targets = load_csv(
        TARGETS_PATH
    )

    if len(manifest) != 7400:
        raise RuntimeError(
            f"Expected 7,400 manifest rows, "
            f"found {len(manifest):,}."
        )

    required_columns = {
        "sample_id",
        "subject",
        "subject_number",
        "sentence_id",
        "repetition_id",
        "filename",
        "relative_path",
        "bytes",
        "sha256",
    }

    actual_columns = set(
        manifest[0].keys()
    )

    missing_columns = (
        required_columns
        - actual_columns
    )

    if missing_columns:
        raise RuntimeError(
            "Manifest missing columns: "
            f"{sorted(missing_columns)}"
        )

    target_by_id = {
        int(row["sentence_id"]): row
        for row in targets
    }

    if sorted(target_by_id) != list(
        range(1, 301)
    ):
        raise RuntimeError(
            "Stage 10 target IDs are not 1..300."
        )

    sample_ids = [
        row["sample_id"]
        for row in manifest
    ]

    if len(set(sample_ids)) != 7400:
        raise RuntimeError(
            "Manifest sample IDs are not unique."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print("MANIFEST CONTRACT")
    print("-" * 78)

    print(
        "Manifest rows:",
        len(manifest),
    )

    print(
        "Manifest hash match:",
        True,
    )

    # ---------------------------------------------------------
    # Byte-level duplicate groups
    # ---------------------------------------------------------

    by_byte_hash = defaultdict(
        list
    )

    for row in manifest:
        by_byte_hash[
            row["sha256"].lower()
        ].append(
            row
        )

    byte_duplicate_hashes = sorted(
        hash_value
        for hash_value, members
        in by_byte_hash.items()
        if len(members) > 1
    )

    byte_group_id_by_hash = {
        hash_value: (
            f"bytedup_{index:04d}"
        )
        for index, hash_value
        in enumerate(
            byte_duplicate_hashes,
            start=1,
        )
    }

    byte_group_rows = []

    for hash_value in (
        byte_duplicate_hashes
    ):
        members = sorted(
            by_byte_hash[
                hash_value
            ],
            key=lambda row: row[
                "sample_id"
            ],
        )

        subjects = sorted(
            {
                row["subject"]
                for row in members
            }
        )

        sentence_ids = sorted(
            {
                int(
                    row["sentence_id"]
                )
                for row in members
            }
        )

        gloss_targets = sorted(
            {
                target_by_id[
                    int(
                        row[
                            "sentence_id"
                        ]
                    )
                ]["gloss_raw"]
                for row in members
            }
        )

        english_targets = sorted(
            {
                target_by_id[
                    int(
                        row[
                            "sentence_id"
                        ]
                    )
                ]["english_raw"]
                for row in members
            }
        )

        byte_group_rows.append(
            {
                "byte_duplicate_group_id": (
                    byte_group_id_by_hash[
                        hash_value
                    ]
                ),
                "byte_sha256": (
                    hash_value
                ),
                "group_size": (
                    len(members)
                ),
                "subjects_json": json.dumps(
                    subjects
                ),
                "sentence_ids_json": json.dumps(
                    sentence_ids
                ),
                "sample_ids_json": json.dumps(
                    [
                        row["sample_id"]
                        for row in members
                    ]
                ),
                "cross_subject": (
                    len(subjects) > 1
                ),
                "cross_sentence_id": (
                    len(sentence_ids) > 1
                ),
                "gloss_target_conflict": (
                    len(
                        gloss_targets
                    ) > 1
                ),
                "english_target_conflict": (
                    len(
                        english_targets
                    ) > 1
                ),
            }
        )

    byte_fieldnames = [
        "byte_duplicate_group_id",
        "byte_sha256",
        "group_size",
        "subjects_json",
        "sentence_ids_json",
        "sample_ids_json",
        "cross_subject",
        "cross_sentence_id",
        "gloss_target_conflict",
        "english_target_conflict",
    ]

    with BYTE_GROUPS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=byte_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            byte_group_rows
        )

    print()
    print("BYTE DUPLICATES")
    print("-" * 78)

    print(
        "Duplicate groups:",
        len(
            byte_duplicate_hashes
        ),
    )

    print(
        "Recordings in duplicate groups:",
        sum(
            len(
                by_byte_hash[
                    hash_value
                ]
            )
            for hash_value
            in byte_duplicate_hashes
        ),
    )

    # ---------------------------------------------------------
    # Canonical numeric hashes
    # ---------------------------------------------------------

    numeric_hash_by_sample = {}

    by_numeric_hash = defaultdict(
        list
    )

    print()
    print("NUMERIC HASHING")
    print("-" * 78)

    sorted_manifest = sorted(
        manifest,
        key=lambda row: row[
            "sample_id"
        ],
    )

    for index, row in enumerate(
        sorted_manifest,
        start=1,
    ):
        path = (
            PROJECT_ROOT
            / Path(
                row[
                    "relative_path"
                ]
            )
        )

        array = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float64,
            ndmin=2,
        )

        if (
            array.ndim != 2
            or array.shape[1]
            != FEATURE_WIDTH
        ):
            raise RuntimeError(
                f"Unexpected shape for "
                f"{row['sample_id']}: "
                f"{array.shape}"
            )

        if not np.isfinite(
            array
        ).all():
            raise RuntimeError(
                f"Non-finite values in "
                f"{row['sample_id']}"
            )

        numeric_hash = (
            canonical_numeric_hash(
                array
            )
        )

        numeric_hash_by_sample[
            row["sample_id"]
        ] = numeric_hash

        by_numeric_hash[
            numeric_hash
        ].append(
            row
        )

        if (
            index == 1
            or index % 250 == 0
            or index
            == len(
                sorted_manifest
            )
        ):
            print(
                f"Processed "
                f"{index:>4,} / "
                f"{len(sorted_manifest):,}"
            )

    numeric_duplicate_hashes = sorted(
        hash_value
        for hash_value, members
        in by_numeric_hash.items()
        if len(members) > 1
    )

    numeric_group_id_by_hash = {
        hash_value: (
            f"numdup_{index:04d}"
        )
        for index, hash_value
        in enumerate(
            numeric_duplicate_hashes,
            start=1,
        )
    }

    # ---------------------------------------------------------
    # Build numeric group metadata
    # ---------------------------------------------------------

    numeric_group_rows = []
    numeric_member_rows = []
    conflict_rows = []

    cross_subject_group_count = 0
    cross_sentence_group_count = 0
    gloss_conflict_group_count = 0
    english_conflict_group_count = 0
    same_target_different_id_count = 0
    byte_identical_numeric_group_count = 0

    for numeric_hash in (
        numeric_duplicate_hashes
    ):
        members = sorted(
            by_numeric_hash[
                numeric_hash
            ],
            key=lambda row: row[
                "sample_id"
            ],
        )

        group_id = (
            numeric_group_id_by_hash[
                numeric_hash
            ]
        )

        subjects = sorted(
            {
                row["subject"]
                for row in members
            }
        )

        sentence_ids = sorted(
            {
                int(
                    row["sentence_id"]
                )
                for row in members
            }
        )

        byte_hashes = sorted(
            {
                row["sha256"].lower()
                for row in members
            }
        )

        gloss_targets = sorted(
            {
                target_by_id[
                    int(
                        row[
                            "sentence_id"
                        ]
                    )
                ]["gloss_raw"]
                for row in members
            }
        )

        english_targets = sorted(
            {
                target_by_id[
                    int(
                        row[
                            "sentence_id"
                        ]
                    )
                ]["english_raw"]
                for row in members
            }
        )

        cross_subject = (
            len(subjects) > 1
        )

        cross_sentence = (
            len(sentence_ids) > 1
        )

        gloss_conflict = (
            len(gloss_targets) > 1
        )

        english_conflict = (
            len(english_targets) > 1
        )

        same_target_different_id = (
            cross_sentence
            and not gloss_conflict
            and not english_conflict
        )

        all_byte_identical = (
            len(byte_hashes) == 1
        )

        if cross_subject:
            cross_subject_group_count += 1

        if cross_sentence:
            cross_sentence_group_count += 1

        if gloss_conflict:
            gloss_conflict_group_count += 1

        if english_conflict:
            english_conflict_group_count += 1

        if same_target_different_id:
            same_target_different_id_count += 1

        if all_byte_identical:
            byte_identical_numeric_group_count += 1

        numeric_group_rows.append(
            {
                "numeric_duplicate_group_id": (
                    group_id
                ),
                "numeric_sha256": (
                    numeric_hash
                ),
                "group_size": (
                    len(members)
                ),
                "subjects_json": json.dumps(
                    subjects
                ),
                "sentence_ids_json": json.dumps(
                    sentence_ids
                ),
                "sample_ids_json": json.dumps(
                    [
                        row["sample_id"]
                        for row in members
                    ]
                ),
                "unique_byte_hashes": (
                    len(byte_hashes)
                ),
                "all_members_byte_identical": (
                    all_byte_identical
                ),
                "cross_subject": (
                    cross_subject
                ),
                "cross_sentence_id": (
                    cross_sentence
                ),
                "gloss_target_conflict": (
                    gloss_conflict
                ),
                "english_target_conflict": (
                    english_conflict
                ),
                "same_target_different_sentence_ids": (
                    same_target_different_id
                ),
                "gloss_targets_json": (
                    json.dumps(
                        gloss_targets,
                        ensure_ascii=False,
                    )
                ),
                "english_targets_json": (
                    json.dumps(
                        english_targets,
                        ensure_ascii=False,
                    )
                ),
            }
        )

        for row in members:
            sentence_id = int(
                row["sentence_id"]
            )

            target = target_by_id[
                sentence_id
            ]

            numeric_member_rows.append(
                {
                    "numeric_duplicate_group_id": (
                        group_id
                    ),
                    "numeric_sha256": (
                        numeric_hash
                    ),
                    "sample_id": (
                        row["sample_id"]
                    ),
                    "subject": (
                        row["subject"]
                    ),
                    "sentence_id": (
                        sentence_id
                    ),
                    "repetition_id": (
                        int(
                            row[
                                "repetition_id"
                            ]
                        )
                    ),
                    "relative_path": (
                        row[
                            "relative_path"
                        ]
                    ),
                    "byte_sha256": (
                        row[
                            "sha256"
                        ].lower()
                    ),
                    "gloss_raw": (
                        target[
                            "gloss_raw"
                        ]
                    ),
                    "english_raw": (
                        target[
                            "english_raw"
                        ]
                    ),
                }
            )

        if (
            gloss_conflict
            or english_conflict
        ):
            conflict_rows.append(
                {
                    "numeric_duplicate_group_id": (
                        group_id
                    ),
                    "numeric_sha256": (
                        numeric_hash
                    ),
                    "group_size": (
                        len(members)
                    ),
                    "subjects_json": json.dumps(
                        subjects
                    ),
                    "sentence_ids_json": json.dumps(
                        sentence_ids
                    ),
                    "gloss_targets_json": (
                        json.dumps(
                            gloss_targets,
                            ensure_ascii=False,
                        )
                    ),
                    "english_targets_json": (
                        json.dumps(
                            english_targets,
                            ensure_ascii=False,
                        )
                    ),
                    "sample_ids_json": (
                        json.dumps(
                            [
                                row[
                                    "sample_id"
                                ]
                                for row
                                in members
                            ]
                        )
                    ),
                }
            )

    numeric_fieldnames = [
        "numeric_duplicate_group_id",
        "numeric_sha256",
        "group_size",
        "subjects_json",
        "sentence_ids_json",
        "sample_ids_json",
        "unique_byte_hashes",
        "all_members_byte_identical",
        "cross_subject",
        "cross_sentence_id",
        "gloss_target_conflict",
        "english_target_conflict",
        "same_target_different_sentence_ids",
        "gloss_targets_json",
        "english_targets_json",
    ]

    with NUMERIC_GROUPS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=numeric_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            numeric_group_rows
        )

    member_fieldnames = [
        "numeric_duplicate_group_id",
        "numeric_sha256",
        "sample_id",
        "subject",
        "sentence_id",
        "repetition_id",
        "relative_path",
        "byte_sha256",
        "gloss_raw",
        "english_raw",
    ]

    with MEMBERS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=member_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            numeric_member_rows
        )

    conflict_fieldnames = [
        "numeric_duplicate_group_id",
        "numeric_sha256",
        "group_size",
        "subjects_json",
        "sentence_ids_json",
        "gloss_targets_json",
        "english_targets_json",
        "sample_ids_json",
    ]

    with CONFLICTS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=conflict_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            conflict_rows
        )

    # ---------------------------------------------------------
    # Full split-group assignment for Stage 13+
    # ---------------------------------------------------------

    assignment_rows = []

    for row in sorted_manifest:
        sample_id = row[
            "sample_id"
        ]

        numeric_hash = (
            numeric_hash_by_sample[
                sample_id
            ]
        )

        group_size = len(
            by_numeric_hash[
                numeric_hash
            ]
        )

        is_duplicate = (
            group_size > 1
        )

        if is_duplicate:
            split_group_id = (
                numeric_group_id_by_hash[
                    numeric_hash
                ]
            )
        else:
            split_group_id = (
                "singleton__"
                + sample_id
            )

        assignment_rows.append(
            {
                "sample_id": (
                    sample_id
                ),
                "subject": (
                    row["subject"]
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
                "split_group_id": (
                    split_group_id
                ),
                "numeric_sha256": (
                    numeric_hash
                ),
                "numeric_group_size": (
                    group_size
                ),
                "is_numeric_duplicate": (
                    is_duplicate
                ),
                "byte_sha256": (
                    row[
                        "sha256"
                    ].lower()
                ),
            }
        )

    assignment_fieldnames = [
        "sample_id",
        "subject",
        "sentence_id",
        "repetition_id",
        "split_group_id",
        "numeric_sha256",
        "numeric_group_size",
        "is_numeric_duplicate",
        "byte_sha256",
    ]

    with ASSIGNMENT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=assignment_fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            assignment_rows
        )

    # ---------------------------------------------------------
    # Summary
    # ---------------------------------------------------------

    byte_duplicate_member_count = sum(
        len(
            by_byte_hash[
                hash_value
            ]
        )
        for hash_value
        in byte_duplicate_hashes
    )

    numeric_duplicate_member_count = sum(
        len(
            by_numeric_hash[
                hash_value
            ]
        )
        for hash_value
        in numeric_duplicate_hashes
    )

    numeric_only_group_count = sum(
        1
        for row in numeric_group_rows
        if not row[
            "all_members_byte_identical"
        ]
    )

    maximum_numeric_group_size = (
        max(
            (
                len(
                    by_numeric_hash[
                        hash_value
                    ]
                )
                for hash_value
                in numeric_duplicate_hashes
            ),
            default=1,
        )
    )

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 12,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "manifest": {
            "rows": (
                len(manifest)
            ),
            "sha256_match": True,
            "sha256": (
                actual_manifest_hash
            ),
        },

        "byte_duplicates": {
            "groups": (
                len(
                    byte_duplicate_hashes
                )
            ),
            "recordings_in_groups": (
                byte_duplicate_member_count
            ),
        },

        "numeric_duplicates": {
            "groups": (
                len(
                    numeric_duplicate_hashes
                )
            ),
            "recordings_in_groups": (
                numeric_duplicate_member_count
            ),
            "maximum_group_size": (
                maximum_numeric_group_size
            ),
            "groups_all_members_byte_identical": (
                byte_identical_numeric_group_count
            ),
            "numeric_duplicate_groups_not_all_byte_identical": (
                numeric_only_group_count
            ),
            "cross_subject_groups": (
                cross_subject_group_count
            ),
            "cross_sentence_id_groups": (
                cross_sentence_group_count
            ),
            "same_target_different_sentence_id_groups": (
                same_target_different_id_count
            ),
        },

        "target_conflicts": {
            "gloss_conflict_groups": (
                gloss_conflict_group_count
            ),
            "english_conflict_groups": (
                english_conflict_group_count
            ),
            "conflict_table_rows": (
                len(
                    conflict_rows
                )
            ),
        },

        "split_policy": {
            "known_signer": (
                "Every exact numeric duplicate group "
                "is indivisible and must remain wholly "
                "within one partition."
            ),
            "singleton_samples": (
                "Each nonduplicate recording receives "
                "its own singleton split_group_id."
            ),
            "cross_subject_duplicates": (
                "If present, LOSO handling must be "
                "decided explicitly after this audit; "
                "subject boundaries must not be silently "
                "overridden."
            ),
            "target_conflicts": (
                "If present, conflicting duplicate "
                "groups require explicit review before "
                "any split generation."
            ),
        },

        "scope": {
            "exact_byte_duplicates": True,
            "exact_parsed_numeric_duplicates": True,
            "approximate_near_duplicates": False,
            "raw_files_modified": False,
            "splits_created": False,
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
    # Documentation
    # ---------------------------------------------------------

    markdown = f"""# Stage 12 — Duplicate Audit

## Purpose

Detect exact duplicate recordings before any train/test folds are
created.

## Manifest

Recordings: {len(manifest):,}

Stage 4 manifest hash verified: yes

## Byte duplicates

Exact byte-level duplicate groups:

{len(byte_duplicate_hashes):,}

Recordings inside byte-level duplicate groups:

{byte_duplicate_member_count:,}

## Numeric duplicates

Exact parsed-numeric duplicate groups:

{len(numeric_duplicate_hashes):,}

Recordings inside numeric duplicate groups:

{numeric_duplicate_member_count:,}

Maximum numeric duplicate group size:

{maximum_numeric_group_size:,}

Numeric duplicate groups that are not entirely byte-identical:

{numeric_only_group_count:,}

## Cross-boundary risks

Cross-subject numeric duplicate groups:

{cross_subject_group_count:,}

Cross-sentence-ID numeric duplicate groups:

{cross_sentence_group_count:,}

Same-target but different-sentence-ID numeric duplicate groups:

{same_target_different_id_count:,}

Gloss-target conflict groups:

{gloss_conflict_group_count:,}

English-target conflict groups:

{english_conflict_group_count:,}

## Canonical numeric equality

Each CSV is parsed as a `T x 38` float64 matrix.

The numeric duplicate hash contains:

- matrix row count,
- matrix column count,
- canonical little-endian float64 values,
- negative zero normalized to positive zero.

No tolerance or approximate similarity is used.

Therefore Stage 12 detects exact equality after numerical parsing, not
near-duplicate motion sequences.

## Split rule

For known-signer folds, every exact numeric duplicate group is
indivisible.

All members of a duplicate group must be assigned to the same
partition.

Nonduplicate recordings receive their own singleton split group.

## LOSO rule

If a numeric duplicate group crosses subject boundaries, Stage 12 does
not silently resolve it.

LOSO handling must preserve the signer definition and explicitly deal
with any such leakage risk.

## Scope

No train/test split is created in Stage 12.

No raw recording is edited or removed.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console report
    # ---------------------------------------------------------

    print()
    print("NUMERIC DUPLICATES")
    print("-" * 78)

    print(
        "Duplicate groups:",
        len(
            numeric_duplicate_hashes
        ),
    )

    print(
        "Recordings in groups:",
        numeric_duplicate_member_count,
    )

    print(
        "Maximum group size:",
        maximum_numeric_group_size,
    )

    print(
        "Numeric-only groups:",
        numeric_only_group_count,
    )

    print()
    print("DUPLICATE RISK CLASSIFICATION")
    print("-" * 78)

    print(
        "Cross-subject groups:",
        cross_subject_group_count,
    )

    print(
        "Cross-sentence-ID groups:",
        cross_sentence_group_count,
    )

    print(
        "Same target / different ID groups:",
        same_target_different_id_count,
    )

    print(
        "Gloss target conflicts:",
        gloss_conflict_group_count,
    )

    print(
        "English target conflicts:",
        english_conflict_group_count,
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    for path in [
        BYTE_GROUPS_PATH,
        NUMERIC_GROUPS_PATH,
        MEMBERS_PATH,
        ASSIGNMENT_PATH,
        CONFLICTS_PATH,
        SUMMARY_PATH,
        DOC_PATH,
    ]:
        print(
            "Saved:",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    print()

    stage12_pass = (
        len(manifest) == 7400
        and actual_manifest_hash
        == recorded_manifest_hash
        and len(
            assignment_rows
        ) == 7400
        and len(
            {
                row[
                    "sample_id"
                ]
                for row
                in assignment_rows
            }
        ) == 7400
        and all(
            row[
                "numeric_group_size"
            ] >= 1
            for row
            in assignment_rows
        )
    )

    if stage12_pass:
        print(
            "FINAL STATUS: "
            "STAGE 12 BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 12 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

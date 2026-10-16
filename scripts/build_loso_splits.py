from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage12_split_group_assignment.csv"
)

STAGE12_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage12_duplicate_summary.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
)

LONG_SPLIT_PATH = (
    OUTPUT_DIR
    / "stage14_loso_assignment.csv"
)

SUMMARY_TABLE_PATH = (
    OUTPUT_DIR
    / "stage14_loso_summary.csv"
)

SUMMARY_JSON_PATH = (
    OUTPUT_DIR
    / "stage14_loso_summary.json"
)

HASH_PATH = (
    OUTPUT_DIR
    / "stage14_loso_assignment.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_14_LOSO_SPLIT_DESIGN.md"
)

SUBJECTS = [
    "subject1",
    "subject2",
    "subject3",
]


def load_csv(path: Path) -> list[dict]:
    with path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(
            csv.DictReader(handle)
        )


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
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


def main() -> None:
    print("=" * 78)
    print("STAGE 14 — LOSO SPLIT DESIGN")
    print("=" * 78)

    rows = load_csv(
        ASSIGNMENT_PATH
    )

    stage12 = load_json(
        STAGE12_SUMMARY_PATH
    )

    if len(rows) != 7400:
        raise RuntimeError(
            f"Expected 7,400 recordings, "
            f"found {len(rows):,}."
        )

    if (
        stage12["numeric_duplicates"]
        ["cross_subject_groups"]
        != 0
    ):
        raise RuntimeError(
            "Cannot construct clean LOSO because "
            "cross-subject numeric duplicates exist."
        )

    sample_ids = [
        row["sample_id"]
        for row in rows
    ]

    if len(set(sample_ids)) != 7400:
        raise RuntimeError(
            "Sample IDs are not unique."
        )

    observed_subjects = sorted(
        {
            row["subject"]
            for row in rows
        }
    )

    if observed_subjects != SUBJECTS:
        raise RuntimeError(
            f"Unexpected subjects: {observed_subjects}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    long_rows = []
    summary_rows = []

    for held_out_subject in SUBJECTS:

        loso_id = (
            "loso_"
            + held_out_subject
        )

        train_subjects = [
            subject
            for subject in SUBJECTS
            if subject
            != held_out_subject
        ]

        test_rows = [
            row
            for row in rows
            if row["subject"]
            == held_out_subject
        ]

        train_rows = [
            row
            for row in rows
            if row["subject"]
            != held_out_subject
        ]

        train_sample_ids = {
            row["sample_id"]
            for row in train_rows
        }

        test_sample_ids = {
            row["sample_id"]
            for row in test_rows
        }

        overlap = (
            train_sample_ids
            & test_sample_ids
        )

        if overlap:
            raise RuntimeError(
                f"Train/test sample overlap in "
                f"{loso_id}: "
                f"{sorted(overlap)[:10]}"
            )

        train_subject_set = sorted(
            {
                row["subject"]
                for row in train_rows
            }
        )

        test_subject_set = sorted(
            {
                row["subject"]
                for row in test_rows
            }
        )

        if train_subject_set != sorted(
            train_subjects
        ):
            raise RuntimeError(
                f"Unexpected train subjects "
                f"for {loso_id}."
            )

        if test_subject_set != [
            held_out_subject
        ]:
            raise RuntimeError(
                f"Unexpected test subject "
                f"for {loso_id}."
            )

        # -----------------------------------------------------
        # Strong split-group leakage check.
        # -----------------------------------------------------

        train_group_ids = {
            row["split_group_id"]
            for row in train_rows
        }

        test_group_ids = {
            row["split_group_id"]
            for row in test_rows
        }

        duplicate_group_overlap = (
            train_group_ids
            & test_group_ids
        )

        if duplicate_group_overlap:
            raise RuntimeError(
                f"Split-group overlap in {loso_id}: "
                f"{sorted(duplicate_group_overlap)[:10]}"
            )

        # -----------------------------------------------------
        # Expand to long-format frozen assignment.
        # -----------------------------------------------------

        for row in sorted(
            rows,
            key=lambda item: (
                item["subject"],
                int(
                    item[
                        "sentence_id"
                    ]
                ),
                int(
                    item[
                        "repetition_id"
                    ]
                ),
                item[
                    "sample_id"
                ],
            ),
        ):
            role = (
                "test"
                if row["subject"]
                == held_out_subject
                else "train"
            )

            long_rows.append(
                {
                    "loso_id": (
                        loso_id
                    ),
                    "held_out_subject": (
                        held_out_subject
                    ),
                    "sample_id": (
                        row["sample_id"]
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
                        row[
                            "split_group_id"
                        ]
                    ),
                    "is_numeric_duplicate": (
                        row[
                            "is_numeric_duplicate"
                        ]
                    ),
                    "role": (
                        role
                    ),
                }
            )

        summary_rows.append(
            {
                "loso_id": (
                    loso_id
                ),
                "held_out_subject": (
                    held_out_subject
                ),
                "training_subjects": (
                    "+".join(
                        train_subjects
                    )
                ),
                "train_recordings": (
                    len(train_rows)
                ),
                "test_recordings": (
                    len(test_rows)
                ),
                "train_unique_templates": (
                    len(
                        {
                            int(
                                row[
                                    "sentence_id"
                                ]
                            )
                            for row
                            in train_rows
                        }
                    )
                ),
                "test_unique_templates": (
                    len(
                        {
                            int(
                                row[
                                    "sentence_id"
                                ]
                            )
                            for row
                            in test_rows
                        }
                    )
                ),
                "train_subject_count": (
                    len(
                        train_subject_set
                    )
                ),
                "test_subject_count": (
                    len(
                        test_subject_set
                    )
                ),
                "sample_overlap": (
                    len(overlap)
                ),
                "split_group_overlap": (
                    len(
                        duplicate_group_overlap
                    )
                ),
            }
        )

    # ---------------------------------------------------------
    # Long assignment checks.
    # ---------------------------------------------------------

    if len(long_rows) != (
        len(rows)
        * len(SUBJECTS)
    ):
        raise RuntimeError(
            "Unexpected LOSO long-table size."
        )

    for held_out_subject in SUBJECTS:
        loso_id = (
            "loso_"
            + held_out_subject
        )

        run_rows = [
            row
            for row in long_rows
            if row["loso_id"]
            == loso_id
        ]

        if len(run_rows) != 7400:
            raise RuntimeError(
                f"{loso_id} does not contain "
                f"7,400 assignments."
            )

        run_samples = {
            row["sample_id"]
            for row in run_rows
        }

        if len(run_samples) != 7400:
            raise RuntimeError(
                f"{loso_id} does not contain "
                f"7,400 unique sample IDs."
            )

        bad_test = [
            row
            for row in run_rows
            if (
                row["role"]
                == "test"
                and row["subject"]
                != held_out_subject
            )
        ]

        bad_train = [
            row
            for row in run_rows
            if (
                row["role"]
                == "train"
                and row["subject"]
                == held_out_subject
            )
        ]

        if bad_test or bad_train:
            raise RuntimeError(
                f"Subject-role violation in "
                f"{loso_id}."
            )

    # ---------------------------------------------------------
    # Save frozen assignments.
    # ---------------------------------------------------------

    with LONG_SPLIT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "loso_id",
                "held_out_subject",
                "sample_id",
                "subject",
                "sentence_id",
                "repetition_id",
                "split_group_id",
                "is_numeric_duplicate",
                "role",
            ],
        )

        writer.writeheader()
        writer.writerows(
            long_rows
        )

    with SUMMARY_TABLE_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    assignment_hash = sha256_file(
        LONG_SPLIT_PATH
    )

    HASH_PATH.write_text(
        assignment_hash + "\n",
        encoding="utf-8",
    )

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 14,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "split_name": (
            "three_subject_loso"
        ),
        "recordings": 7400,
        "subjects": SUBJECTS,
        "loso_runs": (
            summary_rows
        ),
        "long_assignment_rows": (
            len(long_rows)
        ),
        "assignment_sha256": (
            assignment_hash
        ),
        "duplicate_safety": {
            "stage12_cross_subject_numeric_duplicate_groups": (
                stage12[
                    "numeric_duplicates"
                ][
                    "cross_subject_groups"
                ]
            ),
            "loso_split_group_overlap": {
                row["loso_id"]: (
                    row[
                        "split_group_overlap"
                    ]
                )
                for row in summary_rows
            },
        },
        "scientific_interpretation": (
            "Each LOSO run tests an entirely unseen signer. "
            "Template and token support are not assumed "
            "equivalent across runs."
        ),
        "support_analysis": (
            "Deferred to Stage 15. "
            "Stage 14 defines only signer-based train/test membership."
        ),
        "normalization_policy": (
            "No normalization statistics are calculated in Stage 14. "
            "All future preprocessing statistics must use only "
            "the training subjects of the corresponding LOSO run."
        ),
    }

    SUMMARY_JSON_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 14 — LOSO Split Design

## Purpose

Freeze the three signer-held-out train/test partitions.

Stage 14 defines signer membership only.

Support-aware evaluation subsets are defined later in Stage 15.

## LOSO runs

### LOSO Subject 1

Test subject:

`subject1`

Training subjects:

`subject2 + subject3`

Test recordings:

6,000

Training recordings:

1,400

### LOSO Subject 2

Test subject:

`subject2`

Training subjects:

`subject1 + subject3`

Test recordings:

1,000

Training recordings:

6,400

### LOSO Subject 3

Test subject:

`subject3`

Training subjects:

`subject1 + subject2`

Test recordings:

400

Training recordings:

7,000

## Duplicate safety

Stage 12 found zero cross-subject exact numeric duplicate groups.

Therefore signer boundaries do not force an exact numeric duplicate
across LOSO train and test.

## Interpretation

Every LOSO test partition contains a signer completely absent from
training.

However, the three runs do not have equivalent template or vocabulary
support.

That distinction is formalised in Stage 15.

## Preprocessing rule

No normalization, filtering statistics or model-selection decisions are
calculated here.

Future LOSO preprocessing must be fit only on the corresponding
training subjects.

## Frozen assignment

Assignment SHA-256:

`{assignment_hash}`
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("LOSO RUNS")
    print("-" * 78)

    for row in summary_rows:
        print(
            f"{row['loso_id']}: "
            f"held_out={row['held_out_subject']}, "
            f"train_subjects={row['training_subjects']}, "
            f"train={row['train_recordings']}, "
            f"test={row['test_recordings']}, "
            f"train_templates={row['train_unique_templates']}, "
            f"test_templates={row['test_unique_templates']}"
        )

    print()
    print("LEAKAGE CHECKS")
    print("-" * 78)

    for row in summary_rows:
        print(
            f"{row['loso_id']}: "
            f"sample_overlap={row['sample_overlap']}, "
            f"split_group_overlap={row['split_group_overlap']}"
        )

    print()
    print("LONG ASSIGNMENT")
    print("-" * 78)

    print(
        "Rows:",
        len(long_rows),
    )

    print(
        "Expected:",
        7400 * 3,
    )

    print()
    print("FROZEN HASH")
    print("-" * 78)

    print(
        assignment_hash
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    for path in [
        LONG_SPLIT_PATH,
        SUMMARY_TABLE_PATH,
        SUMMARY_JSON_PATH,
        HASH_PATH,
        DOC_PATH,
    ]:
        print(
            "Saved:",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    stage14_pass = (
        len(long_rows) == 22200
        and len(summary_rows) == 3
        and all(
            row[
                "sample_overlap"
            ] == 0
            for row
            in summary_rows
        )
        and all(
            row[
                "split_group_overlap"
            ] == 0
            for row
            in summary_rows
        )
        and stage12[
            "numeric_duplicates"
        ][
            "cross_subject_groups"
        ] == 0
    )

    print()

    if stage14_pass:
        print(
            "FINAL STATUS: "
            "STAGE 14 BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 14 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

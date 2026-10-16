from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
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

FOLD_ASSIGNMENT_PATH = (
    OUTPUT_DIR
    / "stage13_known_signer_5fold.csv"
)

FOLD_SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage13_fold_summary.csv"
)

STRATUM_SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage13_stratum_balance.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage13_known_signer_5fold_summary.json"
)

HASH_PATH = (
    OUTPUT_DIR
    / "stage13_known_signer_5fold.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_13_KNOWN_SIGNER_5FOLD.md"
)

N_FOLDS = 5
SPLIT_SEED = 20260829


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


def stable_integer(text: str) -> int:
    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def main() -> None:
    print("=" * 78)
    print("STAGE 13 — KNOWN-SIGNER FIVE-FOLD SPLIT")
    print("=" * 78)

    rows = load_csv(
        ASSIGNMENT_PATH
    )

    stage12 = load_json(
        STAGE12_SUMMARY_PATH
    )

    if len(rows) != 7400:
        raise RuntimeError(
            f"Expected 7,400 Stage-12 assignments, "
            f"found {len(rows):,}."
        )

    if (
        stage12["numeric_duplicates"]
        ["cross_subject_groups"]
        != 0
    ):
        raise RuntimeError(
            "Cross-subject duplicate groups exist."
        )

    if (
        stage12["numeric_duplicates"]
        ["cross_sentence_id_groups"]
        != 0
    ):
        raise RuntimeError(
            "Cross-sentence duplicate groups exist."
        )

    if (
        stage12["target_conflicts"]
        ["gloss_conflict_groups"]
        != 0
    ):
        raise RuntimeError(
            "Gloss-target duplicate conflicts exist."
        )

    if (
        stage12["target_conflicts"]
        ["english_conflict_groups"]
        != 0
    ):
        raise RuntimeError(
            "English-target duplicate conflicts exist."
        )

    sample_ids = [
        row["sample_id"]
        for row in rows
    ]

    if len(set(sample_ids)) != 7400:
        raise RuntimeError(
            "Sample IDs are not unique."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Build atomic split groups.
    #
    # Every split_group_id must belong to exactly one
    # subject + sentence stratum.
    # ---------------------------------------------------------

    group_members = defaultdict(
        list
    )

    for row in rows:
        group_members[
            row["split_group_id"]
        ].append(
            row
        )

    if len(group_members) != 7397:
        raise RuntimeError(
            f"Expected 7,397 Stage-12 split groups, "
            f"found {len(group_members):,}."
        )

    group_records = []

    for split_group_id, members in (
        group_members.items()
    ):
        subjects = {
            row["subject"]
            for row in members
        }

        sentence_ids = {
            int(row["sentence_id"])
            for row in members
        }

        if len(subjects) != 1:
            raise RuntimeError(
                f"Split group {split_group_id} "
                f"crosses subjects."
            )

        if len(sentence_ids) != 1:
            raise RuntimeError(
                f"Split group {split_group_id} "
                f"crosses sentence IDs."
            )

        subject = next(
            iter(subjects)
        )

        sentence_id = next(
            iter(sentence_ids)
        )

        group_records.append(
            {
                "split_group_id": (
                    split_group_id
                ),
                "subject": (
                    subject
                ),
                "sentence_id": (
                    sentence_id
                ),
                "size": (
                    len(members)
                ),
                "members": (
                    members
                ),
            }
        )

    # ---------------------------------------------------------
    # Stratify by subject + sentence.
    # ---------------------------------------------------------

    strata = defaultdict(
        list
    )

    for group in group_records:
        key = (
            group["subject"],
            group["sentence_id"],
        )

        strata[key].append(
            group
        )

    if len(strata) != 368:
        raise RuntimeError(
            f"Expected 368 supported "
            f"subject-template strata, "
            f"found {len(strata)}."
        )

    too_small = [
        {
            "subject": subject,
            "sentence_id": sentence_id,
            "split_groups": len(groups),
            "recordings": sum(
                group["size"]
                for group in groups
            ),
        }
        for (
            subject,
            sentence_id
        ), groups in strata.items()
        if len(groups) < N_FOLDS
    ]

    if too_small:
        raise RuntimeError(
            "At least one subject-template stratum "
            "contains fewer than five atomic groups: "
            f"{too_small[:10]}"
        )

    # ---------------------------------------------------------
    # Deterministic grouped balancing.
    #
    # Larger groups first. Ties are resolved with a
    # stable SHA-256 value based on the frozen seed.
    #
    # For each stratum, assign the next group to the
    # currently smallest fold by recording count.
    # ---------------------------------------------------------

    fold_by_group = {}

    stratum_summary_rows = []

    for (
        subject,
        sentence_id
    ) in sorted(
        strata.keys(),
        key=lambda item: (
            item[0],
            item[1],
        ),
    ):
        groups = strata[
            (
                subject,
                sentence_id,
            )
        ]

        ordered_groups = sorted(
            groups,
            key=lambda group: (
                -group["size"],
                stable_integer(
                    f"{SPLIT_SEED}|"
                    f"{subject}|"
                    f"{sentence_id}|"
                    f"{group['split_group_id']}"
                ),
            ),
        )

        fold_sample_counts = {
            fold: 0
            for fold in range(
                1,
                N_FOLDS + 1,
            )
        }

        fold_group_counts = {
            fold: 0
            for fold in range(
                1,
                N_FOLDS + 1,
            )
        }

        for group in ordered_groups:
            best_fold = min(
                range(
                    1,
                    N_FOLDS + 1,
                ),
                key=lambda fold: (
                    fold_sample_counts[
                        fold
                    ],
                    fold_group_counts[
                        fold
                    ],
                    stable_integer(
                        f"{SPLIT_SEED}|"
                        f"{subject}|"
                        f"{sentence_id}|"
                        f"{group['split_group_id']}|"
                        f"fold={fold}"
                    ),
                ),
            )

            fold_by_group[
                group[
                    "split_group_id"
                ]
            ] = best_fold

            fold_sample_counts[
                best_fold
            ] += group["size"]

            fold_group_counts[
                best_fold
            ] += 1

        sample_counts = [
            fold_sample_counts[
                fold
            ]
            for fold in range(
                1,
                N_FOLDS + 1,
            )
        ]

        group_counts = [
            fold_group_counts[
                fold
            ]
            for fold in range(
                1,
                N_FOLDS + 1,
            )
        ]

        stratum_summary_rows.append(
            {
                "subject": (
                    subject
                ),
                "sentence_id": (
                    sentence_id
                ),
                "recordings": (
                    sum(sample_counts)
                ),
                "atomic_split_groups": (
                    sum(group_counts)
                ),
                "fold1_recordings": (
                    sample_counts[0]
                ),
                "fold2_recordings": (
                    sample_counts[1]
                ),
                "fold3_recordings": (
                    sample_counts[2]
                ),
                "fold4_recordings": (
                    sample_counts[3]
                ),
                "fold5_recordings": (
                    sample_counts[4]
                ),
                "minimum_fold_recordings": (
                    min(sample_counts)
                ),
                "maximum_fold_recordings": (
                    max(sample_counts)
                ),
                "recording_imbalance": (
                    max(sample_counts)
                    - min(sample_counts)
                ),
                "represented_in_all_folds": (
                    all(
                        count > 0
                        for count
                        in sample_counts
                    )
                ),
            }
        )

    if len(fold_by_group) != len(
        group_members
    ):
        raise RuntimeError(
            "Not every split group received a fold."
        )

    # ---------------------------------------------------------
    # Expand group fold assignment back to recordings.
    # ---------------------------------------------------------

    output_rows = []

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
            item["sample_id"],
        ),
    ):
        fold = fold_by_group[
            row[
                "split_group_id"
            ]
        ]

        output_rows.append(
            {
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
                "outer_fold": (
                    fold
                ),
            }
        )

    # ---------------------------------------------------------
    # Strong duplicate leakage check.
    # ---------------------------------------------------------

    folds_per_group = defaultdict(
        set
    )

    for row in output_rows:
        folds_per_group[
            row["split_group_id"]
        ].add(
            int(
                row["outer_fold"]
            )
        )

    split_duplicate_groups = [
        split_group_id
        for (
            split_group_id,
            fold_set
        ) in folds_per_group.items()
        if len(fold_set) != 1
    ]

    if split_duplicate_groups:
        raise RuntimeError(
            "Atomic duplicate groups were split: "
            f"{split_duplicate_groups[:10]}"
        )

    # ---------------------------------------------------------
    # Verify each subject-template occurs in all five folds.
    # ---------------------------------------------------------

    folds_per_stratum = defaultdict(
        set
    )

    for row in output_rows:
        key = (
            row["subject"],
            int(
                row[
                    "sentence_id"
                ]
            ),
        )

        folds_per_stratum[
            key
        ].add(
            int(
                row[
                    "outer_fold"
                ]
            )
        )

    incomplete_strata = [
        {
            "subject": key[0],
            "sentence_id": key[1],
            "folds": sorted(
                fold_set
            ),
        }
        for key, fold_set
        in folds_per_stratum.items()
        if fold_set
        != set(
            range(
                1,
                N_FOLDS + 1,
            )
        )
    ]

    if incomplete_strata:
        raise RuntimeError(
            "Some subject-template strata do not "
            "appear in all five folds: "
            f"{incomplete_strata[:10]}"
        )

    # ---------------------------------------------------------
    # Save assignment.
    # ---------------------------------------------------------

    with FOLD_ASSIGNMENT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "subject",
                "sentence_id",
                "repetition_id",
                "split_group_id",
                "is_numeric_duplicate",
                "outer_fold",
            ],
        )

        writer.writeheader()
        writer.writerows(
            output_rows
        )

    # ---------------------------------------------------------
    # Per-fold summary.
    # ---------------------------------------------------------

    fold_summary_rows = []

    for fold in range(
        1,
        N_FOLDS + 1,
    ):
        test_rows = [
            row
            for row in output_rows
            if int(
                row["outer_fold"]
            ) == fold
        ]

        train_rows = [
            row
            for row in output_rows
            if int(
                row["outer_fold"]
            ) != fold
        ]

        subject_counts = {
            subject: sum(
                row["subject"]
                == subject
                for row in test_rows
            )
            for subject in [
                "subject1",
                "subject2",
                "subject3",
            ]
        }

        fold_summary_rows.append(
            {
                "outer_fold": (
                    fold
                ),
                "test_recordings": (
                    len(test_rows)
                ),
                "train_recordings": (
                    len(train_rows)
                ),
                "test_subject1": (
                    subject_counts[
                        "subject1"
                    ]
                ),
                "test_subject2": (
                    subject_counts[
                        "subject2"
                    ]
                ),
                "test_subject3": (
                    subject_counts[
                        "subject3"
                    ]
                ),
                "test_unique_templates_global": (
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
                "test_subject_template_strata": (
                    len(
                        {
                            (
                                row[
                                    "subject"
                                ],
                                int(
                                    row[
                                        "sentence_id"
                                    ]
                                ),
                            )
                            for row
                            in test_rows
                        }
                    )
                ),
            }
        )

    with FOLD_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                fold_summary_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            fold_summary_rows
        )

    with STRATUM_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                stratum_summary_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            stratum_summary_rows
        )

    # ---------------------------------------------------------
    # Frozen hash.
    # ---------------------------------------------------------

    fold_hash = sha256_file(
        FOLD_ASSIGNMENT_PATH
    )

    HASH_PATH.write_text(
        fold_hash + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Aggregate validation.
    # ---------------------------------------------------------

    fold_sizes = [
        row[
            "test_recordings"
        ]
        for row
        in fold_summary_rows
    ]

    subject_fold_counts = {}

    for subject in [
        "subject1",
        "subject2",
        "subject3",
    ]:
        key = (
            "test_"
            + subject
        )

        subject_fold_counts[
            subject
        ] = [
            row[key]
            for row
            in fold_summary_rows
        ]

    max_stratum_imbalance = max(
        row[
            "recording_imbalance"
        ]
        for row
        in stratum_summary_rows
    )

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 13,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "split_name": (
            "known_signer_5fold"
        ),
        "n_folds": (
            N_FOLDS
        ),
        "seed": (
            SPLIT_SEED
        ),
        "recordings": (
            len(output_rows)
        ),
        "atomic_split_groups": (
            len(group_members)
        ),
        "subject_template_strata": (
            len(strata)
        ),
        "stratification": (
            "subject_plus_sentence_id"
        ),
        "atomic_unit": (
            "stage12_split_group_id"
        ),
        "fold_test_sizes": (
            fold_sizes
        ),
        "fold_subject_test_sizes": (
            subject_fold_counts
        ),
        "all_samples_assigned_once": (
            len(output_rows)
            == len(
                {
                    row[
                        "sample_id"
                    ]
                    for row
                    in output_rows
                }
            )
        ),
        "duplicate_groups_split_across_folds": (
            len(
                split_duplicate_groups
            )
        ),
        "strata_missing_any_fold": (
            len(
                incomplete_strata
            )
        ),
        "maximum_within_stratum_recording_imbalance": (
            max_stratum_imbalance
        ),
        "assignment_sha256": (
            fold_hash
        ),
        "interpretation": (
            "Known-signer repetition generalisation. "
            "The same subjects and supported sentence "
            "templates are intentionally represented "
            "across outer folds."
        ),
        "not_claimed": [
            "unseen signer generalisation",
            "unseen complete-template generalisation",
            "open-vocabulary ASL recognition",
        ],
        "outer_test_policy": (
            "For outer fold k, test is outer_fold == k "
            "and outer training pool is outer_fold != k."
        ),
        "validation_policy": (
            "No validation split is defined in Stage 13. "
            "Inner selection is created later without "
            "using the outer test fold."
        ),
    }

    SUMMARY_PATH.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Human-readable record.
    # ---------------------------------------------------------

    markdown = f"""# Stage 13 — Known-Signer Five-Fold Split

## Purpose

Create the frozen outer five-fold partition for known-signer
repetition generalisation.

## Dataset

Recordings: {len(output_rows):,}

Atomic split groups: {len(group_members):,}

Subject-template strata: {len(strata):,}

## Stratification

The split is stratified by:

`subject + sentence_id`

The atomic assignment unit is:

`stage12_split_group_id`

Therefore exact duplicate recordings identified in Stage 12 cannot be
separated between outer train and outer test.

## Outer-fold interpretation

For fold `k`:

- test = recordings with `outer_fold == k`
- outer training pool = recordings with `outer_fold != k`

No validation partition is created here.

Inner model-selection partitions are created later and may use only the
outer training pool.

## Scientific meaning

This is a known-signer experiment.

The same signer identities and supported sentence/template identities
are intentionally represented across folds.

The experiment measures recognition of held-out repetitions.

It must not be described as unseen-signer or unseen-template
generalisation.

## Determinism

Frozen split seed:

`{SPLIT_SEED}`

Assignment SHA-256:

`{fold_hash}`

## Leakage safeguards

Exact numeric duplicate groups split across folds:

{len(split_duplicate_groups)}

Subject-template strata missing a fold:

{len(incomplete_strata)}

## Deferred decisions

- inner validation/model selection
- preprocessing statistics
- CTC feasibility
- model training

These are handled in later stages.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console report.
    # ---------------------------------------------------------

    print()
    print("SPLIT CONTRACT")
    print("-" * 78)

    print(
        "Recordings:",
        len(output_rows),
    )

    print(
        "Atomic split groups:",
        len(group_members),
    )

    print(
        "Subject-template strata:",
        len(strata),
    )

    print(
        "Outer folds:",
        N_FOLDS,
    )

    print(
        "Frozen seed:",
        SPLIT_SEED,
    )

    print()
    print("FOLD SIZES")
    print("-" * 78)

    for row in (
        fold_summary_rows
    ):
        print(
            f"Fold {row['outer_fold']}: "
            f"test={row['test_recordings']}, "
            f"train={row['train_recordings']}, "
            f"S1={row['test_subject1']}, "
            f"S2={row['test_subject2']}, "
            f"S3={row['test_subject3']}"
        )

    print()
    print("LEAKAGE / COVERAGE CHECKS")
    print("-" * 78)

    print(
        "Duplicate groups split:",
        len(
            split_duplicate_groups
        ),
    )

    print(
        "Strata missing any fold:",
        len(
            incomplete_strata
        ),
    )

    print(
        "Maximum within-stratum imbalance:",
        max_stratum_imbalance,
    )

    print()
    print("FROZEN HASH")
    print("-" * 78)

    print(
        fold_hash
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    for path in [
        FOLD_ASSIGNMENT_PATH,
        FOLD_SUMMARY_PATH,
        STRATUM_SUMMARY_PATH,
        SUMMARY_PATH,
        HASH_PATH,
        DOC_PATH,
    ]:
        print(
            "Saved:",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    stage13_pass = (
        len(output_rows) == 7400
        and len(group_members) == 7397
        and len(strata) == 368
        and len(
            split_duplicate_groups
        ) == 0
        and len(
            incomplete_strata
        ) == 0
        and all(
            row[
                "represented_in_all_folds"
            ]
            for row
            in stratum_summary_rows
        )
        and len(
            {
                row[
                    "sample_id"
                ]
                for row
                in output_rows
            }
        ) == 7400
    )

    print()

    if stage13_pass:
        print(
            "FINAL STATUS: "
            "STAGE 13 BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 13 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

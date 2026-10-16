from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGETS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

DUPLICATE_GROUPS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_duplicate_label_groups.json"
)

RECORDING_ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage12_split_group_assignment.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
)

GROUPS_PATH = (
    OUTPUT_DIR
    / "stage16_template_holdout_candidate_groups.csv"
)

TOKEN_FREQUENCY_PATH = (
    OUTPUT_DIR
    / "stage16_token_template_group_frequency.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage16_template_holdout_feasibility_summary.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_16A_TEMPLATE_HOLDOUT_FEASIBILITY.md"
)


SUBJECTS = [
    "subject1",
    "subject2",
    "subject3",
]

SUBJECT_SHORT = {
    "subject1": "S1",
    "subject2": "S2",
    "subject3": "S3",
}


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


def main() -> None:
    print("=" * 78)
    print("STAGE 16A — TEMPLATE-HOLDOUT FEASIBILITY AUDIT")
    print("=" * 78)

    targets = load_csv(
        TARGETS_PATH
    )

    recordings = load_csv(
        RECORDING_ASSIGNMENT_PATH
    )

    duplicate_contract = load_json(
        DUPLICATE_GROUPS_PATH
    )

    if len(targets) != 300:
        raise RuntimeError(
            f"Expected 300 sentence targets, "
            f"found {len(targets)}."
        )

    if len(recordings) != 7400:
        raise RuntimeError(
            f"Expected 7,400 recordings, "
            f"found {len(recordings)}."
        )

    target_by_id = {
        int(row["sentence_id"]): row
        for row in targets
    }

    if sorted(target_by_id) != list(
        range(1, 301)
    ):
        raise RuntimeError(
            "Sentence IDs are not exactly 1..300."
        )

    # ---------------------------------------------------------
    # Build label-equivalence groups.
    #
    # Exact duplicate target IDs must always stay together.
    # ---------------------------------------------------------

    explicit_groups = []

    used_ids = set()

    for source_group in duplicate_contract[
        "groups"
    ]:
        ids = sorted(
            int(x)
            for x in source_group[
                "sentence_ids"
            ]
        )

        if not ids:
            raise RuntimeError(
                "Empty duplicate-label group."
            )

        overlap = (
            used_ids
            & set(ids)
        )

        if overlap:
            raise RuntimeError(
                f"Duplicate equivalence IDs overlap: "
                f"{sorted(overlap)}"
            )

        gloss_values = {
            target_by_id[
                sentence_id
            ]["gloss_raw"]
            for sentence_id in ids
        }

        english_values = {
            target_by_id[
                sentence_id
            ]["english_raw"]
            for sentence_id in ids
        }

        if len(gloss_values) != 1:
            raise RuntimeError(
                f"Duplicate-label group {ids} "
                "does not share one gloss target."
            )

        if len(english_values) != 1:
            raise RuntimeError(
                f"Duplicate-label group {ids} "
                "does not share one English target."
            )

        explicit_groups.append(
            ids
        )

        used_ids.update(
            ids
        )

    raw_groups = list(
        explicit_groups
    )

    for sentence_id in range(
        1,
        301,
    ):
        if sentence_id not in used_ids:
            raw_groups.append(
                [sentence_id]
            )

    raw_groups = sorted(
        raw_groups,
        key=lambda ids: (
            min(ids),
            len(ids),
            ids,
        ),
    )

    if len(raw_groups) != 299:
        raise RuntimeError(
            f"Expected 299 template-equivalence groups, "
            f"found {len(raw_groups)}."
        )

    flattened_ids = [
        sentence_id
        for ids in raw_groups
        for sentence_id in ids
    ]

    if sorted(flattened_ids) != list(
        range(1, 301)
    ):
        raise RuntimeError(
            "Template-equivalence groups do not "
            "partition IDs 1..300 exactly once."
        )

    group_records = []

    sentence_to_group = {}

    for index, ids in enumerate(
        raw_groups,
        start=1,
    ):
        group_id = (
            f"tplgrp_{index:03d}"
        )

        for sentence_id in ids:
            sentence_to_group[
                sentence_id
            ] = group_id

        gloss_values = sorted(
            {
                target_by_id[
                    sentence_id
                ]["gloss_raw"]
                for sentence_id in ids
            }
        )

        english_values = sorted(
            {
                target_by_id[
                    sentence_id
                ]["english_raw"]
                for sentence_id in ids
            }
        )

        tokens = set()

        for sentence_id in ids:
            tokens.update(
                json.loads(
                    target_by_id[
                        sentence_id
                    ][
                        "gloss_tokens_json"
                    ]
                )
            )

        group_records.append(
            {
                "template_group_id": (
                    group_id
                ),
                "sentence_ids": (
                    ids
                ),
                "sentence_id_count": (
                    len(ids)
                ),
                "gloss_values": (
                    gloss_values
                ),
                "english_values": (
                    english_values
                ),
                "tokens": (
                    sorted(tokens)
                ),
            }
        )

    # ---------------------------------------------------------
    # Recording counts by equivalence group and subject.
    # ---------------------------------------------------------

    group_subject_recordings = {
        group[
            "template_group_id"
        ]: Counter()
        for group in group_records
    }

    for row in recordings:
        sentence_id = int(
            row["sentence_id"]
        )

        group_id = sentence_to_group[
            sentence_id
        ]

        subject = row[
            "subject"
        ]

        if subject not in SUBJECTS:
            raise RuntimeError(
                f"Unexpected subject: {subject}"
            )

        group_subject_recordings[
            group_id
        ][subject] += 1

    # ---------------------------------------------------------
    # Count in how many EQUIVALENCE GROUPS each token appears.
    #
    # This matters because IDs 39 and 83 cannot support each
    # other when that equivalence group is held out.
    # ---------------------------------------------------------

    token_group_ids = defaultdict(
        set
    )

    token_sentence_ids = defaultdict(
        set
    )

    for group in group_records:
        group_id = group[
            "template_group_id"
        ]

        for token in group[
            "tokens"
        ]:
            token_group_ids[
                token
            ].add(
                group_id
            )

        for sentence_id in group[
            "sentence_ids"
        ]:
            tokens = json.loads(
                target_by_id[
                    sentence_id
                ][
                    "gloss_tokens_json"
                ]
            )

            for token in set(
                tokens
            ):
                token_sentence_ids[
                    token
                ].add(
                    sentence_id
                )

    global_tokens = sorted(
        token_group_ids
    )

    if len(global_tokens) != 455:
        raise RuntimeError(
            f"Expected 455 gloss tokens, "
            f"found {len(global_tokens)}."
        )

    token_frequency_rows = []

    for token in global_tokens:
        group_ids = sorted(
            token_group_ids[
                token
            ]
        )

        sentence_ids = sorted(
            token_sentence_ids[
                token
            ]
        )

        token_frequency_rows.append(
            {
                "token": token,
                "template_group_frequency": (
                    len(group_ids)
                ),
                "sentence_id_frequency": (
                    len(sentence_ids)
                ),
                "template_group_ids_json": (
                    json.dumps(
                        group_ids
                    )
                ),
                "sentence_ids_json": (
                    json.dumps(
                        sentence_ids
                    )
                ),
                "globally_unique_to_one_template_group": (
                    len(group_ids) == 1
                ),
            }
        )

    # ---------------------------------------------------------
    # Determine whether each group is individually holdout-safe.
    #
    # This means: if ONLY this equivalence group is removed,
    # every target token still exists in another group.
    #
    # Multi-group joint safety is checked later in Stage 16B.
    # ---------------------------------------------------------

    candidate_rows = []

    for group in group_records:
        group_id = group[
            "template_group_id"
        ]

        tokens = group[
            "tokens"
        ]

        token_frequencies = [
            len(
                token_group_ids[
                    token
                ]
            )
            for token in tokens
        ]

        unique_tokens = sorted(
            token
            for token in tokens
            if len(
                token_group_ids[
                    token
                ]
            ) == 1
        )

        individually_safe = (
            len(unique_tokens) == 0
        )

        counts = (
            group_subject_recordings[
                group_id
            ]
        )

        supported_subjects = [
            subject
            for subject in SUBJECTS
            if counts[
                subject
            ] > 0
        ]

        support_signature = (
            "+".join(
                SUBJECT_SHORT[
                    subject
                ]
                for subject
                in supported_subjects
            )
        )

        candidate_rows.append(
            {
                "template_group_id": (
                    group_id
                ),
                "sentence_ids_json": (
                    json.dumps(
                        group[
                            "sentence_ids"
                        ]
                    )
                ),
                "sentence_id_count": (
                    group[
                        "sentence_id_count"
                    ]
                ),
                "gloss_targets_json": (
                    json.dumps(
                        group[
                            "gloss_values"
                        ],
                        ensure_ascii=False,
                    )
                ),
                "english_targets_json": (
                    json.dumps(
                        group[
                            "english_values"
                        ],
                        ensure_ascii=False,
                    )
                ),
                "unique_gloss_token_count": (
                    len(tokens)
                ),
                "minimum_token_template_group_frequency": (
                    min(
                        token_frequencies
                    )
                    if token_frequencies
                    else 0
                ),
                "globally_unique_token_count": (
                    len(
                        unique_tokens
                    )
                ),
                "globally_unique_tokens_json": (
                    json.dumps(
                        unique_tokens,
                        ensure_ascii=False,
                    )
                ),
                "individually_holdout_safe": (
                    individually_safe
                ),
                "subject_support_signature": (
                    support_signature
                ),
                "supported_subject_count": (
                    len(
                        supported_subjects
                    )
                ),
                "subject1_recordings": (
                    counts[
                        "subject1"
                    ]
                ),
                "subject2_recordings": (
                    counts[
                        "subject2"
                    ]
                ),
                "subject3_recordings": (
                    counts[
                        "subject3"
                    ]
                ),
                "total_recordings": (
                    sum(
                        counts.values()
                    )
                ),
            }
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with GROUPS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                candidate_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            candidate_rows
        )

    with TOKEN_FREQUENCY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                token_frequency_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            token_frequency_rows
        )

    # ---------------------------------------------------------
    # Summary.
    # ---------------------------------------------------------

    safe_groups = [
        row
        for row in candidate_rows
        if row[
            "individually_holdout_safe"
        ]
    ]

    unsafe_groups = [
        row
        for row in candidate_rows
        if not row[
            "individually_holdout_safe"
        ]
    ]

    unique_tokens = [
        row
        for row in token_frequency_rows
        if row[
            "globally_unique_to_one_template_group"
        ]
    ]

    signature_counts = defaultdict(
        lambda: {
            "groups": 0,
            "safe_groups": 0,
            "recordings": 0,
            "safe_group_recordings": 0,
        }
    )

    for row in candidate_rows:
        signature = row[
            "subject_support_signature"
        ]

        item = signature_counts[
            signature
        ]

        item[
            "groups"
        ] += 1

        item[
            "recordings"
        ] += int(
            row[
                "total_recordings"
            ]
        )

        if row[
            "individually_holdout_safe"
        ]:
            item[
                "safe_groups"
            ] += 1

            item[
                "safe_group_recordings"
            ] += int(
                row[
                    "total_recordings"
                ]
            )

    duplicate_equivalence_rows = [
        row
        for row in candidate_rows
        if int(
            row[
                "sentence_id_count"
            ]
        ) > 1
    ]

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": "16A",
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "sentence_ids": 300,
        "template_equivalence_groups": (
            len(
                candidate_rows
            )
        ),
        "multi_id_equivalence_groups": (
            len(
                duplicate_equivalence_rows
            )
        ),
        "global_gloss_tokens": (
            len(
                global_tokens
            )
        ),
        "globally_unique_token_count": (
            len(
                unique_tokens
            )
        ),
        "individually_holdout_safe_groups": (
            len(
                safe_groups
            )
        ),
        "individually_holdout_unsafe_groups": (
            len(
                unsafe_groups
            )
        ),
        "subject_support_signatures": (
            dict(
                sorted(
                    signature_counts.items()
                )
            )
        ),
        "duplicate_label_rule": (
            "Sentence IDs 39 and 83 are one "
            "indivisible template-equivalence group."
        ),
        "individual_safety_definition": (
            "A template-equivalence group is individually "
            "holdout-safe only if every target gloss token "
            "occurs in at least one other template-equivalence "
            "group."
        ),
        "important_limitation": (
            "Individual safety does not guarantee that "
            "multiple safe groups may all be held out "
            "simultaneously. Final Stage 16B selection must "
            "recompute token support jointly after removing "
            "the entire proposed test set."
        ),
        "split_created": False,
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

    markdown = f"""# Stage 16A — Template-Holdout Feasibility Audit

## Purpose

Determine which complete sentence/template groups are structurally
eligible for a vocabulary-supported template-holdout experiment.

No train/test split is created in Stage 16A.

## Label-equivalence groups

Sentence IDs:

300

Effective template-equivalence groups:

{len(candidate_rows)}

Sentence IDs 39 and 83 form one indivisible group because their gloss
and English targets are identical.

## Vocabulary

Global gloss tokens:

{len(global_tokens)}

Tokens occurring in only one template-equivalence group:

{len(unique_tokens)}

## Individual holdout feasibility

Individually safe groups:

{len(safe_groups)}

Individually unsafe groups:

{len(unsafe_groups)}

A group is individually safe only when every one of its target gloss
tokens occurs in at least one other template-equivalence group.

## Important limitation

Individual safety is not sufficient for selecting multiple test
templates simultaneously.

If two candidate groups are the only remaining support for one
another's token, holding both out together would remove that token from
training.

Therefore Stage 16B must verify vocabulary support after removing the
entire proposed held-out group set.

## Scientific target

The final template-holdout experiment will test unseen complete
sentence/templates while retaining training support for every target
gloss token.

It is not an open-vocabulary experiment.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("EQUIVALENCE CONTRACT")
    print("-" * 78)

    print(
        "Sentence IDs:",
        300,
    )

    print(
        "Template-equivalence groups:",
        len(
            candidate_rows
        ),
    )

    print(
        "Multi-ID groups:",
        len(
            duplicate_equivalence_rows
        ),
    )

    print()
    print("TOKEN SUPPORT")
    print("-" * 78)

    print(
        "Global gloss tokens:",
        len(
            global_tokens
        ),
    )

    print(
        "Tokens unique to one template group:",
        len(
            unique_tokens
        ),
    )

    print()
    print("INDIVIDUAL HOLDOUT FEASIBILITY")
    print("-" * 78)

    print(
        "Safe groups:",
        len(
            safe_groups
        ),
    )

    print(
        "Unsafe groups:",
        len(
            unsafe_groups
        ),
    )

    print()
    print("SUBJECT-SUPPORT SIGNATURES")
    print("-" * 78)

    for signature in sorted(
        signature_counts
    ):
        item = signature_counts[
            signature
        ]

        print(
            f"{signature}: "
            f"groups={item['groups']}, "
            f"safe={item['safe_groups']}, "
            f"recordings={item['recordings']}, "
            f"safe_recordings="
            f"{item['safe_group_recordings']}"
        )

    print()
    print("DUPLICATE-LABEL EQUIVALENCE")
    print("-" * 78)

    for row in (
        duplicate_equivalence_rows
    ):
        print(
            row[
                "template_group_id"
            ],
            row[
                "sentence_ids_json"
            ],
            row[
                "gloss_targets_json"
            ],
        )

    print()
    print("OUTPUTS")
    print("-" * 78)

    for path in [
        GROUPS_PATH,
        TOKEN_FREQUENCY_PATH,
        SUMMARY_PATH,
        DOC_PATH,
    ]:
        print(
            "Saved:",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    stage16a_pass = (
        len(candidate_rows) == 299
        and len(flattened_ids) == 300
        and len(
            set(
                flattened_ids
            )
        ) == 300
        and len(global_tokens) == 455
        and len(
            duplicate_equivalence_rows
        ) == 1
        and sum(
            int(
                row[
                    "total_recordings"
                ]
            )
            for row
            in candidate_rows
        ) == 7400
    )

    print()

    if stage16a_pass:
        print(
            "FINAL STATUS: "
            "STAGE 16A FEASIBILITY PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 16A FEASIBILITY FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

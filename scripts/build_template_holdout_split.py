from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CANDIDATES_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage16_template_holdout_candidate_groups.csv"
)

TARGETS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

RECORDINGS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage12_split_group_assignment.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
)

SELECTED_PATH = (
    OUTPUT_DIR
    / "stage16_template_holdout_selected_groups.csv"
)

ASSIGNMENT_PATH = (
    OUTPUT_DIR
    / "stage16_template_holdout_assignment.csv"
)

SUBJECT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage16_template_holdout_subject_summary.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage16_template_holdout_summary.json"
)

HASH_PATH = (
    OUTPUT_DIR
    / "stage16_template_holdout_assignment.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_16_TEMPLATE_HOLDOUT_SPLIT.md"
)


TARGET_TEST_GROUPS = 21

SELECTION_SEED = 20260830

MAX_SELECTION_ATTEMPTS = 5000

SIGNATURE_QUOTAS = {
    "S1": 15,
    "S1+S2": 4,
    "S1+S3": 2,
}

SUBJECTS = [
    "subject1",
    "subject2",
    "subject3",
]


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


def sha256_file(
    path: Path,
) -> str:

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


def stable_integer(
    text: str,
) -> int:

    digest = hashlib.sha256(
        text.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="big",
        signed=False,
    )


def parse_bool(
    value: str,
) -> bool:

    return (
        str(value)
        .strip()
        .lower()
        == "true"
    )


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 16B — "
        "VOCABULARY-SUPPORTED TEMPLATE HOLDOUT"
    )
    print("=" * 78)

    candidates = load_csv(
        CANDIDATES_PATH
    )

    targets = load_csv(
        TARGETS_PATH
    )

    recordings = load_csv(
        RECORDINGS_PATH
    )

    if len(candidates) != 299:

        raise RuntimeError(
            f"Expected 299 template groups, "
            f"found {len(candidates)}."
        )

    if len(targets) != 300:

        raise RuntimeError(
            f"Expected 300 sentence targets, "
            f"found {len(targets)}."
        )

    if len(recordings) != 7400:

        raise RuntimeError(
            f"Expected 7,400 recordings, "
            f"found {len(recordings):,}."
        )

    # ---------------------------------------------------------
    # Sentence targets.
    # ---------------------------------------------------------

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

    tokens_by_sentence = {}

    global_tokens = set()

    for sentence_id, row in (
        target_by_id.items()
    ):

        tokens = tuple(
            json.loads(
                row[
                    "gloss_tokens_json"
                ]
            )
        )

        if not tokens:

            raise RuntimeError(
                f"Sentence {sentence_id} "
                "contains zero gloss tokens."
            )

        tokens_by_sentence[
            sentence_id
        ] = tokens

        global_tokens.update(
            tokens
        )

    if len(global_tokens) != 455:

        raise RuntimeError(
            f"Expected 455 global gloss tokens, "
            f"found {len(global_tokens)}."
        )

    # ---------------------------------------------------------
    # Parse template groups.
    # ---------------------------------------------------------

    group_by_id = {}

    sentence_to_group = {}

    all_sentence_ids = []

    safe_groups = []

    for row in candidates:

        group_id = row[
            "template_group_id"
        ]

        if group_id in group_by_id:

            raise RuntimeError(
                f"Duplicate template group ID: "
                f"{group_id}"
            )

        sentence_ids = [
            int(value)
            for value in json.loads(
                row[
                    "sentence_ids_json"
                ]
            )
        ]

        if not sentence_ids:

            raise RuntimeError(
                f"Empty template group: "
                f"{group_id}"
            )

        for sentence_id in sentence_ids:

            if sentence_id in sentence_to_group:

                raise RuntimeError(
                    f"Sentence ID {sentence_id} "
                    "appears in multiple groups."
                )

            sentence_to_group[
                sentence_id
            ] = group_id

            all_sentence_ids.append(
                sentence_id
            )

        group_tokens = set()

        for sentence_id in sentence_ids:

            group_tokens.update(
                tokens_by_sentence[
                    sentence_id
                ]
            )

        parsed = {
            "template_group_id": (
                group_id
            ),

            "sentence_ids": (
                sentence_ids
            ),

            "tokens": (
                group_tokens
            ),

            "subject_support_signature": (
                row[
                    "subject_support_signature"
                ]
            ),

            "individually_safe": (
                parse_bool(
                    row[
                        "individually_holdout_safe"
                    ]
                )
            ),

            "row": (
                row
            ),
        }

        group_by_id[
            group_id
        ] = parsed

        if parsed[
            "individually_safe"
        ]:

            safe_groups.append(
                parsed
            )

    if sorted(
        all_sentence_ids
    ) != list(
        range(1, 301)
    ):

        raise RuntimeError(
            "Template groups do not partition "
            "sentence IDs 1..300."
        )

    if len(safe_groups) != 103:

        raise RuntimeError(
            f"Expected 103 individually safe groups, "
            f"found {len(safe_groups)}."
        )

    # ---------------------------------------------------------
    # Verify special 39/83 equivalence group.
    # ---------------------------------------------------------

    group_39 = sentence_to_group[
        39
    ]

    group_83 = sentence_to_group[
        83
    ]

    if group_39 != group_83:

        raise RuntimeError(
            "Sentence IDs 39 and 83 are not "
            "in the same equivalence group."
        )

    special_group = group_by_id[
        group_39
    ]

    if sorted(
        special_group[
            "sentence_ids"
        ]
    ) != [
        39,
        83,
    ]:

        raise RuntimeError(
            "39/83 equivalence group contains "
            "unexpected IDs."
        )

    # ---------------------------------------------------------
    # Candidate-pool quota validation.
    # ---------------------------------------------------------

    safe_by_signature = Counter(
        group[
            "subject_support_signature"
        ]
        for group in safe_groups
    )

    for signature, quota in (
        SIGNATURE_QUOTAS.items()
    ):

        available = safe_by_signature[
            signature
        ]

        if available < quota:

            raise RuntimeError(
                f"Not enough safe groups for "
                f"{signature}: "
                f"need {quota}, "
                f"have {available}."
            )

    if sum(
        SIGNATURE_QUOTAS.values()
    ) != TARGET_TEST_GROUPS:

        raise RuntimeError(
            "Signature quotas do not sum "
            "to target test-group count."
        )

    # ---------------------------------------------------------
    # Joint holdout safety.
    #
    # The whole proposed test-group collection is removed.
    #
    # Every target token in the proposed test collection
    # must still exist somewhere in remaining training
    # templates.
    # ---------------------------------------------------------

    all_sentence_id_set = set(
        range(1, 301)
    )

    def jointly_safe(
        proposed_groups: list[dict],
    ) -> bool:

        test_sentence_ids = set()

        for group in proposed_groups:

            test_sentence_ids.update(
                group[
                    "sentence_ids"
                ]
            )

        train_sentence_ids = (
            all_sentence_id_set
            - test_sentence_ids
        )

        training_tokens = set()

        for sentence_id in (
            train_sentence_ids
        ):

            training_tokens.update(
                tokens_by_sentence[
                    sentence_id
                ]
            )

        proposed_test_tokens = set()

        for sentence_id in (
            test_sentence_ids
        ):

            proposed_test_tokens.update(
                tokens_by_sentence[
                    sentence_id
                ]
            )

        return (
            proposed_test_tokens
            <= training_tokens
        )

    # ---------------------------------------------------------
    # Deterministic search.
    #
    # Multiple stable attempts are allowed so a locally valid
    # early choice cannot permanently block a later quota.
    # ---------------------------------------------------------

    selected = None
    selected_attempt = None

    eligible_groups = [
        group
        for group in safe_groups
        if group[
            "subject_support_signature"
        ] in SIGNATURE_QUOTAS
    ]

    for attempt in range(
        MAX_SELECTION_ATTEMPTS
    ):

        ordered = sorted(
            eligible_groups,
            key=lambda group: (
                stable_integer(
                    f"{SELECTION_SEED}|"
                    f"attempt={attempt}|"
                    f"{group['template_group_id']}"
                ),
                group[
                    "template_group_id"
                ],
            ),
        )

        proposed_selection = []

        selected_signature_counts = (
            Counter()
        )

        for group in ordered:

            signature = group[
                "subject_support_signature"
            ]

            if (
                selected_signature_counts[
                    signature
                ]
                >= SIGNATURE_QUOTAS[
                    signature
                ]
            ):

                continue

            trial = (
                proposed_selection
                + [
                    group
                ]
            )

            if not jointly_safe(
                trial
            ):

                continue

            proposed_selection.append(
                group
            )

            selected_signature_counts[
                signature
            ] += 1

            quotas_complete = all(
                selected_signature_counts[
                    signature
                ]
                == quota
                for signature, quota
                in SIGNATURE_QUOTAS.items()
            )

            if quotas_complete:

                selected = (
                    proposed_selection
                )

                selected_attempt = (
                    attempt
                )

                break

        if selected is not None:
            break

    if selected is None:

        raise RuntimeError(
            "Could not construct a jointly "
            "vocabulary-supported 21-group holdout "
            f"after {MAX_SELECTION_ATTEMPTS} "
            "deterministic attempts."
        )

    if len(selected) != (
        TARGET_TEST_GROUPS
    ):

        raise RuntimeError(
            "Incorrect number of selected groups."
        )

    selected_signature_counts = (
        Counter(
            group[
                "subject_support_signature"
            ]
            for group in selected
        )
    )

    for signature, quota in (
        SIGNATURE_QUOTAS.items()
    ):

        if (
            selected_signature_counts[
                signature
            ]
            != quota
        ):

            raise RuntimeError(
                f"Quota mismatch for "
                f"{signature}."
            )

    if not jointly_safe(
        selected
    ):

        raise RuntimeError(
            "Final selected set is not "
            "jointly vocabulary-supported."
        )

    selected_group_ids = {
        group[
            "template_group_id"
        ]
        for group in selected
    }

    test_sentence_ids = set()

    for group in selected:

        test_sentence_ids.update(
            group[
                "sentence_ids"
            ]
        )

    train_sentence_ids = (
        all_sentence_id_set
        - test_sentence_ids
    )

    # ---------------------------------------------------------
    # Final vocabulary support.
    # ---------------------------------------------------------

    training_tokens = set()

    for sentence_id in (
        train_sentence_ids
    ):

        training_tokens.update(
            tokens_by_sentence[
                sentence_id
            ]
        )

    test_tokens = set()

    for sentence_id in (
        test_sentence_ids
    ):

        test_tokens.update(
            tokens_by_sentence[
                sentence_id
            ]
        )

    unsupported_test_tokens = (
        test_tokens
        - training_tokens
    )

    if unsupported_test_tokens:

        raise RuntimeError(
            "Final test set contains tokens absent "
            "from training: "
            f"{sorted(unsupported_test_tokens)}"
        )

    if len(training_tokens) != 455:

        raise RuntimeError(
            f"Expected all 455 global gloss tokens "
            f"to remain in training, "
            f"found {len(training_tokens)}."
        )

    # ---------------------------------------------------------
    # Save selected template groups.
    # ---------------------------------------------------------

    selected_rows = []

    for rank, group in enumerate(
        selected,
        start=1,
    ):

        source = group[
            "row"
        ]

        selected_rows.append(
            {
                "selection_rank": (
                    rank
                ),

                "template_group_id": (
                    group[
                        "template_group_id"
                    ]
                ),

                "sentence_ids_json": (
                    source[
                        "sentence_ids_json"
                    ]
                ),

                "gloss_targets_json": (
                    source[
                        "gloss_targets_json"
                    ]
                ),

                "english_targets_json": (
                    source[
                        "english_targets_json"
                    ]
                ),

                "subject_support_signature": (
                    source[
                        "subject_support_signature"
                    ]
                ),

                "subject1_recordings": (
                    source[
                        "subject1_recordings"
                    ]
                ),

                "subject2_recordings": (
                    source[
                        "subject2_recordings"
                    ]
                ),

                "subject3_recordings": (
                    source[
                        "subject3_recordings"
                    ]
                ),

                "total_recordings": (
                    source[
                        "total_recordings"
                    ]
                ),

                "minimum_token_template_group_frequency": (
                    source[
                        "minimum_token_template_group_frequency"
                    ]
                ),

                "joint_holdout_safe": (
                    True
                ),
            }
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SELECTED_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                selected_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            selected_rows
        )

    # ---------------------------------------------------------
    # Expand split to all recordings.
    # ---------------------------------------------------------

    assignment_rows = []

    for row in sorted(
        recordings,
        key=lambda item: (
            item[
                "subject"
            ],
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

        sentence_id = int(
            row[
                "sentence_id"
            ]
        )

        template_group_id = (
            sentence_to_group[
                sentence_id
            ]
        )

        role = (
            "test"
            if template_group_id
            in selected_group_ids
            else "train"
        )

        assignment_rows.append(
            {
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
                    sentence_id
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

                "template_group_id": (
                    template_group_id
                ),

                "role": (
                    role
                ),
            }
        )

    if len(
        assignment_rows
    ) != 7400:

        raise RuntimeError(
            "Final assignment does not "
            "contain 7,400 rows."
        )

    if len(
        {
            row[
                "sample_id"
            ]
            for row in assignment_rows
        }
    ) != 7400:

        raise RuntimeError(
            "Final assignment sample IDs "
            "are not unique."
        )

    # ---------------------------------------------------------
    # Exact template leakage check.
    # ---------------------------------------------------------

    train_rows = [
        row
        for row in assignment_rows
        if row[
            "role"
        ] == "train"
    ]

    test_rows = [
        row
        for row in assignment_rows
        if row[
            "role"
        ] == "test"
    ]

    training_group_ids = {
        row[
            "template_group_id"
        ]
        for row in train_rows
    }

    testing_group_ids = {
        row[
            "template_group_id"
        ]
        for row in test_rows
    }

    template_group_overlap = (
        training_group_ids
        & testing_group_ids
    )

    if template_group_overlap:

        raise RuntimeError(
            "Template-equivalence group leakage: "
            f"{sorted(template_group_overlap)}"
        )

    if testing_group_ids != (
        selected_group_ids
    ):

        raise RuntimeError(
            "Observed test template groups do not "
            "match selected template groups."
        )

    # ---------------------------------------------------------
    # Stage-12 duplicate split-group safety.
    # ---------------------------------------------------------

    roles_by_split_group = {}

    for row in assignment_rows:

        split_group_id = row[
            "split_group_id"
        ]

        roles_by_split_group.setdefault(
            split_group_id,
            set(),
        ).add(
            row[
                "role"
            ]
        )

    split_group_role_violations = [
        split_group_id
        for split_group_id, roles
        in roles_by_split_group.items()
        if len(roles) != 1
    ]

    if split_group_role_violations:

        raise RuntimeError(
            "Stage-12 split groups cross "
            "train/test in Stage 16: "
            f"{split_group_role_violations[:10]}"
        )

    # ---------------------------------------------------------
    # 39/83 equivalence safety.
    # ---------------------------------------------------------

    role_39 = {
        row[
            "role"
        ]
        for row in assignment_rows
        if row[
            "sentence_id"
        ] == 39
    }

    role_83 = {
        row[
            "role"
        ]
        for row in assignment_rows
        if row[
            "sentence_id"
        ] == 83
    }

    if (
        len(role_39) != 1
        or len(role_83) != 1
        or role_39 != role_83
    ):

        raise RuntimeError(
            "Sentence IDs 39 and 83 do not "
            "share one split role."
        )

    # ---------------------------------------------------------
    # Subject-level summaries.
    # ---------------------------------------------------------

    subject_summary_rows = []

    for subject in SUBJECTS:

        subject_train = [
            row
            for row in train_rows
            if row[
                "subject"
            ] == subject
        ]

        subject_test = [
            row
            for row in test_rows
            if row[
                "subject"
            ] == subject
        ]

        subject_summary_rows.append(
            {
                "subject": (
                    subject
                ),

                "train_recordings": (
                    len(
                        subject_train
                    )
                ),

                "test_recordings": (
                    len(
                        subject_test
                    )
                ),

                "test_sentence_ids": (
                    len(
                        {
                            row[
                                "sentence_id"
                            ]
                            for row
                            in subject_test
                        }
                    )
                ),

                "test_template_groups": (
                    len(
                        {
                            row[
                                "template_group_id"
                            ]
                            for row
                            in subject_test
                        }
                    )
                ),
            }
        )

    if any(
        row[
            "test_recordings"
        ] == 0
        for row in subject_summary_rows
    ):

        raise RuntimeError(
            "At least one subject has zero "
            "template-holdout test recordings."
        )

    # ---------------------------------------------------------
    # Save assignment + subject summary.
    # ---------------------------------------------------------

    with ASSIGNMENT_PATH.open(
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
                "template_group_id",
                "role",
            ],
        )

        writer.writeheader()

        writer.writerows(
            assignment_rows
        )

    with SUBJECT_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                subject_summary_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            subject_summary_rows
        )

    # ---------------------------------------------------------
    # Frozen hashes.
    # ---------------------------------------------------------

    assignment_hash = (
        sha256_file(
            ASSIGNMENT_PATH
        )
    )

    selected_groups_hash = (
        sha256_file(
            SELECTED_PATH
        )
    )

    candidate_source_hash = (
        sha256_file(
            CANDIDATES_PATH
        )
    )

    HASH_PATH.write_text(
        assignment_hash + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Summary contract.
    # ---------------------------------------------------------

    selected_39_83 = (
        group_39
        in selected_group_ids
    )

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 16,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "split_name": (
            "vocabulary_supported_template_holdout"
        ),

        "selection_seed": (
            SELECTION_SEED
        ),

        "selection_attempt": (
            selected_attempt
        ),

        "eligible_template_groups": (
            len(
                safe_groups
            )
        ),

        "selected_template_groups": (
            len(
                selected_group_ids
            )
        ),

        "selected_sentence_ids": (
            len(
                test_sentence_ids
            )
        ),

        "signature_quotas": (
            SIGNATURE_QUOTAS
        ),

        "selected_signature_counts": (
            dict(
                selected_signature_counts
            )
        ),

        "recordings": {
            "total": (
                len(
                    assignment_rows
                )
            ),

            "train": (
                len(
                    train_rows
                )
            ),

            "test": (
                len(
                    test_rows
                )
            ),
        },

        "subjects": (
            subject_summary_rows
        ),

        "vocabulary": {
            "global_gloss_tokens": (
                len(
                    global_tokens
                )
            ),

            "training_gloss_tokens": (
                len(
                    training_tokens
                )
            ),

            "test_target_gloss_tokens": (
                len(
                    test_tokens
                )
            ),

            "unsupported_test_gloss_tokens": (
                len(
                    unsupported_test_tokens
                )
            ),

            "closed_vocabulary_supported": (
                len(
                    unsupported_test_tokens
                ) == 0
            ),
        },

        "leakage_checks": {
            "train_test_template_group_overlap": (
                len(
                    template_group_overlap
                )
            ),

            "stage12_split_group_role_violations": (
                len(
                    split_group_role_violations
                )
            ),

            "sentence_39_83_same_group": (
                group_39
                == group_83
            ),

            "sentence_39_83_same_role": (
                role_39
                == role_83
            ),

            "sentence_39_83_selected_for_test": (
                selected_39_83
            ),
        },

        "source_hashes": {
            "stage16a_candidate_groups_sha256": (
                candidate_source_hash
            ),
        },

        "output_hashes": {
            "selected_groups_sha256": (
                selected_groups_hash
            ),

            "assignment_sha256": (
                assignment_hash
            ),
        },

        "scientific_interpretation": (
            "Complete held-out template-equivalence "
            "groups are absent from recognition training, "
            "while every target gloss token remains "
            "supported by the remaining training templates."
        ),

        "not_claimed": [
            "open-vocabulary recognition",
            "unseen gloss-token recognition",
            "unseen signer recognition",
        ],

        "language_stage_rule": (
            "When this split is used later for retrieval "
            "or generative English correction evaluation, "
            "the held-out template groups must also be "
            "excluded from retrieval references and "
            "language-stage training examples."
        ),
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

    markdown = f"""# Stage 16 — Vocabulary-Supported Template Holdout

## Purpose

Create one frozen complete-template holdout experiment.

The test templates are absent from recognition training while every
target gloss token remains supported by the remaining training
templates.

## Feasibility basis

Stage 16A found:

- 299 template-equivalence groups,
- 103 individually holdout-safe groups,
- 196 individually unsafe groups.

The final test set is selected only from the safe candidate pool.

## Test size

Eligible safe groups:

103

Selected test groups:

{len(selected_group_ids)}

The test set therefore contains approximately 20% of the eligible
closed-vocabulary template groups.

It is not 20% of all 299 template-equivalence groups.

## Subject-support quotas

S1-only test groups:

{selected_signature_counts['S1']}

S1+S2 test groups:

{selected_signature_counts['S1+S2']}

S1+S3 test groups:

{selected_signature_counts['S1+S3']}

This guarantees template-holdout test recordings for all three
subjects.

## Joint vocabulary-support rule

The entire proposed test set is removed simultaneously before support
is checked.

All target gloss tokens occurring in the held-out test groups must
remain present in at least one remaining training template.

Global gloss vocabulary:

{len(global_tokens)}

Gloss tokens still represented in training:

{len(training_tokens)}

Unsupported held-out target gloss tokens:

{len(unsupported_test_tokens)}

## Template leakage

Training/test template-group overlap:

{len(template_group_overlap)}

Complete held-out template-equivalence groups therefore do not occur
in recognition training.

## Duplicate safety

Stage-12 exact numeric split-group role violations:

{len(split_group_role_violations)}

## Duplicate-label equivalence

Sentence IDs 39 and 83 remain one indivisible equivalence group.

Selected for test:

{selected_39_83}

Both IDs therefore always remain on the same side of the split.

## Scientific interpretation

This experiment evaluates generalisation to unseen complete
sentence/templates under supported gloss vocabulary.

It does not test unseen gloss tokens.

It does not represent unseen-signer evaluation because signer identity
is not held out.

## Language-stage leakage rule

When this template-holdout split is used later for retrieval or
generative English correction, the same held-out template groups must
also be absent from retrieval references and language-stage training
examples.

## Frozen assignment

Selection seed:

{SELECTION_SEED}

Successful deterministic attempt:

{selected_attempt}

Assignment SHA-256:

`{assignment_hash}`
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console report.
    # ---------------------------------------------------------

    print()
    print("SELECTION CONTRACT")
    print("-" * 78)

    print(
        "Eligible safe groups:",
        len(
            safe_groups
        ),
    )

    print(
        "Selected test groups:",
        len(
            selected_group_ids
        ),
    )

    print(
        "Selected sentence IDs:",
        len(
            test_sentence_ids
        ),
    )

    print(
        "Selection seed:",
        SELECTION_SEED,
    )

    print(
        "Successful attempt:",
        selected_attempt,
    )

    print()
    print("SIGNATURE QUOTAS")
    print("-" * 78)

    for signature in [
        "S1",
        "S1+S2",
        "S1+S3",
    ]:

        print(
            f"{signature}: "
            f"{selected_signature_counts[signature]}"
        )

    print()
    print("RECORDING COUNTS")
    print("-" * 78)

    print(
        "Train:",
        len(
            train_rows
        ),
    )

    print(
        "Test:",
        len(
            test_rows
        ),
    )

    print()
    print("BY SUBJECT")
    print("-" * 78)

    for row in (
        subject_summary_rows
    ):

        print(
            f"{row['subject']}: "
            f"train={row['train_recordings']}, "
            f"test={row['test_recordings']}, "
            f"test_sentence_ids="
            f"{row['test_sentence_ids']}, "
            f"test_template_groups="
            f"{row['test_template_groups']}"
        )

    print()
    print("VOCABULARY SUPPORT")
    print("-" * 78)

    print(
        "Global tokens:",
        len(
            global_tokens
        ),
    )

    print(
        "Training tokens:",
        len(
            training_tokens
        ),
    )

    print(
        "Test target tokens:",
        len(
            test_tokens
        ),
    )

    print(
        "Unsupported test tokens:",
        len(
            unsupported_test_tokens
        ),
    )

    print()
    print("LEAKAGE CHECKS")
    print("-" * 78)

    print(
        "Train/test template-group overlap:",
        len(
            template_group_overlap
        ),
    )

    print(
        "Stage-12 split-group role violations:",
        len(
            split_group_role_violations
        ),
    )

    print(
        "39/83 same equivalence group:",
        group_39 == group_83,
    )

    print(
        "39/83 same role:",
        role_39 == role_83,
    )

    print(
        "39/83 selected for test:",
        selected_39_83,
    )

    print()
    print("SELECTED GROUPS")
    print("-" * 78)

    for row in selected_rows:

        print(
            f"{int(row['selection_rank']):>2}: "
            f"{row['template_group_id']} | "
            f"{row['subject_support_signature']} | "
            f"IDs={row['sentence_ids_json']} | "
            f"{row['gloss_targets_json']}"
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
        SELECTED_PATH,
        ASSIGNMENT_PATH,
        SUBJECT_SUMMARY_PATH,
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

    stage16_pass = (
        len(
            selected_group_ids
        ) == 21

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

        and len(
            training_tokens
        ) == 455

        and len(
            unsupported_test_tokens
        ) == 0

        and len(
            template_group_overlap
        ) == 0

        and len(
            split_group_role_violations
        ) == 0

        and group_39
        == group_83

        and role_39
        == role_83

        and all(
            row[
                "test_recordings"
            ] > 0
            for row
            in subject_summary_rows
        )
    )

    print()

    if stage16_pass:

        print(
            "FINAL STATUS: "
            "STAGE 16 BUILD PASSED"
        )

    else:

        print(
            "FINAL STATUS: "
            "STAGE 16 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

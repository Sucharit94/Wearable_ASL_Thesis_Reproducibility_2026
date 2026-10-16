from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "raw_manifest.csv"
)

TARGETS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

VOCAB_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_gloss_vocabulary.json"
)

DUPLICATE_GROUPS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_duplicate_label_groups.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
)

TEMPLATE_SUPPORT_PATH = (
    OUTPUT_DIR
    / "stage11_subject_template_support.csv"
)

TOKEN_SUPPORT_PATH = (
    OUTPUT_DIR
    / "stage11_subject_token_support.csv"
)

LOSO_PREVIEW_PATH = (
    OUTPUT_DIR
    / "stage11_loso_support_preview.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage11_support_summary.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_11_SUBJECT_TEMPLATE_TOKEN_SUPPORT.md"
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


def main() -> None:
    print("=" * 78)
    print("STAGE 11 — SUBJECT / TEMPLATE / TOKEN SUPPORT AUDIT")
    print("=" * 78)

    manifest = load_csv(
        MANIFEST_PATH
    )

    targets = load_csv(
        TARGETS_PATH
    )

    vocabulary = load_json(
        VOCAB_PATH
    )

    duplicate_groups = load_json(
        DUPLICATE_GROUPS_PATH
    )

    if len(manifest) != 7400:
        raise RuntimeError(
            f"Expected 7,400 recordings, found {len(manifest):,}."
        )

    if len(targets) != 300:
        raise RuntimeError(
            f"Expected 300 sentence targets, found {len(targets):,}."
        )

    if (
        vocabulary["nonblank_token_count"]
        != 455
    ):
        raise RuntimeError(
            "Expected 455 nonblank gloss tokens."
        )

    target_by_id = {
        int(row["sentence_id"]): row
        for row in targets
    }

    if sorted(target_by_id) != list(
        range(1, 301)
    ):
        raise RuntimeError(
            "Sentence target IDs are not exactly 1..300."
        )

    # ---------------------------------------------------------
    # Recording support by subject/template
    # ---------------------------------------------------------

    recordings_per_subject_template = (
        Counter()
    )

    subject_recording_count = Counter()

    for row in manifest:
        subject = row["subject"]
        sentence_id = int(
            row["sentence_id"]
        )

        if subject not in SUBJECTS:
            raise RuntimeError(
                f"Unexpected subject: {subject}"
            )

        recordings_per_subject_template[
            (subject, sentence_id)
        ] += 1

        subject_recording_count[
            subject
        ] += 1

    # ---------------------------------------------------------
    # Template support table
    # ---------------------------------------------------------

    template_rows = []

    templates_by_subject = {
        subject: set()
        for subject in SUBJECTS
    }

    for sentence_id in range(
        1,
        301,
    ):
        target = target_by_id[
            sentence_id
        ]

        row = {
            "sentence_id": sentence_id,
            "gloss_raw": target[
                "gloss_raw"
            ],
            "english_raw": target[
                "english_raw"
            ],
        }

        supported_subject_count = 0

        for subject in SUBJECTS:
            count = (
                recordings_per_subject_template[
                    (subject, sentence_id)
                ]
            )

            row[
                f"{subject}_recordings"
            ] = count

            supported = (
                count > 0
            )

            row[
                f"{subject}_supported"
            ] = supported

            if supported:
                templates_by_subject[
                    subject
                ].add(
                    sentence_id
                )

                supported_subject_count += 1

        row[
            "supported_subject_count"
        ] = (
            supported_subject_count
        )

        row[
            "supported_by_all_subjects"
        ] = (
            supported_subject_count
            == len(SUBJECTS)
        )

        template_rows.append(
            row
        )

    with TEMPLATE_SUPPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                template_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            template_rows
        )

    # ---------------------------------------------------------
    # Token support derived from templates available to subject
    # ---------------------------------------------------------

    template_tokens = {}

    all_tokens = set()

    for sentence_id, target in (
        target_by_id.items()
    ):
        tokens = tuple(
            json.loads(
                target[
                    "gloss_tokens_json"
                ]
            )
        )

        template_tokens[
            sentence_id
        ] = tokens

        all_tokens.update(
            tokens
        )

    if len(all_tokens) != 455:
        raise RuntimeError(
            f"Expected 455 global tokens, "
            f"observed {len(all_tokens)}."
        )

    tokens_by_subject = {
        subject: set()
        for subject in SUBJECTS
    }

    token_template_count_by_subject = {
        subject: Counter()
        for subject in SUBJECTS
    }

    token_recording_count_by_subject = {
        subject: Counter()
        for subject in SUBJECTS
    }

    for subject in SUBJECTS:
        for sentence_id in (
            templates_by_subject[
                subject
            ]
        ):
            unique_tokens = set(
                template_tokens[
                    sentence_id
                ]
            )

            recording_count = (
                recordings_per_subject_template[
                    (subject, sentence_id)
                ]
            )

            for token in unique_tokens:
                tokens_by_subject[
                    subject
                ].add(
                    token
                )

                token_template_count_by_subject[
                    subject
                ][token] += 1

                token_recording_count_by_subject[
                    subject
                ][token] += (
                    recording_count
                )

    token_rows = []

    for token in sorted(
        all_tokens
    ):
        row = {
            "token": token,
        }

        supported_subject_count = 0

        for subject in SUBJECTS:
            supported = (
                token
                in tokens_by_subject[
                    subject
                ]
            )

            row[
                f"{subject}_supported"
            ] = supported

            row[
                f"{subject}_template_count"
            ] = (
                token_template_count_by_subject[
                    subject
                ][token]
            )

            row[
                f"{subject}_recording_count"
            ] = (
                token_recording_count_by_subject[
                    subject
                ][token]
            )

            if supported:
                supported_subject_count += 1

        row[
            "supported_subject_count"
        ] = (
            supported_subject_count
        )

        row[
            "supported_by_all_subjects"
        ] = (
            supported_subject_count
            == len(SUBJECTS)
        )

        token_rows.append(
            row
        )

    with TOKEN_SUPPORT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                token_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            token_rows
        )

    # ---------------------------------------------------------
    # LOSO support preview
    #
    # This does NOT create LOSO splits.
    #
    # It only measures whether held-out subject content
    # exists in the union of the other two subjects.
    # ---------------------------------------------------------

    loso_rows = []

    for held_out in SUBJECTS:
        training_subjects = [
            subject
            for subject in SUBJECTS
            if subject != held_out
        ]

        training_templates = set().union(
            *[
                templates_by_subject[
                    subject
                ]
                for subject
                in training_subjects
            ]
        )

        training_tokens = set().union(
            *[
                tokens_by_subject[
                    subject
                ]
                for subject
                in training_subjects
            ]
        )

        held_templates = (
            templates_by_subject[
                held_out
            ]
        )

        held_tokens = (
            tokens_by_subject[
                held_out
            ]
        )

        template_supported = (
            held_templates
            & training_templates
        )

        template_unsupported = (
            held_templates
            - training_templates
        )

        token_supported = (
            held_tokens
            & training_tokens
        )

        token_unsupported = (
            held_tokens
            - training_tokens
        )

        held_recordings = [
            row
            for row in manifest
            if row["subject"]
            == held_out
        ]

        recordings_template_supported = 0
        recordings_all_tokens_supported = 0

        for recording in (
            held_recordings
        ):
            sentence_id = int(
                recording[
                    "sentence_id"
                ]
            )

            if (
                sentence_id
                in training_templates
            ):
                recordings_template_supported += 1

            target_tokens = set(
                template_tokens[
                    sentence_id
                ]
            )

            if target_tokens.issubset(
                training_tokens
            ):
                recordings_all_tokens_supported += 1

        loso_rows.append(
            {
                "held_out_subject": (
                    held_out
                ),
                "training_subjects": (
                    "+".join(
                        training_subjects
                    )
                ),
                "held_out_recordings": (
                    len(
                        held_recordings
                    )
                ),
                "held_out_templates": (
                    len(
                        held_templates
                    )
                ),
                "template_supported_count": (
                    len(
                        template_supported
                    )
                ),
                "template_unsupported_count": (
                    len(
                        template_unsupported
                    )
                ),
                "template_unsupported_ids_json": (
                    json.dumps(
                        sorted(
                            template_unsupported
                        )
                    )
                ),
                "held_out_tokens": (
                    len(
                        held_tokens
                    )
                ),
                "token_supported_count": (
                    len(
                        token_supported
                    )
                ),
                "token_unsupported_count": (
                    len(
                        token_unsupported
                    )
                ),
                "token_unsupported_json": (
                    json.dumps(
                        sorted(
                            token_unsupported
                        )
                    )
                ),
                "recordings_template_supported": (
                    recordings_template_supported
                ),
                "recordings_all_tokens_supported": (
                    recordings_all_tokens_supported
                ),
            }
        )

    with LOSO_PREVIEW_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                loso_rows[0].keys()
            ),
        )

        writer.writeheader()
        writer.writerows(
            loso_rows
        )

    # ---------------------------------------------------------
    # Pairwise overlaps
    # ---------------------------------------------------------

    pairwise = {}

    for i, subject_a in enumerate(
        SUBJECTS
    ):
        for subject_b in (
            SUBJECTS[
                i + 1:
            ]
        ):
            key = (
                f"{subject_a}__"
                f"{subject_b}"
            )

            template_intersection = (
                templates_by_subject[
                    subject_a
                ]
                & templates_by_subject[
                    subject_b
                ]
            )

            token_intersection = (
                tokens_by_subject[
                    subject_a
                ]
                & tokens_by_subject[
                    subject_b
                ]
            )

            pairwise[key] = {
                "shared_templates": (
                    len(
                        template_intersection
                    )
                ),
                "subject_a_templates": (
                    len(
                        templates_by_subject[
                            subject_a
                        ]
                    )
                ),
                "subject_b_templates": (
                    len(
                        templates_by_subject[
                            subject_b
                        ]
                    )
                ),
                "shared_tokens": (
                    len(
                        token_intersection
                    )
                ),
                "subject_a_tokens": (
                    len(
                        tokens_by_subject[
                            subject_a
                        ]
                    )
                ),
                "subject_b_tokens": (
                    len(
                        tokens_by_subject[
                            subject_b
                        ]
                    )
                ),
            }

    all_subject_templates = (
        set.intersection(
            *[
                templates_by_subject[
                    subject
                ]
                for subject in SUBJECTS
            ]
        )
    )

    all_subject_tokens = (
        set.intersection(
            *[
                tokens_by_subject[
                    subject
                ]
                for subject in SUBJECTS
            ]
        )
    )

    subject_summary = {}

    for subject in SUBJECTS:
        counts = [
            recordings_per_subject_template[
                (subject, sentence_id)
            ]
            for sentence_id in (
                templates_by_subject[
                    subject
                ]
            )
        ]

        subject_summary[
            subject
        ] = {
            "recordings": (
                subject_recording_count[
                    subject
                ]
            ),
            "templates": len(
                templates_by_subject[
                    subject
                ]
            ),
            "gloss_tokens": len(
                tokens_by_subject[
                    subject
                ]
            ),
            "minimum_recordings_per_supported_template": (
                min(counts)
                if counts
                else 0
            ),
            "maximum_recordings_per_supported_template": (
                max(counts)
                if counts
                else 0
            ),
        }

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 11,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "global": {
            "recordings": (
                len(manifest)
            ),
            "templates": (
                len(targets)
            ),
            "gloss_tokens": (
                len(all_tokens)
            ),
            "templates_supported_by_all_subjects": (
                len(
                    all_subject_templates
                )
            ),
            "tokens_supported_by_all_subjects": (
                len(
                    all_subject_tokens
                )
            ),
        },
        "subjects": (
            subject_summary
        ),
        "pairwise_overlap": (
            pairwise
        ),
        "duplicate_label_equivalence_groups": (
            duplicate_groups[
                "groups"
            ]
        ),
        "loso_preview": (
            loso_rows
        ),
        "important_scope_note": (
            "Stage 11 is descriptive only. "
            "It does not create folds or LOSO splits."
        ),
        "future_rules": {
            "duplicate_label_equivalence_grouping": (
                "IDs 39 and 83 must remain together "
                "for experiments claiming unseen "
                "complete-template generalisation."
            ),
            "loso_support_definitions": (
                "Formalised later in Stage 15."
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

    markdown = f"""# Stage 11 — Subject / Template / Token Support Audit

## Purpose

Measure descriptive support before creating any train/test split.

## Global source

Recordings: {len(manifest):,}

Sentence/template IDs: {len(targets):,}

Gloss vocabulary: {len(all_tokens):,}

## Subject coverage

"""

    for subject in SUBJECTS:
        item = subject_summary[
            subject
        ]

        markdown += (
            f"### {subject}\n\n"
            f"Recordings: {item['recordings']:,}\n\n"
            f"Supported templates: {item['templates']:,}\n\n"
            f"Supported gloss tokens: {item['gloss_tokens']:,}\n\n"
        )

    markdown += f"""
## Cross-subject support

Templates supported by all three subjects:

{len(all_subject_templates):,}

Gloss tokens supported by all three subjects:

{len(all_subject_tokens):,}

## Duplicate-label rule

Sentence IDs 39 and 83 carry the same gloss and English targets.

They remain distinct dataset IDs, but they must remain in the same
partition for experiments claiming unseen complete-template
generalisation.

## LOSO interpretation

The LOSO preview measures support only.

It does not create the final LOSO partitions.

Stage 15 will formally define:

- LOSO-All,
- LOSO-Template-Supported,
- LOSO-Token-Supported.

## Scope boundary

No folds or train/test assignments are generated in Stage 11.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("GLOBAL")
    print("-" * 78)

    print(
        "Recordings:",
        len(manifest),
    )

    print(
        "Templates:",
        len(targets),
    )

    print(
        "Gloss tokens:",
        len(all_tokens),
    )

    print()
    print("BY SUBJECT")
    print("-" * 78)

    for subject in SUBJECTS:
        item = subject_summary[
            subject
        ]

        print(
            f"{subject}: "
            f"recordings={item['recordings']:,}, "
            f"templates={item['templates']}, "
            f"tokens={item['gloss_tokens']}, "
            f"reps/template="
            f"{item['minimum_recordings_per_supported_template']}"
            f".."
            f"{item['maximum_recordings_per_supported_template']}"
        )

    print()
    print("SUPPORTED BY ALL THREE SUBJECTS")
    print("-" * 78)

    print(
        "Templates:",
        len(
            all_subject_templates
        ),
    )

    print(
        "Gloss tokens:",
        len(
            all_subject_tokens
        ),
    )

    print()
    print("PAIRWISE OVERLAP")
    print("-" * 78)

    for key, value in (
        pairwise.items()
    ):
        print(
            f"{key}: "
            f"templates="
            f"{value['shared_templates']}, "
            f"tokens="
            f"{value['shared_tokens']}"
        )

    print()
    print("LOSO SUPPORT PREVIEW")
    print("-" * 78)

    for row in loso_rows:
        print(
            f"{row['held_out_subject']}: "
            f"recordings={row['held_out_recordings']}, "
            f"templates="
            f"{row['held_out_templates']}, "
            f"unsupported_templates="
            f"{row['template_unsupported_count']}, "
            f"tokens="
            f"{row['held_out_tokens']}, "
            f"unsupported_tokens="
            f"{row['token_unsupported_count']}, "
            f"template-supported recordings="
            f"{row['recordings_template_supported']}, "
            f"all-token-supported recordings="
            f"{row['recordings_all_tokens_supported']}"
        )

    print()
    print("DUPLICATE LABEL EQUIVALENCE")
    print("-" * 78)

    for group in (
        duplicate_groups[
            "groups"
        ]
    ):
        print(
            f"{group['group_id']}: "
            f"sentence IDs "
            f"{group['sentence_ids']}"
        )

    print()
    print(
        "Saved:",
        TEMPLATE_SUPPORT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        TOKEN_SUPPORT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        LOSO_PREVIEW_PATH.relative_to(
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

    stage11_pass = (
        len(manifest) == 7400
        and len(targets) == 300
        and len(all_tokens) == 455
        and sum(
            subject_recording_count.values()
        ) == 7400
        and all(
            subject
            in subject_summary
            for subject in SUBJECTS
        )
        and len(
            template_rows
        ) == 300
        and len(
            token_rows
        ) == 455
        and len(
            loso_rows
        ) == 3
    )

    print()

    if stage11_pass:
        print(
            "FINAL STATUS: STAGE 11 BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: STAGE 11 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

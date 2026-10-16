from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE14_ASSIGNMENT_PATH = (
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

TARGETS_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

STAGE11_PREVIEW_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage11_loso_support_preview.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
)

SUPPORT_ASSIGNMENT_PATH = (
    OUTPUT_DIR
    / "stage15_loso_support_assignment.csv"
)

SUPPORT_SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage15_loso_support_summary.csv"
)

DEFINITIONS_PATH = (
    OUTPUT_DIR
    / "stage15_loso_support_definitions.json"
)

HASH_PATH = (
    OUTPUT_DIR
    / "stage15_loso_support_assignment.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_15_LOSO_SUPPORT_DEFINITIONS.md"
)


LOSO_IDS = [
    "loso_subject1",
    "loso_subject2",
    "loso_subject3",
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


def parse_bool(text: str) -> bool:
    return text.strip().lower() == "true"


def main() -> None:
    print("=" * 78)
    print("STAGE 15 — LOSO SUPPORT DEFINITIONS")
    print("=" * 78)

    # ---------------------------------------------------------
    # Verify frozen Stage-14 assignment.
    # ---------------------------------------------------------

    recorded_stage14_hash = (
        read_recorded_hash(
            STAGE14_HASH_PATH
        )
    )

    actual_stage14_hash = (
        sha256_file(
            STAGE14_ASSIGNMENT_PATH
        )
    )

    if (
        recorded_stage14_hash
        != actual_stage14_hash
    ):
        raise RuntimeError(
            "Stage 14 LOSO assignment hash mismatch."
        )

    assignment = load_csv(
        STAGE14_ASSIGNMENT_PATH
    )

    targets = load_csv(
        TARGETS_PATH
    )

    stage11_preview = load_csv(
        STAGE11_PREVIEW_PATH
    )

    if len(assignment) != 22200:
        raise RuntimeError(
            f"Expected 22,200 Stage-14 rows, "
            f"found {len(assignment):,}."
        )

    if len(targets) != 300:
        raise RuntimeError(
            f"Expected 300 targets, "
            f"found {len(targets):,}."
        )

    target_by_id = {
        int(row["sentence_id"]): row
        for row in targets
    }

    if sorted(target_by_id) != list(
        range(1, 301)
    ):
        raise RuntimeError(
            "Target sentence IDs are not exactly 1..300."
        )

    target_tokens = {}

    for sentence_id, row in (
        target_by_id.items()
    ):
        tokens = tuple(
            json.loads(
                row["gloss_tokens_json"]
            )
        )

        if not tokens:
            raise RuntimeError(
                f"Sentence {sentence_id} "
                "has no gloss tokens."
            )

        target_tokens[
            sentence_id
        ] = tokens

    stage11_by_subject = {
        row["held_out_subject"]: row
        for row in stage11_preview
    }

    # ---------------------------------------------------------
    # Derive support from each LOSO TRAINING SET ONLY.
    # ---------------------------------------------------------

    support_rows = []
    summary_rows = []

    for loso_id in LOSO_IDS:

        run_rows = [
            row
            for row in assignment
            if row["loso_id"]
            == loso_id
        ]

        if len(run_rows) != 7400:
            raise RuntimeError(
                f"{loso_id} does not contain 7,400 rows."
            )

        train_rows = [
            row
            for row in run_rows
            if row["role"] == "train"
        ]

        test_rows = [
            row
            for row in run_rows
            if row["role"] == "test"
        ]

        held_out_subjects = sorted(
            {
                row["held_out_subject"]
                for row in run_rows
            }
        )

        if len(
            held_out_subjects
        ) != 1:
            raise RuntimeError(
                f"Unexpected held-out subject "
                f"metadata in {loso_id}."
            )

        held_out_subject = (
            held_out_subjects[0]
        )

        training_template_ids = {
            int(row["sentence_id"])
            for row in train_rows
        }

        test_template_ids = {
            int(row["sentence_id"])
            for row in test_rows
        }

        training_tokens = set()

        for sentence_id in (
            training_template_ids
        ):
            training_tokens.update(
                target_tokens[
                    sentence_id
                ]
            )

        test_tokens = set()

        for sentence_id in (
            test_template_ids
        ):
            test_tokens.update(
                target_tokens[
                    sentence_id
                ]
            )

        unsupported_test_tokens = (
            test_tokens
            - training_tokens
        )

        supported_test_templates = (
            test_template_ids
            & training_template_ids
        )

        unsupported_test_templates = (
            test_template_ids
            - training_template_ids
        )

        template_supported_recordings = 0
        token_supported_recordings = 0
        token_supported_novel_template_recordings = 0
        token_unsupported_recordings = 0

        test_template_supported_ids = set()
        test_token_supported_template_ids = set()
        token_supported_novel_template_ids = set()
        token_unsupported_template_ids = set()

        for row in sorted(
            test_rows,
            key=lambda item: (
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

            sentence_id = int(
                row["sentence_id"]
            )

            tokens = target_tokens[
                sentence_id
            ]

            template_supported = (
                sentence_id
                in training_template_ids
            )

            unsupported_tokens = sorted(
                {
                    token
                    for token in tokens
                    if token
                    not in training_tokens
                }
            )

            all_tokens_supported = (
                len(
                    unsupported_tokens
                ) == 0
            )

            # Exact template support necessarily implies
            # every token of that template exists in training.
            if (
                template_supported
                and not all_tokens_supported
            ):
                raise RuntimeError(
                    "Logical support violation: "
                    f"{loso_id}, sentence {sentence_id}"
                )

            if template_supported:
                support_class = (
                    "template_supported"
                )

                template_supported_recordings += 1

                token_supported_recordings += 1

                test_template_supported_ids.add(
                    sentence_id
                )

                test_token_supported_template_ids.add(
                    sentence_id
                )

            elif all_tokens_supported:
                support_class = (
                    "token_supported_novel_template"
                )

                token_supported_recordings += 1

                token_supported_novel_template_recordings += 1

                test_token_supported_template_ids.add(
                    sentence_id
                )

                token_supported_novel_template_ids.add(
                    sentence_id
                )

            else:
                support_class = (
                    "token_unsupported"
                )

                token_unsupported_recordings += 1

                token_unsupported_template_ids.add(
                    sentence_id
                )

            support_rows.append(
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
                        sentence_id
                    ),
                    "repetition_id": (
                        int(
                            row[
                                "repetition_id"
                            ]
                        )
                    ),
                    "gloss_raw": (
                        target_by_id[
                            sentence_id
                        ][
                            "gloss_raw"
                        ]
                    ),
                    "gloss_token_count": (
                        len(tokens)
                    ),
                    "loso_all": (
                        True
                    ),
                    "template_supported": (
                        template_supported
                    ),
                    "all_tokens_supported": (
                        all_tokens_supported
                    ),
                    "unsupported_token_count": (
                        len(
                            unsupported_tokens
                        )
                    ),
                    "unsupported_tokens_json": (
                        json.dumps(
                            unsupported_tokens,
                            ensure_ascii=False,
                        )
                    ),
                    "support_class": (
                        support_class
                    ),
                }
            )

        # -----------------------------------------------------
        # Basic nesting.
        # -----------------------------------------------------

        if (
            template_supported_recordings
            > token_supported_recordings
        ):
            raise RuntimeError(
                f"Subset nesting failed in {loso_id}."
            )

        if (
            token_supported_recordings
            + token_unsupported_recordings
            != len(test_rows)
        ):
            raise RuntimeError(
                f"Token-support partition failed "
                f"in {loso_id}."
            )

        if (
            template_supported_recordings
            + token_supported_novel_template_recordings
            != token_supported_recordings
        ):
            raise RuntimeError(
                f"Token-supported decomposition "
                f"failed in {loso_id}."
            )

        # -----------------------------------------------------
        # Stage-11 independent consistency check.
        # -----------------------------------------------------

        stage11 = stage11_by_subject[
            held_out_subject
        ]

        expected_template_supported_recordings = int(
            stage11[
                "recordings_template_supported"
            ]
        )

        expected_token_supported_recordings = int(
            stage11[
                "recordings_all_tokens_supported"
            ]
        )

        expected_unsupported_templates = int(
            stage11[
                "template_unsupported_count"
            ]
        )

        expected_unsupported_tokens = int(
            stage11[
                "token_unsupported_count"
            ]
        )

        if (
            template_supported_recordings
            != expected_template_supported_recordings
        ):
            raise RuntimeError(
                f"Stage-11 template-support mismatch "
                f"for {loso_id}: "
                f"{template_supported_recordings} vs "
                f"{expected_template_supported_recordings}"
            )

        if (
            token_supported_recordings
            != expected_token_supported_recordings
        ):
            raise RuntimeError(
                f"Stage-11 token-support mismatch "
                f"for {loso_id}: "
                f"{token_supported_recordings} vs "
                f"{expected_token_supported_recordings}"
            )

        if (
            len(
                unsupported_test_templates
            )
            != expected_unsupported_templates
        ):
            raise RuntimeError(
                f"Stage-11 unsupported-template "
                f"mismatch for {loso_id}."
            )

        if (
            len(
                unsupported_test_tokens
            )
            != expected_unsupported_tokens
        ):
            raise RuntimeError(
                f"Stage-11 unsupported-token "
                f"mismatch for {loso_id}."
            )

        summary_rows.append(
            {
                "loso_id": (
                    loso_id
                ),
                "held_out_subject": (
                    held_out_subject
                ),
                "train_recordings": (
                    len(train_rows)
                ),
                "test_recordings_loso_all": (
                    len(test_rows)
                ),
                "training_templates": (
                    len(
                        training_template_ids
                    )
                ),
                "test_templates": (
                    len(
                        test_template_ids
                    )
                ),
                "template_supported_test_templates": (
                    len(
                        supported_test_templates
                    )
                ),
                "template_unsupported_test_templates": (
                    len(
                        unsupported_test_templates
                    )
                ),
                "training_gloss_tokens": (
                    len(
                        training_tokens
                    )
                ),
                "test_gloss_tokens": (
                    len(
                        test_tokens
                    )
                ),
                "unsupported_test_gloss_tokens": (
                    len(
                        unsupported_test_tokens
                    )
                ),
                "loso_all_recordings": (
                    len(test_rows)
                ),
                "template_supported_recordings": (
                    template_supported_recordings
                ),
                "token_supported_recordings": (
                    token_supported_recordings
                ),
                "token_supported_novel_template_recordings": (
                    token_supported_novel_template_recordings
                ),
                "token_unsupported_recordings": (
                    token_unsupported_recordings
                ),
                "template_supported_template_ids_json": (
                    json.dumps(
                        sorted(
                            test_template_supported_ids
                        )
                    )
                ),
                "token_supported_template_ids_json": (
                    json.dumps(
                        sorted(
                            test_token_supported_template_ids
                        )
                    )
                ),
                "token_supported_novel_template_ids_json": (
                    json.dumps(
                        sorted(
                            token_supported_novel_template_ids
                        )
                    )
                ),
                "token_unsupported_template_ids_json": (
                    json.dumps(
                        sorted(
                            token_unsupported_template_ids
                        )
                    )
                ),
                "unsupported_test_tokens_json": (
                    json.dumps(
                        sorted(
                            unsupported_test_tokens
                        ),
                        ensure_ascii=False,
                    )
                ),
                "stage11_consistency": (
                    True
                ),
            }
        )

    # ---------------------------------------------------------
    # Across all LOSO runs, every original recording is
    # a held-out test item exactly once.
    # ---------------------------------------------------------

    if len(support_rows) != 7400:
        raise RuntimeError(
            f"Expected 7,400 held-out support rows, "
            f"found {len(support_rows):,}."
        )

    support_sample_ids = [
        row["sample_id"]
        for row in support_rows
    ]

    if len(set(
        support_sample_ids
    )) != 7400:
        raise RuntimeError(
            "Held-out support rows do not contain "
            "7,400 unique samples."
        )

    # ---------------------------------------------------------
    # Save support assignment.
    # ---------------------------------------------------------

    with SUPPORT_ASSIGNMENT_PATH.open(
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
                "gloss_raw",
                "gloss_token_count",
                "loso_all",
                "template_supported",
                "all_tokens_supported",
                "unsupported_token_count",
                "unsupported_tokens_json",
                "support_class",
            ],
        )

        writer.writeheader()
        writer.writerows(
            support_rows
        )

    with SUPPORT_SUMMARY_PATH.open(
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

    support_hash = sha256_file(
        SUPPORT_ASSIGNMENT_PATH
    )

    HASH_PATH.write_text(
        support_hash + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Frozen definitions.
    # ---------------------------------------------------------

    definitions = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 15,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "source_stage14_sha256": (
            actual_stage14_hash
        ),
        "support_assignment_sha256": (
            support_hash
        ),

        "definitions": {
            "LOSO-All": {
                "rule": (
                    "Every recording belonging to the "
                    "held-out subject."
                ),
                "requires_exact_template_in_training": (
                    False
                ),
                "requires_all_target_tokens_in_training": (
                    False
                ),
            },

            "LOSO-Template-Supported": {
                "rule": (
                    "Held-out recordings whose exact "
                    "sentence_id occurs at least once "
                    "in the LOSO training partition."
                ),
                "requires_exact_template_in_training": (
                    True
                ),
                "requires_all_target_tokens_in_training": (
                    True
                ),
            },

            "LOSO-Token-Supported": {
                "rule": (
                    "Held-out recordings for which "
                    "every target gloss token occurs "
                    "in at least one template in the "
                    "LOSO training partition."
                ),
                "requires_exact_template_in_training": (
                    False
                ),
                "requires_all_target_tokens_in_training": (
                    True
                ),
            },

            "Token-Supported-Novel-Template": {
                "rule": (
                    "Diagnostic subset of LOSO-Token-Supported "
                    "where the exact sentence_id is absent "
                    "from training."
                ),
                "primary_report_subset": (
                    False
                ),
            },

            "Token-Unsupported": {
                "rule": (
                    "Diagnostic subset containing at least "
                    "one target gloss token with zero "
                    "training support in that LOSO run."
                ),
                "primary_report_subset": (
                    False
                ),
            },
        },

        "nesting_contract": (
            "LOSO-Template-Supported is a subset of "
            "LOSO-Token-Supported, which is a subset "
            "of LOSO-All."
        ),

        "training_contract": (
            "Support subsets are evaluation filters only. "
            "For a given held-out signer, the same trained "
            "LOSO recognizer is evaluated on all applicable "
            "support views; separate models are not trained "
            "for each support subset."
        ),

        "support_source_contract": (
            "Template and token support are determined "
            "only from the corresponding LOSO training "
            "partition. Global vocabulary membership "
            "does not count as LOSO training support."
        ),

        "reporting_contract": (
            "Report LOSO-All for every signer. "
            "Report support-aware subsets alongside it. "
            "Do not treat the three LOSO directions as "
            "equivalent label-support conditions."
        ),

        "summary": (
            summary_rows
        ),
    }

    DEFINITIONS_PATH.write_text(
        json.dumps(
            definitions,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Documentation.
    # ---------------------------------------------------------

    markdown = """# Stage 15 — LOSO Support Definitions

## Purpose

Freeze support-aware evaluation definitions for the three LOSO runs.

Support is determined only from the training partition of the
corresponding held-out-signer run.

## LOSO-All

Includes every recording from the held-out subject.

This is the complete unseen-signer evaluation.

It may also contain templates or gloss tokens absent from training.

## LOSO-Template-Supported

Includes held-out recordings whose exact sentence ID occurs in the
training partition.

This evaluates unseen-signer recognition while keeping the complete
sentence/template supported by training.

## LOSO-Token-Supported

Includes held-out recordings for which every target gloss token occurs
somewhere in the training partition.

The exact complete sentence does not need to have occurred in training.

Therefore this set can contain novel combinations of individually
supported gloss tokens.

## Nesting

`LOSO-Template-Supported`

is a subset of

`LOSO-Token-Supported`

which is a subset of

`LOSO-All`.

## Token-supported novel-template diagnostic

A held-out recording may have every gloss token supported while its
complete sentence/template is absent from training.

These cases are recorded separately because they test unseen signer
plus unseen template composition without introducing unseen gloss
tokens.

## Token-unsupported diagnostic

A held-out recording is token-unsupported if at least one target gloss
token never occurs in that LOSO training partition.

These recordings remain part of LOSO-All.

They are not deleted.

## Model-training rule

Support subsets do not create separate training jobs.

For each held-out signer, one recognizer is trained using the complete
LOSO training partition.

Its held-out predictions are then evaluated on:

- LOSO-All,
- LOSO-Template-Supported,
- LOSO-Token-Supported.

Thus support-aware reporting changes the evaluation denominator, not
the trained recognizer.

## Global vocabulary

The Stage-10 global CTC vocabulary remains the common output schema.

However, the existence of a token in that global vocabulary does not
mean the token has training support in a particular LOSO run.

## Reporting rule

The three held-out-signer directions must be reported separately.

In particular, LOSO Subject 1 contains substantial template and token
support shift, whereas LOSO Subjects 2 and 3 have different support
conditions.

No unsupported sample is silently removed from LOSO-All.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console report.
    # ---------------------------------------------------------

    print()
    print("FROZEN DEFINITIONS")
    print("-" * 78)

    print(
        "LOSO-All: every held-out recording"
    )

    print(
        "LOSO-Template-Supported: "
        "exact sentence_id exists in training"
    )

    print(
        "LOSO-Token-Supported: "
        "every target gloss token exists in training"
    )

    print()
    print("SUBSET NESTING")
    print("-" * 78)

    print(
        "Template-Supported "
        "⊆ Token-Supported "
        "⊆ All"
    )

    print()
    print("SUPPORT COUNTS")
    print("-" * 78)

    for row in summary_rows:
        print(
            f"{row['loso_id']}: "
            f"All={row['loso_all_recordings']}, "
            f"Template={row['template_supported_recordings']}, "
            f"Token={row['token_supported_recordings']}, "
            f"TokenNovelTemplate="
            f"{row['token_supported_novel_template_recordings']}, "
            f"TokenUnsupported="
            f"{row['token_unsupported_recordings']}"
        )

    print()
    print("UNIQUE SUPPORT COVERAGE")
    print("-" * 78)

    for row in summary_rows:
        print(
            f"{row['loso_id']}: "
            f"train_templates={row['training_templates']}, "
            f"unsupported_test_templates="
            f"{row['template_unsupported_test_templates']}, "
            f"train_tokens={row['training_gloss_tokens']}, "
            f"unsupported_test_tokens="
            f"{row['unsupported_test_gloss_tokens']}"
        )

    print()
    print("STAGE-11 CONSISTENCY")
    print("-" * 78)

    for row in summary_rows:
        print(
            f"{row['loso_id']}: "
            f"{row['stage11_consistency']}"
        )

    print()
    print("SUPPORT ASSIGNMENT")
    print("-" * 78)

    print(
        "Rows:",
        len(support_rows),
    )

    print(
        "Unique held-out samples:",
        len(
            set(
                support_sample_ids
            )
        ),
    )

    print()
    print("FROZEN HASH")
    print("-" * 78)

    print(
        support_hash
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    for path in [
        SUPPORT_ASSIGNMENT_PATH,
        SUPPORT_SUMMARY_PATH,
        DEFINITIONS_PATH,
        HASH_PATH,
        DOC_PATH,
    ]:
        print(
            "Saved:",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    stage15_pass = (
        len(support_rows) == 7400
        and len(
            set(
                support_sample_ids
            )
        ) == 7400
        and len(summary_rows) == 3
        and all(
            bool(
                row[
                    "stage11_consistency"
                ]
            )
            for row in summary_rows
        )
        and all(
            row[
                "template_supported_recordings"
            ]
            <= row[
                "token_supported_recordings"
            ]
            <= row[
                "loso_all_recordings"
            ]
            for row in summary_rows
        )
    )

    print()

    if stage15_pass:
        print(
            "FINAL STATUS: "
            "STAGE 15 BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 15 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

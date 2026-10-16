from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gu_author_dataset"
)

GLOSS_PATH = RAW_ROOT / "label_no_order.txt"
ENGLISH_PATH = RAW_ROOT / "label_order.txt"

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "raw_manifest.csv"
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
    / "labels"
)

CATALOG_PATH = (
    OUTPUT_DIR
    / "stage10_sentence_targets.csv"
)

VOCAB_PATH = (
    OUTPUT_DIR
    / "stage10_gloss_vocabulary.json"
)

TOKEN_TO_ID_PATH = (
    OUTPUT_DIR
    / "stage10_token_to_id.json"
)

ID_TO_TOKEN_PATH = (
    OUTPUT_DIR
    / "stage10_id_to_token.json"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage10_label_vocabulary_summary.json"
)

CATALOG_HASH_PATH = (
    OUTPUT_DIR
    / "stage10_sentence_targets.sha256"
)

VOCAB_HASH_PATH = (
    OUTPUT_DIR
    / "stage10_gloss_vocabulary.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_10_LABELS_AND_VOCABULARY.md"
)

BLANK_TOKEN = "<blank>"
BLANK_ID = 0


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


def load_json(path: Path) -> dict:
    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def read_lines(path: Path) -> list[str]:
    return path.read_text(
        encoding="utf-8-sig"
    ).splitlines()


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
    print("STAGE 10B — LABELS + FRESH CTC VOCABULARY")
    print("=" * 78)

    # ---------------------------------------------------------
    # Source guards
    # ---------------------------------------------------------

    contract = load_json(
        SOURCE_CONTRACT_PATH
    )

    expected_gloss_hash = (
        contract["labels"]
        ["recognition"]["sha256"]
        .lower()
    )

    expected_english_hash = (
        contract["labels"]
        ["english"]["sha256"]
        .lower()
    )

    actual_gloss_hash = sha256_file(
        GLOSS_PATH
    )

    actual_english_hash = sha256_file(
        ENGLISH_PATH
    )

    if (
        actual_gloss_hash
        != expected_gloss_hash
    ):
        raise RuntimeError(
            "label_no_order.txt hash changed "
            "since Stage 3."
        )

    if (
        actual_english_hash
        != expected_english_hash
    ):
        raise RuntimeError(
            "label_order.txt hash changed "
            "since Stage 3."
        )

    print("Gloss source hash:   PASS")
    print("English source hash: PASS")

    # ---------------------------------------------------------
    # Load labels
    # ---------------------------------------------------------

    gloss_lines = read_lines(
        GLOSS_PATH
    )

    english_lines = read_lines(
        ENGLISH_PATH
    )

    if len(gloss_lines) != 300:
        raise RuntimeError(
            f"Expected 300 gloss labels, "
            f"found {len(gloss_lines)}."
        )

    if len(english_lines) != 300:
        raise RuntimeError(
            f"Expected 300 English labels, "
            f"found {len(english_lines)}."
        )

    if any(
        not line.strip()
        for line in gloss_lines
    ):
        raise RuntimeError(
            "Blank gloss label detected."
        )

    if any(
        not line.strip()
        for line in english_lines
    ):
        raise RuntimeError(
            "Blank English label detected."
        )

    # ---------------------------------------------------------
    # Verify recording sentence IDs
    # ---------------------------------------------------------

    manifest = load_manifest()

    sentence_ids = sorted(
        {
            int(row["sentence_id"])
            for row in manifest
        }
    )

    expected_ids = list(
        range(1, 301)
    )

    if sentence_ids != expected_ids:
        raise RuntimeError(
            "Recording sentence IDs are not "
            "exactly 1..300."
        )

    # ---------------------------------------------------------
    # Freeze positional mapping:
    #
    # sentence_id n ↔ physical line n
    # ---------------------------------------------------------

    template_rows = []

    token_counter = Counter()

    lines_with_irregular_whitespace = []

    for sentence_id in range(
        1,
        301,
    ):
        line_index = (
            sentence_id - 1
        )

        gloss_raw = (
            gloss_lines[
                line_index
            ]
        )

        english_raw = (
            english_lines[
                line_index
            ]
        )

        gloss_tokens = (
            gloss_raw.split()
        )

        if not gloss_tokens:
            raise RuntimeError(
                f"Sentence {sentence_id} "
                f"has zero gloss tokens."
            )

        canonical_gloss = " ".join(
            gloss_tokens
        )

        if canonical_gloss != gloss_raw:
            lines_with_irregular_whitespace.append(
                sentence_id
            )

        token_counter.update(
            gloss_tokens
        )

        template_rows.append(
            {
                "sentence_id": sentence_id,
                "physical_line_number": (
                    sentence_id
                ),
                "gloss_raw": gloss_raw,
                "english_raw": english_raw,
                "gloss_token_count": (
                    len(gloss_tokens)
                ),
                "gloss_tokens_json": (
                    json.dumps(
                        gloss_tokens,
                        ensure_ascii=False,
                    )
                ),
            }
        )

    if BLANK_TOKEN in token_counter:
        raise RuntimeError(
            f"Reserved blank token "
            f"{BLANK_TOKEN!r} occurs in "
            f"the source gloss vocabulary."
        )

    # ---------------------------------------------------------
    # Fresh deterministic vocabulary
    # ---------------------------------------------------------

    nonblank_tokens = sorted(
        token_counter.keys()
    )

    token_to_id = {
        BLANK_TOKEN: BLANK_ID
    }

    for index, token in enumerate(
        nonblank_tokens,
        start=1,
    ):
        token_to_id[token] = index

    id_to_token = {
        str(index): token
        for token, index
        in token_to_id.items()
    }

    vocabulary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 10,
        "vocabulary_type": (
            "global_dataset_gloss_schema"
        ),
        "source": (
            "label_no_order.txt"
        ),
        "tokenization": (
            "whitespace_split_preserve_token_text"
        ),
        "blank_token": (
            BLANK_TOKEN
        ),
        "blank_id": (
            BLANK_ID
        ),
        "nonblank_token_count": (
            len(nonblank_tokens)
        ),
        "ctc_output_class_count": (
            len(nonblank_tokens) + 1
        ),
        "nonblank_tokens_sorted": (
            nonblank_tokens
        ),
        "reserved_tokens": {
            "unk": None,
            "bos": None,
            "eos": None,
            "pad": None,
        },
        "important_scope_note": (
            "This vocabulary defines the dataset "
            "label schema. Training-support analysis "
            "for LOSO and template holdout remains "
            "split-specific and is evaluated later."
        ),
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Add encoded targets
    # ---------------------------------------------------------

    for row in template_rows:
        tokens = json.loads(
            row[
                "gloss_tokens_json"
            ]
        )

        token_ids = [
            token_to_id[token]
            for token in tokens
        ]

        row[
            "gloss_token_ids_json"
        ] = json.dumps(
            token_ids
        )

    with CATALOG_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sentence_id",
                "physical_line_number",
                "gloss_raw",
                "english_raw",
                "gloss_token_count",
                "gloss_tokens_json",
                "gloss_token_ids_json",
            ],
        )

        writer.writeheader()
        writer.writerows(
            template_rows
        )

    VOCAB_PATH.write_text(
        json.dumps(
            vocabulary,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    TOKEN_TO_ID_PATH.write_text(
        json.dumps(
            token_to_id,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    ID_TO_TOKEN_PATH.write_text(
        json.dumps(
            id_to_token,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Hash generated contracts
    # ---------------------------------------------------------

    catalog_hash = sha256_file(
        CATALOG_PATH
    )

    vocabulary_hash = sha256_file(
        VOCAB_PATH
    )

    CATALOG_HASH_PATH.write_text(
        catalog_hash + "\n",
        encoding="utf-8",
    )

    VOCAB_HASH_PATH.write_text(
        vocabulary_hash + "\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Dataset-level label statistics
    # ---------------------------------------------------------

    gloss_lengths = [
        row[
            "gloss_token_count"
        ]
        for row in template_rows
    ]

    duplicate_gloss_strings = {
        text: count
        for text, count in Counter(
            gloss_lines
        ).items()
        if count > 1
    }

    duplicate_english_strings = {
        text: count
        for text, count in Counter(
            english_lines
        ).items()
        if count > 1
    }

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 10,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "source_integrity": {
            "label_no_order_sha256": (
                actual_gloss_hash
            ),
            "label_order_sha256": (
                actual_english_hash
            ),
            "stage3_hashes_match": True,
        },

        "mapping_contract": {
            "sentence_ids": (
                "1..300"
            ),
            "physical_label_lines": (
                "1..300"
            ),
            "mapping": (
                "sentence_id_n_maps_to_physical_line_n"
            ),
            "evidence_status": (
                "POSITIONAL_DATASET_CONVENTION"
            ),
            "explicit_mapping_file_present": False,
        },

        "templates": {
            "count": (
                len(template_rows)
            ),
            "minimum_gloss_tokens": (
                min(gloss_lengths)
            ),
            "maximum_gloss_tokens": (
                max(gloss_lengths)
            ),
            "mean_gloss_tokens": (
                sum(gloss_lengths)
                / len(gloss_lengths)
            ),
            "duplicate_gloss_string_count": (
                len(
                    duplicate_gloss_strings
                )
            ),
            "duplicate_english_string_count": (
                len(
                    duplicate_english_strings
                )
            ),
        },

        "tokenization": {
            "rule": (
                "split_on_whitespace"
            ),
            "underscores_preserved": True,
            "case_preserved": True,
            "punctuation_preserved": True,
            "irregular_whitespace_lines": (
                lines_with_irregular_whitespace
            ),
        },

        "vocabulary": {
            "blank_token": (
                BLANK_TOKEN
            ),
            "blank_id": (
                BLANK_ID
            ),
            "nonblank_tokens": (
                len(nonblank_tokens)
            ),
            "ctc_output_classes": (
                len(nonblank_tokens) + 1
            ),
            "unk_token": None,
            "bos_token": None,
            "eos_token": None,
            "pad_token": None,
        },

        "generated_contracts": {
            "sentence_catalog": (
                "artifacts/labels/"
                "stage10_sentence_targets.csv"
            ),
            "sentence_catalog_sha256": (
                catalog_hash
            ),
            "vocabulary": (
                "artifacts/labels/"
                "stage10_gloss_vocabulary.json"
            ),
            "vocabulary_sha256": (
                vocabulary_hash
            ),
        },

        "important_evaluation_rule": (
            "The global vocabulary defines the dataset "
            "output schema only. Whether a token or "
            "template is actually supported by a "
            "particular training split must be determined "
            "from that training split, not from the "
            "existence of the global vocabulary entry."
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
    # Human-readable record
    # ---------------------------------------------------------

    markdown = f"""# Stage 10 — Labels and Fresh CTC Vocabulary

## Source label files

Recognition/gloss target:

`label_no_order.txt`

English target:

`label_order.txt`

Both source hashes still match the immutable Stage 3 contract.

## Sentence mapping

The dataset uses recording sentence IDs `1..300`.

Both label files contain 300 non-empty physical lines.

The frozen operational convention is:

`sentence_id n -> physical label line n`

This is recorded as a positional dataset convention. No separate
author-provided ID-to-label mapping file is present in the project
source.

## Gloss tokenization

Gloss strings are split on whitespace.

Token spelling is otherwise preserved.

Underscores remain inside a token.

Example:

`whats_up`

is one token.

## CTC vocabulary

CTC blank token:

`{BLANK_TOKEN}`

CTC blank ID:

`{BLANK_ID}`

Nonblank gloss tokens:

{len(nonblank_tokens):,}

Total CTC output classes:

{len(nonblank_tokens) + 1:,}

No UNK, BOS, EOS or PAD token is added to the CTC vocabulary.

## Evaluation note

The vocabulary defines the global dataset label schema.

It does not imply that every token has training support in every
LOSO or template-holdout split.

Split-specific token support must be calculated from the corresponding
training partition later.

## Generated files

`artifacts/labels/stage10_sentence_targets.csv`

`artifacts/labels/stage10_gloss_vocabulary.json`

`artifacts/labels/stage10_token_to_id.json`

`artifacts/labels/stage10_id_to_token.json`

`artifacts/labels/stage10_label_vocabulary_summary.json`

`artifacts/labels/stage10_sentence_targets.sha256`

`artifacts/labels/stage10_gloss_vocabulary.sha256`
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("LABEL CONTRACT")
    print("-" * 78)

    print(
        "Templates:",
        len(template_rows),
    )

    print(
        "Sentence IDs:",
        "1..300",
    )

    print(
        "Mapping:",
        "sentence_id n -> physical line n",
    )

    print()
    print("GLOSS TARGET LENGTHS")
    print("-" * 78)

    print(
        "Minimum tokens:",
        min(gloss_lengths),
    )

    print(
        "Maximum tokens:",
        max(gloss_lengths),
    )

    print(
        "Mean tokens:",
        round(
            sum(gloss_lengths)
            / len(gloss_lengths),
            4,
        ),
    )

    print()
    print("VOCABULARY")
    print("-" * 78)

    print(
        "Blank token:",
        BLANK_TOKEN,
    )

    print(
        "Blank ID:",
        BLANK_ID,
    )

    print(
        "Nonblank tokens:",
        len(nonblank_tokens),
    )

    print(
        "CTC output classes:",
        len(nonblank_tokens) + 1,
    )

    print(
        "UNK token:",
        "NONE",
    )

    print(
        "BOS/EOS:",
        "NONE",
    )

    print()
    print("DUPLICATE LABEL STRINGS")
    print("-" * 78)

    print(
        "Duplicate gloss strings:",
        len(
            duplicate_gloss_strings
        ),
    )

    print(
        "Duplicate English strings:",
        len(
            duplicate_english_strings
        ),
    )

    print()
    print("GENERATED HASHES")
    print("-" * 78)

    print(
        "Sentence catalog:",
        catalog_hash,
    )

    print(
        "Vocabulary:",
        vocabulary_hash,
    )

    print()
    print(
        "Saved:",
        CATALOG_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        VOCAB_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        TOKEN_TO_ID_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        ID_TO_TOKEN_PATH.relative_to(
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

    stage10_pass = (
        len(template_rows) == 300
        and sentence_ids
        == list(range(1, 301))
        and BLANK_ID == 0
        and len(nonblank_tokens) > 0
        and len(token_to_id)
        == len(nonblank_tokens) + 1
        and token_to_id[
            BLANK_TOKEN
        ] == 0
        and len(set(
            token_to_id.values()
        ))
        == len(token_to_id)
    )

    if stage10_pass:
        print(
            "FINAL STATUS: "
            "STAGE 10 LABEL/VOCAB BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 10 LABEL/VOCAB BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

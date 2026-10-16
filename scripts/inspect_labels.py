from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

GLOSS_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gu_author_dataset"
    / "label_no_order.txt"
)

ENGLISH_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gu_author_dataset"
    / "label_order.txt"
)

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "raw_manifest.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage10_label_discovery.json"
)


def read_lines(path: Path) -> list[str]:
    return path.read_text(
        encoding="utf-8-sig"
    ).splitlines()


def main() -> None:
    print("=" * 78)
    print("STAGE 10A — LABEL DISCOVERY")
    print("=" * 78)

    gloss_lines = read_lines(
        GLOSS_PATH
    )

    english_lines = read_lines(
        ENGLISH_PATH
    )

    manifest = pd.read_csv(
        MANIFEST_PATH
    )

    sentence_ids = sorted(
        int(x)
        for x in manifest[
            "sentence_id"
        ].unique()
    )

    if not sentence_ids:
        raise RuntimeError(
            "No sentence IDs found."
        )

    expected_range = list(
        range(
            sentence_ids[0],
            sentence_ids[-1] + 1,
        )
    )

    missing_ids = [
        x
        for x in expected_range
        if x not in sentence_ids
    ]

    counts = Counter(
        int(x)
        for x in manifest[
            "sentence_id"
        ]
    )

    gloss_blank = sum(
        not line.strip()
        for line in gloss_lines
    )

    english_blank = sum(
        not line.strip()
        for line in english_lines
    )

    report = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": "10A",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "label_files": {
            "gloss": (
                "data/raw/gu_author_dataset/"
                "label_no_order.txt"
            ),
            "english": (
                "data/raw/gu_author_dataset/"
                "label_order.txt"
            ),
        },

        "line_counts": {
            "gloss": len(
                gloss_lines
            ),
            "english": len(
                english_lines
            ),
            "same_count": (
                len(gloss_lines)
                == len(english_lines)
            ),
        },

        "blank_lines": {
            "gloss": gloss_blank,
            "english": english_blank,
        },

        "recording_sentence_ids": {
            "unique_count": len(
                sentence_ids
            ),
            "minimum": min(
                sentence_ids
            ),
            "maximum": max(
                sentence_ids
            ),
            "missing_within_min_max": (
                missing_ids
            ),
        },

        "recording_support": {
            "minimum_recordings_per_sentence": (
                min(counts.values())
            ),
            "maximum_recordings_per_sentence": (
                max(counts.values())
            ),
        },

        "first_five_pairs": [
            {
                "physical_line_number": (
                    i + 1
                ),
                "gloss": gloss_lines[i],
                "english": english_lines[i],
            }
            for i in range(
                min(
                    5,
                    len(gloss_lines),
                    len(english_lines),
                )
            )
        ],

        "important_decision": (
            "No vocabulary IDs are generated "
            "until label-file indexing and "
            "sentence-ID correspondence are "
            "verified."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    OUTPUT_PATH.write_text(
        json.dumps(
            report,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    print()
    print(
        "Gloss lines:",
        len(gloss_lines),
    )

    print(
        "English lines:",
        len(english_lines),
    )

    print(
        "Same line count:",
        len(gloss_lines)
        == len(english_lines),
    )

    print(
        "Gloss blank lines:",
        gloss_blank,
    )

    print(
        "English blank lines:",
        english_blank,
    )

    print()
    print(
        "Unique recording sentence IDs:",
        len(sentence_ids),
    )

    print(
        "Minimum sentence ID:",
        min(sentence_ids),
    )

    print(
        "Maximum sentence ID:",
        max(sentence_ids),
    )

    print(
        "Missing IDs inside range:",
        len(missing_ids),
    )

    if missing_ids:
        print(
            "Missing IDs:",
            missing_ids,
        )

    print()
    print(
        "Minimum recordings/template:",
        min(counts.values()),
    )

    print(
        "Maximum recordings/template:",
        max(counts.values()),
    )

    print()
    print("FIRST FIVE PAIRS")
    print("-" * 78)

    for i in range(
        min(
            5,
            len(gloss_lines),
            len(english_lines),
        )
    ):
        print(
            f"{i + 1:>3}: "
            f"GLOSS   = {gloss_lines[i]}"
        )

        print(
            "     "
            f"ENGLISH = {english_lines[i]}"
        )

    print()
    print(
        "Saved:",
        OUTPUT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print()
    print(
        "FINAL STATUS: "
        "STAGE 10A DISCOVERY COMPLETE"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()

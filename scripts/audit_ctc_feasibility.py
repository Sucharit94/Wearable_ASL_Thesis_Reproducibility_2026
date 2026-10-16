from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


from src.evaluation.ctc_feasibility import (
    adjacent_repeat_count,
    full_window_count,
    minimum_ctc_timesteps,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CACHE_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.csv"
)

CACHE_INDEX_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.sha256"
)

TARGET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

TARGET_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.sha256"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
)

SAMPLE_AUDIT_PATH = (
    OUTPUT_DIR
    / "stage20_ctc_sample_audit.csv"
)

STRIDE_SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage20_stride_feasibility.csv"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage20_ctc_feasibility_contract.json"
)

AUDIT_HASH_PATH = (
    OUTPUT_DIR
    / "stage20_ctc_sample_audit.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_20_CTC_FEASIBILITY.md"
)


REFERENCE_WINDOW_LENGTH = 36
REFERENCE_STRIDE = 18

CANDIDATE_STRIDES = list(
    range(
        REFERENCE_STRIDE,
        0,
        -1,
    )
)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        while True:

            chunk = handle.read(
                1024 * 1024
            )

            if not chunk:
                break

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_recorded_hash(
    path: Path,
) -> str:

    value = (
        path.read_text(
            encoding="utf-8-sig"
        )
        .strip()
        .split()[0]
        .lower()
    )

    if (
        len(value) != 64
        or any(
            character not in
            "0123456789abcdef"
            for character in value
        )
    ):

        raise RuntimeError(
            f"Invalid SHA-256 record: {path}"
        )

    return value


def verify_hash(
    file_path: Path,
    hash_path: Path,
) -> str:

    expected = read_recorded_hash(
        hash_path
    )

    actual = sha256_file(
        file_path
    )

    if expected != actual:

        raise RuntimeError(
            f"Hash mismatch: {file_path}"
        )

    return actual


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


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 20 — CTC FEASIBILITY ENGINE"
    )
    print("=" * 78)

    upstream_hashes = {
        "stage17_cache_index": (
            verify_hash(
                CACHE_INDEX_PATH,
                CACHE_INDEX_HASH_PATH,
            )
        ),

        "stage10_targets": (
            verify_hash(
                TARGET_PATH,
                TARGET_HASH_PATH,
            )
        ),
    }

    index_rows = load_csv(
        CACHE_INDEX_PATH
    )

    target_rows = load_csv(
        TARGET_PATH
    )

    if len(index_rows) != 7400:

        raise RuntimeError(
            f"Expected 7,400 recordings, "
            f"found {len(index_rows)}."
        )

    if len(target_rows) != 300:

        raise RuntimeError(
            f"Expected 300 target rows, "
            f"found {len(target_rows)}."
        )

    target_by_sentence = {}

    for row in target_rows:

        sentence_id = int(
            row[
                "sentence_id"
            ]
        )

        target_ids = tuple(
            int(value)
            for value in json.loads(
                row[
                    "gloss_token_ids_json"
                ]
            )
        )

        target_by_sentence[
            sentence_id
        ] = {
            "target_ids": (
                target_ids
            ),

            "gloss_raw": (
                row[
                    "gloss_raw"
                ]
            ),

            "target_length": (
                len(
                    target_ids
                )
            ),

            "adjacent_repeats": (
                adjacent_repeat_count(
                    target_ids
                )
            ),

            "minimum_ctc_timesteps": (
                minimum_ctc_timesteps(
                    target_ids
                )
            ),
        }

    if sorted(
        target_by_sentence
    ) != list(
        range(
            1,
            301,
        )
    ):

        raise RuntimeError(
            "Sentence target IDs are not 1..300."
        )

    maximum_target_length = max(
        target[
            "target_length"
        ]
        for target
        in target_by_sentence.values()
    )

    maximum_adjacent_repeats = max(
        target[
            "adjacent_repeats"
        ]
        for target
        in target_by_sentence.values()
    )

    maximum_minimum_ctc = max(
        target[
            "minimum_ctc_timesteps"
        ]
        for target
        in target_by_sentence.values()
    )

    # ---------------------------------------------------------
    # Build base per-recording records.
    # ---------------------------------------------------------

    base_rows = []

    direct_infeasible = 0

    for row in index_rows:

        sample_id = row[
            "sample_id"
        ]

        subject = row[
            "subject"
        ]

        sentence_id = int(
            row[
                "sentence_id"
            ]
        )

        input_length = int(
            row[
                "timesteps"
            ]
        )

        target = target_by_sentence[
            sentence_id
        ]

        minimum_required = (
            target[
                "minimum_ctc_timesteps"
            ]
        )

        direct_margin = (
            input_length
            - minimum_required
        )

        direct_feasible = (
            direct_margin >= 0
        )

        if not direct_feasible:
            direct_infeasible += 1

        base_rows.append(
            {
                "sample_id": (
                    sample_id
                ),

                "subject": (
                    subject
                ),

                "sentence_id": (
                    sentence_id
                ),

                "input_timesteps": (
                    input_length
                ),

                "gloss_raw": (
                    target[
                        "gloss_raw"
                    ]
                ),

                "target_length": (
                    target[
                        "target_length"
                    ]
                ),

                "adjacent_repeat_pairs": (
                    target[
                        "adjacent_repeats"
                    ]
                ),

                "minimum_ctc_timesteps": (
                    minimum_required
                ),

                "direct_margin": (
                    direct_margin
                ),

                "direct_feasible": (
                    direct_feasible
                ),
            }
        )

    if direct_infeasible != 0:

        raise RuntimeError(
            "At least one raw T x 38 sequence is "
            "CTC-infeasible before model-specific "
            "downsampling."
        )

    # ---------------------------------------------------------
    # Evaluate every candidate stride.
    # ---------------------------------------------------------

    stride_summaries = []

    for stride in CANDIDATE_STRIDES:

        infeasible = 0
        minimum_margin = None
        maximum_margin = None
        window_counts = []

        infeasible_by_subject = Counter()

        for row in base_rows:

            windows = full_window_count(
                row[
                    "input_timesteps"
                ],
                window_length=(
                    REFERENCE_WINDOW_LENGTH
                ),
                stride=stride,
            )

            margin = (
                windows
                - row[
                    "minimum_ctc_timesteps"
                ]
            )

            window_counts.append(
                windows
            )

            if (
                minimum_margin is None
                or margin
                < minimum_margin
            ):
                minimum_margin = margin

            if (
                maximum_margin is None
                or margin
                > maximum_margin
            ):
                maximum_margin = margin

            if margin < 0:

                infeasible += 1

                infeasible_by_subject[
                    row[
                        "subject"
                    ]
                ] += 1

        stride_summaries.append(
            {
                "window_length": (
                    REFERENCE_WINDOW_LENGTH
                ),

                "stride": (
                    stride
                ),

                "recordings": (
                    len(
                        base_rows
                    )
                ),

                "infeasible_recordings": (
                    infeasible
                ),

                "feasible_recordings": (
                    len(
                        base_rows
                    )
                    - infeasible
                ),

                "minimum_output_windows": (
                    min(
                        window_counts
                    )
                ),

                "maximum_output_windows": (
                    max(
                        window_counts
                    )
                ),

                "minimum_ctc_margin": (
                    minimum_margin
                ),

                "maximum_ctc_margin": (
                    maximum_margin
                ),

                "subject1_infeasible": (
                    infeasible_by_subject[
                        "subject1"
                    ]
                ),

                "subject2_infeasible": (
                    infeasible_by_subject[
                        "subject2"
                    ]
                ),

                "subject3_infeasible": (
                    infeasible_by_subject[
                        "subject3"
                    ]
                ),
            }
        )

    safe_strides = [
        int(
            row[
                "stride"
            ]
        )
        for row in stride_summaries
        if int(
            row[
                "infeasible_recordings"
            ]
        ) == 0
    ]

    if not safe_strides:

        raise RuntimeError(
            "No stride from 1..18 produces "
            "complete CTC feasibility with "
            "window length 36."
        )

    selected_stride = max(
        safe_strides
    )

    reference_summary = next(
        row
        for row in stride_summaries
        if int(
            row[
                "stride"
            ]
        ) == REFERENCE_STRIDE
    )

    selected_summary = next(
        row
        for row in stride_summaries
        if int(
            row[
                "stride"
            ]
        ) == selected_stride
    )

    # ---------------------------------------------------------
    # Final per-recording audit with reference + selected.
    # ---------------------------------------------------------

    audit_rows = []

    selected_failures = 0

    reference_failures = 0

    for row in sorted(
        base_rows,
        key=lambda item: (
            item[
                "sample_id"
            ]
        ),
    ):

        reference_windows = (
            full_window_count(
                row[
                    "input_timesteps"
                ],
                window_length=(
                    REFERENCE_WINDOW_LENGTH
                ),
                stride=(
                    REFERENCE_STRIDE
                ),
            )
        )

        selected_windows = (
            full_window_count(
                row[
                    "input_timesteps"
                ],
                window_length=(
                    REFERENCE_WINDOW_LENGTH
                ),
                stride=(
                    selected_stride
                ),
            )
        )

        reference_margin = (
            reference_windows
            - row[
                "minimum_ctc_timesteps"
            ]
        )

        selected_margin = (
            selected_windows
            - row[
                "minimum_ctc_timesteps"
            ]
        )

        reference_feasible = (
            reference_margin >= 0
        )

        selected_feasible = (
            selected_margin >= 0
        )

        if not reference_feasible:
            reference_failures += 1

        if not selected_feasible:
            selected_failures += 1

        audit_rows.append(
            {
                **row,

                "window_length": (
                    REFERENCE_WINDOW_LENGTH
                ),

                "reference_stride": (
                    REFERENCE_STRIDE
                ),

                "reference_window_count": (
                    reference_windows
                ),

                "reference_ctc_margin": (
                    reference_margin
                ),

                "reference_feasible": (
                    reference_feasible
                ),

                "selected_stride": (
                    selected_stride
                ),

                "selected_window_count": (
                    selected_windows
                ),

                "selected_ctc_margin": (
                    selected_margin
                ),

                "selected_feasible": (
                    selected_feasible
                ),
            }
        )

    if selected_failures != 0:

        raise RuntimeError(
            "Selected stride still contains "
            "CTC-infeasible recordings."
        )

    if (
        reference_failures
        != int(
            reference_summary[
                "infeasible_recordings"
            ]
        )
    ):

        raise RuntimeError(
            "Reference failure count mismatch."
        )

    # ---------------------------------------------------------
    # Save outputs.
    # ---------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SAMPLE_AUDIT_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                audit_rows[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            audit_rows
        )

    with STRIDE_SUMMARY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(
                stride_summaries[
                    0
                ].keys()
            ),
        )

        writer.writeheader()

        writer.writerows(
            stride_summaries
        )

    audit_hash = sha256_file(
        SAMPLE_AUDIT_PATH
    )

    AUDIT_HASH_PATH.write_text(
        audit_hash + "\n",
        encoding="utf-8",
    )

    selection_reason = (
        "reference_stride_is_safe"
        if selected_stride
        == REFERENCE_STRIDE
        else
        "reference_stride_is_infeasible_"
        "so_largest_globally_safe_stride_selected"
    )

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": (
            20
        ),

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "ctc_minimum_length_rule": (
            "target_length + "
            "adjacent_repeated_target_pairs"
        ),

        "recordings": (
            len(
                audit_rows
            )
        ),

        "targets": {
            "sentence_templates": (
                len(
                    target_by_sentence
                )
            ),

            "maximum_target_length": (
                maximum_target_length
            ),

            "maximum_adjacent_repeat_pairs": (
                maximum_adjacent_repeats
            ),

            "maximum_minimum_ctc_timesteps": (
                maximum_minimum_ctc
            ),
        },

        "direct_sequence": {
            "interpretation": (
                "Raw source-level feasibility only; "
                "model-specific output downsampling "
                "is checked later."
            ),

            "infeasible_recordings": (
                direct_infeasible
            ),
        },

        "windowing": {
            "window_definition": (
                "full_non_padded_sliding_windows"
            ),

            "formula": (
                "1 + floor((T - W) / S) "
                "for T >= W; otherwise 0"
            ),

            "window_length": (
                REFERENCE_WINDOW_LENGTH
            ),

            "reference_stride": (
                REFERENCE_STRIDE
            ),

            "reference_infeasible_recordings": (
                reference_failures
            ),

            "selected_stride": (
                selected_stride
            ),

            "selected_infeasible_recordings": (
                selected_failures
            ),

            "selection_rule": (
                "largest integer stride <= 18 "
                "with zero infeasible recordings "
                "across all 7,400 samples"
            ),

            "selection_reason": (
                selection_reason
            ),

            "applies_to": [
                "G1 near-Gu window adapter",
                "C4 Tiny-Swin temporal-feature window adapter",
            ],
        },

        "reference_summary": (
            reference_summary
        ),

        "selected_summary": (
            selected_summary
        ),

        "sample_policy": {
            "silently_drop_infeasible_samples": (
                False
            ),

            "training_sample_count_preserved": (
                True
            ),
        },

        "hashes": {
            "upstream": (
                upstream_hashes
            ),

            "sample_audit_sha256": (
                audit_hash
            ),
        },
    }

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 20 — CTC Feasibility

## Minimum CTC length

For target sequence `y`, the minimum CTC temporal length is:

`target length + number of adjacent repeated target pairs`

This accounts for the blank required to separate identical adjacent
target labels.

## Dataset

Recordings audited:

{len(audit_rows):,}

Sentence templates:

{len(target_by_sentence)}

Maximum target-token length:

{maximum_target_length}

Maximum adjacent repeated-pair count:

{maximum_adjacent_repeats}

Maximum minimum CTC requirement:

{maximum_minimum_ctc}

## Direct source sequences

CTC-infeasible raw source sequences:

{direct_infeasible}

This is only source-level feasibility.

Later model-specific downsampling must still propagate valid output
lengths exactly.

## Windowed G1/C4 input

Window length:

{REFERENCE_WINDOW_LENGTH}

Reference stride:

{REFERENCE_STRIDE}

Reference-stride infeasible recordings:

{reference_failures}

## Frozen selected stride

Selected stride:

{selected_stride}

Selected-stride infeasible recordings:

{selected_failures}

Selection rule:

Choose the largest integer stride not greater than 18 that provides
zero CTC-infeasible recordings over all 7,400 samples.

## Scientific policy

No recording may be silently removed merely because a temporal
windowing configuration is CTC-infeasible.

The Stage-20 selected window rule is therefore frozen before model
training.

## Output-length boundary

Stage 20 freezes input/window-level feasibility.

Stage 22 will propagate exact valid lengths through each implemented
model architecture.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("TARGET REQUIREMENTS")
    print("-" * 78)

    print(
        "Recordings:",
        len(
            audit_rows
        ),
    )

    print(
        "Sentence templates:",
        len(
            target_by_sentence
        ),
    )

    print(
        "Maximum target length:",
        maximum_target_length,
    )

    print(
        "Maximum adjacent repeat pairs:",
        maximum_adjacent_repeats,
    )

    print(
        "Maximum minimum CTC timesteps:",
        maximum_minimum_ctc,
    )

    print()
    print("DIRECT SOURCE FEASIBILITY")
    print("-" * 78)

    print(
        "Infeasible recordings:",
        direct_infeasible,
    )

    print()
    print("REFERENCE WINDOWING")
    print("-" * 78)

    print(
        "Window length:",
        REFERENCE_WINDOW_LENGTH,
    )

    print(
        "Reference stride:",
        REFERENCE_STRIDE,
    )

    print(
        "Reference infeasible recordings:",
        reference_failures,
    )

    print(
        "Reference minimum CTC margin:",
        reference_summary[
            "minimum_ctc_margin"
        ],
    )

    print()
    print("STRIDE FEASIBILITY")
    print("-" * 78)

    for row in stride_summaries:

        print(
            f"stride={int(row['stride']):>2} | "
            f"infeasible="
            f"{int(row['infeasible_recordings']):>4} | "
            f"min_windows="
            f"{int(row['minimum_output_windows']):>3} | "
            f"min_margin="
            f"{int(row['minimum_ctc_margin']):>3}"
        )

    print()
    print("FROZEN WINDOW DECISION")
    print("-" * 78)

    print(
        "Selected stride:",
        selected_stride,
    )

    print(
        "Selected infeasible recordings:",
        selected_failures,
    )

    print(
        "Selection reason:",
        selection_reason,
    )

    print()
    print("FROZEN SAMPLE-AUDIT HASH")
    print("-" * 78)

    print(
        audit_hash
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    for path in [
        SAMPLE_AUDIT_PATH,
        STRIDE_SUMMARY_PATH,
        CONTRACT_PATH,
        AUDIT_HASH_PATH,
        DOC_PATH,
    ]:

        print(
            "Saved:",
            path.relative_to(
                PROJECT_ROOT
            ),
        )

    stage20_pass = all(
        [
            len(
                audit_rows
            ) == 7400,

            direct_infeasible == 0,

            selected_failures == 0,

            1 <= selected_stride <= 18,

            int(
                selected_summary[
                    "infeasible_recordings"
                ]
            ) == 0,
        ]
    )

    print()

    if stage20_pass:
        print(
            "FINAL STATUS: "
            "STAGE 20 BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 20 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

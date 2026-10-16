from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

from src.evaluation.metrics import (
    score_batch,
    score_sequence,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TARGET_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "labels"
    / "stage10_sentence_targets.csv"
)

STAGE21_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
    / "stage21_ctc_loss_decoder_contract.json"
)

STAGE22_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "ctc"
    / "stage22_sequence_length_contract.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "metrics"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage23_recognition_metrics_contract.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_23_RECOGNITION_METRICS.md"
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


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 23 — RECOGNITION METRICS "
        "AND ERROR ACCOUNTING"
    )
    print("=" * 78)

    stage21 = json.loads(
        STAGE21_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage22 = json.loads(
        STAGE22_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    decoder_ignores_padding = (
        stage21[
            "decoder"
        ][
            "padded_timesteps_decoded"
        ]
        is False
    )

    same_length_for_loss_decoder = (
        stage22[
            "principles"
        ][
            "same_final_length_used_for_ctc_loss_and_decoder"
        ]
        is True
    )

    if not decoder_ignores_padding:
        raise RuntimeError(
            "Stage-21 decoder dependency failed."
        )

    if not same_length_for_loss_decoder:
        raise RuntimeError(
            "Stage-22 length dependency failed."
        )

    with TARGET_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:

        rows = list(
            csv.DictReader(handle)
        )

    if len(rows) != 300:
        raise RuntimeError(
            "Expected 300 frozen sentence targets."
        )

    targets = []

    for row in rows:

        target = tuple(
            int(value)
            for value in json.loads(
                row[
                    "gloss_token_ids_json"
                ]
            )
        )

        if not target:
            raise RuntimeError(
                "Empty frozen target found."
            )

        if min(target) <= 0:
            raise RuntimeError(
                "Blank, padding or negative ID "
                "found in a frozen target."
            )

        if max(target) >= 456:
            raise RuntimeError(
                "Frozen target ID exceeds the "
                "456-class output schema."
            )

        targets.append(
            target
        )

    perfect = score_batch(
        targets,
        targets,
    )

    perfect_aggregate = perfect[
        "aggregate"
    ]

    if (
        perfect_aggregate[
            "total_edits"
        ]
        != 0
    ):
        raise RuntimeError(
            "Perfect-control metric test failed."
        )

    synthetic_references = [
        (
            1,
            2,
        ),
        (
            3,
            4,
            5,
        ),
        (
            6,
            7,
            8,
        ),
        (
            9,
            10,
            11,
        ),
    ]

    synthetic_hypotheses = [
        (
            1,
            2,
        ),
        (
            3,
            99,
            5,
        ),
        (
            6,
            8,
        ),
        (
            9,
            12,
            10,
            11,
        ),
    ]

    synthetic = score_batch(
        synthetic_references,
        synthetic_hypotheses,
    )

    synthetic_aggregate = synthetic[
        "aggregate"
    ]

    tie_probe = score_sequence(
        (
            1,
            2,
        ),
        (
            2,
            1,
        ),
    )

    expected_synthetic = {
        "reference_tokens": 11,
        "substitutions": 1,
        "deletions": 1,
        "insertions": 1,
        "total_edits": 3,
        "sentence_errors": 3,
    }

    for key, expected in (
        expected_synthetic.items()
    ):

        observed = int(
            synthetic_aggregate[
                key
            ]
        )

        if observed != expected:

            raise RuntimeError(
                "Synthetic metric mismatch for "
                f"{key}: {observed} != {expected}."
            )

    if not math.isclose(
        synthetic_aggregate[
            "pooled_wer"
        ],
        3 / 11,
        rel_tol=0,
        abs_tol=1e-12,
    ):
        raise RuntimeError(
            "Synthetic pooled WER mismatch."
        )

    if tie_probe[
        "substitutions"
    ] != 2:
        raise RuntimeError(
            "Metric tie-breaking policy changed."
        )

    implementation_hashes = {
        "metrics": sha256_file(
            PROJECT_ROOT
            / "src"
            / "evaluation"
            / "metrics.py"
        ),
        "tests": sha256_file(
            PROJECT_ROOT
            / "tests"
            / "test_recognition_metrics.py"
        ),
        "verifier": sha256_file(
            PROJECT_ROOT
            / "scripts"
            / "verify_recognition_metrics.py"
        ),
    }

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 23,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "metric_definitions": {
            "unit": "gloss_token_id",
            "pooled_wer": (
                "sum(S+D+I)/sum(reference_tokens)"
            ),
            "macro_sample_wer_is_headline": False,
            "ser": (
                "sentence_errors/number_of_samples"
            ),
            "exact_sentence_accuracy": (
                "1-SER"
            ),
            "word_accuracy_field": (
                "one_minus_wer"
            ),
            "one_minus_wer_may_be_negative": True,
        },
        "alignment": {
            "algorithm": (
                "token_level_levenshtein"
            ),
            "precedence": [
                "exact_match",
                "substitution",
                "deletion",
                "insertion",
            ],
            "tie_probe_substitutions": (
                tie_probe[
                    "substitutions"
                ]
            ),
            "per_sample_alignment_saved": True,
            "per_sample_sdi_saved": True,
        },
        "perfect_control": {
            "sentence_templates": len(
                targets
            ),
            "reference_tokens": (
                perfect_aggregate[
                    "reference_tokens"
                ]
            ),
            "total_edits": (
                perfect_aggregate[
                    "total_edits"
                ]
            ),
            "pooled_wer": (
                perfect_aggregate[
                    "pooled_wer"
                ]
            ),
            "ser": (
                perfect_aggregate[
                    "ser"
                ]
            ),
            "exact_sentence_accuracy": (
                perfect_aggregate[
                    "exact_sentence_accuracy"
                ]
            ),
        },
        "synthetic": {
            "samples": (
                synthetic_aggregate[
                    "samples"
                ]
            ),
            "reference_tokens": (
                synthetic_aggregate[
                    "reference_tokens"
                ]
            ),
            "substitutions": (
                synthetic_aggregate[
                    "substitutions"
                ]
            ),
            "deletions": (
                synthetic_aggregate[
                    "deletions"
                ]
            ),
            "insertions": (
                synthetic_aggregate[
                    "insertions"
                ]
            ),
            "total_edits": (
                synthetic_aggregate[
                    "total_edits"
                ]
            ),
            "pooled_wer": (
                synthetic_aggregate[
                    "pooled_wer"
                ]
            ),
            "sentence_errors": (
                synthetic_aggregate[
                    "sentence_errors"
                ]
            ),
            "ser": (
                synthetic_aggregate[
                    "ser"
                ]
            ),
            "exact_sentence_accuracy": (
                synthetic_aggregate[
                    "exact_sentence_accuracy"
                ]
            ),
        },
        "dependencies": {
            "stage21_decoder_ignores_padding": (
                decoder_ignores_padding
            ),
            "stage22_same_length_for_loss_decoder": (
                same_length_for_loss_decoder
            ),
            "stage22_direct_length_mismatches": (
                stage22[
                    "direct_sequence"
                ][
                    "mismatches"
                ]
            ),
            "stage22_window_infeasible": (
                stage22[
                    "g1_c4_window_adapter"
                ][
                    "infeasible_recordings"
                ]
            ),
        },
        "hashes": implementation_hashes,
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    CONTRACT_PATH.write_text(
        json.dumps(
            contract,
            indent=2,
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 23 — Recognition Metrics

## Headline recognition metric

Pooled word error rate is calculated as:

`sum(S + D + I) / sum(reference gloss tokens)`

The unweighted average of per-sample WER values is retained only as a
diagnostic and is not the headline result.

## Sentence metrics

Sentence error rate:

`sentences containing at least one error / number of sentences`

Exact sentence accuracy:

`1 - sentence error rate`

## Alignment

Token-level Levenshtein alignment uses the frozen tie-breaking order:

1. exact match;
2. substitution;
3. deletion;
4. insertion.

This makes substitution, deletion and insertion reporting reproducible.

## Perfect frozen-target control

Sentence templates:

{len(targets)}

Reference tokens:

{perfect_aggregate["reference_tokens"]}

Total errors:

{perfect_aggregate["total_edits"]}

Pooled WER:

{perfect_aggregate["pooled_wer"]}

Exact sentence accuracy:

{perfect_aggregate["exact_sentence_accuracy"]}

## Synthetic controlled-error test

Substitutions:

{synthetic_aggregate["substitutions"]}

Deletions:

{synthetic_aggregate["deletions"]}

Insertions:

{synthetic_aggregate["insertions"]}

Pooled WER:

{synthetic_aggregate["pooled_wer"]}

SER:

{synthetic_aggregate["ser"]}

Exact sentence accuracy:

{synthetic_aggregate["exact_sentence_accuracy"]}

## Dependency protection

Stage-21 padded outputs excluded:

{decoder_ignores_padding}

Stage-22 same final length supplied to loss and decoder:

{same_length_for_loss_decoder}
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("METRIC DEFINITIONS")
    print("-" * 78)
    print(
        "Pooled WER:",
        "sum(S+D+I)/sum(reference tokens)",
    )
    print(
        "SER:",
        "sentence errors / samples",
    )
    print(
        "Exact sentence accuracy:",
        "1 - SER",
    )
    print(
        "Macro sample WER headline:",
        False,
    )

    print()
    print("ALIGNMENT POLICY")
    print("-" * 78)
    print(
        "Algorithm:",
        "token-level Levenshtein",
    )
    print(
        "Tie order:",
        "match -> substitution -> deletion -> insertion",
    )
    print(
        "Tie-probe substitutions:",
        tie_probe[
            "substitutions"
        ],
    )

    print()
    print("PERFECT TARGET CONTROL")
    print("-" * 78)
    print(
        "Sentence templates:",
        len(targets),
    )
    print(
        "Reference tokens:",
        perfect_aggregate[
            "reference_tokens"
        ],
    )
    print(
        "Total edits:",
        perfect_aggregate[
            "total_edits"
        ],
    )
    print(
        "Pooled WER:",
        perfect_aggregate[
            "pooled_wer"
        ],
    )
    print(
        "SER:",
        perfect_aggregate[
            "ser"
        ],
    )
    print(
        "Exact sentence accuracy:",
        perfect_aggregate[
            "exact_sentence_accuracy"
        ],
    )

    print()
    print("SYNTHETIC ERROR ACCOUNTING")
    print("-" * 78)
    print(
        "Substitutions:",
        synthetic_aggregate[
            "substitutions"
        ],
    )
    print(
        "Deletions:",
        synthetic_aggregate[
            "deletions"
        ],
    )
    print(
        "Insertions:",
        synthetic_aggregate[
            "insertions"
        ],
    )
    print(
        "Pooled WER:",
        synthetic_aggregate[
            "pooled_wer"
        ],
    )
    print(
        "SER:",
        synthetic_aggregate[
            "ser"
        ],
    )
    print(
        "Exact sentence accuracy:",
        synthetic_aggregate[
            "exact_sentence_accuracy"
        ],
    )

    print()
    print("DEPENDENCIES")
    print("-" * 78)
    print(
        "Stage-21 decoder ignores padding:",
        decoder_ignores_padding,
    )
    print(
        "Stage-22 same loss/decode length:",
        same_length_for_loss_decoder,
    )
    print(
        "Stage-22 direct mismatches:",
        stage22[
            "direct_sequence"
        ][
            "mismatches"
        ],
    )
    print(
        "Stage-22 window infeasible:",
        stage22[
            "g1_c4_window_adapter"
        ][
            "infeasible_recordings"
        ],
    )

    print()
    print("OUTPUTS")
    print("-" * 78)
    print(
        "Saved:",
        CONTRACT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )
    print(
        "Saved:",
        DOC_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    stage23_pass = all(
        [
            len(targets) == 300,
            perfect_aggregate[
                "total_edits"
            ] == 0,
            perfect_aggregate[
                "pooled_wer"
            ] == 0,
            perfect_aggregate[
                "ser"
            ] == 0,
            perfect_aggregate[
                "exact_sentence_accuracy"
            ] == 1,
            synthetic_aggregate[
                "substitutions"
            ] == 1,
            synthetic_aggregate[
                "deletions"
            ] == 1,
            synthetic_aggregate[
                "insertions"
            ] == 1,
            math.isclose(
                synthetic_aggregate[
                    "pooled_wer"
                ],
                3 / 11,
                rel_tol=0,
                abs_tol=1e-12,
            ),
            decoder_ignores_padding,
            same_length_for_loss_decoder,
        ]
    )

    print()

    if stage23_pass:
        print(
            "FINAL STATUS: "
            "STAGE 23 BUILD PASSED"
        )
    else:
        print(
            "FINAL STATUS: "
            "STAGE 23 BUILD FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage5_recording_integrity.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
)

FIGURE_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "figures"
)

SUMMARY_JSON = (
    OUTPUT_DIR
    / "stage6_sequence_length_summary.json"
)

SUBJECT_SUMMARY_CSV = (
    OUTPUT_DIR
    / "stage6_subject_length_summary.csv"
)

SHORTEST_CSV = (
    OUTPUT_DIR
    / "stage6_shortest_recordings.csv"
)

LONGEST_CSV = (
    OUTPUT_DIR
    / "stage6_longest_recordings.csv"
)

THRESHOLD_CSV = (
    OUTPUT_DIR
    / "stage6_length_threshold_counts.csv"
)

HISTOGRAM_PATH = (
    FIGURE_DIR
    / "stage6_length_histogram.png"
)

BOXPLOT_PATH = (
    FIGURE_DIR
    / "stage6_length_boxplot_by_subject.png"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_6_SEQUENCE_LENGTH_AUDIT.md"
)


PERCENTILES = [
    1,
    5,
    10,
    25,
    50,
    75,
    90,
    95,
    99,
]

THRESHOLDS = [
    100,
    150,
    200,
    250,
    300,
    400,
    500,
    600,
    700,
    800,
]


def describe_lengths(values: np.ndarray) -> dict:
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    result = {
        "count": int(values.size),
        "min": int(values.min()),
        "max": int(values.max()),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "std_population": float(
            values.std(ddof=0)
        ),
        "std_sample": float(
            values.std(ddof=1)
        ),
    }

    percentile_values = np.percentile(
        values,
        PERCENTILES,
    )

    for p, value in zip(
        PERCENTILES,
        percentile_values,
    ):
        result[f"p{p:02d}"] = float(value)

    return result


def main() -> None:
    print("=" * 78)
    print("STAGE 6 — SEQUENCE-LENGTH AUDIT")
    print("=" * 78)

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"Missing Stage 5 input: {INPUT_PATH}"
        )

    df = pd.read_csv(INPUT_PATH)

    if len(df) != 7400:
        raise RuntimeError(
            f"Expected 7,400 rows, found {len(df):,}."
        )

    if (
        df["integrity_status"]
        .ne("PASS")
        .any()
    ):
        raise RuntimeError(
            "Stage 5 contains failed recordings."
        )

    if df["rows"].isna().any():
        raise RuntimeError(
            "Missing sequence lengths detected."
        )

    df["rows"] = df["rows"].astype(int)

    if (df["rows"] <= 0).any():
        raise RuntimeError(
            "Non-positive sequence length detected."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    lengths = df["rows"].to_numpy()

    overall = describe_lengths(lengths)

    subject_rows = []
    subject_summary = {}

    for subject, group in (
        df.groupby("subject", sort=True)
    ):
        stats = describe_lengths(
            group["rows"].to_numpy()
        )

        subject_summary[subject] = stats

        subject_rows.append(
            {
                "subject": subject,
                **stats,
            }
        )

    pd.DataFrame(
        subject_rows
    ).to_csv(
        SUBJECT_SUMMARY_CSV,
        index=False,
    )

    shortest = (
        df.sort_values(
            [
                "rows",
                "subject",
                "sentence_id",
                "repetition_id",
            ],
            ascending=True,
        )
        .head(20)
        [
            [
                "sample_id",
                "subject",
                "sentence_id",
                "repetition_id",
                "rows",
                "relative_path",
            ]
        ]
    )

    shortest.to_csv(
        SHORTEST_CSV,
        index=False,
    )

    longest = (
        df.sort_values(
            [
                "rows",
                "subject",
                "sentence_id",
                "repetition_id",
            ],
            ascending=[
                False,
                True,
                True,
                True,
            ],
        )
        .head(20)
        [
            [
                "sample_id",
                "subject",
                "sentence_id",
                "repetition_id",
                "rows",
                "relative_path",
            ]
        ]
    )

    longest.to_csv(
        LONGEST_CSV,
        index=False,
    )

    threshold_rows = []

    for threshold in THRESHOLDS:
        below = int(
            (df["rows"] < threshold).sum()
        )

        at_or_below = int(
            (df["rows"] <= threshold).sum()
        )

        above = int(
            (df["rows"] > threshold).sum()
        )

        threshold_rows.append(
            {
                "threshold": threshold,
                "count_below": below,
                "percent_below": (
                    100.0
                    * below
                    / len(df)
                ),
                "count_at_or_below": at_or_below,
                "percent_at_or_below": (
                    100.0
                    * at_or_below
                    / len(df)
                ),
                "count_above": above,
                "percent_above": (
                    100.0
                    * above
                    / len(df)
                ),
            }
        )

    pd.DataFrame(
        threshold_rows
    ).to_csv(
        THRESHOLD_CSV,
        index=False,
    )

    # ---------------------------------------------------------
    # Histogram
    # ---------------------------------------------------------

    plt.figure(
        figsize=(9, 5)
    )

    plt.hist(
        df["rows"],
        bins=50,
    )

    plt.xlabel(
        "Sequence length T (timesteps)"
    )

    plt.ylabel(
        "Number of recordings"
    )

    plt.title(
        "Stage 6 — Raw Sequence-Length Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        HISTOGRAM_PATH,
        dpi=200,
    )

    plt.close()

    # ---------------------------------------------------------
    # Subject boxplot
    # ---------------------------------------------------------

    subject_order = sorted(
        df["subject"].unique()
    )

    boxplot_data = [
        df.loc[
            df["subject"] == subject,
            "rows",
        ].to_numpy()
        for subject in subject_order
    ]

    plt.figure(
        figsize=(8, 5)
    )

    plt.boxplot(
        boxplot_data,
        tick_labels=subject_order,
    )

    plt.xlabel(
        "Subject"
    )

    plt.ylabel(
        "Sequence length T (timesteps)"
    )

    plt.title(
        "Stage 6 — Sequence Length by Subject"
    )

    plt.tight_layout()

    plt.savefig(
        BOXPLOT_PATH,
        dpi=200,
    )

    plt.close()

    shortest_record = (
        shortest.iloc[0].to_dict()
    )

    longest_record = (
        longest.iloc[0].to_dict()
    )

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 6,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "source": (
            "artifacts/audits/"
            "stage5_recording_integrity.csv"
        ),
        "recordings": int(len(df)),
        "feature_width": 38,
        "overall": overall,
        "by_subject": subject_summary,
        "shortest_recording": (
            shortest_record
        ),
        "longest_recording": (
            longest_record
        ),
        "paper_context": {
            "purpose": (
                "Observed source-data lengths "
                "will be reported independently "
                "of approximate ranges described "
                "in the Gu paper."
            ),
            "raw_data_modified_to_match_paper": False,
        },
        "decisions_deferred": {
            "ctc_feasibility": "Stage 20",
            "gu_window_schedule": (
                "Evaluated later after label "
                "targets are available."
            ),
            "truncation": (
                "No truncation performed."
            ),
        },
    }

    SUMMARY_JSON.write_text(
        json.dumps(
            summary,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 6 — Sequence-Length Audit

## Purpose

Measure the observed variable sequence length `T` for all fresh raw
recordings.

The sensor tensor remains:

`X in R^(T x 38)`

where `T` varies between recordings.

## Dataset

Recordings analysed: {len(df):,}

Feature width: 38

## Overall observed sequence-length statistics

Minimum: {overall["min"]}

Maximum: {overall["max"]}

Mean: {overall["mean"]:.3f}

Median: {overall["median"]:.3f}

Sample standard deviation: {overall["std_sample"]:.3f}

P01: {overall["p01"]:.3f}

P05: {overall["p05"]:.3f}

P10: {overall["p10"]:.3f}

P25: {overall["p25"]:.3f}

P50: {overall["p50"]:.3f}

P75: {overall["p75"]:.3f}

P90: {overall["p90"]:.3f}

P95: {overall["p95"]:.3f}

P99: {overall["p99"]:.3f}

## Shortest recording

Sample ID: `{shortest_record["sample_id"]}`

Subject: `{shortest_record["subject"]}`

Length: {shortest_record["rows"]}

## Longest recording

Sample ID: `{longest_record["sample_id"]}`

Subject: `{longest_record["subject"]}`

Length: {longest_record["rows"]}

## Interpretation boundary

The observed raw-data distribution is the project source of truth.

No recording is truncated, stretched or resampled merely to agree with
a range reported in the publication.

Detailed CTC feasibility is not evaluated in this stage because target
gloss sequences have not yet been frozen.

## Outputs

`artifacts/audits/stage6_sequence_length_summary.json`

`artifacts/audits/stage6_subject_length_summary.csv`

`artifacts/audits/stage6_shortest_recordings.csv`

`artifacts/audits/stage6_longest_recordings.csv`

`artifacts/audits/stage6_length_threshold_counts.csv`

`artifacts/figures/stage6_length_histogram.png`

`artifacts/figures/stage6_length_boxplot_by_subject.png`
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("OVERALL")
    print("-" * 78)

    print(
        f"Recordings:  {len(df):,}"
    )
    print(
        f"Minimum T:   {overall['min']}"
    )
    print(
        f"Maximum T:   {overall['max']}"
    )
    print(
        f"Mean T:      {overall['mean']:.3f}"
    )
    print(
        f"Median T:    {overall['median']:.3f}"
    )
    print(
        f"Sample SD:   {overall['std_sample']:.3f}"
    )

    print()
    print("PERCENTILES")
    print("-" * 78)

    for p in PERCENTILES:
        print(
            f"P{p:02d}: "
            f"{overall[f'p{p:02d}']:.3f}"
        )

    print()
    print("BY SUBJECT")
    print("-" * 78)

    for row in subject_rows:
        print(
            f"{row['subject']}: "
            f"n={row['count']:,}, "
            f"min={row['min']}, "
            f"median={row['median']:.3f}, "
            f"mean={row['mean']:.3f}, "
            f"max={row['max']}"
        )

    print()
    print("EXTREMES")
    print("-" * 78)

    print(
        "Shortest:",
        shortest_record["sample_id"],
        "T=",
        shortest_record["rows"],
    )

    print(
        "Longest: ",
        longest_record["sample_id"],
        "T=",
        longest_record["rows"],
    )

    print()
    print("Saved:", SUMMARY_JSON.relative_to(PROJECT_ROOT))
    print("Saved:", SUBJECT_SUMMARY_CSV.relative_to(PROJECT_ROOT))
    print("Saved:", SHORTEST_CSV.relative_to(PROJECT_ROOT))
    print("Saved:", LONGEST_CSV.relative_to(PROJECT_ROOT))
    print("Saved:", THRESHOLD_CSV.relative_to(PROJECT_ROOT))
    print("Saved:", HISTOGRAM_PATH.relative_to(PROJECT_ROOT))
    print("Saved:", BOXPLOT_PATH.relative_to(PROJECT_ROOT))
    print("Saved:", DOC_PATH.relative_to(PROJECT_ROOT))

    print()
    print("FINAL STATUS: STAGE 6 BUILD PASSED")
    print("=" * 78)


if __name__ == "__main__":
    main()

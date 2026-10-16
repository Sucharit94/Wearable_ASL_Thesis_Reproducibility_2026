from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

MANIFEST_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "manifests"
    / "raw_manifest.csv"
)

STAGE7_SUMMARY_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage7_feature_semantics_summary.json"
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

FEATURE_STATS_PATH = (
    OUTPUT_DIR
    / "stage8_rotation_feature_statistics.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage8_rotation_angle_summary.json"
)

MAX_VALUE_FIGURE = (
    FIGURE_DIR
    / "stage8_max_absolute_value_by_column.png"
)

MAX_JUMP_FIGURE = (
    FIGURE_DIR
    / "stage8_max_frame_jump_by_column.png"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_8_ROTATION_ANGLE_AUDIT.md"
)

FEATURES = 38

VALUE_THRESHOLDS = {
    "abs_gt_pi": math.pi,
    "abs_gt_2pi": 2.0 * math.pi,
    "abs_gt_180": 180.0,
    "abs_gt_360": 360.0,
}

JUMP_THRESHOLDS = {
    "jump_gt_90": 90.0,
    "jump_gt_180": 180.0,
    "jump_gt_270": 270.0,
    "jump_gt_300": 300.0,
    "jump_gt_350": 350.0,
}


def load_manifest() -> list[dict]:
    with MANIFEST_PATH.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    print("=" * 78)
    print("STAGE 8 — ROTATION / ANGLE AUDIT")
    print("=" * 78)

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Missing Stage 4 manifest: {MANIFEST_PATH}"
        )

    if not STAGE7_SUMMARY_PATH.exists():
        raise FileNotFoundError(
            f"Missing Stage 7 summary: {STAGE7_SUMMARY_PATH}"
        )

    stage7 = json.loads(
        STAGE7_SUMMARY_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if (
        stage7["feature_width"] != FEATURES
    ):
        raise RuntimeError(
            "Stage 7 feature width is not 38."
        )

    if (
        stage7["csv_column_mapping"]
        ["semantic_mappings_verified"]
        != 0
    ):
        raise RuntimeError(
            "Stage 7 mapping state changed unexpectedly."
        )

    manifest = load_manifest()

    if len(manifest) != 7400:
        raise RuntimeError(
            f"Expected 7,400 recordings, "
            f"found {len(manifest):,}."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    FIGURE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Streaming feature statistics
    # ---------------------------------------------------------

    value_count = np.zeros(
        FEATURES,
        dtype=np.int64,
    )

    value_sum = np.zeros(
        FEATURES,
        dtype=np.float64,
    )

    value_sum_sq = np.zeros(
        FEATURES,
        dtype=np.float64,
    )

    value_min = np.full(
        FEATURES,
        np.inf,
        dtype=np.float64,
    )

    value_max = np.full(
        FEATURES,
        -np.inf,
        dtype=np.float64,
    )

    max_abs_value = np.zeros(
        FEATURES,
        dtype=np.float64,
    )

    value_threshold_counts = {
        name: np.zeros(
            FEATURES,
            dtype=np.int64,
        )
        for name in VALUE_THRESHOLDS
    }

    transition_count = np.zeros(
        FEATURES,
        dtype=np.int64,
    )

    max_abs_jump = np.zeros(
        FEATURES,
        dtype=np.float64,
    )

    jump_threshold_counts = {
        name: np.zeros(
            FEATURES,
            dtype=np.int64,
        )
        for name in JUMP_THRESHOLDS
    }

    candidate_wrap_crossings = np.zeros(
        FEATURES,
        dtype=np.int64,
    )

    total_timesteps = 0

    print()
    print("Reading and auditing all raw recordings...")
    print()

    for index, row in enumerate(
        manifest,
        start=1,
    ):
        path = (
            PROJECT_ROOT
            / Path(row["relative_path"])
        )

        array = np.loadtxt(
            path,
            delimiter=",",
            dtype=np.float64,
            ndmin=2,
        )

        if (
            array.ndim != 2
            or array.shape[1] != FEATURES
        ):
            raise RuntimeError(
                f"Unexpected shape for "
                f"{row['sample_id']}: "
                f"{array.shape}"
            )

        if not np.isfinite(array).all():
            raise RuntimeError(
                f"Non-finite values detected in "
                f"{row['sample_id']}."
            )

        n = array.shape[0]

        total_timesteps += n

        value_count += n

        value_sum += array.sum(
            axis=0,
            dtype=np.float64,
        )

        value_sum_sq += np.square(
            array,
            dtype=np.float64,
        ).sum(
            axis=0,
            dtype=np.float64,
        )

        value_min = np.minimum(
            value_min,
            array.min(axis=0),
        )

        value_max = np.maximum(
            value_max,
            array.max(axis=0),
        )

        abs_array = np.abs(array)

        max_abs_value = np.maximum(
            max_abs_value,
            abs_array.max(axis=0),
        )

        for name, threshold in (
            VALUE_THRESHOLDS.items()
        ):
            value_threshold_counts[
                name
            ] += (
                abs_array > threshold
            ).sum(axis=0)

        # -----------------------------------------------------
        # Frame-to-frame changes
        # -----------------------------------------------------

        if n >= 2:
            previous = array[:-1]
            current = array[1:]

            diff = current - previous
            abs_diff = np.abs(diff)

            transition_count += (
                n - 1
            )

            max_abs_jump = np.maximum(
                max_abs_jump,
                abs_diff.max(axis=0),
            )

            for name, threshold in (
                JUMP_THRESHOLDS.items()
            ):
                jump_threshold_counts[
                    name
                ] += (
                    abs_diff > threshold
                ).sum(axis=0)

            # Candidate degree-like ±180 wrap:
            #
            # previous >= +150 and current <= -150
            # or the reverse,
            # combined with a >=300 raw-unit jump.
            #
            # This is only a diagnostic definition.
            # It does NOT prove the units are degrees.

            opposite_boundary = (
                (
                    (previous >= 150.0)
                    & (current <= -150.0)
                )
                |
                (
                    (previous <= -150.0)
                    & (current >= 150.0)
                )
            )

            large_boundary_jump = (
                abs_diff >= 300.0
            )

            candidate_wrap_crossings += (
                opposite_boundary
                & large_boundary_jump
            ).sum(axis=0)

        if (
            index == 1
            or index % 250 == 0
            or index == len(manifest)
        ):
            print(
                f"Processed {index:>4,} / "
                f"{len(manifest):,}"
            )

    # ---------------------------------------------------------
    # Final statistics
    # ---------------------------------------------------------

    means = (
        value_sum
        / value_count
    )

    variance = (
        value_sum_sq
        / value_count
        - np.square(means)
    )

    variance = np.maximum(
        variance,
        0.0,
    )

    std_population = np.sqrt(
        variance
    )

    feature_rows = []

    for i in range(FEATURES):
        row = {
            "csv_column_index_zero_based": i,
            "semantic_identity": "UNVERIFIED",
            "value_count": int(
                value_count[i]
            ),
            "min": float(
                value_min[i]
            ),
            "max": float(
                value_max[i]
            ),
            "mean": float(
                means[i]
            ),
            "std_population": float(
                std_population[i]
            ),
            "max_absolute_value": float(
                max_abs_value[i]
            ),
            "transition_count": int(
                transition_count[i]
            ),
            "max_absolute_frame_jump": float(
                max_abs_jump[i]
            ),
            "candidate_pm180_wrap_crossings": int(
                candidate_wrap_crossings[i]
            ),
        }

        for name in VALUE_THRESHOLDS:
            row[name] = int(
                value_threshold_counts[
                    name
                ][i]
            )

        for name in JUMP_THRESHOLDS:
            row[name] = int(
                jump_threshold_counts[
                    name
                ][i]
            )

        feature_rows.append(row)

    fieldnames = list(
        feature_rows[0].keys()
    )

    with FEATURE_STATS_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(
            feature_rows
        )

    # ---------------------------------------------------------
    # Dataset-level evidence
    # ---------------------------------------------------------

    global_max_abs_value = float(
        max_abs_value.max()
    )

    global_max_jump = float(
        max_abs_jump.max()
    )

    total_values = int(
        value_count.sum()
    )

    total_transitions = int(
        transition_count.sum()
    )

    total_wrap_candidates = int(
        candidate_wrap_crossings.sum()
    )

    wrap_candidate_rate = (
        total_wrap_candidates
        / total_transitions
        if total_transitions > 0
        else 0.0
    )

    total_jump_gt_180 = int(
        jump_threshold_counts[
            "jump_gt_180"
        ].sum()
    )

    total_jump_gt_300 = int(
        jump_threshold_counts[
            "jump_gt_300"
        ].sum()
    )

    total_abs_gt_2pi = int(
        value_threshold_counts[
            "abs_gt_2pi"
        ].sum()
    )

    fraction_abs_gt_2pi = (
        total_abs_gt_2pi
        / total_values
    )

    # Empirical scale heuristic only.
    #
    # Conventional bounded Euler angles expressed in radians
    # are ordinarily on the scale of pi, rather than tens or
    # hundreds. This does NOT independently prove export units.

    if (
        global_max_abs_value
        > 4.0 * 2.0 * math.pi
    ):
        empirical_scale = (
            "STRONGLY_DEGREE_LIKE"
        )
    elif (
        global_max_abs_value
        > 2.0 * math.pi
    ):
        empirical_scale = (
            "MORE_DEGREE_LIKE_THAN_STANDARD_RADIAN_RANGE"
        )
    else:
        empirical_scale = (
            "INCONCLUSIVE"
        )

    summary = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 8,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "recordings_audited": (
            len(manifest)
        ),
        "feature_width": (
            FEATURES
        ),
        "total_timesteps": (
            total_timesteps
        ),
        "total_scalar_values": (
            total_values
        ),
        "total_feature_transitions": (
            total_transitions
        ),
        "global_min": float(
            value_min.min()
        ),
        "global_max": float(
            value_max.max()
        ),
        "global_max_absolute_value": (
            global_max_abs_value
        ),
        "global_max_absolute_frame_jump": (
            global_max_jump
        ),
        "values_abs_gt_2pi": (
            total_abs_gt_2pi
        ),
        "fraction_values_abs_gt_2pi": (
            fraction_abs_gt_2pi
        ),
        "jumps_gt_180_raw_units": (
            total_jump_gt_180
        ),
        "jumps_gt_300_raw_units": (
            total_jump_gt_300
        ),
        "candidate_pm180_wrap_crossings": (
            total_wrap_candidates
        ),
        "candidate_pm180_wrap_rate": (
            wrap_candidate_rate
        ),
        "unit_evidence": {
            "paper_explicit_degree_or_radian_unit": False,
            "empirical_scale_assessment": (
                empirical_scale
            ),
            "interpretation": (
                "The numerical scale is diagnostic evidence "
                "only and is not treated as independent proof "
                "of the export unit."
            ),
        },
        "wrap_diagnostic": {
            "definition": (
                "Opposite signs beyond +/-150 with "
                "absolute frame jump >=300 raw units."
            ),
            "this_definition_assumes_degree_like_scale": True,
            "candidate_count": (
                total_wrap_candidates
            ),
        },
        "semantic_mapping_status": (
            "UNVERIFIED_FOR_ALL_38_COLUMNS"
        ),
        "preprocessing_decision": {
            "raw_rotation_primary": (
                "DEFERRED_PENDING_STAGE_8_REVIEW"
            ),
            "sin_cos_encoding": (
                "DEFERRED_PENDING_STAGE_8_REVIEW"
            ),
            "in_place_modification": False,
        },
        "raw_files_modified": False,
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
    # Figures
    # ---------------------------------------------------------

    columns = np.arange(
        FEATURES
    )

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        columns,
        max_abs_value,
    )

    plt.xlabel(
        "CSV column index (zero-based)"
    )

    plt.ylabel(
        "Maximum absolute raw value"
    )

    plt.title(
        "Stage 8 — Maximum Absolute Rotation Value"
    )

    plt.tight_layout()

    plt.savefig(
        MAX_VALUE_FIGURE,
        dpi=200,
    )

    plt.close()

    plt.figure(
        figsize=(10, 5)
    )

    plt.bar(
        columns,
        max_abs_jump,
    )

    plt.xlabel(
        "CSV column index (zero-based)"
    )

    plt.ylabel(
        "Maximum absolute one-frame change"
    )

    plt.title(
        "Stage 8 — Maximum Frame-to-Frame Change"
    )

    plt.tight_layout()

    plt.savefig(
        MAX_JUMP_FIGURE,
        dpi=200,
    )

    plt.close()

    markdown = f"""# Stage 8 — Rotation / Angle Audit

## Evidence state

Gu et al. describe the retained values as calculated joint rotation
coordinates.

The current source material does not explicitly establish whether the
export values are degrees or radians.

Therefore numerical scale is treated as empirical evidence, not as
standalone proof of units.

## Dataset

Recordings audited: {len(manifest):,}

Feature width: {FEATURES}

Total timesteps: {total_timesteps:,}

Total scalar feature values: {total_values:,}

## Raw numerical range

Global minimum: {float(value_min.min())}

Global maximum: {float(value_max.max())}

Global maximum absolute value: {global_max_abs_value}

Empirical scale assessment:

`{empirical_scale}`

## Frame-to-frame behaviour

Maximum absolute one-frame change:

{global_max_jump}

Jumps greater than 180 raw units:

{total_jump_gt_180:,}

Jumps greater than 300 raw units:

{total_jump_gt_300:,}

Candidate +/-180 wrap crossings:

{total_wrap_candidates:,}

Candidate wrap rate:

{wrap_candidate_rate:.12f}

## Important limitation

Candidate wrap detection uses a degree-like diagnostic definition:
opposite signs beyond +/-150 together with a jump of at least 300 raw
units.

This diagnostic must not itself be cited as proof that the export unit
is degrees.

## Column semantics

Exact semantic identities of physical CSV columns 0-37 remain
unverified.

No anatomical label is inferred from numerical behaviour.

## Preprocessing

The primary raw-versus-sin/cos decision is intentionally deferred until
the Stage 8 audit is reviewed.

No raw data were modified.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("DATASET SCALE")
    print("-" * 78)

    print(
        f"Recordings:                "
        f"{len(manifest):,}"
    )

    print(
        f"Total timesteps:           "
        f"{total_timesteps:,}"
    )

    print(
        f"Total scalar values:       "
        f"{total_values:,}"
    )

    print(
        f"Global minimum:            "
        f"{float(value_min.min())}"
    )

    print(
        f"Global maximum:            "
        f"{float(value_max.max())}"
    )

    print(
        f"Global max abs value:      "
        f"{global_max_abs_value}"
    )

    print(
        f"Fraction |value| > 2π:     "
        f"{fraction_abs_gt_2pi:.6f}"
    )

    print()
    print("EMPIRICAL UNIT-SCALE ASSESSMENT")
    print("-" * 78)

    print(empirical_scale)

    print()
    print("FRAME-TO-FRAME AUDIT")
    print("-" * 78)

    print(
        f"Feature transitions:       "
        f"{total_transitions:,}"
    )

    print(
        f"Maximum absolute jump:     "
        f"{global_max_jump}"
    )

    print(
        f"Jumps >180 raw units:      "
        f"{total_jump_gt_180:,}"
    )

    print(
        f"Jumps >300 raw units:      "
        f"{total_jump_gt_300:,}"
    )

    print(
        f"Candidate ±180 wraps:      "
        f"{total_wrap_candidates:,}"
    )

    print(
        f"Candidate wrap rate:       "
        f"{wrap_candidate_rate:.12f}"
    )

    print()
    print("COLUMN SEMANTICS")
    print("-" * 78)

    print(
        "Exact semantic mapping: UNVERIFIED"
    )

    print()
    print("PREPROCESSING DECISION")
    print("-" * 78)

    print(
        "Raw-vs-sin/cos decision deferred "
        "until Stage 8 results are reviewed."
    )

    print()
    print(
        "Saved:",
        FEATURE_STATS_PATH.relative_to(
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
        MAX_VALUE_FIGURE.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        MAX_JUMP_FIGURE.relative_to(
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

    stage8_pass = (
        len(manifest) == 7400
        and total_timesteps > 0
        and total_values
        == total_timesteps * FEATURES
        and np.isfinite(value_min).all()
        and np.isfinite(value_max).all()
        and np.isfinite(max_abs_jump).all()
    )

    if stage8_pass:
        print(
            "FINAL STATUS: STAGE 8 AUDIT PASSED"
        )
    else:
        print(
            "FINAL STATUS: STAGE 8 AUDIT FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

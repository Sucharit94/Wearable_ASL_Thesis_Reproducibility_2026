from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "gu_author_dataset"
)

REPRESENTATIVE_CSV = (
    RAW_ROOT
    / "csv"
    / "subject1"
    / "1_1.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
)

SEMANTIC_INVENTORY_PATH = (
    OUTPUT_DIR
    / "stage7_semantic_inventory.csv"
)

COLUMN_MAPPING_PATH = (
    OUTPUT_DIR
    / "stage7_csv_column_mapping.csv"
)

SUMMARY_PATH = (
    OUTPUT_DIR
    / "stage7_feature_semantics_summary.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_7_FEATURE_SEMANTICS.md"
)


# This is a semantic inventory taken from Gu et al. Table 2.
#
# IMPORTANT:
# The ordering below is only an internal inventory order.
# It MUST NOT be interpreted as the raw CSV column order.

TABLE2_GROUPS = [
    ("Arm", ["Rotation_Y", "Rotation_X", "Rotation_Z"]),
    ("Forearm", ["Rotation_Y", "Rotation_X", "Rotation_Z"]),
    ("Hand", ["Rotation_Y", "Rotation_X", "Rotation_Z"]),
    ("Hand Thumb 1", ["Rotation_Y", "Rotation_Z"]),
    ("Hand Thumb 2", ["Rotation_Y", "Rotation_Z"]),
    ("Hand Thumb 3", ["Rotation_Y"]),
    ("Hand Index 1", ["Rotation_Z"]),
    ("Hand Index 2", ["Rotation_Z"]),
    ("Hand Middle 1", ["Rotation_Z"]),
    ("Hand Ring 1", ["Rotation_Z"]),
    ("Hand Pinky 1", ["Rotation_Z"]),
]

SIDES = [
    "Right",
    "Left",
]


def main() -> None:
    print("=" * 78)
    print("STAGE 7 — FEATURE-SEMANTICS AUDIT")
    print("=" * 78)

    if not REPRESENTATIVE_CSV.exists():
        raise FileNotFoundError(
            f"Missing representative CSV: {REPRESENTATIVE_CSV}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Verify raw CSV width and absence of a textual header.
    # ---------------------------------------------------------

    first_line = (
        REPRESENTATIVE_CSV
        .read_text(
            encoding="utf-8",
            errors="strict",
        )
        .splitlines()[0]
    )

    raw_tokens = [
        token.strip()
        for token in first_line.split(",")
    ]

    if len(raw_tokens) != 38:
        raise RuntimeError(
            f"Representative CSV first row has "
            f"{len(raw_tokens)} columns, expected 38."
        )

    numeric_values = []

    for token in raw_tokens:
        numeric_values.append(
            float(token)
        )

    raw_first_row_is_numeric = True

    # ---------------------------------------------------------
    # Create semantic inventory from paper Table 2.
    #
    # Inventory order != CSV column order.
    # ---------------------------------------------------------

    semantic_rows = []

    inventory_index = 0

    for joint, axes in TABLE2_GROUPS:
        for side in SIDES:
            for axis in axes:
                semantic_rows.append(
                    {
                        "inventory_index": inventory_index,
                        "side": side,
                        "joint": joint,
                        "coordinate": axis,
                        "semantic_name": (
                            f"{side} {joint} {axis}"
                        ),
                        "paper_source": (
                            "Gu et al. 2024, Table 2"
                        ),
                        "exists_in_38d_set": True,
                        "csv_column_index_verified": False,
                        "csv_column_index": "",
                    }
                )

                inventory_index += 1

    if len(semantic_rows) != 38:
        raise RuntimeError(
            f"Semantic inventory produced "
            f"{len(semantic_rows)} coordinates, "
            f"expected 38."
        )

    with SEMANTIC_INVENTORY_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "inventory_index",
                "side",
                "joint",
                "coordinate",
                "semantic_name",
                "paper_source",
                "exists_in_38d_set",
                "csv_column_index_verified",
                "csv_column_index",
            ],
        )

        writer.writeheader()
        writer.writerows(
            semantic_rows
        )

    # ---------------------------------------------------------
    # Physical CSV columns.
    #
    # DO NOT infer anatomical semantics from Table 2 order.
    # ---------------------------------------------------------

    mapping_rows = []

    for column_index in range(38):
        mapping_rows.append(
            {
                "csv_column_index_zero_based": column_index,
                "csv_column_number_one_based": column_index + 1,
                "raw_order_verified": True,
                "numeric_column_verified": True,
                "semantic_identity_verified": False,
                "side": "",
                "joint": "",
                "coordinate": "",
                "semantic_name": "UNVERIFIED",
                "mapping_evidence": (
                    "No column header or authoritative "
                    "column-order mapping available in "
                    "current source materials."
                ),
            }
        )

    with COLUMN_MAPPING_PATH.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "csv_column_index_zero_based",
                "csv_column_number_one_based",
                "raw_order_verified",
                "numeric_column_verified",
                "semantic_identity_verified",
                "side",
                "joint",
                "coordinate",
                "semantic_name",
                "mapping_evidence",
            ],
        )

        writer.writeheader()
        writer.writerows(
            mapping_rows
        )

    verified_semantic_mapping_count = sum(
        bool(
            row["semantic_identity_verified"]
        )
        for row in mapping_rows
    )

    summary = {
        "project_generation": "fresh_start_2026_v1",
        "stage": 7,
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),
        "feature_width": 38,
        "representative_csv": (
            "data/raw/gu_author_dataset/"
            "csv/subject1/1_1.csv"
        ),
        "representative_first_row_numeric": (
            raw_first_row_is_numeric
        ),
        "representative_first_row_columns": (
            len(raw_tokens)
        ),
        "semantic_inventory": {
            "coordinate_count": (
                len(semantic_rows)
            ),
            "basis": (
                "Gu et al. 2024 Table 2"
            ),
            "meaning": (
                "Verified set of selected joint-rotation "
                "coordinate types."
            ),
        },
        "csv_column_mapping": {
            "physical_columns": 38,
            "semantic_mappings_verified": (
                verified_semantic_mapping_count
            ),
            "semantic_mappings_unverified": (
                38 - verified_semantic_mapping_count
            ),
            "exact_column_order_status": (
                "UNVERIFIED"
            ),
        },
        "terminology": {
            "appropriate": [
                "wearable inertial motion-capture-derived "
                "joint rotation features",
                "38-dimensional joint-rotation feature vector",
                "rotation-coordinate time series",
            ],
            "unsupported_as_input_description": [
                "sEMG",
                "raw accelerometer channels",
                "raw gyroscope channels",
                "raw magnetometer channels",
            ],
        },
        "swin_implication": {
            "anatomical_2d_topology_verified": False,
            "source_order_available": True,
            "primary_safe_representation": (
                "temporal-feature map preserving native "
                "CSV feature order without anatomical claims"
            ),
        },
        "future_mapping_policy": (
            "A semantic column mapping may only be added "
            "if authoritative author metadata, source code, "
            "export configuration, or equivalent direct "
            "evidence becomes available."
        ),
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

    markdown = """# Stage 7 — Feature-Semantics Audit

## Result

The raw recordings contain 38 numeric columns.

Gu et al. (2024), Table 2, establishes the semantic inventory of the
38 selected joint-rotation coordinates.

However, the current author-provided dataset contains no CSV header or
separate feature-order metadata that establishes which exact physical
CSV column corresponds to each semantic coordinate.

Therefore:

- semantic inventory: VERIFIED
- raw physical column order: VERIFIED
- exact semantic identity of CSV columns 0-37: UNVERIFIED
- anatomical 2D column topology: UNVERIFIED

## Safe terminology

The input is described as:

`wearable inertial motion-capture-derived joint-rotation features`

or:

`T x 38 joint-rotation feature sequence`

The project must not describe these CSV inputs as sEMG.

The current files also do not establish that the 38 retained values
are raw accelerometer, gyroscope, or magnetometer channels.

## Swin implication

The primary 2D representation may use the native source feature order
as a temporal-feature map.

It must not claim that vertical feature adjacency represents verified
anatomical adjacency unless authoritative column-order metadata is
obtained later.

## Evidence rule

Table 2 defines membership in the 38-dimensional semantic feature set.

Table 2 printed order is not treated as proof of physical CSV column
order.

No semantic mapping will be inferred merely from numeric magnitude or
correlation.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("RAW CSV")
    print("-" * 78)

    print(
        "First row numeric:",
        raw_first_row_is_numeric,
    )

    print(
        "First-row columns:",
        len(raw_tokens),
    )

    print()
    print("PAPER SEMANTIC INVENTORY")
    print("-" * 78)

    print(
        "Coordinates in inventory:",
        len(semantic_rows),
    )

    print()
    print("CSV COLUMN MAPPING")
    print("-" * 78)

    print(
        "Physical CSV columns:",
        len(mapping_rows),
    )

    print(
        "Verified semantic mappings:",
        verified_semantic_mapping_count,
    )

    print(
        "Unverified semantic mappings:",
        38 - verified_semantic_mapping_count,
    )

    print()
    print(
        "Exact CSV semantic column order: UNVERIFIED"
    )

    print(
        "Anatomical 2D topology: UNVERIFIED"
    )

    print()
    print(
        "SAFE C4 BASIS:"
    )

    print(
        "Temporal-feature map using native "
        "source column order."
    )

    print()
    print(
        "Saved:",
        SEMANTIC_INVENTORY_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        COLUMN_MAPPING_PATH.relative_to(
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

    stage7_pass = (
        raw_first_row_is_numeric
        and len(raw_tokens) == 38
        and len(semantic_rows) == 38
        and len(mapping_rows) == 38
        and verified_semantic_mapping_count == 0
    )

    if stage7_pass:
        print(
            "FINAL STATUS: STAGE 7 AUDIT PASSED"
        )
    else:
        print(
            "FINAL STATUS: STAGE 7 AUDIT FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

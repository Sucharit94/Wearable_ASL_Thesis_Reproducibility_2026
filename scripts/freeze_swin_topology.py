from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAGE7_SUMMARY = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage7_feature_semantics_summary.json"
)

STAGE8_DECISION = (
    PROJECT_ROOT
    / "artifacts"
    / "audits"
    / "stage8_preprocessing_decision.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "design"
)

DECISION_PATH = (
    OUTPUT_DIR
    / "stage9_swin_topology_decision.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_9_SWIN_TOPOLOGY_DECISION.md"
)


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Required input missing: {path}"
        )

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def main() -> None:
    print("=" * 78)
    print("STAGE 9 — SWIN TOPOLOGY DECISION")
    print("=" * 78)

    stage7 = load_json(
        STAGE7_SUMMARY
    )

    stage8 = load_json(
        STAGE8_DECISION
    )

    # ---------------------------------------------------------
    # Evidence guards
    # ---------------------------------------------------------

    if stage7["feature_width"] != 38:
        raise RuntimeError(
            "Stage 7 feature width is not 38."
        )

    mapping = stage7[
        "csv_column_mapping"
    ]

    if (
        mapping[
            "semantic_mappings_verified"
        ]
        != 0
    ):
        raise RuntimeError(
            "Unexpected semantic mapping state."
        )

    if (
        mapping[
            "semantic_mappings_unverified"
        ]
        != 38
    ):
        raise RuntimeError(
            "Expected all 38 CSV semantic mappings "
            "to remain unverified."
        )

    primary = stage8[
        "primary_representation"
    ]

    if (
        primary["name"]
        != "raw_rotation_38d"
    ):
        raise RuntimeError(
            "Stage 8 primary representation changed."
        )

    if (
        primary["input_width"]
        != 38
    ):
        raise RuntimeError(
            "Stage 8 primary width is not 38."
        )

    if not primary[
        "use_raw_rotation_values"
    ]:
        raise RuntimeError(
            "Stage 8 no longer declares raw rotations "
            "as primary."
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Freeze topology decision
    # ---------------------------------------------------------

    decision = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 9,
        "decision_status": "FROZEN",
        "created_at_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "primary_c4_input": {
            "representation_name": (
                "native_temporal_feature_map"
            ),
            "tensor_shape_before_batching": (
                "T x 38"
            ),
            "time_axis": 0,
            "feature_axis": 1,
            "feature_width": 38,
            "source_values": (
                "raw_rotation_38d"
            ),
            "source_feature_order": (
                "native_csv_order"
            ),
            "feature_reordering": False,
            "anatomical_reordering": False,
        },

        "topology_evidence": {
            "semantic_feature_set_known": True,
            "exact_csv_semantic_order_known": False,
            "anatomical_adjacency_verified": False,
            "native_source_order_verified": True,
        },

        "allowed_claims": [
            (
                "The input is a temporal-feature map."
            ),
            (
                "The feature axis preserves native "
                "CSV source order."
            ),
            (
                "Swin windows model local interactions "
                "over time and source-ordered features."
            ),
        ],

        "forbidden_claims": [
            (
                "Neighboring feature columns are "
                "verified anatomical neighbors."
            ),
            (
                "The T x 38 representation is an "
                "anatomical body map."
            ),
            (
                "The representation is a natural image."
            ),
        ],

        "forbidden_primary_transforms": {
            "resize_to_224x224": True,
            "arbitrary_square_resize": True,
            "rgb_conversion": True,
            "anatomical_feature_permutation": True,
            "global_time_pooling_before_ctc": True,
            "interpolation_for_visual_appearance": True,
        },

        "future_padding_policy": {
            "padding_may_be_used_for_batching": True,
            "padding_may_be_used_for_patch_divisibility": True,
            "padding_is_real_data": False,
            "padding_must_not_change_true_sequence_length": True,
            "padding_must_be_length_or_mask_aware": True,
            "exact_rules_deferred": (
                "Stages 19, 22 and 29"
            ),
        },

        "ctc_constraint": {
            "time_information_must_survive_swin": True,
            "global_spatial_pooling_over_time_allowed": False,
            "output_must_support_temporal_ctc_logits": True,
            "exact_output_length_formula_deferred": (
                "Stages 20, 22 and 29"
            ),
        },

        "architecture_parameters_deferred": {
            "patch_size": True,
            "window_size": True,
            "shift_size": True,
            "embedding_dimension": True,
            "number_of_heads": True,
            "number_of_swin_blocks": True,
            "dropout": True,
        },

        "circular_ablation": {
            "primary": False,
            "declared": True,
            "shape_if_used": "T x 76",
            "topology_rule": (
                "Must be separately specified before use; "
                "does not alter the primary T x 38 C4 design."
            ),
        },

        "primary_stage9_decision": (
            "Use the raw T x 38 sequence directly as a "
            "source-order temporal-feature map for C4. "
            "Do not infer anatomical topology."
        ),
    }

    DECISION_PATH.write_text(
        json.dumps(
            decision,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    markdown = """# Stage 9 — Swin Topology Decision

## Frozen primary representation

The primary C4 input is the raw source sequence:

`T x 38`

The first axis is time.

The second axis contains the 38 features in their original CSV order.

## Why this representation is used

Stage 7 established the semantic inventory of the 38 rotation
coordinates but did not establish the exact semantic identity of
physical CSV columns 0-37.

Therefore the project preserves source order and makes no anatomical
adjacency claim.

## Safe interpretation

The C4 input is a:

`temporal-feature map`

Local and shifted Swin windows may model local interactions across time
and source-ordered feature positions.

Feature proximity in this representation does not imply verified
physical or anatomical proximity.

## Forbidden primary representations

The project will not:

- resize recordings to 224 x 224,
- stretch recordings to square images,
- convert the feature map to RGB,
- anatomically reorder columns without direct evidence,
- globally pool away time before CTC,
- interpolate merely for image-like appearance.

## Padding

Padding may later be required for batching or Swin patch/window
divisibility.

Such padding is artificial, must not alter the true sequence length,
and must be handled using explicit lengths or masks.

Exact padding rules are deferred to later implementation stages.

## CTC requirement

C4 must retain an ordered temporal output suitable for CTC.

A conventional image-classification Swin that globally pools the entire
time axis is not acceptable for the primary recognizer.

## Parameters deliberately not chosen in Stage 9

The following remain open:

- patch size,
- window size,
- shift size,
- embedding dimension,
- number of attention heads,
- number of Swin blocks,
- dropout.

These will be determined during C4 implementation and the frozen
training-protocol stages rather than being invented during the topology
audit.

## Circular encoding

The Stage 8 sin/cos representation remains a declared preprocessing
ablation.

It does not replace the primary `T x 38` C4 topology.
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("EVIDENCE")
    print("-" * 78)

    print(
        "Feature width:",
        stage7["feature_width"],
    )

    print(
        "Verified semantic column mappings:",
        mapping[
            "semantic_mappings_verified"
        ],
    )

    print(
        "Unverified semantic column mappings:",
        mapping[
            "semantic_mappings_unverified"
        ],
    )

    print()
    print("PRIMARY C4 TOPOLOGY")
    print("-" * 78)

    print(
        "Input shape: T x 38"
    )

    print(
        "Axis 0: time"
    )

    print(
        "Axis 1: native CSV feature order"
    )

    print(
        "Anatomical feature reordering: NO"
    )

    print(
        "Anatomical adjacency claim: NO"
    )

    print(
        "224x224 resize: NO"
    )

    print(
        "RGB conversion: NO"
    )

    print(
        "Global time pooling before CTC: NO"
    )

    print()
    print("DEFERRED")
    print("-" * 78)

    print(
        "Patch/window/head/depth choices remain "
        "for later implementation stages."
    )

    print()
    print(
        "Saved:",
        DECISION_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        DOC_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    stage9_pass = (
        stage7["feature_width"] == 38
        and mapping[
            "semantic_mappings_verified"
        ] == 0
        and mapping[
            "semantic_mappings_unverified"
        ] == 38
        and primary[
            "input_width"
        ] == 38
        and decision[
            "primary_c4_input"
        ][
            "feature_reordering"
        ] is False
        and decision[
            "topology_evidence"
        ][
            "anatomical_adjacency_verified"
        ] is False
        and decision[
            "ctc_constraint"
        ][
            "time_information_must_survive_swin"
        ] is True
    )

    print()

    if stage9_pass:
        print(
            "FINAL STATUS: STAGE 9 DECISION PASSED"
        )
    else:
        print(
            "FINAL STATUS: STAGE 9 DECISION FAILED"
        )

    print("=" * 78)


if __name__ == "__main__":
    main()

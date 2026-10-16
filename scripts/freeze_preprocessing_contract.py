from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "preprocessing.yaml"
)

IMPLEMENTATION_PATH = (
    PROJECT_ROOT
    / "src"
    / "data"
    / "preprocessing.py"
)

TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_preprocessing.py"
)

PYTEST_CONFIG_PATH = (
    PROJECT_ROOT
    / "pytest.ini"
)

STAGE17_INDEX_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.csv"
)

STAGE17_INDEX_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "cache"
    / "stage17_raw_cache_index.sha256"
)

STAGE13_SPLIT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage13_known_signer_5fold.csv"
)

STAGE13_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage13_known_signer_5fold.sha256"
)

STAGE14_SPLIT_PATH = (
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

STAGE16_SPLIT_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage16_template_holdout_assignment.csv"
)

STAGE16_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "splits"
    / "stage16_template_holdout_assignment.sha256"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "preprocessing"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage18_preprocessing_contract.json"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_18_PREPROCESSING_CONTRACT.md"
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


def recorded_hash(path: Path) -> str:

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
            char not in "0123456789abcdef"
            for char in value
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

    expected = recorded_hash(
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


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 18 — PREPROCESSING CONTRACT"
    )
    print("=" * 78)

    required_local_files = [
        CONFIG_PATH,
        IMPLEMENTATION_PATH,
        TEST_PATH,
        PYTEST_CONFIG_PATH,
    ]

    for path in required_local_files:

        if not path.exists():

            raise FileNotFoundError(
                path
            )

    config = yaml.safe_load(
        CONFIG_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    if not isinstance(
        config,
        dict,
    ):

        raise RuntimeError(
            "Preprocessing YAML did not "
            "load as a mapping."
        )

    primary = config[
        "primary"
    ]

    normalization = primary[
        "normalization"
    ]

    representation = primary[
        "representation"
    ]

    if (
        representation[
            "name"
        ]
        != "raw_rotation_38d"
    ):

        raise RuntimeError(
            "Unexpected primary representation."
        )

    if (
        representation[
            "input_width"
        ]
        != 38
    ):

        raise RuntimeError(
            "Primary input width must be 38."
        )

    if (
        representation[
            "temporal_unwrap"
        ]
        is not False
    ):

        raise RuntimeError(
            "Primary temporal unwrapping "
            "must be disabled."
        )

    if (
        normalization[
            "name"
        ]
        != "featurewise_zscore"
    ):

        raise RuntimeError(
            "Unexpected normalization method."
        )

    if (
        normalization[
            "fit_scope"
        ]
        !=
        "current_training_partition_only"
    ):

        raise RuntimeError(
            "Normalization is not train-only."
        )

    if (
        normalization[
            "aggregation"
        ]
        !=
        "all_valid_training_timesteps_frame_weighted"
    ):

        raise RuntimeError(
            "Unexpected normalization "
            "aggregation."
        )

    if (
        normalization[
            "variance_definition"
        ]
        != "population_ddof_0"
    ):

        raise RuntimeError(
            "Population variance contract "
            "is not frozen correctly."
        )

    if (
        normalization[
            "mean_dtype"
        ]
        != "float64"
    ):

        raise RuntimeError(
            "Mean fitting dtype must "
            "be float64."
        )

    if (
        normalization[
            "variance_dtype"
        ]
        != "float64"
    ):

        raise RuntimeError(
            "Variance fitting dtype must "
            "be float64."
        )

    if (
        normalization[
            "output_dtype"
        ]
        != "float32"
    ):

        raise RuntimeError(
            "Normalized output must "
            "be float32."
        )

    forbidden_primary = [
        "filtering",
        "smoothing",
        "angle_unwrapping",
        "resampling",
        "truncation",
        "augmentation",
        "per_sample_standardization",
    ]

    for key in forbidden_primary:

        if (
            primary[
                "operations"
            ][key]
            is not False
        ):

            raise RuntimeError(
                f"Primary operation unexpectedly "
                f"enabled: {key}"
            )

    # ---------------------------------------------------------
    # Verify upstream frozen evidence.
    # ---------------------------------------------------------

    upstream_hashes = {
        "stage17_cache_index": (
            verify_hash(
                STAGE17_INDEX_PATH,
                STAGE17_INDEX_HASH_PATH,
            )
        ),

        "stage13_known_signer_5fold": (
            verify_hash(
                STAGE13_SPLIT_PATH,
                STAGE13_HASH_PATH,
            )
        ),

        "stage14_loso": (
            verify_hash(
                STAGE14_SPLIT_PATH,
                STAGE14_HASH_PATH,
            )
        ),

        "stage16_template_holdout": (
            verify_hash(
                STAGE16_SPLIT_PATH,
                STAGE16_HASH_PATH,
            )
        ),
    }

    # ---------------------------------------------------------
    # Freeze implementation/config/test hashes.
    # ---------------------------------------------------------

    config_hash = sha256_file(
        CONFIG_PATH
    )

    implementation_hash = sha256_file(
        IMPLEMENTATION_PATH
    )

    test_hash = sha256_file(
        TEST_PATH
    )

    pytest_config_hash = sha256_file(
        PYTEST_CONFIG_PATH
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 18,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "primary_representation": {
            "name": (
                "raw_rotation_38d"
            ),

            "feature_width": (
                38
            ),

            "temporal_length_preserved": (
                True
            ),

            "temporal_unwrap": (
                False
            ),
        },

        "primary_normalization": {
            "name": (
                "featurewise_zscore"
            ),

            "formula": (
                "(x_j - train_mean_j) / "
                "train_std_j"
            ),

            "fit_scope": (
                "current_training_partition_only"
            ),

            "aggregation": (
                "all_valid_training_timesteps_"
                "frame_weighted"
            ),

            "population_ddof": (
                0
            ),

            "fit_dtype": (
                "float64"
            ),

            "output_dtype": (
                "float32"
            ),

            "minimum_allowed_std": (
                1e-8
            ),

            "low_std_policy": (
                "error"
            ),

            "clipping": (
                False
            ),
        },

        "primary_disabled_operations": {
            key: False
            for key in forbidden_primary
        },

        "fit_rules": (
            config[
                "split_fit_rules"
            ]
        ),

        "padding_contract": {
            "normalization_fitted_before_padding": (
                True
            ),

            "future_padding_value_after_normalization": (
                0.0
            ),

            "valid_lengths_required": (
                True
            ),
        },

        "model_adapter_boundary": {
            "C1": (
                "direct normalized T x 38"
            ),

            "C2": (
                "direct normalized T x 38"
            ),

            "C3": (
                "direct normalized T x 38"
            ),

            "G1": (
                "normalized T x 38 then "
                "Stage-20-frozen window adapter"
            ),

            "C4": (
                "normalized native-source-order "
                "T x 38 then Stage-20-frozen "
                "temporal-feature adapter"
            ),
        },

        "circular_ablation": {
            "primary": (
                False
            ),

            "input_width": (
                38
            ),

            "output_width": (
                76
            ),

            "encoding_order": (
                "sin_f0,cos_f0,...,"
                "sin_f37,cos_f37"
            ),

            "normalization": (
                "fit feature-wise z-score "
                "on current training partition "
                "after circular transform"
            ),
        },

        "numeric_stats_frozen_globally": (
            False
        ),

        "numeric_stats_rule": (
            "Normalization numbers are fitted "
            "separately from the exact training "
            "partition of each inner-selection "
            "or final-refit run."
        ),

        "test_execution": {
            "preferred_command": (
                "python -m pytest"
            ),

            "pytest_pythonpath": (
                "."
            ),
        },

        "hashes": {
            "config_sha256": (
                config_hash
            ),

            "implementation_sha256": (
                implementation_hash
            ),

            "test_sha256": (
                test_hash
            ),

            "pytest_config_sha256": (
                pytest_config_hash
            ),

            "upstream": (
                upstream_hashes
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

    markdown = f"""# Stage 18 — Preprocessing Contract

## Primary representation

The primary recognition input remains the supplied 38-dimensional
rotation-coordinate time series.

No temporal angle unwrapping is performed.

## Primary normalization

Each feature is standardized using:

`(x - training_mean) / training_std`

The mean and standard deviation are fitted only from the exact current
training partition.

Statistics are calculated across all valid training time steps.
Therefore the normalization is frame-weighted.

Population standard deviation (`ddof = 0`) is used.

Statistics are accumulated in float64 and normalized sequences are
returned as float32.

## Leakage rule

Outer-test recordings never contribute to outer-fold normalization.

Held-out LOSO subjects never contribute to LOSO normalization.

Held-out template groups never contribute to template-holdout
normalization.

During inner model selection, inner-validation recordings do not
contribute to fitted normalization statistics.

After model-selection decisions are frozen, final refitting may fit
new statistics using the complete outer-training pool while still
excluding the final test partition.

## Primary disabled operations

No filtering, smoothing, angle unwrapping, resampling, truncation,
augmentation or per-sample standardization is used in the primary
architecture comparison.

## Padding

Normalization occurs before batching and padding.

Future padding uses zero in normalized space and must always be paired
with valid sequence lengths or masks.

## Model adapters

C1, C2 and C3 receive the normalized variable-length `T x 38`
sequence directly.

G1 and C4 receive the same normalized sequence before their respective
model-specific temporal/window adapters.

The G1/C4 temporal stride is not selected in Stage 18.

Stage 20 freezes the shared CTC-feasible temporal rule.

## Circular ablation

The Stage-8 circular representation remains optional rather than
primary.

It maps each 38-dimensional degree-scaled rotation vector to an
interleaved 76-dimensional representation:

`sin(f0), cos(f0), ..., sin(f37), cos(f37)`

If evaluated later, its own normalization statistics must again be
fitted only from the corresponding training partition.

## Numeric statistics

Stage 18 freezes the preprocessing algorithm.

It deliberately does not freeze one global vector of means and
standard deviations.

Global-dataset normalization is prohibited because it would allow
held-out data to affect preprocessing.

## Test execution

The project root is exposed to pytest through `pytest.ini`.

The preferred test command is:

`python -m pytest`

## Frozen hashes

Preprocessing config:

`{config_hash}`

Implementation:

`{implementation_hash}`

Unit tests:

`{test_hash}`

Pytest configuration:

`{pytest_config_hash}`
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("PRIMARY CONTRACT")
    print("-" * 78)

    print(
        "Representation:",
        "raw_rotation_38d",
    )

    print(
        "Feature width:",
        38,
    )

    print(
        "Normalization:",
        "featurewise_zscore",
    )

    print(
        "Fit scope:",
        "current_training_partition_only",
    )

    print(
        "Aggregation:",
        "all valid training timesteps "
        "(frame-weighted)",
    )

    print(
        "Population ddof:",
        0,
    )

    print(
        "Fit dtype:",
        "float64",
    )

    print(
        "Output dtype:",
        "float32",
    )

    print()
    print("PRIMARY DISABLED OPERATIONS")
    print("-" * 78)

    for key in forbidden_primary:

        print(
            f"{key}: False"
        )

    print()
    print("UPSTREAM HASH VERIFICATION")
    print("-" * 78)

    for name, value in (
        upstream_hashes.items()
    ):

        print(
            f"{name}: PASS | {value}"
        )

    print()
    print("CONTRACT HASHES")
    print("-" * 78)

    print(
        "Config:",
        config_hash,
    )

    print(
        "Implementation:",
        implementation_hash,
    )

    print(
        "Tests:",
        test_hash,
    )

    print(
        "Pytest config:",
        pytest_config_hash,
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

    print()

    print(
        "FINAL STATUS: "
        "STAGE 18 CONTRACT BUILD PASSED"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()

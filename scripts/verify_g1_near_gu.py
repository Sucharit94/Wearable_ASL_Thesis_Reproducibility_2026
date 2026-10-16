from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from src.models.contracts import (
    load_model_catalog,
    validate_model_output,
)
from src.models.g1_near_gu import (
    G1_CNN_EMBEDDING_SIZE,
    G1_CTC_CLASSES,
    G1_EXPECTED_TRAINABLE_PARAMETERS,
    G1_FEATURE_WIDTH,
    G1_LSTM_HIDDEN_SIZE,
    G1_LSTM_OUTPUT_SIZE,
    G1_WINDOW_LENGTH,
    G1_WINDOW_STRIDE,
    G1NearGu,
    count_trainable_parameters,
)
from src.training.ctc_loss import (
    torch_ctc_loss_from_logits,
)


ROOT = Path(__file__).resolve().parents[1]

CATALOG_PATH = (
    ROOT
    / "configs"
    / "models"
    / "stage25_model_catalog.json"
)

STAGE25_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "stage25_model_interface_contract.json"
)

STAGE25_HASH_PATH = (
    ROOT
    / "artifacts"
    / "models"
    / "stage25_model_catalog.sha256"
)

MODEL_PATH = (
    ROOT
    / "src"
    / "models"
    / "g1_near_gu.py"
)

TEST_PATH = (
    ROOT
    / "tests"
    / "test_g1_near_gu.py"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "models"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage26_g1_model_contract.json"
)

HASH_PATH = (
    OUTPUT_DIR
    / "stage26_g1_model.sha256"
)

DOC_PATH = (
    ROOT
    / "docs"
    / "STAGE_26_G1_NEAR_GU.md"
)


def read_json(
    path: Path,
) -> dict:

    return json.loads(
        path.read_text(
            encoding="utf-8-sig"
        )
    )


def read_recorded_hash(
    path: Path,
) -> str:

    return (
        path.read_text(
            encoding="utf-8-sig"
        )
        .strip()
        .split()[0]
        .lower()
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
                return digest.hexdigest()

            digest.update(
                chunk
            )


def trainable_parameters(
    module: torch.nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in module.parameters()
        if parameter.requires_grad
    )


def main() -> None:

    print("=" * 78)
    print(
        "STAGE 26 — G1 DIMENSION-CONSISTENT "
        "NEAR-GU MODEL"
    )
    print("=" * 78)

    stage25 = read_json(
        STAGE25_PATH
    )

    catalog = load_model_catalog(
        CATALOG_PATH
    )

    specification = catalog[
        "g1"
    ]

    stage25_hash_match = (
        read_recorded_hash(
            STAGE25_HASH_PATH
        )
        == sha256_file(
            CATALOG_PATH
        )
    )

    stage25_dependency = (
        int(
            stage25[
                "stage"
            ]
        ) == 25

        and int(
            stage25[
                "catalog"
            ][
                "model_count"
            ]
        ) == 5

        and stage25[
            "architectures"
        ][
            "families"
        ][
            "g1"
        ] == (
            "window_conv2d_two_bilstm_ctc"
        )

        and stage25[
            "dependencies"
        ][
            "stage22_window_length_safe"
        ] is True

        and stage25[
            "dependencies"
        ][
            "stage24_registry_passed"
        ] is True

        and stage25_hash_match
    )

    if not stage25_dependency:

        raise RuntimeError(
            "Stage-25 model-contract "
            "dependency failed."
        )

    torch.manual_seed(
        2601
    )

    model = G1NearGu()
    model.eval()

    parameter_breakdown = {
        "window_encoder": (
            trainable_parameters(
                model.window_encoder
            )
        ),

        "bilstm1": (
            trainable_parameters(
                model.bilstm1
            )
        ),

        "bilstm2": (
            trainable_parameters(
                model.bilstm2
            )
        ),

        "classifier": (
            trainable_parameters(
                model.classifier
            )
        ),
    }

    total_parameters = (
        count_trainable_parameters(
            model
        )
    )

    expected_breakdown = {
        "window_encoder": 93_120,
        "bilstm1": 2_887_680,
        "bilstm2": 1_576_960,
        "classifier": 233_928,
    }

    parameter_count_passed = (
        parameter_breakdown
        == expected_breakdown

        and total_parameters
        == G1_EXPECTED_TRAINABLE_PARAMETERS

        and sum(
            parameter_breakdown.values()
        ) == total_parameters
    )

    if not parameter_count_passed:

        raise RuntimeError(
            "G1 trainable-parameter "
            "count changed."
        )

    value = torch.zeros(
        2,
        1,
        36,
        38,
    )

    shape_trace = {
        "input": list(
            value.shape
        )
    }

    with torch.no_grad():

        for (
            name,
            layer,
        ) in (
            model
            .window_encoder
            .named_children()
        ):

            value = layer(
                value
            )

            if name in {
                "pool1",
                "pool2",
                "pool3",
            }:

                shape_trace[
                    f"after_{name}"
                ] = list(
                    value.shape
                )

    shape_trace[
        "flattened"
    ] = list(
        value.flatten(
            start_dim=1
        ).shape
    )

    expected_shape_trace = {
        "input": [
            2,
            1,
            36,
            38,
        ],

        "after_pool1": [
            2,
            32,
            18,
            19,
        ],

        "after_pool2": [
            2,
            64,
            6,
            6,
        ],

        "after_pool3": [
            2,
            128,
            3,
            3,
        ],

        "flattened": [
            2,
            1152,
        ],
    }

    shape_trace_passed = (
        shape_trace
        == expected_shape_trace
    )

    if not shape_trace_passed:

        raise RuntimeError(
            "G1 Conv2D shape trace changed."
        )

    torch.manual_seed(
        2602
    )

    probe = torch.randn(
        2,
        90,
        38,
    )

    changed_padding = (
        probe.clone()
    )

    changed_padding[
        0,
        72:,
        :,
    ] = 1000.0

    with torch.no_grad():

        output = model(
            probe,
            [
                72,
                90,
            ],
        )

        changed_output = model(
            changed_padding,
            [
                72,
                90,
            ],
        )

    validated_lengths = (
        validate_model_output(
            output
        )
    )

    output_shape_passed = (
        tuple(
            output.logits_btc.shape
        ) == (
            2,
            4,
            456,
        )
    )

    output_lengths_passed = (
        validated_lengths
        == (
            3,
            4,
        )
    )

    padded_logits_zero = (
        torch.count_nonzero(
            output.logits_btc[
                0,
                3:,
            ]
        ).item()
        == 0
    )

    padding_max_abs_difference = float(
        (
            output.logits_btc[
                0,
                :3,
            ]
            - changed_output.logits_btc[
                0,
                :3,
            ]
        )
        .abs()
        .max()
        .item()
    )

    padding_invariance_passed = (
        padding_max_abs_difference
        == 0.0
    )

    if not all(
        [
            output_shape_passed,
            output_lengths_passed,
            padded_logits_zero,
            padding_invariance_passed,
        ]
    ):

        raise RuntimeError(
            "G1 length-safe forward "
            "verification failed."
        )

    torch.manual_seed(
        2604
    )

    backward_model = G1NearGu()
    backward_model.train()

    backward_output = backward_model(
        torch.randn(
            2,
            90,
            38,
        ),
        [
            72,
            90,
        ],
    )

    targets = torch.tensor(
        [
            [
                1,
                2,
            ],
            [
                3,
                3,
            ],
        ],
        dtype=torch.long,
    )

    target_lengths = torch.tensor(
        [
            2,
            2,
        ],
        dtype=torch.long,
    )

    loss = torch_ctc_loss_from_logits(
        backward_output.logits_btc,
        targets,
        backward_output.output_lengths,
        target_lengths,
    )

    finite_loss = bool(
        torch.isfinite(
            loss
        ).item()
    )

    loss.backward()

    trainable = [
        parameter
        for parameter
        in backward_model.parameters()
        if parameter.requires_grad
    ]

    gradients = [
        parameter.grad
        for parameter in trainable
    ]

    gradient_tensor_count = len(
        gradients
    )

    missing_gradient_count = sum(
        gradient is None
        for gradient in gradients
    )

    nonfinite_gradient_count = sum(
        gradient is not None
        and not torch.isfinite(
            gradient
        ).all().item()
        for gradient in gradients
    )

    gradient_norm = math.sqrt(
        sum(
            float(
                gradient
                .detach()
                .double()
                .square()
                .sum()
                .item()
            )
            for gradient in gradients
            if gradient is not None
        )
    )

    backward_passed = (
        finite_loss

        and gradient_tensor_count > 0

        and missing_gradient_count == 0

        and nonfinite_gradient_count == 0

        and math.isfinite(
            gradient_norm
        )

        and gradient_norm > 0.0
    )

    if not backward_passed:

        raise RuntimeError(
            "G1 CTC backward "
            "verification failed."
        )

    model_hash = sha256_file(
        MODEL_PATH
    )

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 26,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "scientific_identity": {
            "model_id": (
                "g1"
            ),

            "name": (
                "NearGu-CNN2D-BiLSTM-CTC"
            ),

            "claim": (
                "dimension_consistent_"
                "near_gu_implementation"
            ),

            "exact_paper_reproduction_claimed": (
                False
            ),

            "reported_structure_followed": (
                "Conv2D_two_BiLSTM_CTC"
            ),

            "declared_ambiguity_resolution": (
                "256 hidden units per direction, "
                "producing 512 features"
            ),

            "source_feature_description": (
                "38 wearable motion-capture-derived "
                "joint-rotation features"
            ),
        },

        "input": {
            "layout": (
                "B_T_38"
            ),

            "feature_width": (
                G1_FEATURE_WIDTH
            ),

            "window_layout": (
                "N_1_36_38"
            ),

            "window_length": (
                G1_WINDOW_LENGTH
            ),

            "window_stride": (
                G1_WINDOW_STRIDE
            ),

            "full_valid_windows_only": (
                True
            ),

            "window_length_formula": (
                "1 + floor((T - 36) / 18)"
            ),

            "padded_windows_enter_cnn": (
                False
            ),
        },

        "conv2d_encoder": {
            "channels": [
                1,
                32,
                64,
                128,
            ],

            "kernel_sizes": [
                3,
                3,
                3,
            ],

            "strides": [
                1,
                1,
                1,
            ],

            "paddings": [
                1,
                1,
                1,
            ],

            "pool_kernel_strides": [
                [
                    2,
                    2,
                ],
                [
                    3,
                    3,
                ],
                [
                    2,
                    2,
                ],
            ],

            "batch_norm_after_each_convolution": (
                True
            ),

            "activation": (
                "ReLU"
            ),

            "dropout": 0.0,

            "shape_trace": (
                shape_trace
            ),

            "embedding_size": (
                G1_CNN_EMBEDDING_SIZE
            ),

            "shape_trace_passed": (
                shape_trace_passed
            ),
        },

        "recurrent_encoder": {
            "bilstm_layers": 2,

            "hidden_size_per_direction": (
                G1_LSTM_HIDDEN_SIZE
            ),

            "directions": 2,

            "output_features": (
                G1_LSTM_OUTPUT_SIZE
            ),

            "packed_sequence_processing": (
                True
            ),

            "dropout": 0.0,
        },

        "ctc_output": {
            "raw_logits": True,

            "classes": (
                G1_CTC_CLASSES
            ),

            "blank_id": 0,

            "classifier_input_features": (
                G1_LSTM_OUTPUT_SIZE
            ),

            "valid_output_lengths_returned": (
                True
            ),

            "padded_output_logits_zeroed": (
                True
            ),
        },

        "parameters": {
            "breakdown": (
                parameter_breakdown
            ),

            "trainable_total": (
                total_parameters
            ),

            "expected_total": (
                G1_EXPECTED_TRAINABLE_PARAMETERS
            ),

            "count_passed": (
                parameter_count_passed
            ),
        },

        "verification": {
            "probe_input_shape": [
                2,
                90,
                38,
            ],

            "probe_raw_lengths": [
                72,
                90,
            ],

            "probe_output_shape": list(
                output.logits_btc.shape
            ),

            "probe_output_lengths": list(
                validated_lengths
            ),

            "output_shape_passed": (
                output_shape_passed
            ),

            "output_lengths_passed": (
                output_lengths_passed
            ),

            "padded_logits_zero": (
                padded_logits_zero
            ),

            "padding_invariance_passed": (
                padding_invariance_passed
            ),

            "padding_max_abs_difference": (
                padding_max_abs_difference
            ),

            "ctc_loss_value": float(
                loss.detach().item()
            ),

            "ctc_loss_finite": (
                finite_loss
            ),

            "gradient_tensor_count": (
                gradient_tensor_count
            ),

            "missing_gradient_count": (
                missing_gradient_count
            ),

            "nonfinite_gradient_count": (
                nonfinite_gradient_count
            ),

            "gradient_global_norm": (
                gradient_norm
            ),

            "backward_passed": (
                backward_passed
            ),
        },

        "training_defaults": {
            "optimizer": (
                specification.optimizer
            ),

            "learning_rate": (
                specification.learning_rate
            ),

            "epoch_cap": (
                specification.epoch_cap
            ),

            "fixed_epoch_reference": (
                specification.fixed_epoch_reference
            ),

            "physical_batch_size": (
                specification.physical_batch_size
            ),

            "gradient_clip_norm": (
                specification.gradient_clip_norm
            ),

            "primary_augmentation": (
                "none"
            ),
        },

        "runtime": {
            "torch_version": str(
                torch.__version__
            ),

            "cuda_available_locally": (
                torch.cuda.is_available()
            ),

            "verification_device": (
                "cpu"
            ),
        },

        "dependencies": {
            "stage25_contract_passed": (
                stage25_dependency
            ),

            "stage25_catalog_hash_match": (
                stage25_hash_match
            ),

            "stage22_window_rule_preserved": (
                True
            ),

            "stage21_ctc_loss_wrapper_used": (
                True
            ),
        },

        "hashes": {
            "model": (
                model_hash
            ),

            "tests": (
                sha256_file(
                    TEST_PATH
                )
            ),

            "verifier": (
                sha256_file(
                    Path(
                        __file__
                    ).resolve()
                )
            ),
        },
    }

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    DOC_PATH.parent.mkdir(
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

    HASH_PATH.write_text(
        (
            f"{model_hash}  "
            "src/models/g1_near_gu.py\n"
        ),
        encoding="utf-8",
    )

    markdown = f"""# Stage 26 — G1 Dimension-Consistent Near-Gu Model

## Scientific identity

G1 is a dimension-consistent near-Gu implementation following the reported
Conv2D, two-BiLSTM and CTC structure. It is not claimed as an exact paper
reproduction.

The source architecture contains a dimensional ambiguity. This implementation
explicitly uses 256 hidden units per direction, producing 512 output features
from each BiLSTM layer.

The input contains 38 wearable motion-capture-derived joint-rotation features.

## Window adapter

- Window size: 36 x 38
- Temporal stride: 18
- Complete valid windows only
- Padded windows entering Conv2D: False

## Conv2D shape trace

- Input: {shape_trace["input"]}
- After pool 1: {shape_trace["after_pool1"]}
- After pool 2: {shape_trace["after_pool2"]}
- After pool 3: {shape_trace["after_pool3"]}
- Flattened embedding: {shape_trace["flattened"]}

## Recurrent and CTC output

- Two packed bidirectional LSTM layers
- Hidden size: 256 per direction
- Recurrent output size: 512
- Raw CTC classes: 456
- Blank ID: 0

## Parameters

- Conv2D encoder: {parameter_breakdown["window_encoder"]}
- BiLSTM 1: {parameter_breakdown["bilstm1"]}
- BiLSTM 2: {parameter_breakdown["bilstm2"]}
- Classifier: {parameter_breakdown["classifier"]}
- Total: {total_parameters}

## Safety verification

- Padding-invariant valid logits: {padding_invariance_passed}
- Padded output logits zero: {padded_logits_zero}
- Finite CTC loss: {finite_loss}
- Missing gradients: {missing_gradient_count}
- Non-finite gradients: {nonfinite_gradient_count}
- Backward test passed: {backward_passed}

## Runtime

- PyTorch: {torch.__version__}
- Verification device: CPU
- Local CUDA available: {torch.cuda.is_available()}
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    print()
    print("SCIENTIFIC IDENTITY")
    print("-" * 78)

    print(
        "Model ID:",
        "g1",
    )

    print(
        "Claim:",
        "dimension-consistent near-Gu implementation",
    )

    print(
        "Exact paper reproduction claimed:",
        False,
    )

    print(
        "Feature source:",
        "wearable motion-capture-derived joint rotations",
    )

    print()
    print("CONV2D SHAPE TRACE")
    print("-" * 78)

    for (
        name,
        shape,
    ) in shape_trace.items():

        print(
            name + ":",
            shape,
        )

    print()
    print("PARAMETERS")
    print("-" * 78)

    for (
        name,
        count,
    ) in parameter_breakdown.items():

        print(
            name + ":",
            count,
        )

    print(
        "Trainable total:",
        total_parameters,
    )

    print(
        "Expected total:",
        G1_EXPECTED_TRAINABLE_PARAMETERS,
    )

    print(
        "Parameter count passed:",
        parameter_count_passed,
    )

    print()
    print("LENGTH AND PADDING SAFETY")
    print("-" * 78)

    print(
        "Output shape:",
        list(
            output.logits_btc.shape
        ),
    )

    print(
        "Output lengths:",
        list(
            validated_lengths
        ),
    )

    print(
        "Padded logits zero:",
        padded_logits_zero,
    )

    print(
        "Padding max absolute difference:",
        padding_max_abs_difference,
    )

    print(
        "Padding invariance passed:",
        padding_invariance_passed,
    )

    print()
    print("CTC BACKWARD TEST")
    print("-" * 78)

    print(
        "Loss finite:",
        finite_loss,
    )

    print(
        "Gradient tensors:",
        gradient_tensor_count,
    )

    print(
        "Missing gradients:",
        missing_gradient_count,
    )

    print(
        "Non-finite gradients:",
        nonfinite_gradient_count,
    )

    print(
        "Gradient global norm:",
        gradient_norm,
    )

    print(
        "Backward passed:",
        backward_passed,
    )

    print()
    print("RUNTIME")
    print("-" * 78)

    print(
        "PyTorch:",
        torch.__version__,
    )

    print(
        "CUDA available locally:",
        torch.cuda.is_available(),
    )

    print(
        "Verification device:",
        "cpu",
    )

    print()
    print("FROZEN DEPENDENCIES")
    print("-" * 78)

    print(
        "Stage-25 contract passed:",
        stage25_dependency,
    )

    print(
        "Stage-25 catalog hash match:",
        stage25_hash_match,
    )

    print(
        "Stage-22 window rule preserved:",
        True,
    )

    print(
        "Stage-21 CTC loss wrapper used:",
        True,
    )

    print()
    print("OUTPUTS")
    print("-" * 78)

    print(
        "Saved:",
        CONTRACT_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        HASH_PATH.relative_to(
            ROOT
        ),
    )

    print(
        "Saved:",
        DOC_PATH.relative_to(
            ROOT
        ),
    )

    print()

    print(
        "FINAL STATUS: "
        "STAGE 26 BUILD PASSED"
    )

    print("=" * 78)


if __name__ == "__main__":
    main()

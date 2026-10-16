from __future__ import annotations

import hashlib
import json
import math
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path

import numpy as np
import torch

from src.data.lengths import (
    full_window_lengths,
)
from src.models.c4_tiny_swin import (
    C4_EXPECTED_TRAINABLE_PARAMETERS,
    C4TinySwin,
    c4_unfold_windows,
    c4_window_lengths,
    count_c4_trainable_parameters,
    padded_grid_size,
)
from src.models.contracts import (
    load_model_catalog,
    validate_model_output,
)
from src.training.ctc_loss import (
    torch_ctc_loss_from_logits,
)


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)

CATALOG_PATH = (
    PROJECT_ROOT
    / "configs"
    / "models"
    / "stage25_model_catalog.json"
)

CATALOG_HASH_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "stage25_model_catalog.sha256"
)

STAGE26_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "stage26_g1_model_contract.json"
)

G1_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "g1_near_gu.py"
)

STAGE29_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "stage29_c3_model_contract.json"
)

C3_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "c3_transformer.py"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "c4_tiny_swin.py"
)

TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_c4_tiny_swin.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage30_c4_model_contract.json"
)

MODEL_HASH_PATH = (
    OUTPUT_DIR
    / "stage30_c4_model.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_30_C4_TINY_SWIN.md"
)


def sha256_file(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open(
        "rb"
    ) as handle:

        for chunk in iter(
            lambda: handle.read(
                1024 * 1024
            ),
            b"",
        ):

            digest.update(
                chunk
            )

    return digest.hexdigest()


def read_hash(
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


def count_parameters(
    *modules: torch.nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for module in modules
        for parameter
        in module.parameters()
        if parameter.requires_grad
    )


def section(
    title: str,
) -> None:

    print()
    print(
        title
    )
    print(
        "-" * 78
    )


def main() -> None:

    print(
        "=" * 78
    )

    print(
        "STAGE 30 — C4 "
        "TINY-SWIN-BILSTM-CTC MODEL"
    )

    print(
        "=" * 78
    )

    catalog_hash_match = (
        read_hash(
            CATALOG_HASH_PATH
        )
        == sha256_file(
            CATALOG_PATH
        )
    )

    if not catalog_hash_match:

        raise RuntimeError(
            "Stage-25 catalog hash mismatch."
        )

    c4_spec = load_model_catalog(
        CATALOG_PATH
    )[
        "c4"
    ]

    expected_spec = {
        "model_id": "c4",
        "scientific_role": (
            "tiny_swin_window_attention_comparison"
        ),
        "architecture_family": (
            "tiny_swin_bilstm_ctc"
        ),
        "input_adapter": "windowed_2d",
        "feature_width": 38,
        "output_classes": 456,
        "blank_id": 0,
        "target_padding_value": -1,
        "returns_raw_logits": True,
        "valid_output_length_required": True,
        "window_length": 36,
        "window_stride": 18,
        "uses_bilstm": True,
        "uses_self_attention": False,
        "uses_transformer_encoder": False,
        "uses_tiny_swin": True,
        "optimizer": "AdamW",
        "learning_rate": 0.00005,
        "weight_decay": 0.0001,
        "epoch_cap": 80,
        "early_stopping_patience": 8,
        "physical_batch_size": 16,
        "gradient_clip_norm": 5.0,
        "scheduler": "ReduceLROnPlateau",
        "fixed_epoch_reference": False,
    }

    observed_spec = (
        c4_spec.to_dict()
    )

    c4_spec_passed = all(
        observed_spec[
            key
        ] == expected
        for (
            key,
            expected,
        ) in expected_spec.items()
    )

    if not c4_spec_passed:

        raise RuntimeError(
            "Stage-25 C4 specification changed."
        )

    stage26 = json.loads(
        STAGE26_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage26_passed = all(
        [
            int(
                stage26[
                    "stage"
                ]
            ) == 26,

            stage26[
                "scientific_identity"
            ][
                "model_id"
            ] == "g1",

            int(
                stage26[
                    "input"
                ][
                    "window_length"
                ]
            ) == 36,

            int(
                stage26[
                    "input"
                ][
                    "window_stride"
                ]
            ) == 18,

            stage26[
                "input"
            ][
                "full_valid_windows_only"
            ] is True,

            stage26[
                "verification"
            ][
                "padding_invariance_passed"
            ] is True,

            stage26[
                "verification"
            ][
                "backward_passed"
            ] is True,

            stage26[
                "hashes"
            ][
                "model"
            ] == sha256_file(
                G1_PATH
            ),
        ]
    )

    if not stage26_passed:

        raise RuntimeError(
            "Stage-26 G1 window "
            "dependency is invalid."
        )

    stage29 = json.loads(
        STAGE29_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage29_passed = all(
        [
            int(
                stage29[
                    "stage"
                ]
            ) == 29,

            stage29[
                "identity"
            ][
                "model_id"
            ] == "c3",

            stage29[
                "parameter_counts"
            ][
                "passed"
            ] is True,

            stage29[
                "length_and_mask_safety"
            ][
                "transformer_mask_passed"
            ] is True,

            stage29[
                "ctc_backward"
            ][
                "passed"
            ] is True,

            stage29[
                "hashes"
            ][
                "model"
            ] == sha256_file(
                C3_PATH
            ),
        ]
    )

    if not stage29_passed:

        raise RuntimeError(
            "Stage-29 C3 dependency is invalid."
        )

    raw_probe = np.asarray(
        [
            36,
            53,
            54,
            71,
            72,
            81,
            90,
            108,
            831,
        ],
        dtype=np.int64,
    )

    expected_window_lengths = (
        full_window_lengths(
            raw_probe,
            window_length=36,
            stride=18,
        )
    )

    observed_window_lengths = (
        c4_window_lengths(
            raw_probe
        )
        .numpy()
    )

    window_formula_match = (
        observed_window_lengths.tolist()
        == expected_window_lengths.tolist()
    )

    if not window_formula_match:

        raise RuntimeError(
            "C4 window counts disagree "
            "with Stage 22."
        )

    order_source = torch.arange(
        72 * 38,
        dtype=torch.float32,
    ).reshape(
        1,
        72,
        38,
    )

    order_windows = (
        c4_unfold_windows(
            order_source
        )
    )

    feature_order_passed = all(
        [
            list(
                order_windows.shape
            ) == [
                1,
                3,
                36,
                38,
            ],

            torch.equal(
                order_windows[
                    0,
                    0,
                ],
                order_source[
                    0,
                    0:36,
                ],
            ),

            torch.equal(
                order_windows[
                    0,
                    1,
                ],
                order_source[
                    0,
                    18:54,
                ],
            ),

            torch.equal(
                order_windows[
                    0,
                    2,
                ],
                order_source[
                    0,
                    36:72,
                ],
            ),
        ]
    )

    if not feature_order_passed:

        raise RuntimeError(
            "C4 windowing changed "
            "temporal/feature order."
        )

    torch.manual_seed(
        3006
    )

    device = torch.device(
        "cpu"
    )

    model = C4TinySwin().to(
        device
    )

    model.eval()

    parameters = {
        "window_encoder": (
            count_parameters(
                model.window_encoder
            )
        ),
        "bilstm": (
            count_parameters(
                model.bilstm
            )
        ),
        "post_lstm_normalization": (
            count_parameters(
                model.post_lstm_normalization
            )
        ),
        "classifier": (
            count_parameters(
                model.classifier
            )
        ),
        "trainable_total": (
            count_c4_trainable_parameters(
                model
            )
        ),
        "expected_total": (
            C4_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }

    expected_parameters = {
        "window_encoder": 135_796,
        "bilstm": 124_416,
        "post_lstm_normalization": 384,
        "classifier": 88_008,
        "trainable_total": 348_604,
        "expected_total": 348_604,
    }

    parameter_count_passed = (
        parameters
        == expected_parameters
    )

    parameter_budget_passed = (
        parameters[
            "trainable_total"
        ]
        < 1_500_000
    )

    if (
        not parameter_count_passed
        or not parameter_budget_passed
    ):

        raise RuntimeError(
            "C4 parameter contract failed."
        )

    shape_trace = {}

    shifted_calls = {
        "stage1": 0,
        "stage2": 0,
    }

    def patch_hook(
        _module,
        _inputs,
        output,
    ):

        shape_trace[
            "after_patch_projection_nchw"
        ] = list(
            output.shape
        )

    def stage1_hook(
        _module,
        _inputs,
        output,
    ):

        shape_trace[
            "after_stage1_nhwc"
        ] = list(
            output.shape
        )

    def merge_hook(
        _module,
        _inputs,
        output,
    ):

        shape_trace[
            "after_patch_merge_nhwc"
        ] = list(
            output[0].shape
        )

    def stage2_hook(
        _module,
        _inputs,
        output,
    ):

        shape_trace[
            "after_stage2_nhwc"
        ] = list(
            output.shape
        )

    def shifted1_hook(
        _module,
        _inputs,
        _output,
    ):

        shifted_calls[
            "stage1"
        ] += 1

    def shifted2_hook(
        _module,
        _inputs,
        _output,
    ):

        shifted_calls[
            "stage2"
        ] += 1

    handles = [
        (
            model.window_encoder
            .patch_projection
            .register_forward_hook(
                patch_hook
            )
        ),
        (
            model.window_encoder
            .stage1
            .register_forward_hook(
                stage1_hook
            )
        ),
        (
            model.window_encoder
            .patch_merging
            .register_forward_hook(
                merge_hook
            )
        ),
        (
            model.window_encoder
            .stage2
            .register_forward_hook(
                stage2_hook
            )
        ),
        (
            model.window_encoder
            .stage1
            .blocks[1]
            .register_forward_hook(
                shifted1_hook
            )
        ),
        (
            model.window_encoder
            .stage2
            .blocks[1]
            .register_forward_hook(
                shifted2_hook
            )
        ),
    ]

    try:

        with torch.no_grad():

            window_embeddings = (
                model.encode_windows(
                    torch.randn(
                        2,
                        1,
                        36,
                        38,
                        device=device,
                    )
                )
            )

    finally:

        for handle in handles:

            handle.remove()

    shape_trace[
        "window_embedding"
    ] = list(
        window_embeddings.shape
    )

    expected_shape_trace = {
        "after_patch_projection_nchw": [
            2,
            32,
            18,
            19,
        ],
        "after_stage1_nhwc": [
            2,
            18,
            19,
            32,
        ],
        "after_patch_merge_nhwc": [
            2,
            9,
            10,
            64,
        ],
        "after_stage2_nhwc": [
            2,
            9,
            10,
            64,
        ],
        "window_embedding": [
            2,
            64,
        ],
    }

    shape_trace_passed = (
        shape_trace
        == expected_shape_trace
    )

    shifted_blocks_passed = all(
        [
            shifted_calls == {
                "stage1": 1,
                "stage2": 1,
            },

            (
                model.window_encoder
                .stage1
                .blocks[1]
                .shift_size
                == (
                    4,
                    2,
                )
            ),

            (
                model.window_encoder
                .stage2
                .blocks[1]
                .shift_size
                == (
                    4,
                    2,
                )
            ),
        ]
    )

    minimal_grid_padding_passed = all(
        [
            padded_grid_size(
                18,
                19,
            ) == (
                24,
                20,
            ),

            padded_grid_size(
                9,
                10,
            ) == (
                16,
                10,
            ),

            padded_grid_size(
                3,
                2,
            ) == (
                8,
                5,
            ),

            padded_grid_size(
                25,
                21,
            ) == (
                32,
                25,
            ),
        ]
    )

    stage1_attention = (
        model.window_encoder
        .stage1
        .blocks[0]
        .attention
    )

    stage2_attention = (
        model.window_encoder
        .stage2
        .blocks[0]
        .attention
    )

    relative_position_bias_passed = all(
        [
            list(
                stage1_attention
                .relative_position_bias_table
                .shape
            ) == [
                135,
                2,
            ],

            list(
                stage2_attention
                .relative_position_bias_table
                .shape
            ) == [
                135,
                4,
            ],

            list(
                stage1_attention
                .relative_position_index
                .shape
            ) == [
                40,
                40,
            ],

            list(
                stage2_attention
                .relative_position_index
                .shape
            ) == [
                40,
                40,
            ],

            (
                stage1_attention
                .relative_position_bias_table
                .requires_grad
            ),

            (
                stage2_attention
                .relative_position_bias_table
                .requires_grad
            ),
        ]
    )

    spatial_input = torch.randn(
        1,
        7,
        9,
        32,
        device=device,
    )

    spatial_mask = torch.zeros(
        1,
        7,
        9,
        dtype=torch.bool,
        device=device,
    )

    spatial_mask[
        :,
        :5,
        :7,
    ] = True

    changed_spatial = (
        spatial_input.clone()
    )

    changed_spatial[
        ~spatial_mask
    ] = 1000.0

    shifted_block = (
        model.window_encoder
        .stage1
        .blocks[1]
    )

    with torch.no_grad():

        first_spatial = shifted_block(
            spatial_input,
            spatial_mask,
        )

        second_spatial = shifted_block(
            changed_spatial,
            spatial_mask,
        )

    spatial_padding_difference = float(
        (
            first_spatial[
                spatial_mask
            ]
            - second_spatial[
                spatial_mask
            ]
        )
        .abs()
        .max()
        .item()
    )

    padded_spatial_outputs_zero = all(
        [
            torch.count_nonzero(
                first_spatial[
                    ~spatial_mask
                ]
            ).item() == 0,

            torch.count_nonzero(
                second_spatial[
                    ~spatial_mask
                ]
            ).item() == 0,
        ]
    )

    spatial_padding_invariance_passed = (
        spatial_padding_difference == 0.0
        and padded_spatial_outputs_zero
    )

    merge_input = torch.randn(
        1,
        5,
        7,
        32,
        device=device,
    )

    merge_mask = torch.zeros(
        1,
        5,
        7,
        dtype=torch.bool,
        device=device,
    )

    merge_mask[
        :,
        :4,
        :6,
    ] = True

    changed_merge = (
        merge_input.clone()
    )

    changed_merge[
        ~merge_mask
    ] = 1000.0

    with torch.no_grad():

        (
            first_merge,
            first_merge_mask,
        ) = (
            model.window_encoder
            .patch_merging(
                merge_input,
                merge_mask,
            )
        )

        (
            second_merge,
            second_merge_mask,
        ) = (
            model.window_encoder
            .patch_merging(
                changed_merge,
                merge_mask,
            )
        )

    patch_merge_difference = float(
        (
            first_merge
            - second_merge
        )
        .abs()
        .max()
        .item()
    )

    patch_merge_mask_match = (
        torch.equal(
            first_merge_mask,
            second_merge_mask,
        )
    )

    patch_merge_padding_passed = (
        patch_merge_difference == 0.0
        and patch_merge_mask_match
    )

    swin_verification_passed = all(
        [
            shape_trace_passed,
            shifted_blocks_passed,
            minimal_grid_padding_passed,
            relative_position_bias_passed,
            spatial_padding_invariance_passed,
            patch_merge_padding_passed,
            torch.isfinite(
                window_embeddings
            ).all().item(),
        ]
    )

    if not swin_verification_passed:

        raise RuntimeError(
            "C4 Tiny-Swin verification failed."
        )

    shape_input = torch.randn(
        3,
        108,
        38,
        device=device,
    )

    with torch.no_grad():

        shape_output = model(
            shape_input,
            [
                72,
                90,
                108,
            ],
        )

    output_shape = list(
        shape_output.logits_btc.shape
    )

    output_lengths = list(
        validate_model_output(
            shape_output
        )
    )

    positions = torch.arange(
        output_shape[1],
        device=device,
    ).unsqueeze(
        0
    )

    output_padding_mask = (
        positions
        >= shape_output
        .output_lengths
        .to(
            device
        )
        .unsqueeze(
            1
        )
    )

    padded_logits_zero = (
        torch.count_nonzero(
            shape_output.logits_btc[
                output_padding_mask
            ]
        ).item()
        == 0
    )

    raw_padding_input = torch.randn(
        2,
        90,
        38,
        device=device,
    )

    changed_raw_padding = (
        raw_padding_input.clone()
    )

    changed_raw_padding[
        0,
        72:,
        :,
    ] = 1000.0

    with torch.no_grad():

        first_raw = model(
            raw_padding_input,
            [
                72,
                90,
            ],
        )

        second_raw = model(
            changed_raw_padding,
            [
                72,
                90,
            ],
        )

    raw_padding_difference = float(
        (
            first_raw.logits_btc[
                0,
                :3,
            ]
            - second_raw.logits_btc[
                0,
                :3,
            ]
        )
        .abs()
        .max()
        .item()
    )

    raw_padding_invariance_passed = (
        raw_padding_difference
        == 0.0
    )

    length_safety_passed = all(
        [
            window_formula_match,
            feature_order_passed,

            output_shape == [
                3,
                5,
                456,
            ],

            output_lengths == [
                3,
                4,
                5,
            ],

            padded_logits_zero,
            raw_padding_invariance_passed,
        ]
    )

    if not length_safety_passed:

        raise RuntimeError(
            "C4 length/padding safety failed."
        )

    torch.manual_seed(
        3007
    )

    backward_model = (
        C4TinySwin().to(
            device
        )
    )

    backward_model.train()

    backward_output = backward_model(
        torch.randn(
            2,
            90,
            38,
            device=device,
        ),
        [
            72,
            90,
        ],
    )

    loss = torch_ctc_loss_from_logits(
        backward_output.logits_btc,
        torch.tensor(
            [
                [
                    1,
                    2,
                    -1,
                ],
                [
                    3,
                    4,
                    5,
                ],
            ],
            dtype=torch.long,
        ),
        backward_output.output_lengths,
        torch.tensor(
            [
                2,
                3,
            ],
            dtype=torch.long,
        ),
    )

    loss_finite = bool(
        torch.isfinite(
            loss
        ).item()
    )

    if not loss_finite:

        raise RuntimeError(
            "C4 CTC loss is not finite."
        )

    loss.backward()

    gradients = [
        parameter.grad
        for parameter
        in backward_model.parameters()
        if parameter.requires_grad
    ]

    gradient_tensors = len(
        gradients
    )

    missing_gradients = sum(
        gradient is None
        for gradient
        in gradients
    )

    nonfinite_gradients = sum(
        (
            gradient is not None
            and not torch.isfinite(
                gradient
            ).all().item()
        )
        for gradient
        in gradients
    )

    gradient_global_norm = float(
        sum(
            (
                gradient.detach()
                .double()
                .square()
                .sum()
                .item()
            )
            for gradient
            in gradients
            if gradient is not None
        )
        ** 0.5
    )

    backward_passed = all(
        [
            loss_finite,
            gradient_tensors == 73,
            missing_gradients == 0,
            nonfinite_gradients == 0,

            math.isfinite(
                gradient_global_norm
            ),

            gradient_global_norm > 0.0,
        ]
    )

    if not backward_passed:

        raise RuntimeError(
            "C4 CTC backward proof failed."
        )

    training_defaults = {
        "optimizer": (
            c4_spec.optimizer
        ),
        "learning_rate": (
            c4_spec.learning_rate
        ),
        "weight_decay": (
            c4_spec.weight_decay
        ),
        "epoch_cap": (
            c4_spec.epoch_cap
        ),
        "early_stopping_patience": (
            c4_spec.early_stopping_patience
        ),
        "physical_batch_size": (
            c4_spec.physical_batch_size
        ),
        "gradient_clip_norm": (
            c4_spec.gradient_clip_norm
        ),
        "scheduler": (
            c4_spec.scheduler
        ),
        "primary_augmentation": "none",
    }

    hashes = {
        "model": sha256_file(
            MODEL_PATH
        ),
        "tests": sha256_file(
            TEST_PATH
        ),
        "verifier": sha256_file(
            Path(__file__).resolve()
        ),
        "stage25_catalog": sha256_file(
            CATALOG_PATH
        ),
        "stage26_contract": sha256_file(
            STAGE26_PATH
        ),
        "g1_model": sha256_file(
            G1_PATH
        ),
        "stage29_contract": sha256_file(
            STAGE29_PATH
        ),
        "c3_model": sha256_file(
            C3_PATH
        ),
    }

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),

        "stage": 30,

        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),

        "identity": {
            "model_id": "c4",

            "scientific_role": (
                "tiny_swin_window_attention_comparison"
            ),

            "architecture_family": (
                "tiny_swin_bilstm_ctc"
            ),

            "input_adapter": (
                "windowed_2d"
            ),

            "exact_standard_swin_reproduction_claimed": (
                False
            ),
        },

        "input_adapter": {
            "input_layout": "B_T_38",
            "window_layout": "N_1_36_38",
            "window_length": 36,
            "window_stride": 18,

            "window_count_formula": (
                "1 + floor((T - 36) / 18)"
            ),

            "complete_valid_windows_only": (
                True
            ),

            "padded_windows_enter_swin": (
                False
            ),

            "temporal_and_feature_order_preserved": (
                feature_order_passed
            ),

            "feature_axis_permuted": False,
            "resize_to_224": False,
        },

        "architecture": {
            "patch_projection": (
                "Conv2d(1,32,kernel=2,stride=2)"
            ),

            "patch_grid": [
                18,
                19,
            ],

            "stage_dimensions": [
                32,
                64,
            ],

            "stage_depths": [
                2,
                2,
            ],

            "stage_heads": [
                2,
                4,
            ],

            "swin_window_size": [
                8,
                5,
            ],

            "regular_shift": [
                0,
                0,
            ],

            "shifted_shift": [
                4,
                2,
            ],

            "blocks_per_stage": [
                "regular",
                "shifted",
            ],

            "mlp_ratio": 4,
            "activation": "GELU",
            "swin_dropout": 0.10,
            "stochastic_depth": 0.0,
            "relative_position_bias": True,

            "patch_merge": (
                "2x2_32_to_64"
            ),

            "post_merge_grid": [
                9,
                10,
            ],

            "window_pooling": (
                "masked_mean_over_2d_grid"
            ),

            "window_embedding_width": 64,
            "bilstm_layers": 1,

            "bilstm_hidden_per_direction": (
                96
            ),

            "bilstm_output_width": 192,
            "packed_bilstm": True,

            "post_lstm_normalization": (
                "LayerNorm(192)"
            ),

            "post_lstm_dropout": 0.15,
            "output_classes": 456,
            "blank_id": 0,

            "external_swin_library_required": (
                False
            ),
        },

        "design_resolution": {
            "status": (
                "prospective_before_training"
            ),

            "stage25_hybrid_contract_authoritative": (
                True
            ),

            "pure_swin_without_bilstm_selected": (
                False
            ),

            "roadmap_explicit": [
                "36x38_windows_stride_18",
                "tiny_swin_depths_2_2",
                "heads_2_4",
                "embedding_width_24_or_32",
                "one_embedding_per_window",
                "one_layer_bilstm_hidden_96_or_128_per_direction",
                "linear_ctc_head",
                "parameter_budget_below_1_5m",
            ],

            "resolved_before_training": [
                "embedding_width_32",
                "patch_projection_2x2",
                "rectangular_window_8x5_shift_4x2",
                "mlp_ratio_4_gelu_pre_norm",
                "relative_position_bias",
                "bilstm_hidden_96_per_direction",
                "post_lstm_layernorm_dropout_0_15",
                "no_stochastic_depth",
            ],
        },

        "parameter_counts": {
            **parameters,

            "budget_upper_bound": (
                1_500_000
            ),

            "budget_passed": (
                parameter_budget_passed
            ),

            "passed": (
                parameter_count_passed
            ),
        },

        "swin_verification": {
            "shape_trace": (
                shape_trace
            ),

            "shape_trace_passed": (
                shape_trace_passed
            ),

            "stage1_attention_grid": [
                18,
                19,
            ],

            "stage1_minimally_padded_grid": [
                24,
                20,
            ],

            "stage2_attention_grid": [
                9,
                10,
            ],

            "stage2_minimally_padded_grid": [
                16,
                10,
            ],

            "short_grid_probe": [
                3,
                2,
            ],

            "short_grid_padded": [
                8,
                5,
            ],

            "long_grid_probe": [
                25,
                21,
            ],

            "long_grid_padded": [
                32,
                25,
            ],

            "minimal_grid_padding_passed": (
                minimal_grid_padding_passed
            ),

            "shifted_stage1_calls": (
                shifted_calls[
                    "stage1"
                ]
            ),

            "shifted_stage2_calls": (
                shifted_calls[
                    "stage2"
                ]
            ),

            "shifted_blocks_passed": (
                shifted_blocks_passed
            ),

            "stage1_relative_bias_shape": (
                list(
                    stage1_attention
                    .relative_position_bias_table
                    .shape
                )
            ),

            "stage2_relative_bias_shape": (
                list(
                    stage2_attention
                    .relative_position_bias_table
                    .shape
                )
            ),

            "relative_position_bias_passed": (
                relative_position_bias_passed
            ),

            "spatial_padding_max_absolute_difference": (
                spatial_padding_difference
            ),

            "padded_spatial_outputs_zero": (
                padded_spatial_outputs_zero
            ),

            "spatial_padding_invariance_passed": (
                spatial_padding_invariance_passed
            ),

            "patch_merge_max_absolute_difference": (
                patch_merge_difference
            ),

            "patch_merge_mask_match": (
                patch_merge_mask_match
            ),

            "patch_merge_padding_passed": (
                patch_merge_padding_passed
            ),

            "passed": (
                swin_verification_passed
            ),
        },

        "length_and_padding_safety": {
            "probe_raw": (
                raw_probe.tolist()
            ),

            "stage22_expected_windows": (
                expected_window_lengths.tolist()
            ),

            "observed_windows": (
                observed_window_lengths.tolist()
            ),

            "stage22_formula_match": (
                window_formula_match
            ),

            "feature_order_passed": (
                feature_order_passed
            ),

            "output_shape": (
                output_shape
            ),

            "output_lengths": (
                output_lengths
            ),

            "padded_logits_zero": (
                padded_logits_zero
            ),

            "raw_padding_max_absolute_difference": (
                raw_padding_difference
            ),

            "raw_padding_invariance_passed": (
                raw_padding_invariance_passed
            ),

            "passed": (
                length_safety_passed
            ),
        },

        "ctc_backward": {
            "loss": float(
                loss.detach().item()
            ),

            "loss_finite": (
                loss_finite
            ),

            "gradient_tensors": (
                gradient_tensors
            ),

            "missing_gradients": (
                missing_gradients
            ),

            "nonfinite_gradients": (
                nonfinite_gradients
            ),

            "gradient_global_norm": (
                gradient_global_norm
            ),

            "passed": (
                backward_passed
            ),
        },

        "training_defaults": (
            training_defaults
        ),

        "dependencies": {
            "stage21_ctc_loss_wrapper_used": (
                True
            ),

            "stage22_window_formula_match": (
                window_formula_match
            ),

            "stage25_catalog_hash_match": (
                catalog_hash_match
            ),

            "stage25_c4_spec_passed": (
                c4_spec_passed
            ),

            "stage26_window_predecessor_passed": (
                stage26_passed
            ),

            "stage29_immediate_predecessor_passed": (
                stage29_passed
            ),
        },

        "runtime": {
            "pytorch_version": (
                torch.__version__
            ),

            "cuda_available_locally": (
                torch.cuda.is_available()
            ),

            "verification_device": (
                str(
                    device
                )
            ),
        },

        "hashes": (
            hashes
        ),
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

    MODEL_HASH_PATH.write_text(
        (
            hashes[
                "model"
            ]
            + "  "
            + "src/models/c4_tiny_swin.py\n"
        ),
        encoding="utf-8",
    )

    DOC_PATH.write_text(
        f"""# Stage 30 — C4 Tiny-Swin-BiLSTM-CTC

## Scientific identity

C4 is the task-specific Tiny-Swin window-attention comparison. It is
not claimed as an exact reproduction of standard image Swin Transformer.
The frozen Stage-25 catalog is authoritative: the primary C4 family is
`tiny_swin_bilstm_ctc`, so C4 retains a packed BiLSTM after Tiny-Swin.

## Input adapter

- Raw input: `B x T x 38`
- Complete sensor windows: `N x 1 x 36 x 38`
- Temporal stride: 18
- Window count: `1 + floor((T - 36) / 18)`
- Feature-axis permutation: none
- 224 x 224 image resizing: none
- Padded windows entering Tiny-Swin: false

## Tiny-Swin encoder

- Patch projection: Conv2D 1→32, kernel/stride 2 x 2
- Patch grid: 18 x 19
- Stage depths: [2, 2]
- Heads: [2, 4]
- Window: 8 x 5
- Shift: 4 x 2 in the second block of each stage
- Patch merge: 2 x 2, 32→64
- MLP ratio: 4
- Activation: GELU
- Relative-position bias: true
- One masked-mean 64-D embedding per valid sensor window

## Sequence backend and CTC

- Packed one-layer BiLSTM
- Hidden size: 96 per direction
- Output width: 192
- LayerNorm(192), dropout 0.15
- Raw CTC logits: 456 classes
- Blank ID: 0

## Verification

- Trainable parameters: {parameters["trainable_total"]}
- Under 1.5M budget: {parameter_budget_passed}
- Shape trace passed: {shape_trace_passed}
- Shifted block executed in stage 1: {shifted_calls["stage1"] == 1}
- Shifted block executed in stage 2: {shifted_calls["stage2"] == 1}
- Spatial-padding invariance difference: {spatial_padding_difference}
- Patch-merge padding difference: {patch_merge_difference}
- Stage-22 window formula passed: {window_formula_match}
- Feature ordering preserved: {feature_order_passed}
- Output shape: {output_shape}
- Output lengths: {output_lengths}
- Padded logits zero: {padded_logits_zero}
- Raw-padding invariance difference: {raw_padding_difference}
- CTC loss finite: {loss_finite}
- Missing gradients: {missing_gradients}
- Non-finite gradients: {nonfinite_gradients}
- Backward passed: {backward_passed}

## Training defaults

- Optimizer: {c4_spec.optimizer}
- Learning rate: {c4_spec.learning_rate}
- Weight decay: {c4_spec.weight_decay}
- Epoch cap: {c4_spec.epoch_cap}
- Early-stopping patience: {c4_spec.early_stopping_patience}
- Physical batch size: {c4_spec.physical_batch_size}
- Gradient clipping norm: {c4_spec.gradient_clip_norm}
- Scheduler: {c4_spec.scheduler}
- Primary augmentation: none
""",
        encoding="utf-8",
    )

    section(
        "SCIENTIFIC IDENTITY"
    )

    print(
        "Model ID: c4"
    )

    print(
        "Role: Tiny-Swin "
        "window-attention comparison"
    )

    print(
        "Architecture: "
        "tiny_swin_bilstm_ctc"
    )

    print(
        "Exact standard Swin "
        "reproduction claimed: False"
    )

    section(
        "WINDOW ADAPTER"
    )

    print(
        "Input / window: "
        "B x T x 38 / N x 1 x 36 x 38"
    )

    print(
        "Window length / stride: 36 / 18"
    )

    print(
        "Observed Stage-22 counts:",
        observed_window_lengths.tolist(),
    )

    print(
        "Stage-22 formula match:",
        window_formula_match,
    )

    print(
        "Temporal/feature order preserved:",
        feature_order_passed,
    )

    print(
        "Padded windows enter Tiny-Swin:",
        False,
    )

    print(
        "Resize to 224 x 224:",
        False,
    )

    section(
        "TINY-SWIN CONTRACT"
    )

    print(
        "Patch projection: "
        "1 -> 32, kernel/stride 2 x 2"
    )

    print(
        "Stage dimensions / depths / heads: "
        "[32, 64] / [2, 2] / [2, 4]"
    )

    print(
        "Window / shifted offset: "
        "[8, 5] / [4, 2]"
    )

    print(
        "Patch merge: 2 x 2, 32 -> 64"
    )

    print(
        "MLP ratio / dropout: 4 / 0.1"
    )

    print(
        "BiLSTM hidden per direction: 96"
    )

    print(
        "Packed BiLSTM: True"
    )

    section(
        "SHAPE AND SHIFTED-WINDOW PROOF"
    )

    for (
        name,
        shape,
    ) in shape_trace.items():

        print(
            name + ":",
            shape,
        )

    print(
        "Stage-1 padded attention grid:",
        [
            24,
            20,
        ],
    )

    print(
        "Stage-2 padded attention grid:",
        [
            16,
            10,
        ],
    )

    print(
        "Minimal grid padding passed:",
        minimal_grid_padding_passed,
    )

    print(
        "Shifted Stage-1 calls:",
        shifted_calls[
            "stage1"
        ],
    )

    print(
        "Shifted Stage-2 calls:",
        shifted_calls[
            "stage2"
        ],
    )

    print(
        "Shifted blocks passed:",
        shifted_blocks_passed,
    )

    print(
        "Relative-position bias passed:",
        relative_position_bias_passed,
    )

    print(
        "Spatial padding max difference:",
        spatial_padding_difference,
    )

    print(
        "Patch-merge padding max difference:",
        patch_merge_difference,
    )

    print(
        "Swin verification passed:",
        swin_verification_passed,
    )

    section(
        "PARAMETERS"
    )

    for (
        name,
        value,
    ) in parameters.items():

        print(
            (
                name.replace(
                    "_",
                    " ",
                ).title()
                + ":"
            ),
            value,
        )

    print(
        "Under 1.5M budget:",
        parameter_budget_passed,
    )

    print(
        "Parameter count passed:",
        parameter_count_passed,
    )

    section(
        "LENGTH AND PADDING SAFETY"
    )

    print(
        "Output shape:",
        output_shape,
    )

    print(
        "Output lengths:",
        output_lengths,
    )

    print(
        "Padded logits zero:",
        padded_logits_zero,
    )

    print(
        "Raw padding max difference:",
        raw_padding_difference,
    )

    print(
        "Raw padding invariance passed:",
        raw_padding_invariance_passed,
    )

    section(
        "CTC BACKWARD TEST"
    )

    print(
        "Loss finite:",
        loss_finite,
    )

    print(
        "Gradient tensors:",
        gradient_tensors,
    )

    print(
        "Missing gradients:",
        missing_gradients,
    )

    print(
        "Non-finite gradients:",
        nonfinite_gradients,
    )

    print(
        "Gradient global norm:",
        gradient_global_norm,
    )

    print(
        "Backward passed:",
        backward_passed,
    )

    section(
        "TRAINING DEFAULTS"
    )

    for (
        name,
        value,
    ) in training_defaults.items():

        print(
            (
                name.replace(
                    "_",
                    " ",
                ).title()
                + ":"
            ),
            value,
        )

    section(
        "FROZEN DEPENDENCIES"
    )

    print(
        "Stage-21 CTC wrapper used:",
        True,
    )

    print(
        "Stage-22 window formula match:",
        window_formula_match,
    )

    print(
        "Stage-25 catalog hash match:",
        catalog_hash_match,
    )

    print(
        "Stage-25 C4 spec passed:",
        c4_spec_passed,
    )

    print(
        "Stage-26 window predecessor passed:",
        stage26_passed,
    )

    print(
        "Stage-29 immediate predecessor passed:",
        stage29_passed,
    )

    section(
        "RUNTIME"
    )

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
        device,
    )

    section(
        "OUTPUTS"
    )

    print(
        "Saved:",
        CONTRACT_PATH.relative_to(
            PROJECT_ROOT
        ),
    )

    print(
        "Saved:",
        MODEL_HASH_PATH.relative_to(
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
        "STAGE 30 BUILD PASSED"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":

    main()

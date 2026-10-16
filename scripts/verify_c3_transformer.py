from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data.lengths import (
    conv1d_output_lengths,
)
from src.models.c3_transformer import (
    C3_EXPECTED_TRAINABLE_PARAMETERS,
    C3_POSITION_MAX_LENGTH,
    C3Transformer,
    c3_key_padding_mask,
    c3_output_lengths,
    count_c3_trainable_parameters,
)
from src.models.contracts import (
    load_model_catalog,
    validate_model_output,
)
from src.training.ctc_loss import (
    torch_ctc_loss_from_logits,
)


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]

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

STAGE28_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "stage28_c2_model_contract.json"
)

C2_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "c2_self_attention.py"
)

C1_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "c1_compact.py"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "c3_transformer.py"
)

TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_c3_transformer.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage29_c3_model_contract.json"
)

MODEL_HASH_PATH = (
    OUTPUT_DIR
    / "stage29_c3_model.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_29_C3_TRANSFORMER.md"
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
        for parameter in module.parameters()
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
        "STAGE 29 — C3 COMPACT "
        "CONV1D-TRANSFORMER-CTC MODEL"
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

    c3_spec = load_model_catalog(
        CATALOG_PATH
    )[
        "c3"
    ]

    expected_spec = {
        "model_id": "c3",
        "scientific_role": (
            "compact_transformer_comparison"
        ),
        "architecture_family": (
            "conv1d_transformer_encoder_ctc"
        ),
        "input_adapter": "direct_sequence",
        "feature_width": 38,
        "output_classes": 456,
        "blank_id": 0,
        "target_padding_value": -1,
        "uses_bilstm": False,
        "uses_self_attention": False,
        "uses_transformer_encoder": True,
        "uses_tiny_swin": False,
        "optimizer": "AdamW",
        "learning_rate": 0.00005,
        "weight_decay": 0.0001,
        "epoch_cap": 80,
        "early_stopping_patience": 8,
        "physical_batch_size": 32,
        "gradient_clip_norm": 5.0,
        "scheduler": "ReduceLROnPlateau",
        "fixed_epoch_reference": False,
    }

    observed_spec = c3_spec.to_dict()

    c3_spec_passed = all(
        observed_spec[
            key
        ] == expected
        for key, expected
        in expected_spec.items()
    )

    if not c3_spec_passed:
        raise RuntimeError(
            "Stage-25 C3 specification changed."
        )

    stage28 = json.loads(
        STAGE28_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage28_passed = all(
        [
            int(
                stage28[
                    "stage"
                ]
            ) == 28,
            stage28[
                "identity"
            ][
                "model_id"
            ] == "c2",
            stage28[
                "parameter_counts"
            ][
                "passed"
            ] is True,
            stage28[
                "attention_safety"
            ][
                "mask_passed"
            ] is True,
            stage28[
                "hashes"
            ][
                "model"
            ] == sha256_file(
                C2_PATH
            ),
        ]
    )

    if not stage28_passed:
        raise RuntimeError(
            "Stage-28 C2 dependency is invalid."
        )

    raw_probe = np.asarray(
        [
            1,
            2,
            5,
            32,
            36,
            40,
            72,
            90,
            108,
        ],
        dtype=np.int64,
    )

    after_conv1 = conv1d_output_lengths(
        raw_probe,
        kernel_size=5,
        stride=2,
        padding=2,
    )

    expected_lengths = conv1d_output_lengths(
        after_conv1,
        kernel_size=5,
        stride=2,
        padding=2,
    )

    observed_lengths = c3_output_lengths(
        raw_probe
    )

    formula_match = (
        observed_lengths.tolist()
        == expected_lengths.tolist()
    )

    if not formula_match:
        raise RuntimeError(
            "C3 lengths disagree with Stage 22."
        )

    torch.manual_seed(
        2906
    )

    device = torch.device(
        "cpu"
    )

    model = C3Transformer().to(
        device
    )

    model.eval()

    parameters = {
        "temporal_encoder": (
            count_parameters(
                model.conv1,
                model.bn1,
                model.conv2,
                model.bn2,
            )
        ),
        "transformer_layers": (
            count_parameters(
                *model.transformer.layers
            )
        ),
        "transformer_final_norm": (
            count_parameters(
                model.transformer.norm
            )
        ),
        "classifier": (
            count_parameters(
                model.classifier
            )
        ),
        "trainable_total": (
            count_c3_trainable_parameters(
                model
            )
        ),
        "expected_total": (
            C3_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }

    expected_parameters = {
        "temporal_encoder": 43_360,
        "transformer_layers": 149_568,
        "transformer_final_norm": 192,
        "classifier": 44_232,
        "trainable_total": 237_352,
        "expected_total": 237_352,
    }

    parameter_count_passed = (
        parameters
        == expected_parameters
    )

    if not parameter_count_passed:
        raise RuntimeError(
            "C3 parameter-count contract failed."
        )

    positional = model.position_encoding

    position_probe = positional(
        torch.zeros(
            1,
            3,
            96,
            device=device,
        )
    )

    positional_parameter_count = (
        count_parameters(
            positional
        )
    )

    position_zero_even_max = float(
        position_probe[
            0,
            0,
            0::2,
        ]
        .abs()
        .max()
        .item()
    )

    position_zero_odd_error = float(
        (
            position_probe[
                0,
                0,
                1::2,
            ]
            - 1.0
        )
        .abs()
        .max()
        .item()
    )

    position_one_sin_error = abs(
        float(
            position_probe[
                0,
                1,
                0,
            ].item()
        )
        - math.sin(
            1.0
        )
    )

    position_one_cos_error = abs(
        float(
            position_probe[
                0,
                1,
                1,
            ].item()
        )
        - math.cos(
            1.0
        )
    )

    positional_encoding_passed = all(
        [
            list(
                positional.encoding.shape
            ) == [
                1,
                1000,
                96,
            ],
            positional_parameter_count == 0,
            position_zero_even_max == 0.0,
            position_zero_odd_error == 0.0,
            position_one_sin_error < 1e-6,
            position_one_cos_error < 1e-6,
        ]
    )

    if not positional_encoding_passed:
        raise RuntimeError(
            "C3 positional-encoding proof failed."
        )

    mask_probe = c3_key_padding_mask(
        [
            2,
            4,
        ],
        padded_time=4,
        device=device,
    )

    expected_mask = torch.tensor(
        [
            [
                False,
                False,
                True,
                True,
            ],
            [
                False,
                False,
                False,
                False,
            ],
        ],
        device=device,
    )

    exact_mask_passed = torch.equal(
        mask_probe,
        expected_mask,
    )

    if not exact_mask_passed:
        raise RuntimeError(
            "C3 key-padding mask is incorrect."
        )

    transformer_input = torch.randn(
        2,
        6,
        96,
        device=device,
    )

    changed_transformer_input = (
        transformer_input.clone()
    )

    changed_transformer_input[
        0,
        4:,
        :,
    ] = 1000.0

    with torch.no_grad():

        first_transformer = (
            model.encode_transformer(
                transformer_input,
                [
                    4,
                    6,
                ],
            )
        )

        second_transformer = (
            model.encode_transformer(
                changed_transformer_input,
                [
                    4,
                    6,
                ],
            )
        )

    transformer_difference = float(
        (
            first_transformer[
                0,
                :4,
            ]
            - second_transformer[
                0,
                :4,
            ]
        )
        .abs()
        .max()
        .item()
    )

    padded_transformer_zero = all(
        [
            torch.count_nonzero(
                first_transformer[
                    0,
                    4:,
                ]
            ).item() == 0,
            torch.count_nonzero(
                second_transformer[
                    0,
                    4:,
                ]
            ).item() == 0,
        ]
    )

    transformer_mask_passed = (
        transformer_difference == 0.0
        and padded_transformer_zero
    )

    if not transformer_mask_passed:
        raise RuntimeError(
            "C3 Transformer padding-mask proof failed."
        )

    shape_input = torch.randn(
        3,
        40,
        38,
        device=device,
    )

    with torch.no_grad():

        shape_output = model(
            shape_input,
            [
                32,
                36,
                40,
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
        output_shape[
            1
        ],
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
        40,
        38,
        device=device,
    )

    changed_raw_padding = (
        raw_padding_input.clone()
    )

    changed_raw_padding[
        0,
        32:,
        :,
    ] = 1000.0

    with torch.no_grad():

        first_raw = model(
            raw_padding_input,
            [
                32,
                40,
            ],
        )

        second_raw = model(
            changed_raw_padding,
            [
                32,
                40,
            ],
        )

    raw_padding_difference = float(
        (
            first_raw.logits_btc[
                0,
                :8,
            ]
            - second_raw.logits_btc[
                0,
                :8,
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
            output_shape
            == [
                3,
                10,
                456,
            ],
            output_lengths
            == [
                8,
                9,
                10,
            ],
            padded_logits_zero,
            raw_padding_invariance_passed,
            transformer_mask_passed,
        ]
    )

    if not length_safety_passed:
        raise RuntimeError(
            "C3 length/padding safety failed."
        )

    torch.manual_seed(
        2907
    )

    backward_model = C3Transformer().to(
        device
    )

    backward_model.train()

    backward_output = backward_model(
        torch.randn(
            2,
            24,
            38,
            device=device,
        ),
        [
            20,
            24,
        ],
    )

    loss = torch_ctc_loss_from_logits(
        backward_output.logits_btc,
        torch.tensor(
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
        ),
        backward_output.output_lengths,
        torch.tensor(
            [
                2,
                2,
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
            "C3 CTC loss is not finite."
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
        for gradient in gradients
    )

    nonfinite_gradients = sum(
        gradient is not None
        and not torch.isfinite(
            gradient
        ).all().item()
        for gradient in gradients
    )

    gradient_global_norm = float(
        sum(
            gradient
            .detach()
            .double()
            .square()
            .sum()
            .item()
            for gradient in gradients
            if gradient is not None
        )
        ** 0.5
    )

    backward_passed = all(
        [
            loss_finite,
            gradient_tensors == 36,
            missing_gradients == 0,
            nonfinite_gradients == 0,
            np.isfinite(
                gradient_global_norm
            ),
            gradient_global_norm > 0.0,
        ]
    )

    if not backward_passed:
        raise RuntimeError(
            "C3 CTC backward proof failed."
        )

    training_defaults = {
        "optimizer": c3_spec.optimizer,
        "learning_rate": c3_spec.learning_rate,
        "weight_decay": c3_spec.weight_decay,
        "epoch_cap": c3_spec.epoch_cap,
        "early_stopping_patience": (
            c3_spec.early_stopping_patience
        ),
        "physical_batch_size": (
            c3_spec.physical_batch_size
        ),
        "gradient_clip_norm": (
            c3_spec.gradient_clip_norm
        ),
        "scheduler": c3_spec.scheduler,
    }

    hashes = {
        "model": sha256_file(
            MODEL_PATH
        ),
        "tests": sha256_file(
            TEST_PATH
        ),
        "verifier": sha256_file(
            Path(
                __file__
            ).resolve()
        ),
        "shared_c1_components": (
            sha256_file(
                C1_PATH
            )
        ),
        "stage25_catalog": (
            sha256_file(
                CATALOG_PATH
            )
        ),
        "stage28_contract": (
            sha256_file(
                STAGE28_PATH
            )
        ),
    }

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 29,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "identity": {
            "model_id": "c3",
            "scientific_role": (
                "compact_transformer_comparison"
            ),
            "architecture_family": (
                "conv1d_transformer_encoder_ctc"
            ),
            "input_adapter": (
                "direct_sequence"
            ),
        },
        "architecture": {
            "feature_width": 38,
            "output_classes": 456,
            "conv_channels": [
                38,
                64,
                96,
            ],
            "conv_kernel_size": 5,
            "conv_stride": 2,
            "conv_padding": 2,
            "conv_normalization": (
                "length_masked_batch_norm_1d"
            ),
            "conv_activation": "ReLU",
            "conv_dropout": 0.10,
            "d_model": 96,
            "encoder_layers": 2,
            "attention_heads": 4,
            "feedforward_size": 192,
            "transformer_dropout": 0.10,
            "transformer_activation": "GELU",
            "norm_first": True,
            "final_normalization": (
                "LayerNorm(96)"
            ),
            "positional_encoding": (
                "fixed_sinusoidal"
            ),
            "position_max_length": (
                C3_POSITION_MAX_LENGTH
            ),
            "true_key_padding_mask": True,
        },
        "design_resolution": {
            "status": (
                "prospective_before_training"
            ),
            "resolved_before_training": [
                "conv_padding_2",
                "masked_batchnorm_relu_dropout",
                "fixed_sinusoidal_max_length_1000",
                "gelu_activation",
                "pre_norm_layers",
                "final_layernorm_96",
            ],
            "final_layernorm_parameter_reconciliation": (
                192
            ),
        },
        "parameter_counts": {
            **parameters,
            "passed": (
                parameter_count_passed
            ),
        },
        "positional_encoding_safety": {
            "buffer_shape": list(
                positional.encoding.shape
            ),
            "trainable_parameters": (
                positional_parameter_count
            ),
            "position_zero_even_max_abs": (
                position_zero_even_max
            ),
            "position_zero_odd_max_error": (
                position_zero_odd_error
            ),
            "position_one_sin_error": (
                position_one_sin_error
            ),
            "position_one_cos_error": (
                position_one_cos_error
            ),
            "passed": (
                positional_encoding_passed
            ),
        },
        "length_and_mask_safety": {
            "probe_raw": (
                raw_probe.tolist()
            ),
            "probe_after_conv1": (
                after_conv1.tolist()
            ),
            "probe_after_conv2": (
                observed_lengths.tolist()
            ),
            "stage22_formula_match": (
                formula_match
            ),
            "exact_key_padding_mask_passed": (
                exact_mask_passed
            ),
            "transformer_valid_max_difference": (
                transformer_difference
            ),
            "padded_transformer_outputs_zero": (
                padded_transformer_zero
            ),
            "transformer_mask_passed": (
                transformer_mask_passed
            ),
            "output_shape": output_shape,
            "output_lengths": output_lengths,
            "padded_logits_zero": (
                padded_logits_zero
            ),
            "raw_padding_max_absolute_difference": (
                raw_padding_difference
            ),
            "raw_padding_invariance_passed": (
                raw_padding_invariance_passed
            ),
        },
        "ctc_backward": {
            "loss": float(
                loss.detach().item()
            ),
            "loss_finite": loss_finite,
            "gradient_tensors": gradient_tensors,
            "missing_gradients": missing_gradients,
            "nonfinite_gradients": (
                nonfinite_gradients
            ),
            "gradient_global_norm": (
                gradient_global_norm
            ),
            "passed": backward_passed,
        },
        "training_defaults": (
            training_defaults
        ),
        "dependencies": {
            "stage21_ctc_loss_wrapper_used": True,
            "stage22_conv_formula_match": (
                formula_match
            ),
            "stage25_catalog_hash_match": (
                catalog_hash_match
            ),
            "stage25_c3_spec_passed": (
                c3_spec_passed
            ),
            "stage28_predecessor_passed": (
                stage28_passed
            ),
        },
        "runtime": {
            "pytorch_version": torch.__version__,
            "cuda_available_locally": (
                torch.cuda.is_available()
            ),
            "verification_device": str(
                device
            ),
        },
        "hashes": hashes,
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
        hashes[
            "model"
        ]
        + "  src/models/c3_transformer.py\n",
        encoding="utf-8",
    )

    markdown = f"""# Stage 29 — C3 Compact Conv1D-Transformer-CTC

## Identity

C3 is the compact Transformer comparison. It processes direct `T x 38`
sequences through two stride-2 Conv1D blocks, fixed sinusoidal positions,
two pre-normalized Transformer layers and a 456-class CTC head.

## Architecture resolution

The roadmap fixes channels 38 to 64 to 96, kernel 5/stride 2, d-model 96,
two layers, four heads, FFN 192, Transformer dropout 0.10, positional
representation and a padding mask.

Before training, Stage 29 resolves unspecified details as padding 2,
length-masked BatchNorm, ReLU, sinusoidal positions to length 1000,
GELU, pre-normalization and final LayerNorm(96). Final LayerNorm
contributes 192 parameters and reconciles the exact 237,352 total.

## Verification

- Trainable parameters: {parameters["trainable_total"]}
- Parameter check passed: {parameter_count_passed}
- Positional encoding passed: {positional_encoding_passed}
- Exact padding mask passed: {exact_mask_passed}
- Transformer padding difference: {transformer_difference}
- Output shape: {output_shape}
- Output lengths: {output_lengths}
- Padded logits zero: {padded_logits_zero}
- Raw-padding difference: {raw_padding_difference}
- CTC loss finite: {loss_finite}
- Missing gradients: {missing_gradients}
- Non-finite gradients: {nonfinite_gradients}
- Backward passed: {backward_passed}

## Training defaults

- Optimizer: {c3_spec.optimizer}
- Learning rate: {c3_spec.learning_rate}
- Weight decay: {c3_spec.weight_decay}
- Epoch cap: {c3_spec.epoch_cap}
- Early-stopping patience: {c3_spec.early_stopping_patience}
- Physical batch size: {c3_spec.physical_batch_size}
- Gradient clipping norm: {c3_spec.gradient_clip_norm}
- Scheduler: {c3_spec.scheduler}
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    section(
        "SCIENTIFIC IDENTITY"
    )
    print(
        "Model ID: c3"
    )
    print(
        "Role: compact Transformer comparison"
    )
    print(
        "Architecture: "
        "conv1d_transformer_encoder_ctc"
    )

    section(
        "TRANSFORMER CONTRACT"
    )
    print(
        "d_model / heads / FFN: 96 / 4 / 192"
    )
    print(
        "Encoder layers: 2"
    )
    print(
        "Dropout: 0.1"
    )
    print(
        "Activation: GELU"
    )
    print(
        "Norm first: True"
    )
    print(
        "Final normalization: LayerNorm(96)"
    )
    print(
        "Position encoding: fixed sinusoidal, "
        "max length 1000"
    )
    print(
        "True key padding mask: True"
    )

    section(
        "PARAMETERS"
    )
    for name, value in parameters.items():
        print(
            name.replace(
                "_",
                " ",
            ).title()
            + ":",
            value,
        )
    print(
        "Parameter count passed:",
        parameter_count_passed,
    )

    section(
        "POSITIONAL ENCODING"
    )
    print(
        "Buffer shape:",
        list(
            positional.encoding.shape
        ),
    )
    print(
        "Trainable parameters:",
        positional_parameter_count,
    )
    print(
        "Position-zero even max abs:",
        position_zero_even_max,
    )
    print(
        "Position-zero odd max error:",
        position_zero_odd_error,
    )
    print(
        "Position-one sin error:",
        position_one_sin_error,
    )
    print(
        "Position-one cos error:",
        position_one_cos_error,
    )
    print(
        "Positional encoding passed:",
        positional_encoding_passed,
    )

    section(
        "LENGTH AND MASK SAFETY"
    )
    print(
        "After Conv1:",
        after_conv1.tolist(),
    )
    print(
        "After Conv2:",
        observed_lengths.tolist(),
    )
    print(
        "Stage-22 formula match:",
        formula_match,
    )
    print(
        "Exact key padding mask passed:",
        exact_mask_passed,
    )
    print(
        "Transformer valid max difference:",
        transformer_difference,
    )
    print(
        "Padded Transformer outputs zero:",
        padded_transformer_zero,
    )
    print(
        "Transformer mask passed:",
        transformer_mask_passed,
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
    for name, value in training_defaults.items():
        print(
            name.replace(
                "_",
                " ",
            ).title()
            + ":",
            value,
        )

    section(
        "FROZEN DEPENDENCIES"
    )
    print(
        "Stage-21 CTC wrapper used: True"
    )
    print(
        "Stage-22 formula match:",
        formula_match,
    )
    print(
        "Stage-25 catalog hash match:",
        catalog_hash_match,
    )
    print(
        "Stage-25 C3 spec passed:",
        c3_spec_passed,
    )
    print(
        "Stage-28 predecessor passed:",
        stage28_passed,
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
        "FINAL STATUS: STAGE 29 BUILD PASSED"
    )
    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()

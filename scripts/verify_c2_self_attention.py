from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from src.data.lengths import (
    conv1d_output_lengths,
)
from src.models.c2_self_attention import (
    C2_EXPECTED_TRAINABLE_PARAMETERS,
    C2SelfAttention,
    c2_output_lengths,
    count_c2_trainable_parameters,
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

STAGE27_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "stage27_c1_model_contract.json"
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
    / "c2_self_attention.py"
)

TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_c2_self_attention.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage28_c2_model_contract.json"
)

MODEL_HASH_PATH = (
    OUTPUT_DIR
    / "stage28_c2_model.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_28_C2_SELF_ATTENTION.md"
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
        "STAGE 28 — C2 CONV1D-BILSTM "
        "SELF-ATTENTION CTC MODEL"
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

    c2_spec = load_model_catalog(
        CATALOG_PATH
    )[
        "c2"
    ]

    expected_spec = {
        "model_id": "c2",
        "scientific_role": (
            "recurrent_self_attention_comparison"
        ),
        "architecture_family": (
            "conv1d_bilstm_self_attention_ctc"
        ),
        "input_adapter": "direct_sequence",
        "feature_width": 38,
        "output_classes": 456,
        "blank_id": 0,
        "target_padding_value": -1,
        "uses_bilstm": True,
        "uses_self_attention": True,
        "uses_transformer_encoder": False,
        "uses_tiny_swin": False,
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
        c2_spec.to_dict()
    )

    c2_spec_passed = all(
        observed_spec[
            key
        ] == expected
        for key, expected
        in expected_spec.items()
    )

    if not c2_spec_passed:

        raise RuntimeError(
            "Stage-25 C2 specification changed."
        )

    stage27 = json.loads(
        STAGE27_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage27_passed = all(
        [
            int(
                stage27[
                    "stage"
                ]
            ) == 27,
            stage27[
                "identity"
            ][
                "model_id"
            ] == "c1",
            stage27[
                "parameter_counts"
            ][
                "passed"
            ] is True,
            stage27[
                "length_safety"
            ][
                "stage22_formula_match"
            ] is True,
            stage27[
                "hashes"
            ][
                "model"
            ] == sha256_file(
                C1_PATH
            ),
        ]
    )

    if not stage27_passed:

        raise RuntimeError(
            "Stage-27 C1 dependency is invalid."
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

    expected_output_lengths = (
        conv1d_output_lengths(
            after_conv1,
            kernel_size=5,
            stride=2,
            padding=2,
        )
    )

    observed_output_lengths = (
        c2_output_lengths(
            raw_probe
        )
    )

    formula_match = (
        observed_output_lengths.tolist()
        == expected_output_lengths.tolist()
    )

    if not formula_match:

        raise RuntimeError(
            "C2 lengths disagree with Stage 22."
        )

    torch.manual_seed(
        2807
    )

    device = torch.device(
        "cpu"
    )

    model = C2SelfAttention().to(
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
        "bilstm": (
            count_parameters(
                model.bilstm
            )
        ),
        "attention_input_norm": (
            count_parameters(
                model.attention_input_norm
            )
        ),
        "multihead_attention": (
            count_parameters(
                model.self_attention
            )
        ),
        "post_attention_norm": (
            count_parameters(
                model.post_attention_norm
            )
        ),
        "classifier": (
            count_parameters(
                model.classifier
            )
        ),
        "trainable_total": (
            count_c2_trainable_parameters(
                model
            )
        ),
        "expected_total": (
            C2_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }

    expected_parameters = {
        "temporal_encoder": 43_360,
        "bilstm": 231_424,
        "attention_input_norm": 512,
        "multihead_attention": 263_168,
        "post_attention_norm": 512,
        "classifier": 117_192,
        "trainable_total": 656_168,
        "expected_total": 656_168,
    }

    parameter_count_passed = (
        parameters
        == expected_parameters
    )

    if not parameter_count_passed:

        raise RuntimeError(
            "C2 parameter-count contract failed."
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

    padding_mask = (
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
                padding_mask
            ]
        ).item()
        == 0
    )

    padding_input = torch.randn(
        2,
        40,
        38,
        device=device,
    )

    changed_padding = padding_input.clone()

    changed_padding[
        0,
        32:,
        :,
    ] = 1000.0

    with torch.no_grad():

        first_padding = model(
            padding_input,
            [
                32,
                40,
            ],
        )

        second_padding = model(
            changed_padding,
            [
                32,
                40,
            ],
        )

    padding_difference = float(
        (
            first_padding.logits_btc[
                0,
                :8,
            ]
            - second_padding.logits_btc[
                0,
                :8,
            ]
        )
        .abs()
        .max()
        .item()
    )

    padding_invariance_passed = (
        padding_difference
        == 0.0
    )

    attention_input = torch.randn(
        2,
        6,
        256,
        device=device,
    )

    changed_attention_input = (
        attention_input.clone()
    )

    changed_attention_input[
        0,
        4:,
        :,
    ] = 1000.0

    with torch.no_grad():

        (
            first_attention,
            first_weights,
        ) = model.apply_self_attention(
            attention_input,
            [
                4,
                6,
            ],
            return_attention_weights=True,
        )

        (
            second_attention,
            second_weights,
        ) = model.apply_self_attention(
            changed_attention_input,
            [
                4,
                6,
            ],
            return_attention_weights=True,
        )

    attention_difference = float(
        (
            first_attention[
                0,
                :4,
            ]
            - second_attention[
                0,
                :4,
            ]
        )
        .abs()
        .max()
        .item()
    )

    weight_difference = float(
        (
            first_weights[
                0,
                :,
                :4,
            ]
            - second_weights[
                0,
                :,
                :4,
            ]
        )
        .abs()
        .max()
        .item()
    )

    masked_key_weight_count = int(
        torch.count_nonzero(
            first_weights[
                0,
                :,
                :4,
                4:,
            ]
        ).item()
    )

    masked_query_weight_count = int(
        torch.count_nonzero(
            first_weights[
                0,
                :,
                4:,
                :,
            ]
        ).item()
    )

    attention_mask_passed = all(
        [
            list(
                first_weights.shape
            ) == [
                2,
                4,
                6,
                6,
            ],
            attention_difference == 0.0,
            weight_difference == 0.0,
            masked_key_weight_count == 0,
            masked_query_weight_count == 0,
        ]
    )

    if not attention_mask_passed:

        raise RuntimeError(
            "C2 attention-mask proof failed."
        )

    torch.manual_seed(
        2808
    )

    residual_model = C2SelfAttention().to(
        device
    )

    residual_model.eval()

    with torch.no_grad():

        for parameter in (
            residual_model
            .self_attention
            .parameters()
        ):

            parameter.zero_()

    residual_input = torch.randn(
        2,
        5,
        256,
        device=device,
    )

    residual_lengths = torch.tensor(
        [
            3,
            5,
        ],
        dtype=torch.long,
    )

    with torch.no_grad():

        residual_observed, _ = (
            residual_model.apply_self_attention(
                residual_input,
                residual_lengths,
            )
        )

        residual_mask = (
            torch.arange(
                5,
                device=device,
            ).unsqueeze(
                0
            )
            < residual_lengths.to(
                device
            ).unsqueeze(
                1
            )
        )

        residual_base = (
            residual_input.masked_fill(
                ~residual_mask.unsqueeze(
                    -1
                ),
                0.0,
            )
        )

        residual_expected = (
            residual_model
            .post_attention_norm(
                residual_base
            )
        )

        residual_expected = (
            residual_expected.masked_fill(
                ~residual_mask.unsqueeze(
                    -1
                ),
                0.0,
            )
        )

    residual_difference = float(
        (
            residual_observed
            - residual_expected
        )
        .abs()
        .max()
        .item()
    )

    residual_passed = (
        residual_difference
        == 0.0
    )

    if not residual_passed:

        raise RuntimeError(
            "C2 residual-path proof failed."
        )

    length_and_attention_passed = all(
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
            padding_invariance_passed,
            attention_mask_passed,
            residual_passed,
        ]
    )

    if not length_and_attention_passed:

        raise RuntimeError(
            "C2 length/attention safety failed."
        )

    torch.manual_seed(
        2809
    )

    backward_model = C2SelfAttention().to(
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
            "C2 CTC loss is not finite."
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
            gradient_tensors == 26,
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
            "C2 CTC backward proof failed."
        )

    training_defaults = {
        "optimizer": c2_spec.optimizer,
        "learning_rate": c2_spec.learning_rate,
        "weight_decay": c2_spec.weight_decay,
        "epoch_cap": c2_spec.epoch_cap,
        "early_stopping_patience": (
            c2_spec.early_stopping_patience
        ),
        "physical_batch_size": (
            c2_spec.physical_batch_size
        ),
        "gradient_clip_norm": (
            c2_spec.gradient_clip_norm
        ),
        "scheduler": c2_spec.scheduler,
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
        "c1_backbone_source": (
            sha256_file(
                C1_PATH
            )
        ),
        "stage25_catalog": (
            sha256_file(
                CATALOG_PATH
            )
        ),
        "stage27_contract": (
            sha256_file(
                STAGE27_PATH
            )
        ),
    }

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 28,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "identity": {
            "model_id": "c2",
            "scientific_role": (
                "recurrent_self_attention_comparison"
            ),
            "architecture_family": (
                "conv1d_bilstm_self_attention_ctc"
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
            "conv_dropout": 0.10,
            "bilstm_layers": 1,
            "bilstm_hidden_per_direction": 128,
            "bilstm_output_width": 256,
            "packed_bilstm": True,
            "recurrent_dropout": 0.15,
            "attention_input_norm": (
                "LayerNorm(256)"
            ),
            "attention_heads": 4,
            "attention_dropout": 0.10,
            "attention_batch_first": True,
            "true_key_padding_mask": True,
            "residual_connection": True,
            "post_attention_norm": (
                "LayerNorm(256)"
            ),
        },
        "parameter_resolution": {
            "exact_fresh_count": 656_168,
            "expanded_roadmap_count": 656_168,
            "older_under_500k_wording_satisfied": (
                False
            ),
            "resolution": (
                "Preserve the frozen 256-dimensional "
                "four-head architecture and record its "
                "exact count prospectively before training."
            ),
        },
        "parameter_counts": {
            **parameters,
            "passed": (
                parameter_count_passed
            ),
        },
        "length_safety": {
            "probe_raw": (
                raw_probe.tolist()
            ),
            "probe_after_conv1": (
                after_conv1.tolist()
            ),
            "probe_after_conv2": (
                observed_output_lengths.tolist()
            ),
            "stage22_formula_match": (
                formula_match
            ),
            "output_shape": (
                output_shape
            ),
            "output_lengths": (
                output_lengths
            ),
            "masked_batch_normalization": True,
            "packed_bilstm": True,
            "padded_logits_zero": (
                padded_logits_zero
            ),
            "padding_max_absolute_difference": (
                padding_difference
            ),
            "padding_invariance_passed": (
                padding_invariance_passed
            ),
        },
        "attention_safety": {
            "weights_shape": (
                list(
                    first_weights.shape
                )
            ),
            "valid_output_max_difference": (
                attention_difference
            ),
            "valid_weight_max_difference": (
                weight_difference
            ),
            "masked_key_nonzero_weights": (
                masked_key_weight_count
            ),
            "masked_query_nonzero_weights": (
                masked_query_weight_count
            ),
            "mask_passed": (
                attention_mask_passed
            ),
            "zero_attention_residual_max_difference": (
                residual_difference
            ),
            "residual_passed": (
                residual_passed
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
            "stage21_ctc_loss_wrapper_used": True,
            "stage22_conv_formula_match": (
                formula_match
            ),
            "stage25_catalog_hash_match": (
                catalog_hash_match
            ),
            "stage25_c2_spec_passed": (
                c2_spec_passed
            ),
            "stage27_c1_dependency_passed": (
                stage27_passed
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
        hashes[
            "model"
        ]
        + "  src/models/c2_self_attention.py\n",
        encoding="utf-8",
    )

    markdown = f"""# Stage 28 — C2 Conv1D-BiLSTM Self-Attention CTC

## Identity

C2 is the recurrent self-attention comparison. It uses the exact C1
Conv1D and packed-BiLSTM backbone followed by one four-head
self-attention block with true padding masking, a residual connection
and post-normalization.

## Parameter-count resolution

The frozen 256-dimensional four-head architecture contains exactly
{parameters["trainable_total"]} trainable parameters. This matches the
expanded roadmap's 656,168 count.

It does not satisfy the older descriptive wording of under 0.5M.
Architecture identity takes precedence, and this discrepancy is recorded
before training or inspection of experimental results.

## Verification

- Parameter check passed: {parameter_count_passed}
- Output shape: {output_shape}
- Output lengths: {output_lengths}
- Stage-22 formula matched: {formula_match}
- Padded logits zero: {padded_logits_zero}
- Padding invariance difference: {padding_difference}
- Attention mask passed: {attention_mask_passed}
- Masked-key nonzero weights: {masked_key_weight_count}
- Residual test difference: {residual_difference}
- CTC loss finite: {loss_finite}
- Missing gradients: {missing_gradients}
- Non-finite gradients: {nonfinite_gradients}
- Backward test passed: {backward_passed}

## Training defaults

- Optimizer: {c2_spec.optimizer}
- Learning rate: {c2_spec.learning_rate}
- Weight decay: {c2_spec.weight_decay}
- Epoch cap: {c2_spec.epoch_cap}
- Early-stopping patience: {c2_spec.early_stopping_patience}
- Physical batch size: {c2_spec.physical_batch_size}
- Gradient clipping norm: {c2_spec.gradient_clip_norm}
- Scheduler: {c2_spec.scheduler}
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    section(
        "SCIENTIFIC IDENTITY"
    )

    print(
        "Model ID: c2"
    )

    print(
        "Role: recurrent self-attention comparison"
    )

    print(
        "Architecture: "
        "conv1d_bilstm_self_attention_ctc"
    )

    section(
        "ATTENTION CONTRACT"
    )

    print(
        "Embedding width: 256"
    )

    print(
        "Heads: 4"
    )

    print(
        "Attention dropout: 0.1"
    )

    print(
        "True key padding mask: True"
    )

    print(
        "Residual connection: True"
    )

    print(
        "Pre / post normalization: "
        "LayerNorm / LayerNorm"
    )

    section(
        "PARAMETER RESOLUTION"
    )

    for (
        name,
        value,
    ) in parameters.items():

        print(
            name.replace(
                "_",
                " ",
            ).title()
            + ":",
            value,
        )

    print(
        "Older under-0.5M wording satisfied:",
        False,
    )

    print(
        "Exact-count check passed:",
        parameter_count_passed,
    )

    section(
        "LENGTH AND PADDING SAFETY"
    )

    print(
        "After Conv1:",
        after_conv1.tolist(),
    )

    print(
        "After Conv2:",
        observed_output_lengths.tolist(),
    )

    print(
        "Stage-22 formula match:",
        formula_match,
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
        "Padding max absolute difference:",
        padding_difference,
    )

    print(
        "Padding invariance passed:",
        padding_invariance_passed,
    )

    section(
        "ATTENTION MASK AND RESIDUAL"
    )

    print(
        "Attention weights shape:",
        list(
            first_weights.shape
        ),
    )

    print(
        "Valid output max difference:",
        attention_difference,
    )

    print(
        "Valid weight max difference:",
        weight_difference,
    )

    print(
        "Masked-key nonzero weights:",
        masked_key_weight_count,
    )

    print(
        "Masked-query nonzero weights:",
        masked_query_weight_count,
    )

    print(
        "Attention mask passed:",
        attention_mask_passed,
    )

    print(
        "Residual max difference:",
        residual_difference,
    )

    print(
        "Residual passed:",
        residual_passed,
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
        "Stage-21 CTC wrapper used:",
        True,
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
        "Stage-25 C2 spec passed:",
        c2_spec_passed,
    )

    print(
        "Stage-27 C1 dependency passed:",
        stage27_passed,
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
        "STAGE 28 BUILD PASSED"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()

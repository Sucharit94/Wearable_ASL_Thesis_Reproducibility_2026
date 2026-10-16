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
from src.models.c1_compact import (
    C1_EXPECTED_TRAINABLE_PARAMETERS,
    C1_FEATURE_WIDTH,
    C1Compact,
    c1_length_trace,
    count_c1_trainable_parameters,
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

STAGE26_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
    / "stage26_g1_model_contract.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "src"
    / "models"
    / "c1_compact.py"
)

TEST_PATH = (
    PROJECT_ROOT
    / "tests"
    / "test_c1_compact.py"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "models"
)

CONTRACT_PATH = (
    OUTPUT_DIR
    / "stage27_c1_model_contract.json"
)

MODEL_HASH_PATH = (
    OUTPUT_DIR
    / "stage27_c1_model.sha256"
)

DOC_PATH = (
    PROJECT_ROOT
    / "docs"
    / "STAGE_27_C1_COMPACT.md"
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


def parameter_count(
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
        "STAGE 27 — C1 COMPACT "
        "CONV1D-BILSTM-CTC MODEL"
    )
    print(
        "=" * 78
    )

    # ---------------------------------------------------------
    # Stage-25 frozen C1 specification.
    # ---------------------------------------------------------

    recorded_catalog_hash = read_hash(
        CATALOG_HASH_PATH
    )

    actual_catalog_hash = sha256_file(
        CATALOG_PATH
    )

    catalog_hash_match = (
        recorded_catalog_hash
        == actual_catalog_hash
    )

    if not catalog_hash_match:

        raise RuntimeError(
            "Stage-25 model catalog "
            "hash mismatch."
        )

    c1_spec = load_model_catalog(
        CATALOG_PATH
    )[
        "c1"
    ]

    expected_spec = {
        "model_id": "c1",
        "scientific_role": (
            "lightweight_recurrent_baseline"
        ),
        "architecture_family": (
            "conv1d_bilstm_ctc"
        ),
        "input_adapter": (
            "direct_sequence"
        ),
        "feature_width": 38,
        "output_classes": 456,
        "blank_id": 0,
        "target_padding_value": -1,
        "uses_bilstm": True,
        "uses_self_attention": False,
        "uses_transformer_encoder": False,
        "uses_tiny_swin": False,
        "optimizer": "AdamW",
        "learning_rate": 0.00005,
        "weight_decay": 0.0001,
        "epoch_cap": 80,
        "early_stopping_patience": 8,
        "physical_batch_size": 32,
        "gradient_clip_norm": 5.0,
        "scheduler": (
            "ReduceLROnPlateau"
        ),
        "fixed_epoch_reference": False,
    }

    observed_spec = (
        c1_spec.to_dict()
    )

    c1_spec_passed = all(
        observed_spec[
            key
        ] == expected
        for key, expected
        in expected_spec.items()
    )

    if not c1_spec_passed:

        raise RuntimeError(
            "Stage-25 C1 specification changed."
        )

    # ---------------------------------------------------------
    # Sequential Stage-26 dependency.
    # ---------------------------------------------------------

    stage26 = json.loads(
        STAGE26_PATH.read_text(
            encoding="utf-8-sig"
        )
    )

    stage26_passed = (
        int(
            stage26[
                "stage"
            ]
        )
        == 26
    )

    if not stage26_passed:

        raise RuntimeError(
            "Stage-26 predecessor "
            "contract is invalid."
        )

    # ---------------------------------------------------------
    # Stage-22 exact length-formula comparison.
    # ---------------------------------------------------------

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

    expected_conv1 = (
        conv1d_output_lengths(
            raw_probe,
            kernel_size=5,
            stride=2,
            padding=2,
        )
    )

    expected_conv2 = (
        conv1d_output_lengths(
            expected_conv1,
            kernel_size=5,
            stride=2,
            padding=2,
        )
    )

    (
        observed_raw,
        observed_conv1,
        observed_conv2,
    ) = c1_length_trace(
        raw_probe
    )

    formula_match = (
        observed_raw.tolist()
        == raw_probe.tolist()
        and observed_conv1.tolist()
        == expected_conv1.tolist()
        and observed_conv2.tolist()
        == expected_conv2.tolist()
    )

    if not formula_match:

        raise RuntimeError(
            "C1 lengths disagree with "
            "the Stage-22 formula."
        )

    # ---------------------------------------------------------
    # Model construction and parameter audit.
    # ---------------------------------------------------------

    torch.manual_seed(
        2706
    )

    device = torch.device(
        "cpu"
    )

    model = C1Compact().to(
        device
    )

    model.eval()

    parameters = {
        "temporal_encoder": (
            parameter_count(
                model.conv1,
                model.bn1,
                model.conv2,
                model.bn2,
            )
        ),
        "bilstm": (
            parameter_count(
                model.bilstm
            )
        ),
        "post_lstm_norm": (
            parameter_count(
                model.post_lstm_norm
            )
        ),
        "classifier": (
            parameter_count(
                model.classifier
            )
        ),
        "trainable_total": (
            count_c1_trainable_parameters(
                model
            )
        ),
        "expected_total": (
            C1_EXPECTED_TRAINABLE_PARAMETERS
        ),
    }

    parameter_count_passed = (
        parameters
        == {
            "temporal_encoder": 43_360,
            "bilstm": 231_424,
            "post_lstm_norm": 512,
            "classifier": 117_192,
            "trainable_total": 392_488,
            "expected_total": 392_488,
        }
    )

    if not parameter_count_passed:

        raise RuntimeError(
            "C1 parameter-count "
            "contract failed."
        )

    # ---------------------------------------------------------
    # Shape, output length and padding audit.
    # ---------------------------------------------------------

    shape_input = torch.randn(
        3,
        40,
        C1_FEATURE_WIDTH,
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

    output_lengths = list(
        validate_model_output(
            shape_output
        )
    )

    output_shape = list(
        shape_output
        .logits_btc
        .shape
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
        C1_FEATURE_WIDTH,
        device=device,
    )

    changed_padding = (
        padding_input.clone()
    )

    changed_padding[
        0,
        32:,
        :,
    ] = 1000.0

    with torch.no_grad():

        first = model(
            padding_input,
            [
                32,
                40,
            ],
        )

        second = model(
            changed_padding,
            [
                32,
                40,
            ],
        )

    padding_difference = float(
        (
            first.logits_btc[
                0,
                :8,
            ]
            - second.logits_btc[
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
            padding_invariance_passed,
        ]
    )

    if not length_safety_passed:

        raise RuntimeError(
            "C1 length or padding-safety "
            "check failed."
        )

    # ---------------------------------------------------------
    # Real CTC backward proof.
    # ---------------------------------------------------------

    torch.manual_seed(
        2707
    )

    backward_model = C1Compact().to(
        device
    )

    backward_model.train()

    backward_output = backward_model(
        torch.randn(
            2,
            24,
            C1_FEATURE_WIDTH,
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
            "C1 CTC loss is not finite."
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
            gradient_tensors == 20,
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
            "C1 CTC backward check failed."
        )

    # ---------------------------------------------------------
    # Frozen training defaults and file hashes.
    # ---------------------------------------------------------

    training_defaults = {
        "optimizer": (
            c1_spec.optimizer
        ),
        "learning_rate": (
            c1_spec.learning_rate
        ),
        "weight_decay": (
            c1_spec.weight_decay
        ),
        "epoch_cap": (
            c1_spec.epoch_cap
        ),
        "early_stopping_patience": (
            c1_spec
            .early_stopping_patience
        ),
        "physical_batch_size": (
            c1_spec.physical_batch_size
        ),
        "gradient_clip_norm": (
            c1_spec.gradient_clip_norm
        ),
        "scheduler": (
            c1_spec.scheduler
        ),
    }

    hashes = {
        "model": (
            sha256_file(
                MODEL_PATH
            )
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
        "stage25_catalog": (
            actual_catalog_hash
        ),
        "stage26_contract": (
            sha256_file(
                STAGE26_PATH
            )
        ),
    }

    # ---------------------------------------------------------
    # Write contract.
    # ---------------------------------------------------------

    contract = {
        "project_generation": (
            "fresh_start_2026_v1"
        ),
        "stage": 27,
        "created_at_utc": (
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
        "identity": {
            "model_id": "c1",
            "scientific_role": (
                "lightweight_recurrent_baseline"
            ),
            "architecture_family": (
                "conv1d_bilstm_ctc"
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
            "post_lstm_normalization": (
                "LayerNorm(256)"
            ),
            "post_lstm_dropout": 0.15,
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
                observed_conv1.tolist()
            ),
            "probe_after_conv2": (
                observed_conv2.tolist()
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
        "ctc_backward": {
            "loss": (
                float(
                    loss.detach().item()
                )
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
            "stage25_c1_spec_passed": (
                c1_spec_passed
            ),
            "stage26_predecessor_passed": (
                stage26_passed
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
        + "  src/models/c1_compact.py\n",
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Write scientific documentation.
    # ---------------------------------------------------------

    markdown = f"""# Stage 27 — C1 Compact Conv1D-BiLSTM-CTC

## Scientific identity

C1 is the lightweight recurrent baseline. It consumes a direct
variable-length `T x 38` feature sequence and produces raw
`B x T_out x 456` CTC logits.

## Architecture

Two Conv1D blocks use channels 38 to 64 to 96, kernel 5, stride 2,
padding 2, length-masked BatchNorm1D, ReLU and dropout 0.10.

One packed BiLSTM uses 128 hidden units per direction. LayerNorm(256)
and dropout 0.15 precede the 456-class classifier. LayerNorm supplies
the 512 parameters required to reconcile the frozen parameter total.

## Verification

- Trainable parameters: {parameters["trainable_total"]}
- Expected parameters: {parameters["expected_total"]}
- Parameter check passed: {parameter_count_passed}
- Output shape: {output_shape}
- Output lengths: {output_lengths}
- Stage-22 length formula matched: {formula_match}
- Padded logits zero: {padded_logits_zero}
- Padding maximum absolute difference: {padding_difference}
- CTC loss finite: {loss_finite}
- Missing gradients: {missing_gradients}
- Non-finite gradients: {nonfinite_gradients}
- Backward test passed: {backward_passed}

## Frozen training defaults

- Optimizer: {c1_spec.optimizer}
- Learning rate: {c1_spec.learning_rate}
- Weight decay: {c1_spec.weight_decay}
- Epoch cap: {c1_spec.epoch_cap}
- Early-stopping patience: {c1_spec.early_stopping_patience}
- Physical batch size: {c1_spec.physical_batch_size}
- Gradient clipping norm: {c1_spec.gradient_clip_norm}
- Scheduler: {c1_spec.scheduler}

## Runtime

- PyTorch: {torch.__version__}
- CUDA available locally: {torch.cuda.is_available()}
- Verification device: {device}
"""

    DOC_PATH.write_text(
        markdown,
        encoding="utf-8",
    )

    # ---------------------------------------------------------
    # Console report.
    # ---------------------------------------------------------

    section(
        "SCIENTIFIC IDENTITY"
    )

    print(
        "Model ID: c1"
    )

    print(
        "Role: lightweight recurrent baseline"
    )

    print(
        "Architecture: conv1d_bilstm_ctc"
    )

    print(
        "Input adapter: direct_sequence"
    )

    section(
        "ARCHITECTURE"
    )

    print(
        "Input features: 38"
    )

    print(
        "Conv channels: 38 -> 64 -> 96"
    )

    print(
        "Conv kernel / stride / padding: "
        "5 / 2 / 2"
    )

    print(
        "BiLSTM hidden per direction: 128"
    )

    print(
        "Post-BiLSTM normalization: "
        "LayerNorm(256)"
    )

    print(
        "CTC classes: 456"
    )

    section(
        "LENGTH TRACE"
    )

    print(
        "Raw:",
        raw_probe.tolist(),
    )

    print(
        "After Conv1:",
        observed_conv1.tolist(),
    )

    print(
        "After Conv2:",
        observed_conv2.tolist(),
    )

    print(
        "Stage-22 formula match:",
        formula_match,
    )

    section(
        "PARAMETERS"
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
        "Masked BatchNorm:",
        True,
    )

    print(
        "Packed BiLSTM:",
        True,
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
        "Stage-25 C1 spec passed:",
        c1_spec_passed,
    )

    print(
        "Stage-26 predecessor passed:",
        stage26_passed,
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
        "STAGE 27 BUILD PASSED"
    )

    print(
        "=" * 78
    )


if __name__ == "__main__":
    main()

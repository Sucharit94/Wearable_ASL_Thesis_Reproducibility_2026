from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from src.data.lengths import (
    conv1d_output_lengths,
)
from src.models.c3_transformer import (
    C3_ATTENTION_HEADS,
    C3_D_MODEL,
    C3_DROPOUT,
    C3_ENCODER_LAYERS,
    C3_EXPECTED_TRAINABLE_PARAMETERS,
    C3_FEEDFORWARD_SIZE,
    C3_POSITION_MAX_LENGTH,
    C3Transformer,
    SinusoidalPositionalEncoding,
    c3_key_padding_mask,
    c3_output_lengths,
    count_c3_trainable_parameters,
)
from src.models.contracts import (
    validate_model_output,
)
from src.training.ctc_loss import (
    torch_ctc_loss_from_logits,
)


@pytest.fixture(
    scope="module"
)
def model() -> C3Transformer:

    torch.manual_seed(
        2901
    )

    instance = C3Transformer()
    instance.eval()

    return instance


def test_frozen_architecture_and_parameter_count(
    model: C3Transformer,
):

    assert (
        model.conv1.in_channels,
        model.conv1.out_channels,
    ) == (
        38,
        64,
    )

    assert model.conv1.kernel_size == (
        5,
    )

    assert model.conv1.stride == (
        2,
    )

    assert model.conv1.padding == (
        2,
    )

    assert (
        model.conv2.in_channels,
        model.conv2.out_channels,
    ) == (
        64,
        96,
    )

    assert model.conv2.kernel_size == (
        5,
    )

    assert model.conv2.stride == (
        2,
    )

    assert model.conv2.padding == (
        2,
    )

    assert len(
        model.transformer.layers
    ) == C3_ENCODER_LAYERS

    for layer in model.transformer.layers:

        assert (
            layer.self_attn.embed_dim
            == C3_D_MODEL
        )

        assert (
            layer.self_attn.num_heads
            == C3_ATTENTION_HEADS
        )

        assert layer.linear1.in_features == 96

        assert (
            layer.linear1.out_features
            == C3_FEEDFORWARD_SIZE
        )

        assert (
            layer.linear2.in_features
            == C3_FEEDFORWARD_SIZE
        )

        assert layer.linear2.out_features == 96

        assert layer.dropout.p == pytest.approx(
            C3_DROPOUT
        )

        assert layer.norm_first is True

        observed_gelu = layer.activation(
            torch.tensor(
                [
                    -1.0
                ]
            )
        )

        expected_gelu = (
            torch.nn.functional.gelu(
                torch.tensor(
                    [
                        -1.0
                    ]
                )
            )
        )

        torch.testing.assert_close(
            observed_gelu,
            expected_gelu,
        )

    assert (
        model.transformer.norm.normalized_shape
        == (
            96,
        )
    )

    assert model.classifier.in_features == 96
    assert model.classifier.out_features == 456

    assert (
        count_c3_trainable_parameters(
            model
        )
        == C3_EXPECTED_TRAINABLE_PARAMETERS
    )


def test_exact_length_formula_matches_stage22():

    raw = np.asarray(
        [
            1,
            2,
            5,
            72,
            90,
            108,
        ],
        dtype=np.int64,
    )

    after_conv1 = conv1d_output_lengths(
        raw,
        kernel_size=5,
        stride=2,
        padding=2,
    )

    expected = conv1d_output_lengths(
        after_conv1,
        kernel_size=5,
        stride=2,
        padding=2,
    )

    assert c3_output_lengths(
        raw
    ).tolist() == expected.tolist()

    assert c3_output_lengths(
        [
            72,
            90,
            108,
        ]
    ).tolist() == [
        18,
        23,
        27,
    ]


def test_key_padding_mask_is_exact():

    observed = c3_key_padding_mask(
        [
            2,
            4,
        ],
        padded_time=4,
    )

    expected = torch.tensor(
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
        ]
    )

    assert observed.dtype == torch.bool

    assert torch.equal(
        observed,
        expected,
    )


def test_sinusoidal_position_encoding_is_fixed_and_correct():

    encoding = SinusoidalPositionalEncoding(
        d_model=96,
        max_length=1000,
    )

    assert list(
        encoding.parameters()
    ) == []

    assert tuple(
        encoding.encoding.shape
    ) == (
        1,
        1000,
        96,
    )

    observed = encoding(
        torch.zeros(
            1,
            3,
            96,
        )
    )

    torch.testing.assert_close(
        observed[
            0,
            0,
            0::2,
        ],
        torch.zeros(
            48
        ),
    )

    torch.testing.assert_close(
        observed[
            0,
            0,
            1::2,
        ],
        torch.ones(
            48
        ),
    )

    assert observed[
        0,
        1,
        0,
    ].item() == pytest.approx(
        math.sin(
            1.0
        )
    )

    assert observed[
        0,
        1,
        1,
    ].item() == pytest.approx(
        math.cos(
            1.0
        )
    )

    assert not torch.equal(
        observed[
            0,
            0,
        ],
        observed[
            0,
            1,
        ],
    )


def test_position_capacity_is_enforced():

    encoding = SinusoidalPositionalEncoding(
        d_model=96,
        max_length=C3_POSITION_MAX_LENGTH,
    )

    with pytest.raises(
        ValueError,
        match="capacity",
    ):

        encoding(
            torch.zeros(
                1,
                C3_POSITION_MAX_LENGTH + 1,
                96,
            )
        )


def test_transformer_mask_blocks_padded_values(
    model: C3Transformer,
):

    torch.manual_seed(
        2902
    )

    sequence = torch.randn(
        2,
        6,
        96,
    )

    changed = sequence.clone()

    changed[
        0,
        4:,
        :,
    ] = 1000.0

    with torch.no_grad():

        first = model.encode_transformer(
            sequence,
            [
                4,
                6,
            ],
        )

        second = model.encode_transformer(
            changed,
            [
                4,
                6,
            ],
        )

    torch.testing.assert_close(
        first[
            0,
            :4,
        ],
        second[
            0,
            :4,
        ],
        rtol=0.0,
        atol=0.0,
    )

    assert torch.count_nonzero(
        first[
            0,
            4:,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        second[
            0,
            4:,
        ]
    ).item() == 0


def test_forward_shape_and_shared_contract(
    model: C3Transformer,
):

    torch.manual_seed(
        2903
    )

    with torch.no_grad():

        output = model(
            torch.randn(
                3,
                40,
                38,
            ),
            [
                32,
                36,
                40,
            ],
        )

    assert tuple(
        output.logits_btc.shape
    ) == (
        3,
        10,
        456,
    )

    assert validate_model_output(
        output
    ) == (
        8,
        9,
        10,
    )

    assert torch.count_nonzero(
        output.logits_btc[
            0,
            8:,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        output.logits_btc[
            1,
            9:,
        ]
    ).item() == 0


def test_padded_input_tail_does_not_change_valid_logits(
    model: C3Transformer,
):

    torch.manual_seed(
        2904
    )

    original = torch.randn(
        2,
        40,
        38,
    )

    changed = original.clone()

    changed[
        0,
        32:,
        :,
    ] = 1000.0

    with torch.no_grad():

        first = model(
            original,
            [
                32,
                40,
            ],
        )

        second = model(
            changed,
            [
                32,
                40,
            ],
        )

    torch.testing.assert_close(
        first.logits_btc[
            0,
            :8,
        ],
        second.logits_btc[
            0,
            :8,
        ],
        rtol=0.0,
        atol=0.0,
    )


def test_wrong_feature_width_rejected(
    model: C3Transformer,
):

    with pytest.raises(
        ValueError,
        match="38",
    ):

        model(
            torch.zeros(
                1,
                40,
                37,
            ),
            [
                40
            ],
        )


def test_float_lengths_rejected():

    with pytest.raises(
        TypeError,
        match="integers",
    ):

        c3_output_lengths(
            [
                40.0
            ]
        )


def test_nonpositive_length_rejected(
    model: C3Transformer,
):

    with pytest.raises(
        ValueError,
        match="positive",
    ):

        model(
            torch.zeros(
                1,
                40,
                38,
            ),
            [
                0
            ],
        )


def test_length_exceeding_padded_time_rejected(
    model: C3Transformer,
):

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):

        model(
            torch.zeros(
                1,
                40,
                38,
            ),
            [
                41
            ],
        )


def test_nonfinite_input_rejected(
    model: C3Transformer,
):

    x_btf = torch.zeros(
        1,
        40,
        38,
    )

    x_btf[
        0,
        3,
        7,
    ] = torch.inf

    with pytest.raises(
        ValueError,
        match="infinity",
    ):

        model(
            x_btf,
            [
                40
            ],
        )


def test_real_ctc_backward_is_finite_and_reaches_all_parameters():

    torch.manual_seed(
        2905
    )

    model = C3Transformer()
    model.train()

    output = model(
        torch.randn(
            2,
            24,
            38,
        ),
        [
            20,
            24,
        ],
    )

    loss = torch_ctc_loss_from_logits(
        output.logits_btc,
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
        output.output_lengths,
        torch.tensor(
            [
                2,
                2,
            ],
            dtype=torch.long,
        ),
    )

    assert torch.isfinite(
        loss
    ).item()

    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    assert len(
        gradients
    ) == 36

    assert all(
        gradient is not None
        for gradient in gradients
    )

    assert all(
        torch.isfinite(
            gradient
        ).all().item()
        for gradient in gradients
    )

    assert sum(
        float(
            gradient
            .detach()
            .square()
            .sum()
            .item()
        )
        for gradient in gradients
    ) > 0.0

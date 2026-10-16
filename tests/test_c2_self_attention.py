from __future__ import annotations

import numpy as np
import pytest
import torch

from src.data.lengths import (
    conv1d_output_lengths,
)
from src.models.c2_self_attention import (
    C2_ATTENTION_DROPOUT,
    C2_ATTENTION_HEADS,
    C2_EXPECTED_TRAINABLE_PARAMETERS,
    C2SelfAttention,
    c2_output_lengths,
    count_c2_trainable_parameters,
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
def model() -> C2SelfAttention:

    torch.manual_seed(
        2801
    )

    instance = C2SelfAttention()
    instance.eval()

    return instance


def test_frozen_architecture_and_parameter_count(
    model: C2SelfAttention,
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

    assert model.bilstm.input_size == 96
    assert model.bilstm.hidden_size == 128
    assert model.bilstm.bidirectional is True

    assert (
        model.attention_input_norm.normalized_shape
        == (
            256,
        )
    )

    assert model.self_attention.embed_dim == 256

    assert (
        model.self_attention.num_heads
        == C2_ATTENTION_HEADS
    )

    assert (
        model.self_attention.dropout
        == pytest.approx(
            C2_ATTENTION_DROPOUT
        )
    )

    assert model.self_attention.batch_first is True

    assert (
        model.post_attention_norm.normalized_shape
        == (
            256,
        )
    )

    assert model.classifier.in_features == 256
    assert model.classifier.out_features == 456

    assert (
        count_c2_trainable_parameters(
            model
        )
        == C2_EXPECTED_TRAINABLE_PARAMETERS
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

    assert c2_output_lengths(
        raw
    ).tolist() == expected.tolist()

    assert c2_output_lengths(
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


def test_attention_mask_blocks_padded_keys(
    model: C2SelfAttention,
):

    torch.manual_seed(
        2802
    )

    sequence = torch.randn(
        2,
        6,
        256,
    )

    changed = sequence.clone()

    changed[
        0,
        4:,
        :,
    ] = (
        torch.randn(
            2,
            256,
        )
        * 1000.0
    )

    with torch.no_grad():

        (
            first,
            first_weights,
        ) = model.apply_self_attention(
            sequence,
            [
                4,
                6,
            ],
            return_attention_weights=True,
        )

        (
            second,
            second_weights,
        ) = model.apply_self_attention(
            changed,
            [
                4,
                6,
            ],
            return_attention_weights=True,
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

    torch.testing.assert_close(
        first_weights[
            0,
            :,
            :4,
        ],
        second_weights[
            0,
            :,
            :4,
        ],
        rtol=0.0,
        atol=0.0,
    )

    assert tuple(
        first_weights.shape
    ) == (
        2,
        4,
        6,
        6,
    )

    assert torch.count_nonzero(
        first_weights[
            0,
            :,
            :4,
            4:,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        first_weights[
            0,
            :,
            4:,
            :,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        first[
            0,
            4:,
        ]
    ).item() == 0


def test_residual_path_remains_when_attention_is_zero():

    torch.manual_seed(
        2803
    )

    model = C2SelfAttention()
    model.eval()

    with torch.no_grad():

        for parameter in (
            model
            .self_attention
            .parameters()
        ):

            parameter.zero_()

    sequence = torch.randn(
        2,
        5,
        256,
    )

    lengths = [
        3,
        5,
    ]

    with torch.no_grad():

        observed, _ = (
            model.apply_self_attention(
                sequence,
                lengths,
            )
        )

    mask = (
        torch.arange(
            5
        ).unsqueeze(
            0
        )
        < torch.tensor(
            lengths
        ).unsqueeze(
            1
        )
    )

    masked_sequence = sequence.masked_fill(
        ~mask.unsqueeze(
            -1
        ),
        0.0,
    )

    expected = model.post_attention_norm(
        masked_sequence
    )

    expected = expected.masked_fill(
        ~mask.unsqueeze(
            -1
        ),
        0.0,
    )

    torch.testing.assert_close(
        observed,
        expected,
        rtol=0.0,
        atol=0.0,
    )


def test_forward_shape_and_shared_contract(
    model: C2SelfAttention,
):

    torch.manual_seed(
        2804
    )

    x_btf = torch.randn(
        3,
        40,
        38,
    )

    with torch.no_grad():

        output = model(
            x_btf,
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
    model: C2SelfAttention,
):

    torch.manual_seed(
        2805
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


def test_attention_wrong_width_rejected(
    model: C2SelfAttention,
):

    with pytest.raises(
        ValueError,
        match="256",
    ):

        model.apply_self_attention(
            torch.zeros(
                1,
                5,
                255,
            ),
            [
                5
            ],
        )


def test_wrong_input_feature_width_rejected(
    model: C2SelfAttention,
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

        c2_output_lengths(
            [
                40.0
            ]
        )


def test_nonpositive_length_rejected(
    model: C2SelfAttention,
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
    model: C2SelfAttention,
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
    model: C2SelfAttention,
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
        2806
    )

    model = C2SelfAttention()
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
    ) == 26

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

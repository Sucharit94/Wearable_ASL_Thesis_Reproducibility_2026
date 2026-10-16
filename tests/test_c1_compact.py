from __future__ import annotations

from copy import deepcopy

import numpy as np
import pytest
import torch

from src.data.lengths import (
    conv1d_output_lengths,
)
from src.models.c1_compact import (
    C1_CONV_DROPOUT,
    C1_EXPECTED_TRAINABLE_PARAMETERS,
    C1_LSTM_HIDDEN_SIZE,
    C1_POST_LSTM_DROPOUT,
    C1Compact,
    LengthMaskedBatchNorm1d,
    c1_length_trace,
    c1_output_lengths,
    count_c1_trainable_parameters,
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
def model() -> C1Compact:

    torch.manual_seed(
        2701
    )

    instance = C1Compact()
    instance.eval()

    return instance


def test_frozen_architecture_and_parameter_count(
    model: C1Compact,
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

    assert model.dropout1.p == pytest.approx(
        C1_CONV_DROPOUT
    )

    assert model.dropout2.p == pytest.approx(
        C1_CONV_DROPOUT
    )

    assert model.bilstm.input_size == 96

    assert (
        model.bilstm.hidden_size
        == C1_LSTM_HIDDEN_SIZE
    )

    assert model.bilstm.bidirectional is True

    assert (
        model
        .post_lstm_norm
        .normalized_shape
        == (
            256,
        )
    )

    assert (
        model.post_lstm_dropout.p
        == pytest.approx(
            C1_POST_LSTM_DROPOUT
        )
    )

    assert model.classifier.in_features == 256
    assert model.classifier.out_features == 456

    assert (
        count_c1_trainable_parameters(
            model
        )
        == C1_EXPECTED_TRAINABLE_PARAMETERS
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

    expected1 = conv1d_output_lengths(
        raw,
        kernel_size=5,
        stride=2,
        padding=2,
    )

    expected2 = conv1d_output_lengths(
        expected1,
        kernel_size=5,
        stride=2,
        padding=2,
    )

    (
        observed_raw,
        observed1,
        observed2,
    ) = c1_length_trace(
        raw
    )

    assert observed_raw.tolist() == raw.tolist()

    assert (
        observed1.tolist()
        == expected1.tolist()
    )

    assert (
        observed2.tolist()
        == expected2.tolist()
    )

    assert c1_output_lengths(
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


def test_masked_batch_norm_excludes_padded_values():

    torch.manual_seed(
        2702
    )

    first_norm = LengthMaskedBatchNorm1d(
        3
    )

    second_norm = deepcopy(
        first_norm
    )

    first_norm.train()
    second_norm.train()

    first_input = torch.randn(
        2,
        3,
        8,
    )

    second_input = first_input.clone()

    second_input[
        0,
        :,
        5:,
    ] = 1000.0

    lengths = [
        5,
        8,
    ]

    first = first_norm(
        first_input,
        lengths,
    )

    second = second_norm(
        second_input,
        lengths,
    )

    torch.testing.assert_close(
        first[
            0,
            :,
            :5,
        ],
        second[
            0,
            :,
            :5,
        ],
    )

    torch.testing.assert_close(
        first[
            1
        ],
        second[
            1
        ],
    )

    torch.testing.assert_close(
        first_norm.running_mean,
        second_norm.running_mean,
    )

    torch.testing.assert_close(
        first_norm.running_var,
        second_norm.running_var,
    )

    assert torch.count_nonzero(
        first[
            0,
            :,
            5:,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        second[
            0,
            :,
            5:,
        ]
    ).item() == 0


def test_forward_shape_and_shared_contract(
    model: C1Compact,
):

    torch.manual_seed(
        2703
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


def test_padded_tail_does_not_change_valid_logits(
    model: C1Compact,
):

    torch.manual_seed(
        2704
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
    model: C1Compact,
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

        c1_output_lengths(
            [
                40.0
            ]
        )


def test_nonpositive_length_rejected(
    model: C1Compact,
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
    model: C1Compact,
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
    model: C1Compact,
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
        2705
    )

    model = C1Compact()
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
        output.logits_btc,
        targets,
        output.output_lengths,
        target_lengths,
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
    ) == 20

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

    total_squared_gradient = sum(
        float(
            gradient
            .detach()
            .square()
            .sum()
            .item()
        )
        for gradient in gradients
    )

    assert total_squared_gradient > 0.0

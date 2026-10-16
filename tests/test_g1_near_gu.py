from __future__ import annotations

import pytest
import torch

from src.models.contracts import (
    validate_model_output,
)
from src.models.g1_near_gu import (
    G1_CNN_EMBEDDING_SIZE,
    G1_CTC_CLASSES,
    G1_EXPECTED_TRAINABLE_PARAMETERS,
    G1_LSTM_HIDDEN_SIZE,
    G1NearGu,
    count_trainable_parameters,
    g1_window_lengths,
)
from src.training.ctc_loss import (
    torch_ctc_loss_from_logits,
)


@pytest.fixture(
    scope="module"
)
def model() -> G1NearGu:

    torch.manual_seed(
        2601
    )

    instance = G1NearGu()
    instance.eval()

    return instance


def test_frozen_architecture_and_parameter_count(
    model: G1NearGu,
):

    assert (
        model.window_encoder.conv1.in_channels
        == 1
    )

    assert (
        model.window_encoder.conv1.out_channels
        == 32
    )

    assert (
        model.window_encoder.conv2.out_channels
        == 64
    )

    assert (
        model.window_encoder.conv3.out_channels
        == 128
    )

    assert model.bilstm1.input_size == (
        G1_CNN_EMBEDDING_SIZE
    )

    assert model.bilstm1.hidden_size == (
        G1_LSTM_HIDDEN_SIZE
    )

    assert model.bilstm1.bidirectional is True

    assert model.bilstm2.input_size == 512

    assert model.bilstm2.hidden_size == (
        G1_LSTM_HIDDEN_SIZE
    )

    assert model.bilstm2.bidirectional is True

    assert model.classifier.in_features == 512

    assert model.classifier.out_features == (
        G1_CTC_CLASSES
    )

    assert count_trainable_parameters(
        model
    ) == (
        G1_EXPECTED_TRAINABLE_PARAMETERS
    )


def test_window_length_formula():

    observed = g1_window_lengths(
        [
            72,
            234,
            828,
        ]
    )

    assert observed.tolist() == [
        3,
        12,
        45,
    ]


def test_window_encoder_shape(
    model: G1NearGu,
):

    value = torch.zeros(
        2,
        1,
        36,
        38,
    )

    shapes = {}

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

                shapes[
                    name
                ] = tuple(
                    value.shape
                )

    assert shapes == {
        "pool1": (
            2,
            32,
            18,
            19,
        ),

        "pool2": (
            2,
            64,
            6,
            6,
        ),

        "pool3": (
            2,
            128,
            3,
            3,
        ),
    }

    assert tuple(
        value.flatten(
            start_dim=1
        ).shape
    ) == (
        2,
        1152,
    )


def test_forward_shape_and_shared_contract(
    model: G1NearGu,
):

    torch.manual_seed(
        2602
    )

    x_btf = torch.randn(
        3,
        108,
        38,
    )

    with torch.no_grad():

        output = model(
            x_btf,
            [
                72,
                90,
                108,
            ],
        )

    assert tuple(
        output.logits_btc.shape
    ) == (
        3,
        5,
        456,
    )

    assert validate_model_output(
        output
    ) == (
        3,
        4,
        5,
    )

    assert torch.count_nonzero(
        output.logits_btc[
            0,
            3:,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        output.logits_btc[
            1,
            4:,
        ]
    ).item() == 0


def test_padded_tail_does_not_change_valid_logits(
    model: G1NearGu,
):

    torch.manual_seed(
        2603
    )

    original = torch.randn(
        2,
        90,
        38,
    )

    changed = original.clone()

    changed[
        0,
        72:,
        :,
    ] = 1000.0

    with torch.no_grad():

        first = model(
            original,
            [
                72,
                90,
            ],
        )

        second = model(
            changed,
            [
                72,
                90,
            ],
        )

    torch.testing.assert_close(
        first.logits_btc[
            0,
            :3,
        ],
        second.logits_btc[
            0,
            :3,
        ],
        rtol=0.0,
        atol=0.0,
    )


def test_wrong_feature_width_rejected(
    model: G1NearGu,
):

    with pytest.raises(
        ValueError,
        match="38",
    ):

        model(
            torch.zeros(
                1,
                72,
                37,
            ),
            [
                72
            ],
        )


def test_float_lengths_rejected():

    with pytest.raises(
        TypeError,
        match="integers",
    ):

        g1_window_lengths(
            [
                72.0
            ]
        )


def test_too_short_length_rejected(
    model: G1NearGu,
):

    with pytest.raises(
        ValueError,
        match="at least 36",
    ):

        model(
            torch.zeros(
                1,
                72,
                38,
            ),
            [
                35
            ],
        )


def test_length_exceeding_padded_time_rejected(
    model: G1NearGu,
):

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):

        model(
            torch.zeros(
                1,
                72,
                38,
            ),
            [
                73
            ],
        )


def test_nonfinite_input_rejected(
    model: G1NearGu,
):

    x_btf = torch.zeros(
        1,
        72,
        38,
    )

    x_btf[
        0,
        10,
        4,
    ] = torch.nan

    with pytest.raises(
        ValueError,
        match="NaN",
    ):

        model(
            x_btf,
            [
                72
            ],
        )


def test_real_ctc_backward_is_finite_and_reaches_all_parameters():

    torch.manual_seed(
        2604
    )

    model = G1NearGu()
    model.train()

    x_btf = torch.randn(
        2,
        90,
        38,
    )

    output = model(
        x_btf,
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
        output.logits_btc,
        targets,
        output.output_lengths,
        target_lengths,
    )

    assert torch.isfinite(
        loss
    ).item()

    loss.backward()

    trainable = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    gradients = [
        parameter.grad
        for parameter in trainable
    ]

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

    squared_norm = sum(
        float(
            gradient
            .detach()
            .square()
            .sum()
            .item()
        )
        for gradient in gradients
    )

    assert squared_norm > 0.0

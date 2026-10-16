from __future__ import annotations

import pytest
import torch

from src.models.c4_tiny_swin import (
    C4_CTC_CLASSES,
    C4_EXPECTED_TRAINABLE_PARAMETERS,
    C4_FEATURE_WIDTH,
    C4_LSTM_HIDDEN_SIZE,
    C4_PATCH_SIZE,
    C4_STAGE_DEPTHS,
    C4_STAGE_DIMS,
    C4_STAGE_HEADS,
    C4_SWIN_SHIFT_SIZE,
    C4_SWIN_WINDOW_SIZE,
    C4_WINDOW_LENGTH,
    C4_WINDOW_STRIDE,
    C4TinySwin,
    PatchMerging,
    SwinTransformerBlock,
    c4_unfold_windows,
    c4_window_lengths,
    count_c4_trainable_parameters,
    padded_grid_size,
)
from src.models.contracts import (
    validate_model_output,
)
from src.training.ctc_loss import (
    torch_ctc_loss_from_logits,
)


@pytest.fixture(
    scope="module",
)
def model() -> C4TinySwin:

    torch.manual_seed(
        20260826
    )

    instance = C4TinySwin()
    instance.eval()

    return instance


def test_frozen_architecture_and_parameter_count(
    model: C4TinySwin,
):

    assert C4_FEATURE_WIDTH == 38
    assert C4_CTC_CLASSES == 456
    assert C4_WINDOW_LENGTH == 36
    assert C4_WINDOW_STRIDE == 18

    assert C4_PATCH_SIZE == (
        2,
        2,
    )

    assert C4_STAGE_DIMS == (
        32,
        64,
    )

    assert C4_STAGE_DEPTHS == (
        2,
        2,
    )

    assert C4_STAGE_HEADS == (
        2,
        4,
    )

    assert C4_SWIN_WINDOW_SIZE == (
        8,
        5,
    )

    assert C4_SWIN_SHIFT_SIZE == (
        4,
        2,
    )

    assert C4_LSTM_HIDDEN_SIZE == 96

    assert (
        model.bilstm.bidirectional
        is True
    )

    assert model.bilstm.num_layers == 1

    assert (
        count_c4_trainable_parameters(
            model
        )
        == C4_EXPECTED_TRAINABLE_PARAMETERS
    )

    assert (
        C4_EXPECTED_TRAINABLE_PARAMETERS
        == 348_604
    )


def test_parameter_subtotals_are_exact(
    model: C4TinySwin,
):

    counts = {
        "window_encoder": sum(
            parameter.numel()
            for parameter
            in model.window_encoder.parameters()
        ),
        "bilstm": sum(
            parameter.numel()
            for parameter
            in model.bilstm.parameters()
        ),
        "post_lstm_normalization": sum(
            parameter.numel()
            for parameter
            in model.post_lstm_normalization.parameters()
        ),
        "classifier": sum(
            parameter.numel()
            for parameter
            in model.classifier.parameters()
        ),
    }

    assert counts == {
        "window_encoder": 135_796,
        "bilstm": 124_416,
        "post_lstm_normalization": 384,
        "classifier": 88_008,
    }


def test_stage22_window_lengths_are_exact():

    observed = c4_window_lengths(
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
        padded_time=831,
    )

    expected = torch.tensor(
        [
            1,
            1,
            2,
            2,
            3,
            3,
            4,
            5,
            45,
        ]
    )

    torch.testing.assert_close(
        observed,
        expected,
    )


def test_window_unfold_preserves_time_and_feature_order():

    source = torch.arange(
        72 * 38,
        dtype=torch.float32,
    ).reshape(
        1,
        72,
        38,
    )

    windows = c4_unfold_windows(
        source
    )

    assert windows.shape == (
        1,
        3,
        36,
        38,
    )

    torch.testing.assert_close(
        windows[
            0,
            0,
        ],
        source[
            0,
            0:36,
        ],
    )

    torch.testing.assert_close(
        windows[
            0,
            1,
        ],
        source[
            0,
            18:54,
        ],
    )

    torch.testing.assert_close(
        windows[
            0,
            2,
        ],
        source[
            0,
            36:72,
        ],
    )


def test_invalid_raw_lengths_are_rejected():

    with pytest.raises(
        ValueError,
        match="at least 36",
    ):

        c4_window_lengths(
            [
                35
            ]
        )

    with pytest.raises(
        TypeError,
        match="integers",
    ):

        c4_window_lengths(
            [
                36.0
            ]
        )

    with pytest.raises(
        ValueError,
        match="exceeds",
    ):

        c4_window_lengths(
            [
                73
            ],
            padded_time=72,
        )


def test_minimal_swin_grid_padding_is_exact():

    assert padded_grid_size(
        18,
        19,
    ) == (
        24,
        20,
    )

    assert padded_grid_size(
        9,
        10,
    ) == (
        16,
        10,
    )

    assert padded_grid_size(
        3,
        2,
    ) == (
        8,
        5,
    )

    assert padded_grid_size(
        25,
        21,
    ) == (
        32,
        25,
    )


def test_window_encoder_shape(
    model: C4TinySwin,
):

    windows = torch.randn(
        3,
        1,
        36,
        38,
    )

    with torch.no_grad():

        embeddings = (
            model.encode_windows(
                windows
            )
        )

    assert embeddings.shape == (
        3,
        64,
    )

    assert torch.isfinite(
        embeddings
    ).all()


def test_window_encoder_shape_trace(
    model: C4TinySwin,
):

    observed = {}

    handles = [
        (
            model.window_encoder
            .patch_projection
            .register_forward_hook(
                lambda _module, _inputs, output: (
                    observed.__setitem__(
                        "patch",
                        tuple(
                            output.shape
                        ),
                    )
                )
            )
        ),
        (
            model.window_encoder
            .stage1
            .register_forward_hook(
                lambda _module, _inputs, output: (
                    observed.__setitem__(
                        "stage1",
                        tuple(
                            output.shape
                        ),
                    )
                )
            )
        ),
        (
            model.window_encoder
            .patch_merging
            .register_forward_hook(
                lambda _module, _inputs, output: (
                    observed.__setitem__(
                        "merge",
                        tuple(
                            output[0].shape
                        ),
                    )
                )
            )
        ),
        (
            model.window_encoder
            .stage2
            .register_forward_hook(
                lambda _module, _inputs, output: (
                    observed.__setitem__(
                        "stage2",
                        tuple(
                            output.shape
                        ),
                    )
                )
            )
        ),
    ]

    try:

        with torch.no_grad():

            model.encode_windows(
                torch.randn(
                    2,
                    1,
                    36,
                    38,
                )
            )

    finally:

        for handle in handles:

            handle.remove()

    assert observed == {
        "patch": (
            2,
            32,
            18,
            19,
        ),
        "stage1": (
            2,
            18,
            19,
            32,
        ),
        "merge": (
            2,
            9,
            10,
            64,
        ),
        "stage2": (
            2,
            9,
            10,
            64,
        ),
    }


def test_shifted_block_executes_in_both_stages(
    model: C4TinySwin,
):

    calls = {
        "stage1": 0,
        "stage2": 0,
    }

    def stage1_hook(
        _module,
        _inputs,
        _output,
    ):

        calls[
            "stage1"
        ] += 1

    def stage2_hook(
        _module,
        _inputs,
        _output,
    ):

        calls[
            "stage2"
        ] += 1

    handles = [
        (
            model.window_encoder
            .stage1
            .blocks[1]
            .register_forward_hook(
                stage1_hook
            )
        ),
        (
            model.window_encoder
            .stage2
            .blocks[1]
            .register_forward_hook(
                stage2_hook
            )
        ),
    ]

    try:

        with torch.no_grad():

            model.encode_windows(
                torch.randn(
                    1,
                    1,
                    36,
                    38,
                )
            )

    finally:

        for handle in handles:

            handle.remove()

    assert calls == {
        "stage1": 1,
        "stage2": 1,
    }

    assert (
        model.window_encoder
        .stage1
        .blocks[1]
        .shift_size
        == (
            4,
            2,
        )
    )

    assert (
        model.window_encoder
        .stage2
        .blocks[1]
        .shift_size
        == (
            4,
            2,
        )
    )


def test_relative_position_bias_contract(
    model: C4TinySwin,
):

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

    assert (
        stage1_attention
        .relative_position_bias_table
        .shape
        == (
            135,
            2,
        )
    )

    assert (
        stage2_attention
        .relative_position_bias_table
        .shape
        == (
            135,
            4,
        )
    )

    assert (
        stage1_attention
        .relative_position_index
        .shape
        == (
            40,
            40,
        )
    )

    assert (
        stage2_attention
        .relative_position_index
        .shape
        == (
            40,
            40,
        )
    )

    assert (
        stage1_attention
        .relative_position_bias_table
        .requires_grad
    )

    assert (
        stage2_attention
        .relative_position_bias_table
        .requires_grad
    )


def test_shifted_block_ignores_masked_spatial_values():

    torch.manual_seed(
        101
    )

    block = SwinTransformerBlock(
        dim=32,
        num_heads=2,
        window_size=(
            8,
            5,
        ),
        shift_size=(
            4,
            2,
        ),
        mlp_ratio=4,
        dropout=0.0,
    )

    block.eval()

    x = torch.randn(
        1,
        7,
        9,
        32,
    )

    mask = torch.zeros(
        1,
        7,
        9,
        dtype=torch.bool,
    )

    mask[
        :,
        :5,
        :7,
    ] = True

    changed = x.clone()

    changed[
        ~mask
    ] = (
        torch.randn_like(
            changed[
                ~mask
            ]
        )
        * 1000.0
    )

    with torch.no_grad():

        first = block(
            x,
            mask,
        )

        second = block(
            changed,
            mask,
        )

    torch.testing.assert_close(
        first[
            mask
        ],
        second[
            mask
        ],
        rtol=0,
        atol=0,
    )

    assert torch.count_nonzero(
        first[
            ~mask
        ]
    ).item() == 0

    assert torch.count_nonzero(
        second[
            ~mask
        ]
    ).item() == 0


def test_patch_merging_ignores_masked_spatial_values():

    torch.manual_seed(
        102
    )

    merge = PatchMerging(
        input_dim=32
    )

    merge.eval()

    x = torch.randn(
        1,
        5,
        7,
        32,
    )

    mask = torch.zeros(
        1,
        5,
        7,
        dtype=torch.bool,
    )

    mask[
        :,
        :4,
        :6,
    ] = True

    changed = x.clone()

    changed[
        ~mask
    ] = (
        torch.randn_like(
            changed[
                ~mask
            ]
        )
        * 1000.0
    )

    with torch.no_grad():

        (
            first,
            first_mask,
        ) = merge(
            x,
            mask,
        )

        (
            second,
            second_mask,
        ) = merge(
            changed,
            mask,
        )

    torch.testing.assert_close(
        first,
        second,
        rtol=0,
        atol=0,
    )

    assert torch.equal(
        first_mask,
        second_mask,
    )


def test_forward_shape_and_shared_contract(
    model: C4TinySwin,
):

    x = torch.randn(
        3,
        108,
        38,
    )

    with torch.no_grad():

        output = model(
            x,
            [
                72,
                90,
                108,
            ],
        )

    assert output.logits_btc.shape == (
        3,
        5,
        456,
    )

    assert validate_model_output(
        output,
        expected_classes=456,
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


def test_padded_raw_tail_does_not_change_valid_logits(
    model: C4TinySwin,
):

    torch.manual_seed(
        103
    )

    first_input = torch.randn(
        2,
        90,
        38,
    )

    second_input = (
        first_input.clone()
    )

    second_input[
        0,
        72:,
    ] = (
        torch.randn_like(
            second_input[
                0,
                72:,
            ]
        )
        * 1000.0
    )

    with torch.no_grad():

        first = model(
            first_input,
            [
                72,
                90,
            ],
        )

        second = model(
            second_input,
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
        rtol=0,
        atol=0,
    )

    assert torch.count_nonzero(
        first.logits_btc[
            0,
            3:,
        ]
    ).item() == 0

    assert torch.count_nonzero(
        second.logits_btc[
            0,
            3:,
        ]
    ).item() == 0


def test_wrong_feature_width_rejected(
    model: C4TinySwin,
):

    with pytest.raises(
        ValueError,
        match="feature width",
    ):

        model(
            torch.randn(
                1,
                72,
                37,
            ),
            [
                72
            ],
        )


def test_nonfinite_input_rejected(
    model: C4TinySwin,
):

    x = torch.randn(
        1,
        72,
        38,
    )

    x[
        0,
        0,
        0,
    ] = float(
        "nan"
    )

    with pytest.raises(
        ValueError,
        match="NaN or infinity",
    ):

        model(
            x,
            [
                72
            ],
        )


def test_model_rejects_invalid_length_metadata(
    model: C4TinySwin,
):

    x = torch.randn(
        1,
        72,
        38,
    )

    with pytest.raises(
        TypeError,
        match="integers",
    ):

        model(
            x,
            [
                72.0
            ],
        )

    with pytest.raises(
        ValueError,
        match="batch size",
    ):

        model(
            x,
            [
                72,
                72,
            ],
        )


def test_ctc_backward_reaches_every_parameter():

    torch.manual_seed(
        20260826
    )

    model = C4TinySwin()
    model.train()

    x = torch.randn(
        2,
        90,
        38,
    )

    output = model(
        x,
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
                -1,
            ],
            [
                3,
                4,
                5,
            ],
        ],
        dtype=torch.long,
    )

    loss = torch_ctc_loss_from_logits(
        output.logits_btc,
        targets,
        output.output_lengths,
        [
            2,
            3,
        ],
    )

    assert torch.isfinite(
        loss
    )

    loss.backward()

    parameters = [
        parameter
        for parameter
        in model.parameters()
        if parameter.requires_grad
    ]

    assert len(
        parameters
    ) == 73

    assert all(
        parameter.grad is not None
        for parameter
        in parameters
    )

    assert all(
        torch.isfinite(
            parameter.grad
        ).all()
        for parameter
        in parameters
    )

    global_norm = torch.stack(
        [
            parameter.grad.detach().norm(
                2
            )
            for parameter
            in parameters
        ]
    ).norm(
        2
    )

    assert torch.isfinite(
        global_norm
    )

    assert (
        global_norm.item()
        > 0.0
    )

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from src.models.contracts import ModelOutput


C1_FEATURE_WIDTH = 38
C1_CTC_CLASSES = 456

C1_CONV1_CHANNELS = 64
C1_CONV2_CHANNELS = 96
C1_CONV_KERNEL_SIZE = 5
C1_CONV_STRIDE = 2
C1_CONV_PADDING = 2
C1_CONV_DROPOUT = 0.10

C1_LSTM_HIDDEN_SIZE = 128
C1_LSTM_OUTPUT_SIZE = 256
C1_POST_LSTM_DROPOUT = 0.15

C1_EXPECTED_TRAINABLE_PARAMETERS = 392_488


def _as_integer_lengths(
    raw_lengths: Sequence[int] | torch.Tensor,
) -> torch.Tensor:

    try:
        lengths = torch.as_tensor(
            raw_lengths
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            "raw_lengths must be an integer sequence."
        ) from exc

    if lengths.ndim != 1:

        raise ValueError(
            "raw_lengths must be one-dimensional."
        )

    if lengths.numel() == 0:

        raise ValueError(
            "raw_lengths cannot be empty."
        )

    if (
        lengths.dtype == torch.bool
        or torch.is_floating_point(
            lengths
        )
        or torch.is_complex(
            lengths
        )
    ):

        raise TypeError(
            "raw_lengths must contain integers."
        )

    lengths = (
        lengths
        .detach()
        .to(
            device="cpu",
            dtype=torch.long,
        )
    )

    if torch.any(
        lengths <= 0
    ).item():

        raise ValueError(
            "Every raw valid length must be positive."
        )

    return lengths


def _single_conv_lengths(
    lengths: torch.Tensor,
) -> torch.Tensor:

    numerator = (
        lengths
        + 2 * C1_CONV_PADDING
        - (
            C1_CONV_KERNEL_SIZE
            - 1
        )
        - 1
    )

    return torch.div(
        numerator,
        C1_CONV_STRIDE,
        rounding_mode="floor",
    ) + 1


def c1_length_trace(
    raw_lengths: Sequence[int] | torch.Tensor,
    *,
    padded_time: int | None = None,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """
    Return raw, post-Conv1 and post-Conv2
    valid temporal lengths.
    """

    raw = _as_integer_lengths(
        raw_lengths
    )

    if padded_time is not None:

        padded_time = int(
            padded_time
        )

        if padded_time <= 0:

            raise ValueError(
                "Padded input time must be positive."
            )

        if torch.any(
            raw > padded_time
        ).item():

            raise ValueError(
                "A raw valid length exceeds "
                "padded input time."
            )

    after_conv1 = _single_conv_lengths(
        raw
    )

    after_conv2 = _single_conv_lengths(
        after_conv1
    )

    if torch.any(
        after_conv2 <= 0
    ).item():

        raise ValueError(
            "C1 temporal encoder produced "
            "a nonpositive length."
        )

    return (
        raw,
        after_conv1,
        after_conv2,
    )


def c1_output_lengths(
    raw_lengths: Sequence[int] | torch.Tensor,
    *,
    padded_time: int | None = None,
) -> torch.Tensor:
    """
    Return exact valid output lengths after
    both Conv1D layers.
    """

    return c1_length_trace(
        raw_lengths,
        padded_time=padded_time,
    )[2]


def _time_mask(
    lengths: torch.Tensor,
    max_time: int,
    *,
    device: torch.device,
) -> torch.Tensor:

    positions = torch.arange(
        max_time,
        device=device,
    ).unsqueeze(
        0
    )

    return (
        positions
        < lengths.to(
            device
        ).unsqueeze(
            1
        )
    )


class LengthMaskedBatchNorm1d(
    nn.BatchNorm1d
):
    """
    BatchNorm1d whose statistics use only valid
    temporal positions.

    It retains the standard BatchNorm affine
    parameter count while preventing padded
    timesteps from changing batch or running
    statistics.
    """

    def forward(
        self,
        input_bct: torch.Tensor,
        lengths: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:

        if input_bct.ndim != 3:

            raise ValueError(
                "Masked BatchNorm input must "
                "be B x C x T."
            )

        (
            batch_size,
            channels,
            padded_time,
        ) = input_bct.shape

        if channels != self.num_features:

            raise ValueError(
                "Masked BatchNorm channel "
                "count changed."
            )

        valid_lengths = _as_integer_lengths(
            lengths
        )

        if valid_lengths.shape[0] != batch_size:

            raise ValueError(
                "BatchNorm lengths do not "
                "match batch size."
            )

        if torch.any(
            valid_lengths > padded_time
        ).item():

            raise ValueError(
                "A BatchNorm length exceeds "
                "padded time."
            )

        mask_bool = _time_mask(
            valid_lengths,
            padded_time,
            device=input_bct.device,
        ).unsqueeze(
            1
        )

        statistics_dtype = (
            torch.float32
            if input_bct.dtype in {
                torch.float16,
                torch.bfloat16,
            }
            else input_bct.dtype
        )

        values = input_bct.to(
            statistics_dtype
        )

        mask = mask_bool.to(
            statistics_dtype
        )

        valid_count = mask.sum()

        use_batch_statistics = (
            self.training
            or not self.track_running_stats
        )

        if use_batch_statistics:

            if valid_count.item() <= 1:

                raise ValueError(
                    "Masked BatchNorm requires "
                    "more than one valid position."
                )

            mean = (
                (
                    values
                    * mask
                ).sum(
                    dim=(
                        0,
                        2,
                    )
                )
                / valid_count
            )

            centered = (
                values
                - mean.view(
                    1,
                    -1,
                    1,
                )
            )

            variance = (
                (
                    centered.square()
                    * mask
                ).sum(
                    dim=(
                        0,
                        2,
                    )
                )
                / valid_count
            )

            if self.track_running_stats:

                with torch.no_grad():

                    self.num_batches_tracked.add_(
                        1
                    )

                    if self.momentum is None:

                        factor = (
                            1.0
                            / float(
                                self
                                .num_batches_tracked
                                .item()
                            )
                        )

                    else:

                        factor = float(
                            self.momentum
                        )

                    unbiased_variance = (
                        variance.detach()
                        * (
                            valid_count
                            / (
                                valid_count
                                - 1.0
                            )
                        )
                    )

                    self.running_mean.lerp_(
                        mean.detach(),
                        factor,
                    )

                    self.running_var.lerp_(
                        unbiased_variance,
                        factor,
                    )

        else:

            mean = self.running_mean.to(
                statistics_dtype
            )

            variance = self.running_var.to(
                statistics_dtype
            )

        output = (
            values
            - mean.view(
                1,
                -1,
                1,
            )
        ) / torch.sqrt(
            variance.view(
                1,
                -1,
                1,
            )
            + self.eps
        )

        if self.affine:

            output = (
                output
                * self.weight.to(
                    statistics_dtype
                ).view(
                    1,
                    -1,
                    1,
                )
                + self.bias.to(
                    statistics_dtype
                ).view(
                    1,
                    -1,
                    1,
                )
            )

        output = output.masked_fill(
            ~mask_bool,
            0.0,
        )

        return output.to(
            input_bct.dtype
        )


class C1Compact(nn.Module):
    """
    Length-safe compact
    Conv1D -> BiLSTM -> CTC recognizer.
    """

    model_id = "c1"

    architecture_family = (
        "conv1d_bilstm_ctc"
    )

    def __init__(self) -> None:

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=C1_FEATURE_WIDTH,
            out_channels=C1_CONV1_CHANNELS,
            kernel_size=C1_CONV_KERNEL_SIZE,
            stride=C1_CONV_STRIDE,
            padding=C1_CONV_PADDING,
            bias=True,
        )

        self.bn1 = LengthMaskedBatchNorm1d(
            C1_CONV1_CHANNELS
        )

        self.relu1 = nn.ReLU(
            inplace=False
        )

        self.dropout1 = nn.Dropout(
            C1_CONV_DROPOUT
        )

        self.conv2 = nn.Conv1d(
            in_channels=C1_CONV1_CHANNELS,
            out_channels=C1_CONV2_CHANNELS,
            kernel_size=C1_CONV_KERNEL_SIZE,
            stride=C1_CONV_STRIDE,
            padding=C1_CONV_PADDING,
            bias=True,
        )

        self.bn2 = LengthMaskedBatchNorm1d(
            C1_CONV2_CHANNELS
        )

        self.relu2 = nn.ReLU(
            inplace=False
        )

        self.dropout2 = nn.Dropout(
            C1_CONV_DROPOUT
        )

        self.bilstm = nn.LSTM(
            input_size=C1_CONV2_CHANNELS,
            hidden_size=C1_LSTM_HIDDEN_SIZE,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        # LayerNorm contributes 512 parameters,
        # reconciling the frozen 392,488 total.
        self.post_lstm_norm = nn.LayerNorm(
            C1_LSTM_OUTPUT_SIZE,
            eps=1e-5,
            elementwise_affine=True,
        )

        self.post_lstm_dropout = nn.Dropout(
            C1_POST_LSTM_DROPOUT
        )

        self.classifier = nn.Linear(
            C1_LSTM_OUTPUT_SIZE,
            C1_CTC_CLASSES,
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(self) -> None:

        for convolution in (
            self.conv1,
            self.conv2,
        ):

            nn.init.kaiming_normal_(
                convolution.weight,
                mode="fan_out",
                nonlinearity="relu",
            )

            nn.init.zeros_(
                convolution.bias
            )

        for normalization in (
            self.bn1,
            self.bn2,
        ):

            nn.init.ones_(
                normalization.weight
            )

            nn.init.zeros_(
                normalization.bias
            )

            normalization.running_mean.zero_()
            normalization.running_var.fill_(
                1.0
            )

            normalization.num_batches_tracked.zero_()

        for (
            name,
            parameter,
        ) in self.bilstm.named_parameters():

            if "weight_ih" in name:

                nn.init.xavier_uniform_(
                    parameter
                )

            elif "weight_hh" in name:

                nn.init.orthogonal_(
                    parameter
                )

            elif "bias" in name:

                nn.init.zeros_(
                    parameter
                )

        nn.init.ones_(
            self.post_lstm_norm.weight
        )

        nn.init.zeros_(
            self.post_lstm_norm.bias
        )

        nn.init.xavier_uniform_(
            self.classifier.weight
        )

        nn.init.zeros_(
            self.classifier.bias
        )

    def forward(
        self,
        x_btf: torch.Tensor,
        raw_lengths: Sequence[int] | torch.Tensor,
    ) -> ModelOutput:

        if not torch.is_tensor(
            x_btf
        ):

            raise TypeError(
                "x_btf must be a torch.Tensor."
            )

        if x_btf.ndim != 3:

            raise ValueError(
                "x_btf must have shape B x T x 38."
            )

        (
            batch_size,
            padded_time,
            feature_width,
        ) = x_btf.shape

        if batch_size <= 0:

            raise ValueError(
                "C1 input batch cannot be empty."
            )

        if feature_width != C1_FEATURE_WIDTH:

            raise ValueError(
                "C1 input feature width must equal 38."
            )

        if not torch.is_floating_point(
            x_btf
        ):

            raise TypeError(
                "x_btf must contain "
                "floating-point values."
            )

        if not torch.isfinite(
            x_btf
        ).all().item():

            raise ValueError(
                "C1 input contains NaN or infinity."
            )

        (
            raw,
            length1,
            length2,
        ) = c1_length_trace(
            raw_lengths,
            padded_time=padded_time,
        )

        if raw.shape[0] != batch_size:

            raise ValueError(
                "raw_lengths does not match "
                "the C1 batch size."
            )

        raw_mask = _time_mask(
            raw,
            padded_time,
            device=x_btf.device,
        )

        sequence = x_btf.masked_fill(
            ~raw_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        sequence = (
            sequence
            .transpose(
                1,
                2,
            )
            .contiguous()
        )

        sequence = self.conv1(
            sequence
        )

        sequence = self.bn1(
            sequence,
            length1,
        )

        sequence = self.relu1(
            sequence
        )

        sequence = self.dropout1(
            sequence
        )

        mask1 = _time_mask(
            length1,
            sequence.shape[2],
            device=sequence.device,
        )

        sequence = sequence.masked_fill(
            ~mask1.unsqueeze(
                1
            ),
            0.0,
        )

        sequence = self.conv2(
            sequence
        )

        sequence = self.bn2(
            sequence,
            length2,
        )

        sequence = self.relu2(
            sequence
        )

        sequence = self.dropout2(
            sequence
        )

        mask2 = _time_mask(
            length2,
            sequence.shape[2],
            device=sequence.device,
        )

        sequence = sequence.masked_fill(
            ~mask2.unsqueeze(
                1
            ),
            0.0,
        )

        sequence = (
            sequence
            .transpose(
                1,
                2,
            )
            .contiguous()
        )

        packed = pack_padded_sequence(
            sequence,
            length2,
            batch_first=True,
            enforce_sorted=False,
        )

        packed, _ = self.bilstm(
            packed
        )

        recurrent, _ = pad_packed_sequence(
            packed,
            batch_first=True,
            total_length=sequence.shape[1],
        )

        recurrent = self.post_lstm_norm(
            recurrent
        )

        recurrent = self.post_lstm_dropout(
            recurrent
        )

        recurrent = recurrent.masked_fill(
            ~mask2.unsqueeze(
                -1
            ),
            0.0,
        )

        logits = self.classifier(
            recurrent
        )

        logits = logits.masked_fill(
            ~mask2.unsqueeze(
                -1
            ),
            0.0,
        )

        return ModelOutput(
            logits_btc=logits,
            output_lengths=length2,
        )


def count_c1_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

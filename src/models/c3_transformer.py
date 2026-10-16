from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn

from src.models.c1_compact import (
    C1_CONV1_CHANNELS,
    C1_CONV2_CHANNELS,
    C1_CONV_DROPOUT,
    C1_CONV_KERNEL_SIZE,
    C1_CONV_PADDING,
    C1_CONV_STRIDE,
    C1_CTC_CLASSES,
    C1_FEATURE_WIDTH,
    LengthMaskedBatchNorm1d,
    c1_length_trace,
)
from src.models.contracts import ModelOutput


C3_D_MODEL = 96
C3_ATTENTION_HEADS = 4
C3_FEEDFORWARD_SIZE = 192
C3_ENCODER_LAYERS = 2
C3_DROPOUT = 0.10
C3_POSITION_MAX_LENGTH = 1000
C3_EXPECTED_TRAINABLE_PARAMETERS = 237_352


def c3_output_lengths(
    raw_lengths: Sequence[int] | torch.Tensor,
    *,
    padded_time: int | None = None,
) -> torch.Tensor:
    """Return C3 lengths after its two Conv1D layers."""

    return c1_length_trace(
        raw_lengths,
        padded_time=padded_time,
    )[2]


def _integer_lengths(
    lengths: Sequence[int] | torch.Tensor,
    *,
    batch_size: int | None = None,
    padded_time: int | None = None,
) -> torch.Tensor:

    try:
        values = torch.as_tensor(
            lengths
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            "lengths must be an integer sequence."
        ) from exc

    if values.ndim != 1:
        raise ValueError(
            "lengths must be one-dimensional."
        )

    if values.numel() == 0:
        raise ValueError(
            "lengths cannot be empty."
        )

    if (
        values.dtype == torch.bool
        or torch.is_floating_point(values)
        or torch.is_complex(values)
    ):
        raise TypeError(
            "lengths must contain integers."
        )

    values = (
        values
        .detach()
        .to(
            device="cpu",
            dtype=torch.long,
        )
    )

    if (
        batch_size is not None
        and values.shape[0] != batch_size
    ):
        raise ValueError(
            "lengths does not match batch size."
        )

    if torch.any(
        values <= 0
    ).item():
        raise ValueError(
            "Every valid length must be positive."
        )

    if (
        padded_time is not None
        and torch.any(
            values > padded_time
        ).item()
    ):
        raise ValueError(
            "A valid length exceeds padded time."
        )

    return values


def c3_key_padding_mask(
    output_lengths: Sequence[int] | torch.Tensor,
    *,
    padded_time: int,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """
    Return B x T Boolean mask.

    True means the Transformer must ignore
    that padded temporal position.
    """

    padded_time = int(
        padded_time
    )

    if padded_time <= 0:
        raise ValueError(
            "padded_time must be positive."
        )

    lengths = _integer_lengths(
        output_lengths,
        padded_time=padded_time,
    )

    positions = torch.arange(
        padded_time,
        device=device,
    ).unsqueeze(
        0
    )

    return (
        positions
        >= lengths.to(
            device
        ).unsqueeze(
            1
        )
    )


class SinusoidalPositionalEncoding(nn.Module):
    """
    Fixed sinusoidal positional representation.

    This module contains no trainable parameters.
    """

    def __init__(
        self,
        d_model: int = C3_D_MODEL,
        max_length: int = C3_POSITION_MAX_LENGTH,
    ) -> None:

        super().__init__()

        d_model = int(
            d_model
        )

        max_length = int(
            max_length
        )

        if d_model <= 0:
            raise ValueError(
                "d_model must be positive."
            )

        if max_length <= 0:
            raise ValueError(
                "max_length must be positive."
            )

        position = torch.arange(
            max_length,
            dtype=torch.float32,
        ).unsqueeze(
            1
        )

        frequency = torch.exp(
            torch.arange(
                0,
                d_model,
                2,
                dtype=torch.float32,
            )
            * (
                -math.log(
                    10000.0
                )
                / d_model
            )
        )

        encoding = torch.zeros(
            max_length,
            d_model,
            dtype=torch.float32,
        )

        encoding[
            :,
            0::2,
        ] = torch.sin(
            position
            * frequency
        )

        if d_model > 1:

            odd_width = encoding[
                :,
                1::2,
            ].shape[1]

            encoding[
                :,
                1::2,
            ] = torch.cos(
                position
                * frequency[
                    :odd_width
                ]
            )

        self.d_model = d_model
        self.max_length = max_length

        self.register_buffer(
            "encoding",
            encoding.unsqueeze(
                0
            ),
            persistent=True,
        )

    def forward(
        self,
        sequence_btd: torch.Tensor,
    ) -> torch.Tensor:

        if not torch.is_tensor(
            sequence_btd
        ):
            raise TypeError(
                "sequence_btd must be a torch.Tensor."
            )

        if sequence_btd.ndim != 3:
            raise ValueError(
                "sequence_btd must have shape B x T x D."
            )

        if sequence_btd.shape[2] != self.d_model:
            raise ValueError(
                "Position-encoding feature width changed."
            )

        if sequence_btd.shape[1] > self.max_length:
            raise ValueError(
                "Sequence exceeds positional-encoding capacity."
            )

        if not torch.is_floating_point(
            sequence_btd
        ):
            raise TypeError(
                "Position-encoding input must be floating point."
            )

        positional = self.encoding[
            :,
            :sequence_btd.shape[1],
            :,
        ].to(
            device=sequence_btd.device,
            dtype=sequence_btd.dtype,
        )

        return (
            sequence_btd
            + positional
        )


class C3Transformer(nn.Module):
    """
    Length-safe compact
    Conv1D-Transformer-CTC recognizer.
    """

    model_id = "c3"

    architecture_family = (
        "conv1d_transformer_encoder_ctc"
    )

    def __init__(self) -> None:

        super().__init__()

        self.conv1 = nn.Conv1d(
            C1_FEATURE_WIDTH,
            C1_CONV1_CHANNELS,
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
            C1_CONV1_CHANNELS,
            C1_CONV2_CHANNELS,
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

        self.position_encoding = (
            SinusoidalPositionalEncoding(
                d_model=C3_D_MODEL,
                max_length=C3_POSITION_MAX_LENGTH,
            )
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=C3_D_MODEL,
            nhead=C3_ATTENTION_HEADS,
            dim_feedforward=C3_FEEDFORWARD_SIZE,
            dropout=C3_DROPOUT,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=C3_ENCODER_LAYERS,
            norm=nn.LayerNorm(
                C3_D_MODEL
            ),
            enable_nested_tensor=False,
        )

        self.classifier = nn.Linear(
            C3_D_MODEL,
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

        for layer in self.transformer.layers:

            nn.init.xavier_uniform_(
                layer.self_attn.in_proj_weight
            )

            nn.init.zeros_(
                layer.self_attn.in_proj_bias
            )

            nn.init.xavier_uniform_(
                layer.self_attn.out_proj.weight
            )

            nn.init.zeros_(
                layer.self_attn.out_proj.bias
            )

            nn.init.xavier_uniform_(
                layer.linear1.weight
            )

            nn.init.zeros_(
                layer.linear1.bias
            )

            nn.init.xavier_uniform_(
                layer.linear2.weight
            )

            nn.init.zeros_(
                layer.linear2.bias
            )

            nn.init.ones_(
                layer.norm1.weight
            )

            nn.init.zeros_(
                layer.norm1.bias
            )

            nn.init.ones_(
                layer.norm2.weight
            )

            nn.init.zeros_(
                layer.norm2.bias
            )

        nn.init.ones_(
            self.transformer.norm.weight
        )

        nn.init.zeros_(
            self.transformer.norm.bias
        )

        nn.init.xavier_uniform_(
            self.classifier.weight
        )

        nn.init.zeros_(
            self.classifier.bias
        )

    def encode_transformer(
        self,
        sequence_btd: torch.Tensor,
        output_lengths: Sequence[int] | torch.Tensor,
    ) -> torch.Tensor:
        """
        Add positional information and encode only
        the valid temporal region.
        """

        if not torch.is_tensor(
            sequence_btd
        ):
            raise TypeError(
                "sequence_btd must be a torch.Tensor."
            )

        if sequence_btd.ndim != 3:
            raise ValueError(
                "sequence_btd must have shape B x T x 96."
            )

        (
            batch_size,
            padded_time,
            width,
        ) = sequence_btd.shape

        if width != C3_D_MODEL:
            raise ValueError(
                "Transformer feature width must equal 96."
            )

        if not torch.is_floating_point(
            sequence_btd
        ):
            raise TypeError(
                "Transformer input must be floating point."
            )

        if not torch.isfinite(
            sequence_btd
        ).all().item():
            raise ValueError(
                "Transformer input contains NaN or infinity."
            )

        lengths = _integer_lengths(
            output_lengths,
            batch_size=batch_size,
            padded_time=padded_time,
        )

        padding_mask = c3_key_padding_mask(
            lengths,
            padded_time=padded_time,
            device=sequence_btd.device,
        )

        sequence = sequence_btd.masked_fill(
            padding_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        sequence = self.position_encoding(
            sequence
        )

        sequence = sequence.masked_fill(
            padding_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        encoded = self.transformer(
            sequence,
            src_key_padding_mask=padding_mask,
        )

        return encoded.masked_fill(
            padding_mask.unsqueeze(
                -1
            ),
            0.0,
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
                "C3 input batch cannot be empty."
            )

        if feature_width != C1_FEATURE_WIDTH:
            raise ValueError(
                "C3 input feature width must equal 38."
            )

        if not torch.is_floating_point(
            x_btf
        ):
            raise TypeError(
                "x_btf must contain floating-point values."
            )

        if not torch.isfinite(
            x_btf
        ).all().item():
            raise ValueError(
                "C3 input contains NaN or infinity."
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
                "the C3 batch size."
            )

        raw_padding = c3_key_padding_mask(
            raw,
            padded_time=padded_time,
            device=x_btf.device,
        )

        sequence = x_btf.masked_fill(
            raw_padding.unsqueeze(
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

        padding1 = c3_key_padding_mask(
            length1,
            padded_time=sequence.shape[2],
            device=sequence.device,
        )

        sequence = sequence.masked_fill(
            padding1.unsqueeze(
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

        padding2 = c3_key_padding_mask(
            length2,
            padded_time=sequence.shape[2],
            device=sequence.device,
        )

        sequence = sequence.masked_fill(
            padding2.unsqueeze(
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

        encoded = self.encode_transformer(
            sequence,
            length2,
        )

        logits = self.classifier(
            encoded
        )

        logits = logits.masked_fill(
            padding2.unsqueeze(
                -1
            ),
            0.0,
        )

        return ModelOutput(
            logits_btc=logits,
            output_lengths=length2,
        )


def count_c3_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

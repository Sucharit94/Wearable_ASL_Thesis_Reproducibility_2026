from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from src.models.c1_compact import (
    C1_CONV1_CHANNELS,
    C1_CONV2_CHANNELS,
    C1_CONV_DROPOUT,
    C1_CONV_KERNEL_SIZE,
    C1_CONV_PADDING,
    C1_CONV_STRIDE,
    C1_CTC_CLASSES,
    C1_FEATURE_WIDTH,
    C1_LSTM_HIDDEN_SIZE,
    C1_LSTM_OUTPUT_SIZE,
    C1_POST_LSTM_DROPOUT,
    LengthMaskedBatchNorm1d,
    c1_length_trace,
)
from src.models.contracts import ModelOutput


C2_ATTENTION_HEADS = 4
C2_ATTENTION_DROPOUT = 0.10
C2_EXPECTED_TRAINABLE_PARAMETERS = 656_168


def c2_output_lengths(
    raw_lengths: Sequence[int] | torch.Tensor,
    *,
    padded_time: int | None = None,
) -> torch.Tensor:
    """
    C2 preserves the exact two-Conv1D
    C1 temporal-length rule.
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


def _validate_attention_lengths(
    output_lengths: Sequence[int] | torch.Tensor,
    *,
    batch_size: int,
    padded_time: int,
) -> torch.Tensor:

    try:
        lengths = torch.as_tensor(
            output_lengths
        )

    except (
        TypeError,
        ValueError,
    ) as exc:

        raise TypeError(
            "output_lengths must be an integer sequence."
        ) from exc

    if lengths.ndim != 1:

        raise ValueError(
            "output_lengths must be one-dimensional."
        )

    if lengths.shape[0] != batch_size:

        raise ValueError(
            "output_lengths does not match batch size."
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
            "output_lengths must contain integers."
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
            "Every valid output length must be positive."
        )

    if torch.any(
        lengths > padded_time
    ).item():

        raise ValueError(
            "A valid output length exceeds padded time."
        )

    return lengths


class C2SelfAttention(nn.Module):
    """
    Length-safe Conv1D-BiLSTM-self-attention
    CTC recognizer.
    """

    model_id = "c2"

    architecture_family = (
        "conv1d_bilstm_self_attention_ctc"
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

        self.bilstm = nn.LSTM(
            input_size=C1_CONV2_CHANNELS,
            hidden_size=C1_LSTM_HIDDEN_SIZE,
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        self.recurrent_dropout = nn.Dropout(
            C1_POST_LSTM_DROPOUT
        )

        self.attention_input_norm = nn.LayerNorm(
            C1_LSTM_OUTPUT_SIZE
        )

        self.self_attention = nn.MultiheadAttention(
            embed_dim=C1_LSTM_OUTPUT_SIZE,
            num_heads=C2_ATTENTION_HEADS,
            dropout=C2_ATTENTION_DROPOUT,
            bias=True,
            batch_first=True,
        )

        self.post_attention_norm = nn.LayerNorm(
            C1_LSTM_OUTPUT_SIZE
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

        for normalization in (
            self.attention_input_norm,
            self.post_attention_norm,
        ):

            nn.init.ones_(
                normalization.weight
            )

            nn.init.zeros_(
                normalization.bias
            )

        nn.init.xavier_uniform_(
            self.self_attention.in_proj_weight
        )

        nn.init.zeros_(
            self.self_attention.in_proj_bias
        )

        nn.init.xavier_uniform_(
            self.self_attention.out_proj.weight
        )

        nn.init.zeros_(
            self.self_attention.out_proj.bias
        )

        nn.init.xavier_uniform_(
            self.classifier.weight
        )

        nn.init.zeros_(
            self.classifier.bias
        )

    def apply_self_attention(
        self,
        sequence_btd: torch.Tensor,
        output_lengths: Sequence[int] | torch.Tensor,
        *,
        return_attention_weights: bool = False,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor | None,
    ]:
        """
        Apply pre-normalized self-attention using
        a true padding mask and residual path.
        """

        if not torch.is_tensor(
            sequence_btd
        ):

            raise TypeError(
                "sequence_btd must be a torch.Tensor."
            )

        if sequence_btd.ndim != 3:

            raise ValueError(
                "sequence_btd must have "
                "shape B x T x 256."
            )

        (
            batch_size,
            padded_time,
            width,
        ) = sequence_btd.shape

        if width != C1_LSTM_OUTPUT_SIZE:

            raise ValueError(
                "Attention feature width must equal 256."
            )

        if not torch.is_floating_point(
            sequence_btd
        ):

            raise TypeError(
                "Attention input must be floating point."
            )

        if not torch.isfinite(
            sequence_btd
        ).all().item():

            raise ValueError(
                "Attention input contains NaN or infinity."
            )

        lengths = _validate_attention_lengths(
            output_lengths,
            batch_size=batch_size,
            padded_time=padded_time,
        )

        valid_mask = _time_mask(
            lengths,
            padded_time,
            device=sequence_btd.device,
        )

        sequence = sequence_btd.masked_fill(
            ~valid_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        normalized = self.attention_input_norm(
            sequence
        )

        normalized = normalized.masked_fill(
            ~valid_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        (
            attention_output,
            attention_weights,
        ) = self.self_attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid_mask,
            need_weights=return_attention_weights,
            average_attn_weights=False,
        )

        attention_output = (
            attention_output.masked_fill(
                ~valid_mask.unsqueeze(
                    -1
                ),
                0.0,
            )
        )

        output = self.post_attention_norm(
            sequence
            + attention_output
        )

        output = output.masked_fill(
            ~valid_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        if attention_weights is not None:

            attention_weights = (
                attention_weights.masked_fill(
                    ~valid_mask[
                        :,
                        None,
                        :,
                        None,
                    ],
                    0.0,
                )
            )

        return (
            output,
            attention_weights,
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
                "C2 input batch cannot be empty."
            )

        if feature_width != C1_FEATURE_WIDTH:

            raise ValueError(
                "C2 input feature width must equal 38."
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
                "C2 input contains NaN or infinity."
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
                "the C2 batch size."
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

        recurrent = self.recurrent_dropout(
            recurrent
        )

        recurrent = recurrent.masked_fill(
            ~mask2.unsqueeze(
                -1
            ),
            0.0,
        )

        attended, _ = self.apply_self_attention(
            recurrent,
            length2,
        )

        logits = self.classifier(
            attended
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


def count_c2_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

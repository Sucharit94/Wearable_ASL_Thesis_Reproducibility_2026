from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from src.models.contracts import (
    ModelOutput,
)


G1_FEATURE_WIDTH = 38
G1_CTC_CLASSES = 456

G1_WINDOW_LENGTH = 36
G1_WINDOW_STRIDE = 18

G1_LSTM_HIDDEN_SIZE = 256
G1_LSTM_OUTPUT_SIZE = 512

G1_CNN_EMBEDDING_SIZE = 1152

G1_EXPECTED_TRAINABLE_PARAMETERS = (
    4_791_688
)


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
            "raw_lengths must be "
            "an integer sequence."
        ) from exc

    if lengths.ndim != 1:

        raise ValueError(
            "raw_lengths must be "
            "one-dimensional."
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

    return (
        lengths
        .detach()
        .to(
            device="cpu",
            dtype=torch.long,
        )
    )


def g1_window_lengths(
    raw_lengths: Sequence[int] | torch.Tensor,
    *,
    padded_time: int | None = None,
) -> torch.Tensor:
    """
    Return exact complete-window counts for the
    frozen G1 window-length 36 / stride 18 rule.
    """

    lengths = _as_integer_lengths(
        raw_lengths
    )

    if torch.any(
        lengths < G1_WINDOW_LENGTH
    ).item():

        bad = torch.nonzero(
            lengths < G1_WINDOW_LENGTH,
            as_tuple=False,
        ).flatten().tolist()

        raise ValueError(
            "Every G1 input must contain "
            "at least 36 valid timesteps; "
            f"invalid indices: {bad}."
        )

    if padded_time is not None:

        padded_time = int(
            padded_time
        )

        if padded_time < G1_WINDOW_LENGTH:

            raise ValueError(
                "Padded input time must "
                "be at least 36."
            )

        if torch.any(
            lengths > padded_time
        ).item():

            raise ValueError(
                "A raw valid length exceeds "
                "padded input time."
            )

    return (
        1
        + torch.div(
            lengths - G1_WINDOW_LENGTH,
            G1_WINDOW_STRIDE,
            rounding_mode="floor",
        )
    )


class G1NearGu(
    nn.Module
):
    """
    Dimension-consistent near-Gu
    Conv2D -> 2xBiLSTM -> CTC model.

    This is not claimed as an exact reproduction.

    It follows the reported Conv2D/BiLSTM/CTC
    structure while resolving the dimensional
    ambiguity as 256 hidden units per direction,
    producing 512 BiLSTM output features.

    Input:
        x_btf:
            Padded float tensor B x T x 38.

        raw_lengths:
            Exact valid raw length per sample.

    Output:
        ModelOutput containing:
            raw B x Tout x 456 logits;
            exact valid complete-window counts.

        Padded output logits are explicitly zeroed.
    """

    model_id = "g1"

    architecture_family = (
        "window_conv2d_two_bilstm_ctc"
    )

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.window_encoder = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            in_channels=1,
                            out_channels=32,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        "bn1",
                        nn.BatchNorm2d(
                            32,
                            eps=1e-5,
                            momentum=0.1,
                            affine=True,
                            track_running_stats=True,
                        ),
                    ),
                    (
                        "relu1",
                        nn.ReLU(
                            inplace=False
                        ),
                    ),
                    (
                        "pool1",
                        nn.MaxPool2d(
                            kernel_size=2,
                            stride=2,
                            ceil_mode=False,
                        ),
                    ),
                    (
                        "conv2",
                        nn.Conv2d(
                            in_channels=32,
                            out_channels=64,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        "bn2",
                        nn.BatchNorm2d(
                            64,
                            eps=1e-5,
                            momentum=0.1,
                            affine=True,
                            track_running_stats=True,
                        ),
                    ),
                    (
                        "relu2",
                        nn.ReLU(
                            inplace=False
                        ),
                    ),
                    (
                        "pool2",
                        nn.MaxPool2d(
                            kernel_size=3,
                            stride=3,
                            ceil_mode=False,
                        ),
                    ),
                    (
                        "conv3",
                        nn.Conv2d(
                            in_channels=64,
                            out_channels=128,
                            kernel_size=3,
                            stride=1,
                            padding=1,
                            bias=True,
                        ),
                    ),
                    (
                        "bn3",
                        nn.BatchNorm2d(
                            128,
                            eps=1e-5,
                            momentum=0.1,
                            affine=True,
                            track_running_stats=True,
                        ),
                    ),
                    (
                        "relu3",
                        nn.ReLU(
                            inplace=False
                        ),
                    ),
                    (
                        "pool3",
                        nn.MaxPool2d(
                            kernel_size=2,
                            stride=2,
                            ceil_mode=False,
                        ),
                    ),
                ]
            )
        )

        self.bilstm1 = nn.LSTM(
            input_size=(
                G1_CNN_EMBEDDING_SIZE
            ),
            hidden_size=(
                G1_LSTM_HIDDEN_SIZE
            ),
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        self.bilstm2 = nn.LSTM(
            input_size=(
                G1_LSTM_OUTPUT_SIZE
            ),
            hidden_size=(
                G1_LSTM_HIDDEN_SIZE
            ),
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        self.classifier = nn.Linear(
            in_features=(
                G1_LSTM_OUTPUT_SIZE
            ),
            out_features=(
                G1_CTC_CLASSES
            ),
            bias=True,
        )

        self.reset_parameters()

    def reset_parameters(
        self,
    ) -> None:

        for module in (
            self.window_encoder.modules()
        ):

            if isinstance(
                module,
                nn.Conv2d,
            ):

                nn.init.kaiming_normal_(
                    module.weight,
                    mode="fan_out",
                    nonlinearity="relu",
                )

                nn.init.zeros_(
                    module.bias
                )

            elif isinstance(
                module,
                nn.BatchNorm2d,
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

        for recurrent in (
            self.bilstm1,
            self.bilstm2,
        ):

            for (
                name,
                parameter,
            ) in recurrent.named_parameters():

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

        nn.init.xavier_uniform_(
            self.classifier.weight
        )

        nn.init.zeros_(
            self.classifier.bias
        )

    def encode_windows(
        self,
        windows_bchw: torch.Tensor,
    ) -> torch.Tensor:

        if not torch.is_tensor(
            windows_bchw
        ):

            raise TypeError(
                "windows_bchw must be "
                "a torch.Tensor."
            )

        if windows_bchw.ndim != 4:

            raise ValueError(
                "windows_bchw must have shape "
                "N x 1 x 36 x 38."
            )

        if tuple(
            windows_bchw.shape[
                1:
            ]
        ) != (
            1,
            36,
            38,
        ):

            raise ValueError(
                "Every G1 window must have "
                "shape 1 x 36 x 38."
            )

        encoded = self.window_encoder(
            windows_bchw
        )

        if tuple(
            encoded.shape[
                1:
            ]
        ) != (
            128,
            3,
            3,
        ):

            raise RuntimeError(
                "G1 Conv2D encoder did not "
                "produce 128 x 3 x 3."
            )

        return encoded.flatten(
            start_dim=1
        )

    def forward(
        self,
        x_btf: torch.Tensor,
        raw_lengths: (
            Sequence[int]
            | torch.Tensor
        ),
    ) -> ModelOutput:

        if not torch.is_tensor(
            x_btf
        ):

            raise TypeError(
                "x_btf must be "
                "a torch.Tensor."
            )

        if x_btf.ndim != 3:

            raise ValueError(
                "x_btf must have shape "
                "B x T x 38."
            )

        (
            batch_size,
            padded_time,
            feature_width,
        ) = x_btf.shape

        if batch_size <= 0:

            raise ValueError(
                "G1 input batch cannot be empty."
            )

        if feature_width != (
            G1_FEATURE_WIDTH
        ):

            raise ValueError(
                "G1 input feature width "
                "must equal 38."
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
                "G1 input contains "
                "NaN or infinity."
            )

        window_lengths = (
            g1_window_lengths(
                raw_lengths,
                padded_time=padded_time,
            )
        )

        if window_lengths.shape[
            0
        ] != batch_size:

            raise ValueError(
                "raw_lengths does not match "
                "the G1 batch size."
            )

        padded_window_count = (
            1
            + (
                padded_time
                - G1_WINDOW_LENGTH
            )
            // G1_WINDOW_STRIDE
        )

        # Tensor.unfold produces:
        # B x Nwindows x 38 x 36.
        windows = x_btf.unfold(
            dimension=1,
            size=G1_WINDOW_LENGTH,
            step=G1_WINDOW_STRIDE,
        )

        # Convert to:
        # B x Nwindows x 36 x 38.
        windows = windows.permute(
            0,
            1,
            3,
            2,
        ).contiguous()

        if windows.shape[
            1
        ] != padded_window_count:

            raise RuntimeError(
                "Unexpected padded "
                "G1 window count."
            )

        positions = torch.arange(
            padded_window_count,
            device=x_btf.device,
        ).unsqueeze(
            0
        )

        valid_mask = (
            positions
            < window_lengths
            .to(
                x_btf.device
            )
            .unsqueeze(
                1
            )
        )

        # Only complete valid windows enter the
        # Conv2D/BatchNorm encoder. Padded windows
        # cannot affect BatchNorm statistics.
        valid_windows = windows[
            valid_mask
        ].unsqueeze(
            1
        )

        valid_embeddings = (
            self.encode_windows(
                valid_windows
            )
        )

        flat_indices = torch.nonzero(
            valid_mask.reshape(
                -1
            ),
            as_tuple=False,
        ).flatten()

        flat_embeddings = (
            valid_embeddings.new_zeros(
                (
                    batch_size
                    * padded_window_count,
                    G1_CNN_EMBEDDING_SIZE,
                )
            )
        )

        flat_embeddings = (
            flat_embeddings.index_copy(
                0,
                flat_indices,
                valid_embeddings,
            )
        )

        sequence = flat_embeddings.reshape(
            batch_size,
            padded_window_count,
            G1_CNN_EMBEDDING_SIZE,
        )

        # Packed processing prevents padded windows
        # from changing either BiLSTM.
        packed = pack_padded_sequence(
            sequence,
            window_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        packed, _ = self.bilstm1(
            packed
        )

        packed, _ = self.bilstm2(
            packed
        )

        recurrent, _ = (
            pad_packed_sequence(
                packed,
                batch_first=True,
                total_length=(
                    padded_window_count
                ),
            )
        )

        logits = self.classifier(
            recurrent
        )

        # The decoder and CTCLoss already use exact
        # lengths, but zeroing padded logits creates
        # an additional safety boundary.
        logits = logits.masked_fill(
            ~valid_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        return ModelOutput(
            logits_btc=logits,
            output_lengths=window_lengths,
        )


def count_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

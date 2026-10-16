from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

from src.models.contracts import ModelOutput


C4_FEATURE_WIDTH = 38
C4_CTC_CLASSES = 456
C4_WINDOW_LENGTH = 36
C4_WINDOW_STRIDE = 18

C4_PATCH_SIZE = (
    2,
    2,
)

C4_STAGE_DIMS = (
    32,
    64,
)

C4_STAGE_DEPTHS = (
    2,
    2,
)

C4_STAGE_HEADS = (
    2,
    4,
)

C4_SWIN_WINDOW_SIZE = (
    8,
    5,
)

C4_SWIN_SHIFT_SIZE = (
    4,
    2,
)

C4_MLP_RATIO = 4
C4_SWIN_DROPOUT = 0.10

C4_LSTM_HIDDEN_SIZE = 96
C4_LSTM_OUTPUT_SIZE = 192
C4_POST_LSTM_DROPOUT = 0.15

C4_EXPECTED_TRAINABLE_PARAMETERS = (
    348_604
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

    return lengths.detach().to(
        device="cpu",
        dtype=torch.long,
    )


def c4_window_lengths(
    raw_lengths: Sequence[int] | torch.Tensor,
    *,
    padded_time: int | None = None,
) -> torch.Tensor:
    """
    Exact complete-window counts for the frozen
    36-frame window and stride-18 rule.
    """

    lengths = _as_integer_lengths(
        raw_lengths
    )

    if torch.any(
        lengths < C4_WINDOW_LENGTH
    ).item():

        bad = torch.nonzero(
            lengths < C4_WINDOW_LENGTH,
            as_tuple=False,
        ).flatten().tolist()

        raise ValueError(
            "Every C4 input must contain at least "
            "36 valid timesteps; "
            f"invalid indices: {bad}."
        )

    if padded_time is not None:

        padded_time = int(
            padded_time
        )

        if padded_time < C4_WINDOW_LENGTH:

            raise ValueError(
                "Padded input time must be "
                "at least 36."
            )

        if torch.any(
            lengths > padded_time
        ).item():

            raise ValueError(
                "A raw valid length exceeds "
                "padded input time."
            )

    return 1 + torch.div(
        lengths - C4_WINDOW_LENGTH,
        C4_WINDOW_STRIDE,
        rounding_mode="floor",
    )


def c4_unfold_windows(
    x_btf: torch.Tensor,
) -> torch.Tensor:
    """
    Preserve temporal and feature ordering while
    creating all padded-batch 36 x 38 windows.
    """

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

    if x_btf.shape[1] < (
        C4_WINDOW_LENGTH
    ):

        raise ValueError(
            "Padded input time must be "
            "at least 36."
        )

    if x_btf.shape[2] != (
        C4_FEATURE_WIDTH
    ):

        raise ValueError(
            "C4 input feature width "
            "must equal 38."
        )

    windows = x_btf.unfold(
        dimension=1,
        size=C4_WINDOW_LENGTH,
        step=C4_WINDOW_STRIDE,
    )

    return windows.permute(
        0,
        1,
        3,
        2,
    ).contiguous()


def padded_grid_size(
    height: int,
    width: int,
    window_size: tuple[int, int] = (
        C4_SWIN_WINDOW_SIZE
    ),
) -> tuple[int, int]:
    """
    Return the minimally padded grid required
    for Swin window partitioning.
    """

    height = int(
        height
    )

    width = int(
        width
    )

    window_height = int(
        window_size[0]
    )

    window_width = int(
        window_size[1]
    )

    if height <= 0 or width <= 0:

        raise ValueError(
            "Grid dimensions must be positive."
        )

    if (
        window_height <= 0
        or window_width <= 0
    ):

        raise ValueError(
            "Swin window dimensions "
            "must be positive."
        )

    padded_height = (
        (
            height
            + window_height
            - 1
        )
        // window_height
        * window_height
    )

    padded_width = (
        (
            width
            + window_width
            - 1
        )
        // window_width
        * window_width
    )

    return (
        padded_height,
        padded_width,
    )


def _window_partition(
    x_bhwc: torch.Tensor,
    window_size: tuple[int, int],
) -> torch.Tensor:

    (
        batch_size,
        height,
        width,
        channels,
    ) = x_bhwc.shape

    (
        window_height,
        window_width,
    ) = window_size

    if (
        height % window_height
        or width % window_width
    ):

        raise ValueError(
            "Grid must be divisible by "
            "the Swin window size."
        )

    windows = x_bhwc.view(
        batch_size,
        height // window_height,
        window_height,
        width // window_width,
        window_width,
        channels,
    )

    windows = windows.permute(
        0,
        1,
        3,
        2,
        4,
        5,
    ).contiguous()

    return windows.view(
        -1,
        window_height * window_width,
        channels,
    )


def _window_reverse(
    windows_bnc: torch.Tensor,
    *,
    batch_size: int,
    padded_height: int,
    padded_width: int,
    window_size: tuple[int, int],
) -> torch.Tensor:

    (
        window_height,
        window_width,
    ) = window_size

    channels = windows_bnc.shape[-1]

    x = windows_bnc.view(
        batch_size,
        padded_height // window_height,
        padded_width // window_width,
        window_height,
        window_width,
        channels,
    )

    x = x.permute(
        0,
        1,
        3,
        2,
        4,
        5,
    ).contiguous()

    return x.view(
        batch_size,
        padded_height,
        padded_width,
        channels,
    )


class WindowAttention(
    nn.Module
):
    """
    Multi-head attention inside one rectangular
    2D Swin window.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        dropout: float,
    ) -> None:

        super().__init__()

        if dim % num_heads:

            raise ValueError(
                "Swin embedding dimension must "
                "be divisible by heads."
            )

        self.dim = int(
            dim
        )

        self.num_heads = int(
            num_heads
        )

        self.window_size = (
            int(
                window_size[0]
            ),
            int(
                window_size[1]
            ),
        )

        self.head_dim = (
            self.dim
            // self.num_heads
        )

        self.scale = (
            self.head_dim
            ** -0.5
        )

        self.qkv = nn.Linear(
            self.dim,
            3 * self.dim,
            bias=True,
        )

        self.attention_dropout = (
            nn.Dropout(
                dropout
            )
        )

        self.projection = nn.Linear(
            self.dim,
            self.dim,
            bias=True,
        )

        self.projection_dropout = (
            nn.Dropout(
                dropout
            )
        )

        (
            window_height,
            window_width,
        ) = self.window_size

        relative_positions = (
            (2 * window_height - 1)
            * (2 * window_width - 1)
        )

        self.relative_position_bias_table = (
            nn.Parameter(
                torch.zeros(
                    relative_positions,
                    self.num_heads,
                )
            )
        )

        coordinates = torch.stack(
            torch.meshgrid(
                torch.arange(
                    window_height
                ),
                torch.arange(
                    window_width
                ),
                indexing="ij",
            )
        )

        coordinates_flat = torch.flatten(
            coordinates,
            start_dim=1,
        )

        relative_coordinates = (
            coordinates_flat[
                :,
                :,
                None,
            ]
            - coordinates_flat[
                :,
                None,
                :,
            ]
        )

        relative_coordinates = (
            relative_coordinates
            .permute(
                1,
                2,
                0,
            )
            .contiguous()
        )

        relative_coordinates[
            :,
            :,
            0,
        ] += (
            window_height
            - 1
        )

        relative_coordinates[
            :,
            :,
            1,
        ] += (
            window_width
            - 1
        )

        relative_coordinates[
            :,
            :,
            0,
        ] *= (
            2 * window_width
            - 1
        )

        relative_position_index = (
            relative_coordinates.sum(
                dim=-1
            )
        )

        self.register_buffer(
            "relative_position_index",
            relative_position_index,
            persistent=True,
        )

        nn.init.trunc_normal_(
            self.relative_position_bias_table,
            std=0.02,
        )

    def forward(
        self,
        x_bnc: torch.Tensor,
        *,
        valid_tokens_bn: torch.Tensor,
        shift_mask_nnn: (
            torch.Tensor
            | None
        ),
        windows_per_image: int,
    ) -> torch.Tensor:

        (
            batch_windows,
            tokens,
            channels,
        ) = x_bnc.shape

        expected_tokens = (
            self.window_size[0]
            * self.window_size[1]
        )

        if channels != self.dim:

            raise ValueError(
                "Window attention received "
                "the wrong embedding width."
            )

        if tokens != expected_tokens:

            raise ValueError(
                "Window attention received "
                "the wrong token count."
            )

        if valid_tokens_bn.shape != (
            batch_windows,
            tokens,
        ):

            raise ValueError(
                "Window token mask shape "
                "is invalid."
            )

        if (
            batch_windows
            % windows_per_image
        ):

            raise ValueError(
                "Window batch cannot be "
                "grouped by image."
            )

        qkv = self.qkv(
            x_bnc
        ).reshape(
            batch_windows,
            tokens,
            3,
            self.num_heads,
            self.head_dim,
        )

        (
            query,
            key,
            value,
        ) = qkv.permute(
            2,
            0,
            3,
            1,
            4,
        ).unbind(
            dim=0
        )

        scores = (
            query
            * self.scale
        ) @ key.transpose(
            -2,
            -1,
        )

        relative_bias = (
            self.relative_position_bias_table[
                self.relative_position_index
                .reshape(
                    -1
                )
            ]
            .view(
                tokens,
                tokens,
                self.num_heads,
            )
        )

        relative_bias = (
            relative_bias
            .permute(
                2,
                0,
                1,
            )
            .contiguous()
        )

        scores = (
            scores
            + relative_bias.unsqueeze(
                0
            )
        )

        safe_valid_tokens = (
            valid_tokens_bn.clone()
        )

        empty_windows = (
            ~safe_valid_tokens.any(
                dim=1
            )
        )

        if torch.any(
            empty_windows
        ).item():

            safe_valid_tokens[
                empty_windows,
                0,
            ] = True

        scores = scores.masked_fill(
            ~safe_valid_tokens[
                :,
                None,
                None,
                :,
            ],
            torch.finfo(
                scores.dtype
            ).min,
        )

        if shift_mask_nnn is not None:

            expected_shape = (
                windows_per_image,
                tokens,
                tokens,
            )

            if (
                shift_mask_nnn.shape
                != expected_shape
            ):

                raise ValueError(
                    "Shifted-window attention "
                    "mask shape is invalid."
                )

            image_batch = (
                batch_windows
                // windows_per_image
            )

            scores = scores.view(
                image_batch,
                windows_per_image,
                self.num_heads,
                tokens,
                tokens,
            )

            scores = (
                scores
                + shift_mask_nnn[
                    None,
                    :,
                    None,
                    :,
                    :,
                ]
            )

            scores = scores.view(
                batch_windows,
                self.num_heads,
                tokens,
                tokens,
            )

        attention = torch.softmax(
            scores,
            dim=-1,
        )

        attention = (
            self.attention_dropout(
                attention
            )
        )

        output = (
            attention @ value
        ).transpose(
            1,
            2,
        ).reshape(
            batch_windows,
            tokens,
            channels,
        )

        output = self.projection(
            output
        )

        output = (
            self.projection_dropout(
                output
            )
        )

        return output.masked_fill(
            ~valid_tokens_bn.unsqueeze(
                -1
            ),
            0.0,
        )


class SwinMLP(
    nn.Module
):

    def __init__(
        self,
        *,
        dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.linear1 = nn.Linear(
            dim,
            hidden_dim,
        )

        self.activation = (
            nn.GELU()
        )

        self.dropout1 = (
            nn.Dropout(
                dropout
            )
        )

        self.linear2 = nn.Linear(
            hidden_dim,
            dim,
        )

        self.dropout2 = (
            nn.Dropout(
                dropout
            )
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        x = self.linear1(
            x
        )

        x = self.activation(
            x
        )

        x = self.dropout1(
            x
        )

        x = self.linear2(
            x
        )

        return self.dropout2(
            x
        )


class SwinTransformerBlock(
    nn.Module
):
    """
    Pre-normalized regular or shifted
    2D Swin block.
    """

    def __init__(
        self,
        *,
        dim: int,
        num_heads: int,
        window_size: tuple[int, int],
        shift_size: tuple[int, int],
        mlp_ratio: int,
        dropout: float,
    ) -> None:

        super().__init__()

        self.dim = int(
            dim
        )

        self.window_size = (
            int(
                window_size[0]
            ),
            int(
                window_size[1]
            ),
        )

        self.shift_size = (
            int(
                shift_size[0]
            ),
            int(
                shift_size[1]
            ),
        )

        if not (
            0
            <= self.shift_size[0]
            < self.window_size[0]
            and
            0
            <= self.shift_size[1]
            < self.window_size[1]
        ):

            raise ValueError(
                "Swin shift must be smaller "
                "than its window."
            )

        self.normalization1 = (
            nn.LayerNorm(
                self.dim
            )
        )

        self.attention = (
            WindowAttention(
                dim=self.dim,
                num_heads=num_heads,
                window_size=(
                    self.window_size
                ),
                dropout=dropout,
            )
        )

        self.normalization2 = (
            nn.LayerNorm(
                self.dim
            )
        )

        self.mlp = SwinMLP(
            dim=self.dim,
            hidden_dim=(
                self.dim
                * int(
                    mlp_ratio
                )
            ),
            dropout=dropout,
        )

    def _shift_mask(
        self,
        *,
        padded_height: int,
        padded_width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:

        (
            shift_height,
            shift_width,
        ) = self.shift_size

        if (
            shift_height == 0
            and shift_width == 0
        ):

            return None

        (
            window_height,
            window_width,
        ) = self.window_size

        region = torch.zeros(
            (
                1,
                padded_height,
                padded_width,
                1,
            ),
            device=device,
            dtype=torch.long,
        )

        height_slices = (
            slice(
                0,
                -window_height,
            ),
            slice(
                -window_height,
                -shift_height,
            ),
            slice(
                -shift_height,
                None,
            ),
        )

        width_slices = (
            slice(
                0,
                -window_width,
            ),
            slice(
                -window_width,
                -shift_width,
            ),
            slice(
                -shift_width,
                None,
            ),
        )

        counter = 0

        for height_slice in (
            height_slices
        ):

            for width_slice in (
                width_slices
            ):

                region[
                    :,
                    height_slice,
                    width_slice,
                    :,
                ] = counter

                counter += 1

        region_windows = (
            _window_partition(
                region,
                self.window_size,
            )
            .squeeze(
                -1
            )
        )

        differences = (
            region_windows.unsqueeze(
                1
            )
            - region_windows.unsqueeze(
                2
            )
        )

        shift_mask = torch.zeros(
            differences.shape,
            device=device,
            dtype=dtype,
        )

        return shift_mask.masked_fill(
            differences != 0,
            -100.0,
        )

    def forward(
        self,
        x_bhwc: torch.Tensor,
        valid_mask_bhw: torch.Tensor,
    ) -> torch.Tensor:

        if x_bhwc.ndim != 4:

            raise ValueError(
                "Swin input must have shape "
                "B x H x W x C."
            )

        (
            batch_size,
            height,
            width,
            channels,
        ) = x_bhwc.shape

        if channels != self.dim:

            raise ValueError(
                "Swin block received the "
                "wrong embedding width."
            )

        if valid_mask_bhw.shape != (
            batch_size,
            height,
            width,
        ):

            raise ValueError(
                "Swin spatial mask shape "
                "is invalid."
            )

        if (
            valid_mask_bhw.dtype
            != torch.bool
        ):

            raise TypeError(
                "Swin spatial mask must "
                "be boolean."
            )

        x_bhwc = x_bhwc.masked_fill(
            ~valid_mask_bhw.unsqueeze(
                -1
            ),
            0.0,
        )

        shortcut = x_bhwc

        normalized = (
            self.normalization1(
                x_bhwc
            )
        )

        (
            padded_height,
            padded_width,
        ) = padded_grid_size(
            height,
            width,
            self.window_size,
        )

        pad_height = (
            padded_height
            - height
        )

        pad_width = (
            padded_width
            - width
        )

        normalized = F.pad(
            normalized,
            (
                0,
                0,
                0,
                pad_width,
                0,
                pad_height,
            ),
            value=0.0,
        )

        padded_mask = torch.zeros(
            (
                batch_size,
                padded_height,
                padded_width,
            ),
            device=x_bhwc.device,
            dtype=torch.bool,
        )

        padded_mask[
            :,
            :height,
            :width,
        ] = valid_mask_bhw

        (
            shift_height,
            shift_width,
        ) = self.shift_size

        if (
            shift_height
            or shift_width
        ):

            shifts = (
                -shift_height,
                -shift_width,
            )

            normalized = torch.roll(
                normalized,
                shifts=shifts,
                dims=(
                    1,
                    2,
                ),
            )

            padded_mask = torch.roll(
                padded_mask,
                shifts=shifts,
                dims=(
                    1,
                    2,
                ),
            )

        windows = _window_partition(
            normalized,
            self.window_size,
        )

        mask_windows = (
            _window_partition(
                padded_mask.unsqueeze(
                    -1
                ),
                self.window_size,
            )
            .squeeze(
                -1
            )
        )

        windows_per_image = (
            (
                padded_height
                // self.window_size[0]
            )
            * (
                padded_width
                // self.window_size[1]
            )
        )

        shift_mask = self._shift_mask(
            padded_height=(
                padded_height
            ),
            padded_width=(
                padded_width
            ),
            device=x_bhwc.device,
            dtype=x_bhwc.dtype,
        )

        attended_windows = (
            self.attention(
                windows,
                valid_tokens_bn=(
                    mask_windows
                ),
                shift_mask_nnn=(
                    shift_mask
                ),
                windows_per_image=(
                    windows_per_image
                ),
            )
        )

        attended = _window_reverse(
            attended_windows,
            batch_size=batch_size,
            padded_height=(
                padded_height
            ),
            padded_width=(
                padded_width
            ),
            window_size=(
                self.window_size
            ),
        )

        if (
            shift_height
            or shift_width
        ):

            attended = torch.roll(
                attended,
                shifts=(
                    shift_height,
                    shift_width,
                ),
                dims=(
                    1,
                    2,
                ),
            )

        attended = attended[
            :,
            :height,
            :width,
            :,
        ]

        x_bhwc = (
            shortcut
            + attended
        )

        x_bhwc = x_bhwc.masked_fill(
            ~valid_mask_bhw.unsqueeze(
                -1
            ),
            0.0,
        )

        x_bhwc = (
            x_bhwc
            + self.mlp(
                self.normalization2(
                    x_bhwc
                )
            )
        )

        return x_bhwc.masked_fill(
            ~valid_mask_bhw.unsqueeze(
                -1
            ),
            0.0,
        )


class SwinStage(
    nn.Module
):

    def __init__(
        self,
        *,
        dim: int,
        depth: int,
        num_heads: int,
        window_size: tuple[int, int],
        mlp_ratio: int,
        dropout: float,
    ) -> None:

        super().__init__()

        blocks = []

        for index in range(
            int(
                depth
            )
        ):

            shift_size = (
                (
                    0,
                    0,
                )
                if index % 2 == 0
                else C4_SWIN_SHIFT_SIZE
            )

            blocks.append(
                SwinTransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=(
                        window_size
                    ),
                    shift_size=(
                        shift_size
                    ),
                    mlp_ratio=(
                        mlp_ratio
                    ),
                    dropout=dropout,
                )
            )

        self.blocks = nn.ModuleList(
            blocks
        )

    def forward(
        self,
        x_bhwc: torch.Tensor,
        valid_mask_bhw: torch.Tensor,
    ) -> torch.Tensor:

        for block in self.blocks:

            x_bhwc = block(
                x_bhwc,
                valid_mask_bhw,
            )

        return x_bhwc


class PatchMerging(
    nn.Module
):
    """
    Standard 2 x 2 Swin merge with explicit
    spatial-mask propagation.
    """

    def __init__(
        self,
        *,
        input_dim: int,
    ) -> None:

        super().__init__()

        self.input_dim = int(
            input_dim
        )

        self.normalization = (
            nn.LayerNorm(
                4 * self.input_dim
            )
        )

        self.reduction = nn.Linear(
            4 * self.input_dim,
            2 * self.input_dim,
            bias=False,
        )

    def forward(
        self,
        x_bhwc: torch.Tensor,
        valid_mask_bhw: torch.Tensor,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:

        (
            batch_size,
            height,
            width,
            channels,
        ) = x_bhwc.shape

        if channels != self.input_dim:

            raise ValueError(
                "Patch merging received the "
                "wrong embedding width."
            )

        if valid_mask_bhw.shape != (
            batch_size,
            height,
            width,
        ):

            raise ValueError(
                "Patch-merging mask shape "
                "is invalid."
            )

        x_bhwc = x_bhwc.masked_fill(
            ~valid_mask_bhw.unsqueeze(
                -1
            ),
            0.0,
        )

        pad_height = (
            height % 2
        )

        pad_width = (
            width % 2
        )

        x_bhwc = F.pad(
            x_bhwc,
            (
                0,
                0,
                0,
                pad_width,
                0,
                pad_height,
            ),
            value=0.0,
        )

        padded_height = (
            height
            + pad_height
        )

        padded_width = (
            width
            + pad_width
        )

        padded_mask = torch.zeros(
            (
                batch_size,
                padded_height,
                padded_width,
            ),
            device=x_bhwc.device,
            dtype=torch.bool,
        )

        padded_mask[
            :,
            :height,
            :width,
        ] = valid_mask_bhw

        x0 = x_bhwc[
            :,
            0::2,
            0::2,
            :,
        ]

        x1 = x_bhwc[
            :,
            1::2,
            0::2,
            :,
        ]

        x2 = x_bhwc[
            :,
            0::2,
            1::2,
            :,
        ]

        x3 = x_bhwc[
            :,
            1::2,
            1::2,
            :,
        ]

        merged = torch.cat(
            (
                x0,
                x1,
                x2,
                x3,
            ),
            dim=-1,
        )

        merged = (
            self.normalization(
                merged
            )
        )

        merged = self.reduction(
            merged
        )

        m0 = padded_mask[
            :,
            0::2,
            0::2,
        ]

        m1 = padded_mask[
            :,
            1::2,
            0::2,
        ]

        m2 = padded_mask[
            :,
            0::2,
            1::2,
        ]

        m3 = padded_mask[
            :,
            1::2,
            1::2,
        ]

        merged_mask = (
            m0
            | m1
            | m2
            | m3
        )

        merged = merged.masked_fill(
            ~merged_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        return (
            merged,
            merged_mask,
        )


class TinySwinWindowEncoder(
    nn.Module
):
    """
    Encode each complete 36 x 38 sensor
    window into one 64-D vector.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.patch_projection = (
            nn.Conv2d(
                1,
                C4_STAGE_DIMS[0],
                kernel_size=(
                    C4_PATCH_SIZE
                ),
                stride=(
                    C4_PATCH_SIZE
                ),
                padding=0,
                bias=True,
            )
        )

        self.patch_normalization = (
            nn.LayerNorm(
                C4_STAGE_DIMS[0]
            )
        )

        self.stage1 = SwinStage(
            dim=C4_STAGE_DIMS[0],
            depth=C4_STAGE_DEPTHS[0],
            num_heads=(
                C4_STAGE_HEADS[0]
            ),
            window_size=(
                C4_SWIN_WINDOW_SIZE
            ),
            mlp_ratio=C4_MLP_RATIO,
            dropout=C4_SWIN_DROPOUT,
        )

        self.patch_merging = (
            PatchMerging(
                input_dim=(
                    C4_STAGE_DIMS[0]
                )
            )
        )

        self.stage2 = SwinStage(
            dim=C4_STAGE_DIMS[1],
            depth=C4_STAGE_DEPTHS[1],
            num_heads=(
                C4_STAGE_HEADS[1]
            ),
            window_size=(
                C4_SWIN_WINDOW_SIZE
            ),
            mlp_ratio=C4_MLP_RATIO,
            dropout=C4_SWIN_DROPOUT,
        )

        self.final_normalization = (
            nn.LayerNorm(
                C4_STAGE_DIMS[1]
            )
        )

    def forward(
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
            windows_bchw.shape[1:]
        ) != (
            1,
            C4_WINDOW_LENGTH,
            C4_FEATURE_WIDTH,
        ):

            raise ValueError(
                "Every C4 window must have "
                "shape 1 x 36 x 38."
            )

        x = self.patch_projection(
            windows_bchw
        )

        x = x.permute(
            0,
            2,
            3,
            1,
        ).contiguous()

        x = (
            self.patch_normalization(
                x
            )
        )

        valid_mask = torch.ones(
            x.shape[:3],
            device=x.device,
            dtype=torch.bool,
        )

        x = self.stage1(
            x,
            valid_mask,
        )

        (
            x,
            valid_mask,
        ) = self.patch_merging(
            x,
            valid_mask,
        )

        x = self.stage2(
            x,
            valid_mask,
        )

        x = (
            self.final_normalization(
                x
            )
        )

        x = x.masked_fill(
            ~valid_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        denominator = (
            valid_mask.sum(
                dim=(
                    1,
                    2,
                )
            )
            .clamp_min(
                1
            )
            .to(
                dtype=x.dtype
            )
        )

        return (
            x.sum(
                dim=(
                    1,
                    2,
                )
            )
            / denominator.unsqueeze(
                -1
            )
        )


class C4TinySwin(
    nn.Module
):
    """
    C4:
        valid 2D windows
        -> Tiny-Swin
        -> packed BiLSTM
        -> CTC.
    """

    def __init__(
        self,
    ) -> None:

        super().__init__()

        self.window_encoder = (
            TinySwinWindowEncoder()
        )

        self.bilstm = nn.LSTM(
            input_size=(
                C4_STAGE_DIMS[1]
            ),
            hidden_size=(
                C4_LSTM_HIDDEN_SIZE
            ),
            num_layers=1,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=True,
        )

        self.post_lstm_normalization = (
            nn.LayerNorm(
                C4_LSTM_OUTPUT_SIZE
            )
        )

        self.post_lstm_dropout = (
            nn.Dropout(
                C4_POST_LSTM_DROPOUT
            )
        )

        self.classifier = nn.Linear(
            C4_LSTM_OUTPUT_SIZE,
            C4_CTC_CLASSES,
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

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.Linear,
            ):

                nn.init.trunc_normal_(
                    module.weight,
                    std=0.02,
                )

                if module.bias is not None:

                    nn.init.zeros_(
                        module.bias
                    )

            elif isinstance(
                module,
                nn.LayerNorm,
            ):

                nn.init.ones_(
                    module.weight
                )

                nn.init.zeros_(
                    module.bias
                )

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
            self.post_lstm_normalization.weight
        )

        nn.init.zeros_(
            self.post_lstm_normalization.bias
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

        return self.window_encoder(
            windows_bchw
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
                "x_btf must be a torch.Tensor."
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
                "C4 input batch cannot be empty."
            )

        if feature_width != (
            C4_FEATURE_WIDTH
        ):

            raise ValueError(
                "C4 input feature width "
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
                "C4 input contains "
                "NaN or infinity."
            )

        window_lengths = (
            c4_window_lengths(
                raw_lengths,
                padded_time=(
                    padded_time
                ),
            )
        )

        if (
            window_lengths.shape[0]
            != batch_size
        ):

            raise ValueError(
                "raw_lengths does not match "
                "the C4 batch size."
            )

        padded_window_count = (
            1
            + (
                padded_time
                - C4_WINDOW_LENGTH
            )
            // C4_WINDOW_STRIDE
        )

        windows = c4_unfold_windows(
            x_btf
        )

        if (
            windows.shape[1]
            != padded_window_count
        ):

            raise RuntimeError(
                "Unexpected padded "
                "C4 window count."
            )

        positions = torch.arange(
            padded_window_count,
            device=x_btf.device,
        ).unsqueeze(
            0
        )

        valid_mask = (
            positions
            < window_lengths.to(
                x_btf.device
            ).unsqueeze(
                1
            )
        )

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
                    C4_STAGE_DIMS[1],
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

        sequence = (
            flat_embeddings.reshape(
                batch_size,
                padded_window_count,
                C4_STAGE_DIMS[1],
            )
        )

        packed = pack_padded_sequence(
            sequence,
            window_lengths,
            batch_first=True,
            enforce_sorted=False,
        )

        (
            packed,
            _,
        ) = self.bilstm(
            packed
        )

        (
            recurrent,
            _,
        ) = pad_packed_sequence(
            packed,
            batch_first=True,
            total_length=(
                padded_window_count
            ),
        )

        recurrent = (
            self.post_lstm_normalization(
                recurrent
            )
        )

        recurrent = (
            self.post_lstm_dropout(
                recurrent
            )
        )

        logits = self.classifier(
            recurrent
        )

        logits = logits.masked_fill(
            ~valid_mask.unsqueeze(
                -1
            ),
            0.0,
        )

        return ModelOutput(
            logits_btc=logits,
            output_lengths=(
                window_lengths
            ),
        )


def count_c4_trainable_parameters(
    model: nn.Module,
) -> int:

    return sum(
        parameter.numel()
        for parameter
        in model.parameters()
        if parameter.requires_grad
    )

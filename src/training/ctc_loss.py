from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.evaluation.ctc_feasibility import (
    minimum_ctc_timesteps,
)


CTC_BLANK_ID = 0
TARGET_PADDING_VALUE = -1


def validate_target_lengths(
    targets_padded: np.ndarray,
    target_lengths: Sequence[int],
) -> np.ndarray:

    targets = np.asarray(
        targets_padded
    )

    if targets.ndim != 2:

        raise ValueError(
            "targets_padded must be "
            "B x Umax."
        )

    lengths = np.asarray(
        target_lengths,
        dtype=np.int64,
    )

    if lengths.ndim != 1:

        raise ValueError(
            "target_lengths must be 1D."
        )

    if lengths.shape[0] != (
        targets.shape[0]
    ):

        raise ValueError(
            "target_lengths does not "
            "match batch size."
        )

    if np.any(
        lengths <= 0
    ):

        raise ValueError(
            "Every target length must "
            "be positive."
        )

    if np.any(
        lengths
        > targets.shape[1]
    ):

        raise ValueError(
            "Target length exceeds "
            "padded target width."
        )

    return lengths


def pack_ctc_targets(
    targets_padded: np.ndarray,
    target_lengths: Sequence[int],
    *,
    blank_id: int = CTC_BLANK_ID,
    padding_value: int = TARGET_PADDING_VALUE,
) -> np.ndarray:
    """
    Convert B x Umax padded target IDs into the flat
    target vector expected by PyTorch CTCLoss.

    Padding is removed strictly using target_lengths.
    """

    targets = np.asarray(
        targets_padded,
        dtype=np.int64,
    )

    lengths = validate_target_lengths(
        targets,
        target_lengths,
    )

    blank_id = int(
        blank_id
    )

    padding_value = int(
        padding_value
    )

    packed_parts = []

    for batch_index, length in enumerate(
        lengths
    ):

        valid_length = int(
            length
        )

        valid = targets[
            batch_index,
            :valid_length,
        ]

        padded_tail = targets[
            batch_index,
            valid_length:,
        ]

        if np.any(
            valid == padding_value
        ):

            raise ValueError(
                "Target padding occurs inside "
                "the valid target region."
            )

        if np.any(
            valid == blank_id
        ):

            raise ValueError(
                "CTC blank occurs inside "
                "a reference target."
            )

        if np.any(
            valid < 1
        ):

            raise ValueError(
                "Reference target contains "
                "an invalid token ID."
            )

        if padded_tail.size:

            if np.any(
                padded_tail
                != padding_value
            ):

                raise ValueError(
                    "Non-padding token occurs "
                    "after target_length."
                )

        packed_parts.append(
            valid
        )

    packed = np.concatenate(
        packed_parts,
        axis=0,
    )

    return np.asarray(
        packed,
        dtype=np.int64,
    )


def unpack_ctc_targets(
    packed_targets: Sequence[int],
    target_lengths: Sequence[int],
) -> list[tuple[int, ...]]:

    packed = np.asarray(
        packed_targets,
        dtype=np.int64,
    )

    lengths = np.asarray(
        target_lengths,
        dtype=np.int64,
    )

    if packed.ndim != 1:

        raise ValueError(
            "packed_targets must be 1D."
        )

    if lengths.ndim != 1:

        raise ValueError(
            "target_lengths must be 1D."
        )

    if np.any(
        lengths <= 0
    ):

        raise ValueError(
            "Target lengths must be positive."
        )

    if int(
        lengths.sum()
    ) != packed.shape[0]:

        raise ValueError(
            "Packed target size does not equal "
            "sum(target_lengths)."
        )

    output = []

    start = 0

    for length in lengths:

        end = (
            start
            + int(
                length
            )
        )

        output.append(
            tuple(
                int(value)
                for value in packed[
                    start:end
                ]
            )
        )

        start = end

    return output


def validate_ctc_batch_feasibility(
    targets_padded: np.ndarray,
    target_lengths: Sequence[int],
    output_lengths: Sequence[int],
    *,
    blank_id: int = CTC_BLANK_ID,
    padding_value: int = TARGET_PADDING_VALUE,
) -> np.ndarray:
    """
    Verify every sample has enough VALID model output
    positions for a CTC alignment.

    Returns per-sample CTC margins.
    """

    targets = np.asarray(
        targets_padded,
        dtype=np.int64,
    )

    target_lengths_array = (
        validate_target_lengths(
            targets,
            target_lengths,
        )
    )

    output_lengths_array = np.asarray(
        output_lengths,
        dtype=np.int64,
    )

    if output_lengths_array.ndim != 1:

        raise ValueError(
            "output_lengths must be 1D."
        )

    if output_lengths_array.shape[0] != (
        targets.shape[0]
    ):

        raise ValueError(
            "output_lengths does not "
            "match batch size."
        )

    if np.any(
        output_lengths_array <= 0
    ):

        raise ValueError(
            "Output lengths must be positive."
        )

    # Also verifies padding/blank contract.
    packed = pack_ctc_targets(
        targets,
        target_lengths_array,
        blank_id=blank_id,
        padding_value=padding_value,
    )

    unpacked = unpack_ctc_targets(
        packed,
        target_lengths_array,
    )

    margins = []

    for (
        target,
        output_length,
    ) in zip(
        unpacked,
        output_lengths_array,
        strict=True,
    ):

        minimum_required = (
            minimum_ctc_timesteps(
                target
            )
        )

        margin = (
            int(
                output_length
            )
            - minimum_required
        )

        margins.append(
            margin
        )

    margins_array = np.asarray(
        margins,
        dtype=np.int64,
    )

    if np.any(
        margins_array < 0
    ):

        bad_indices = np.flatnonzero(
            margins_array < 0
        ).tolist()

        raise ValueError(
            "CTC-infeasible batch samples "
            f"at indices {bad_indices}."
        )

    return margins_array


def torch_ctc_loss_from_logits(
    logits_btc,
    targets_padded,
    output_lengths,
    target_lengths,
    *,
    blank_id: int = CTC_BLANK_ID,
    padding_value: int = TARGET_PADDING_VALUE,
    reduction: str = "mean",
):
    """
    PyTorch CTC loss wrapper.

    PyTorch is imported lazily because the local CPU
    development environment intentionally does not require
    PyTorch. Final model/runtime execution occurs in Colab.

    logits_btc:
        torch.Tensor with shape B x T x C.

    zero_infinity is deliberately False because impossible
    alignments must not be silently hidden.
    """

    try:
        import torch
        import torch.nn.functional as F

    except ImportError as exc:

        raise RuntimeError(
            "PyTorch is required to execute "
            "the CTC loss wrapper."
        ) from exc

    if not torch.is_tensor(
        logits_btc
    ):

        raise TypeError(
            "logits_btc must be "
            "a torch.Tensor."
        )

    if logits_btc.ndim != 3:

        raise ValueError(
            "logits_btc must have "
            "shape B x T x C."
        )

    batch_size, padded_time, classes = (
        logits_btc.shape
    )

    if not (
        0 <= int(blank_id) < classes
    ):

        raise ValueError(
            "blank_id outside model "
            "output-class range."
        )

    if hasattr(
        targets_padded,
        "detach",
    ):

        targets_numpy = (
            targets_padded
            .detach()
            .cpu()
            .numpy()
        )

    else:

        targets_numpy = np.asarray(
            targets_padded
        )

    if hasattr(
        target_lengths,
        "detach",
    ):

        target_lengths_numpy = (
            target_lengths
            .detach()
            .cpu()
            .numpy()
        )

    else:

        target_lengths_numpy = (
            np.asarray(
                target_lengths
            )
        )

    if hasattr(
        output_lengths,
        "detach",
    ):

        output_lengths_numpy = (
            output_lengths
            .detach()
            .cpu()
            .numpy()
        )

    else:

        output_lengths_numpy = (
            np.asarray(
                output_lengths
            )
        )

    if len(
        output_lengths_numpy
    ) != batch_size:

        raise ValueError(
            "output_lengths does not "
            "match logits batch size."
        )

    if np.any(
        output_lengths_numpy
        > padded_time
    ):

        raise ValueError(
            "Valid model output length exceeds "
            "padded logit time dimension."
        )

    # ---------------------------------------------------------
    # Strong feasibility guard BEFORE PyTorch CTCLoss.
    # ---------------------------------------------------------

    validate_ctc_batch_feasibility(
        targets_numpy,
        target_lengths_numpy,
        output_lengths_numpy,
        blank_id=blank_id,
        padding_value=padding_value,
    )

    flat_targets_numpy = (
        pack_ctc_targets(
            targets_numpy,
            target_lengths_numpy,
            blank_id=blank_id,
            padding_value=padding_value,
        )
    )

    flat_targets = torch.as_tensor(
        flat_targets_numpy,
        dtype=torch.long,
        device=logits_btc.device,
    )

    input_lengths_tensor = (
        torch.as_tensor(
            output_lengths_numpy,
            dtype=torch.long,
            device="cpu",
        )
    )

    target_lengths_tensor = (
        torch.as_tensor(
            target_lengths_numpy,
            dtype=torch.long,
            device="cpu",
        )
    )

    log_probs_tbc = (
        logits_btc
        .log_softmax(
            dim=-1
        )
        .transpose(
            0,
            1,
        )
        .contiguous()
    )

    return F.ctc_loss(
        log_probs_tbc,
        flat_targets,
        input_lengths_tensor,
        target_lengths_tensor,
        blank=int(
            blank_id
        ),
        reduction=reduction,
        zero_infinity=False,
    )

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

import numpy as np


EXPECTED_FEATURE_WIDTH = 38
EXPECTED_CTC_CLASSES = 456
EXPECTED_BLANK_ID = 0
EXPECTED_TARGET_PADDING = -1

EXPECTED_MODEL_IDS = {
    "g1",
    "c1",
    "c2",
    "c3",
    "c4",
}

DIRECT_MODEL_IDS = {
    "c1",
    "c2",
    "c3",
}

WINDOWED_MODEL_IDS = {
    "g1",
    "c4",
}


@dataclass(
    frozen=True
)
class ModelOutput:
    """
    Shared output returned by every recognition model.

    logits_btc:
        Raw, unnormalised logits with shape B x T x C.

    output_lengths:
        Exact valid temporal output length for each sample.
    """

    logits_btc: object
    output_lengths: object


def _numpy_lengths(
    output_lengths,
) -> np.ndarray:

    if hasattr(
        output_lengths,
        "detach",
    ):
        lengths = (
            output_lengths
            .detach()
            .cpu()
            .numpy()
        )
    else:
        lengths = np.asarray(
            output_lengths
        )

    if lengths.ndim != 1:
        raise ValueError(
            "output_lengths must be one-dimensional."
        )

    if (
        np.issubdtype(
            lengths.dtype,
            np.bool_,
        )
        or not np.issubdtype(
            lengths.dtype,
            np.integer,
        )
    ):
        raise TypeError(
            "output_lengths must contain integers."
        )

    return lengths.astype(
        np.int64,
        copy=False,
    )


def _logits_are_finite(
    logits,
) -> bool:

    if hasattr(
        logits,
        "detach",
    ):

        try:
            import torch

        except ImportError as exc:
            raise RuntimeError(
                "PyTorch is required to validate "
                "a torch.Tensor model output."
            ) from exc

        if torch.is_tensor(
            logits
        ):
            return bool(
                torch.isfinite(
                    logits
                ).all().item()
            )

    array = np.asarray(
        logits
    )

    return bool(
        np.isfinite(
            array
        ).all()
    )


def validate_model_output(
    output: ModelOutput,
    *,
    expected_classes: int = EXPECTED_CTC_CLASSES,
) -> tuple[int, ...]:
    """
    Validate the common G1/C1/C2/C3/C4 output contract.

    Returns the exact valid lengths as an immutable tuple.
    """

    if not isinstance(
        output,
        ModelOutput,
    ):
        raise TypeError(
            "Model must return ModelOutput."
        )

    logits = output.logits_btc

    if not hasattr(
        logits,
        "shape",
    ):
        raise TypeError(
            "logits_btc must provide a shape."
        )

    shape = tuple(
        int(value)
        for value in logits.shape
    )

    if len(shape) != 3:
        raise ValueError(
            "logits_btc must have shape B x T x C."
        )

    batch_size, padded_time, classes = shape

    if batch_size <= 0:
        raise ValueError(
            "Model-output batch cannot be empty."
        )

    if padded_time <= 0:
        raise ValueError(
            "Model-output time cannot be empty."
        )

    if classes != int(
        expected_classes
    ):
        raise ValueError(
            "Model-output class count does not "
            f"equal {expected_classes}."
        )

    if not _logits_are_finite(
        logits
    ):
        raise ValueError(
            "Model logits contain NaN or infinity."
        )

    lengths = _numpy_lengths(
        output.output_lengths
    )

    if lengths.shape[0] != batch_size:
        raise ValueError(
            "output_lengths does not match "
            "the model-output batch size."
        )

    if np.any(
        lengths <= 0
    ):
        raise ValueError(
            "Every valid output length "
            "must be positive."
        )

    if np.any(
        lengths > padded_time
    ):
        raise ValueError(
            "A valid output length exceeds "
            "the padded output time."
        )

    return tuple(
        int(value)
        for value in lengths
    )


@dataclass(
    frozen=True
)
class ModelSpec:

    model_id: str
    scientific_role: str
    architecture_family: str
    input_adapter: str

    feature_width: int
    output_classes: int
    blank_id: int
    target_padding_value: int

    returns_raw_logits: bool
    valid_output_length_required: bool

    window_length: int | None
    window_stride: int | None

    uses_bilstm: bool
    uses_self_attention: bool
    uses_transformer_encoder: bool
    uses_tiny_swin: bool

    optimizer: str
    learning_rate: float
    weight_decay: float
    epoch_cap: int
    early_stopping_patience: int
    physical_batch_size: int
    gradient_clip_norm: float
    scheduler: str
    fixed_epoch_reference: bool

    def __post_init__(
        self,
    ) -> None:

        if self.model_id not in (
            EXPECTED_MODEL_IDS
        ):
            raise ValueError(
                "Unknown model ID."
            )

        if self.input_adapter not in {
            "direct_sequence",
            "windowed_2d",
        }:
            raise ValueError(
                "Unknown model input adapter."
            )

        if self.feature_width != (
            EXPECTED_FEATURE_WIDTH
        ):
            raise ValueError(
                "Every model must receive "
                "38 source features."
            )

        if self.output_classes != (
            EXPECTED_CTC_CLASSES
        ):
            raise ValueError(
                "Every model must produce "
                "456 CTC classes."
            )

        if self.blank_id != (
            EXPECTED_BLANK_ID
        ):
            raise ValueError(
                "CTC blank ID must be 0."
            )

        if self.target_padding_value != (
            EXPECTED_TARGET_PADDING
        ):
            raise ValueError(
                "Target padding must be -1."
            )

        if self.returns_raw_logits is not True:
            raise ValueError(
                "Models must return raw logits."
            )

        if (
            self.valid_output_length_required
            is not True
        ):
            raise ValueError(
                "Every model must return "
                "exact valid output lengths."
            )

        if self.input_adapter == (
            "direct_sequence"
        ):

            if (
                self.window_length is not None
                or self.window_stride is not None
            ):
                raise ValueError(
                    "Direct-sequence models cannot "
                    "define window geometry."
                )

        if self.input_adapter == (
            "windowed_2d"
        ):

            if self.window_length != 36:
                raise ValueError(
                    "Windowed models must use "
                    "window length 36."
                )

            if self.window_stride != 18:
                raise ValueError(
                    "Windowed models must use "
                    "the frozen stride 18."
                )

        if self.learning_rate <= 0:
            raise ValueError(
                "Learning rate must be positive."
            )

        if self.weight_decay < 0:
            raise ValueError(
                "Weight decay cannot be negative."
            )

        if self.epoch_cap <= 0:
            raise ValueError(
                "Epoch cap must be positive."
            )

        if self.early_stopping_patience <= 0:
            raise ValueError(
                "Early-stopping patience "
                "must be positive."
            )

        if self.physical_batch_size <= 0:
            raise ValueError(
                "Physical batch size must be positive."
            )

        if self.gradient_clip_norm <= 0:
            raise ValueError(
                "Gradient clipping norm "
                "must be positive."
            )

    def to_dict(
        self,
    ) -> dict:

        return asdict(
            self
        )

    @classmethod
    def from_dict(
        cls,
        payload: Mapping,
    ) -> "ModelSpec":

        required = {
            field.name
            for field in fields(
                cls
            )
        }

        missing = (
            required
            - set(
                payload
            )
        )

        extra = (
            set(
                payload
            )
            - required
        )

        if missing:
            raise ValueError(
                "Model specification is missing: "
                + ", ".join(
                    sorted(
                        missing
                    )
                )
            )

        if extra:
            raise ValueError(
                "Model specification has "
                "unknown fields: "
                + ", ".join(
                    sorted(
                        extra
                    )
                )
            )

        return cls(
            **{
                name: payload[
                    name
                ]
                for name in required
            }
        )


def load_model_catalog(
    path: Path,
) -> dict[str, ModelSpec]:

    payload = json.loads(
        Path(
            path
        ).read_text(
            encoding="utf-8-sig"
        )
    )

    if payload.get(
        "schema_version"
    ) != 1:
        raise ValueError(
            "Unsupported model-catalog schema."
        )

    raw_models = payload.get(
        "models"
    )

    if not isinstance(
        raw_models,
        list,
    ):
        raise ValueError(
            "Model catalog must contain "
            "a models list."
        )

    specifications = {}

    for raw_model in raw_models:

        specification = (
            ModelSpec.from_dict(
                raw_model
            )
        )

        if specification.model_id in (
            specifications
        ):
            raise ValueError(
                "Duplicate model ID in catalog."
            )

        specifications[
            specification.model_id
        ] = specification

    observed_ids = set(
        specifications
    )

    if observed_ids != (
        EXPECTED_MODEL_IDS
    ):
        raise ValueError(
            "Model catalog must contain "
            "exactly G1 and C1-C4."
        )

    if set(
        model_id
        for model_id, specification
        in specifications.items()
        if specification.input_adapter
        == "direct_sequence"
    ) != DIRECT_MODEL_IDS:

        raise ValueError(
            "Direct-sequence model mapping changed."
        )

    if set(
        model_id
        for model_id, specification
        in specifications.items()
        if specification.input_adapter
        == "windowed_2d"
    ) != WINDOWED_MODEL_IDS:

        raise ValueError(
            "Windowed model mapping changed."
        )

    return specifications

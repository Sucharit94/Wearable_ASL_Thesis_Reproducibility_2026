from __future__ import annotations

from torch import nn

from src.models.c1_compact import (
    C1Compact,
    C1_EXPECTED_TRAINABLE_PARAMETERS,
)
from src.models.c2_self_attention import (
    C2SelfAttention,
    C2_EXPECTED_TRAINABLE_PARAMETERS,
)
from src.models.c3_transformer import (
    C3Transformer,
    C3_EXPECTED_TRAINABLE_PARAMETERS,
)
from src.models.c4_tiny_swin import (
    C4TinySwin,
    C4_EXPECTED_TRAINABLE_PARAMETERS,
)
from src.models.g1_near_gu import (
    G1NearGu,
    G1_EXPECTED_TRAINABLE_PARAMETERS,
)


MODEL_IDS = (
    "g1",
    "c1",
    "c2",
    "c3",
    "c4",
)


EXPECTED_TRAINABLE_PARAMETERS = {
    "g1": G1_EXPECTED_TRAINABLE_PARAMETERS,
    "c1": C1_EXPECTED_TRAINABLE_PARAMETERS,
    "c2": C2_EXPECTED_TRAINABLE_PARAMETERS,
    "c3": C3_EXPECTED_TRAINABLE_PARAMETERS,
    "c4": C4_EXPECTED_TRAINABLE_PARAMETERS,
}


MODEL_BUILDERS: dict[str, type[nn.Module]] = {
    "g1": G1NearGu,
    "c1": C1Compact,
    "c2": C2SelfAttention,
    "c3": C3Transformer,
    "c4": C4TinySwin,
}


def build_model(model_id: str) -> nn.Module:
    """Build one fresh, untrained Stage-26--30 recognizer."""

    model_id = str(model_id).strip().lower()

    if model_id not in MODEL_BUILDERS:
        raise ValueError(
            "Unknown model_id. Expected one of: "
            + ", ".join(MODEL_IDS)
            + "."
        )

    return MODEL_BUILDERS[model_id]()


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def validate_model_parameter_count(
    model_id: str,
    model: nn.Module,
) -> int:
    model_id = str(model_id).strip().lower()

    if model_id not in EXPECTED_TRAINABLE_PARAMETERS:
        raise ValueError("Unknown model_id.")

    observed = count_trainable_parameters(model)
    expected = EXPECTED_TRAINABLE_PARAMETERS[model_id]

    if observed != expected:
        raise RuntimeError(
            f"{model_id} parameter count changed: "
            f"expected {expected}, observed {observed}."
        )

    return observed

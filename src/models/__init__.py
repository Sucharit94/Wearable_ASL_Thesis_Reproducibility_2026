from src.models.contracts import (
    ModelOutput,
    ModelSpec,
    load_model_catalog,
    validate_model_output,
)
from src.models.g1_near_gu import (
    G1_CNN_EMBEDDING_SIZE,
    G1_CTC_CLASSES,
    G1_EXPECTED_TRAINABLE_PARAMETERS,
    G1_FEATURE_WIDTH,
    G1_LSTM_HIDDEN_SIZE,
    G1_LSTM_OUTPUT_SIZE,
    G1_WINDOW_LENGTH,
    G1_WINDOW_STRIDE,
    G1NearGu,
    count_trainable_parameters,
    g1_window_lengths,
)


__all__ = [
    "ModelOutput",
    "ModelSpec",
    "load_model_catalog",
    "validate_model_output",
    "G1_CNN_EMBEDDING_SIZE",
    "G1_CTC_CLASSES",
    "G1_EXPECTED_TRAINABLE_PARAMETERS",
    "G1_FEATURE_WIDTH",
    "G1_LSTM_HIDDEN_SIZE",
    "G1_LSTM_OUTPUT_SIZE",
    "G1_WINDOW_LENGTH",
    "G1_WINDOW_STRIDE",
    "G1NearGu",
    "count_trainable_parameters",
    "g1_window_lengths",
]

from src.models.c1_compact import (
    C1Compact,
    c1_length_trace,
    c1_output_lengths,
    count_c1_trainable_parameters,
)

__all__ = [
    *__all__,
    "C1Compact",
    "c1_length_trace",
    "c1_output_lengths",
    "count_c1_trainable_parameters",
]

from src.models.c2_self_attention import (
    C2SelfAttention,
    c2_output_lengths,
    count_c2_trainable_parameters,
)

__all__ = [
    *__all__,
    "C2SelfAttention",
    "c2_output_lengths",
    "count_c2_trainable_parameters",
]

from src.models.c3_transformer import (
    C3Transformer,
    SinusoidalPositionalEncoding,
    c3_key_padding_mask,
    c3_output_lengths,
    count_c3_trainable_parameters,
)

__all__ = [
    *__all__,
    "C3Transformer",
    "SinusoidalPositionalEncoding",
    "c3_key_padding_mask",
    "c3_output_lengths",
    "count_c3_trainable_parameters",
]

from src.models.c4_tiny_swin import (
    C4TinySwin,
    TinySwinWindowEncoder,
    c4_unfold_windows,
    c4_window_lengths,
    count_c4_trainable_parameters,
    padded_grid_size,
)

__all__ = [
    *__all__,
    "C4TinySwin",
    "TinySwinWindowEncoder",
    "c4_unfold_windows",
    "c4_window_lengths",
    "count_c4_trainable_parameters",
    "padded_grid_size",
]

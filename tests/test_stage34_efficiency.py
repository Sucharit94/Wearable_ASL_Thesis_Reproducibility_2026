import json

import pytest
import torch

from src.evaluation.stage34_efficiency import (
    MODEL_ORDER,
    WORKER_PREFIX,
    _median_and_iqr,
    benchmark_model,
    byte_count,
    parameter_and_buffer_bytes,
    serialized_state_dict_bytes,
)
from src.models.factory import EXPECTED_TRAINABLE_PARAMETERS, build_model, count_trainable_parameters


def test_nested_tensor_byte_count() -> None:
    value = {
        "a": torch.zeros(2, 3, dtype=torch.float32),
        "b": [torch.zeros(4, dtype=torch.int64)],
    }
    assert byte_count(value) == 24 + 32


def test_median_and_inclusive_iqr() -> None:
    median, first, third = _median_and_iqr([1.0, 2.0, 3.0, 4.0])
    assert median == pytest.approx(2.5)
    assert first == pytest.approx(1.75)
    assert third == pytest.approx(3.25)


@pytest.mark.parametrize("model_id", MODEL_ORDER)
def test_frozen_parameter_counts_and_sizes(model_id: str) -> None:
    model = build_model(model_id)
    assert count_trainable_parameters(model) == EXPECTED_TRAINABLE_PARAMETERS[model_id]
    parameter_bytes, buffer_bytes = parameter_and_buffer_bytes(model)
    assert parameter_bytes > 0
    assert buffer_bytes >= 0
    assert serialized_state_dict_bytes(model) >= parameter_bytes


def test_small_c1_benchmark_has_no_training_or_gradients() -> None:
    result = benchmark_model(
        "c1",
        warmup_iterations=0,
        timed_iterations={1: 1, 16: 1},
    )
    assert result["model_id"] == "c1"
    assert result["training_started"] is False
    assert result["gradients_created"] is False
    assert result["model_in_eval_mode"] is True
    assert result["batch1"]["latency_median_ms"] > 0.0
    assert result["batch16"]["throughput_samples_per_second"] > 0.0


def test_worker_prefix_payload_is_json_safe() -> None:
    payload = {"model_id": "c4", "training_started": False}
    line = WORKER_PREFIX + json.dumps(payload)
    assert json.loads(line[len(WORKER_PREFIX) :]) == payload

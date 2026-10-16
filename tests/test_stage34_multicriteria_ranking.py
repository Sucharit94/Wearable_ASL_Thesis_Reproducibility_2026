from pathlib import Path

import pytest

from src.evaluation.stage34_ranking import (
    CRITERIA_DIRECTIONS,
    WEIGHT_SCENARIOS,
    lightweight_eligible,
    normalize_utility,
    pareto_frontier,
    quality_eligible,
    statistical_wer_tie_groups,
    weighted_scores,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_all_weight_scenarios_are_complete_and_sum_to_one() -> None:
    for weights in WEIGHT_SCENARIOS.values():
        assert set(weights) == set(CRITERIA_DIRECTIONS)
        assert sum(weights.values()) == pytest.approx(1.0)


def test_minimize_and_maximize_utility_directions() -> None:
    assert normalize_utility([1.0, 2.0, 3.0], direction="minimize") == [1.0, 0.5, 0.0]
    assert normalize_utility([1.0, 2.0, 3.0], direction="maximize") == [0.0, 0.5, 1.0]


def test_equal_values_receive_equal_full_utility() -> None:
    assert normalize_utility([5.0, 5.0], direction="minimize") == [1.0, 1.0]


def _row(model: str, wer: float, accuracy: float, parameters: int, latency: float, throughput: float, memory: float) -> dict:
    return {
        "model_id": model,
        "pooled_wer": wer,
        "exact_sentence_accuracy": accuracy,
        "trainable_parameters": parameters,
        "batch1_latency_median_ms": latency,
        "batch16_throughput_samples_per_second": throughput,
        "incremental_peak_rss_mib": memory,
    }


def test_declared_quality_and_lightweight_thresholds() -> None:
    accepted = _row("good", 0.10, 0.75, 1_000_000, 1.0, 1.0, 1.0)
    rejected = _row("bad", 0.11, 0.74, 1_000_001, 1.0, 1.0, 1.0)
    assert quality_eligible(accepted) is True
    assert lightweight_eligible(accepted) is True
    assert quality_eligible(rejected) is False
    assert lightweight_eligible(rejected) is False


def test_weighted_score_rewards_dominant_model() -> None:
    best = _row("best", 0.01, 0.99, 100, 1.0, 100.0, 10.0)
    worst = _row("worst", 0.09, 0.76, 900_000, 9.0, 10.0, 90.0)
    scores = weighted_scores([best, worst], WEIGHT_SCENARIOS["balanced"])
    assert scores["best"] == pytest.approx(1.0)
    assert scores["worst"] == pytest.approx(0.0)


def test_pareto_frontier_excludes_dominated_model() -> None:
    best = _row("best", 0.01, 0.99, 100, 1.0, 100.0, 10.0)
    tradeoff = _row("tradeoff", 0.02, 0.98, 50, 2.0, 80.0, 12.0)
    dominated = _row("dominated", 0.03, 0.95, 200, 3.0, 70.0, 20.0)
    assert pareto_frontier([best, tradeoff, dominated]) == {"best", "tradeoff"}


def test_invalid_weight_sum_is_rejected() -> None:
    row = _row("one", 0.01, 0.99, 100, 1.0, 100.0, 10.0)
    weights = dict(WEIGHT_SCENARIOS["balanced"])
    weights["pooled_wer"] += 0.1
    with pytest.raises(ValueError, match="sum to one"):
        weighted_scores([row], weights)


def test_missing_peak_rss_is_omitted_and_weights_are_redistributed() -> None:
    best = _row("best", 0.01, 0.99, 100, 1.0, 100.0, 10.0)
    worst = _row("worst", 0.09, 0.76, 900_000, 9.0, 10.0, 90.0)
    best["incremental_peak_rss_mib"] = None
    worst["incremental_peak_rss_mib"] = None
    scores = weighted_scores([best, worst], WEIGHT_SCENARIOS["balanced"])
    assert scores["best"] == pytest.approx(1.0)
    assert scores["worst"] == pytest.approx(0.0)
    assert pareto_frontier([best, worst]) == {"best"}


def test_stage34c_unresolved_wer_pair_is_loaded_not_hardcoded() -> None:
    assert statistical_wer_tie_groups(PROJECT_ROOT) == [["c2", "c4"]]

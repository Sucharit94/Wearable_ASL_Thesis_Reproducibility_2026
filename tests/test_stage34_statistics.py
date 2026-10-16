import numpy as np
import pytest

from src.evaluation.stage34_statistics import (
    bootstrap_two_sided_pvalue,
    exact_mcnemar_pvalue,
    holm_adjust,
    percentile_interval,
)


def test_holm_adjustment_known_values():
    adjusted = holm_adjust([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_adjustment_preserves_input_order():
    adjusted = holm_adjust([0.5, 0.001, 0.02, 1.0])
    assert adjusted[1] <= adjusted[2] <= adjusted[0] <= adjusted[3]


def test_exact_mcnemar_no_discordance_is_one():
    assert exact_mcnemar_pvalue(0, 0) == 1.0


def test_exact_mcnemar_is_symmetric():
    assert exact_mcnemar_pvalue(2, 12) == pytest.approx(
        exact_mcnemar_pvalue(12, 2)
    )


def test_exact_mcnemar_balanced_discordance_is_one():
    assert exact_mcnemar_pvalue(5, 5) == pytest.approx(1.0)


def test_percentile_interval():
    low, high = percentile_interval(np.arange(100, dtype=np.float64))
    assert low == pytest.approx(2.475)
    assert high == pytest.approx(96.525)


def test_bootstrap_pvalue_has_plus_one_protection():
    values = np.ones(999)
    assert bootstrap_two_sided_pvalue(values) == pytest.approx(0.002)

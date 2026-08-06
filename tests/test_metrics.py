"""Every metric on a hand-checked series, and every degenerate case by name.

The degenerate cases carry most of the weight here. A summary table is read once
and believed afterwards, so a metric that returns NaN when the position never
moved does not get noticed as an error -- it gets noticed as a gap, if at all.
Each of the awkward inputs below therefore has an asserted, stated value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from council.backtest.engine import buy_and_hold, run_backtest
from council.backtest.metrics import (
    TRADING_PERIODS_PER_YEAR,
    PerformanceMetrics,
    evaluate,
    evaluate_returns,
)

# mean 0.005, sample standard deviation 0.012909944, downside deviation 0.005.
HAND_CHECKED = np.array([0.01, -0.01, 0.02, 0.00])


def summarise(
    returns: np.ndarray,
    *,
    position: np.ndarray | None = None,
    turnover: float = 0.0,
    periods_per_year: float = TRADING_PERIODS_PER_YEAR,
) -> PerformanceMetrics:
    held = np.ones_like(returns) if position is None else position
    return evaluate_returns(
        net_return=returns,
        position=held,
        turnover=turnover,
        periods_per_year=periods_per_year,
    )


# -- the ordinary case -------------------------------------------------------


def test_total_return_compounds_rather_than_summing() -> None:
    metrics = summarise(np.array([0.5, 0.5]))

    assert metrics.total_return == pytest.approx(1.25)


def test_cagr_annualises_a_two_year_run_to_its_compound_rate() -> None:
    # Two years of identical sessions that together grow capital by 21%.
    periods = int(TRADING_PERIODS_PER_YEAR) * 2
    per_period = 1.21 ** (1.0 / periods) - 1.0

    metrics = summarise(np.full(periods, per_period))

    assert metrics.total_return == pytest.approx(0.21)
    assert metrics.cagr == pytest.approx(0.10)


def test_sharpe_is_the_hand_computed_ratio_scaled_by_the_root_of_the_year() -> None:
    metrics = summarise(HAND_CHECKED)

    # 0.005 / 0.012909944 * sqrt(252)
    assert metrics.sharpe == pytest.approx(6.1481705, rel=1e-6)


def test_sortino_divides_by_the_downside_only_and_so_exceeds_sharpe_here() -> None:
    metrics = summarise(HAND_CHECKED)

    # The single negative period gives sqrt(0.01^2 / 4) = 0.005 of downside.
    assert metrics.sortino == pytest.approx(math.sqrt(252.0), rel=1e-9)
    assert metrics.sortino > metrics.sharpe


def test_max_drawdown_is_reported_as_a_positive_magnitude() -> None:
    metrics = summarise(np.array([0.2, -0.5, 0.1]))

    # 1.2 -> 0.6 is a fall of half.
    assert metrics.max_drawdown == pytest.approx(0.5)


def test_max_drawdown_counts_a_fall_in_the_very_first_period() -> None:
    # The peak the trough is measured against is the starting capital. Without
    # that, a strategy that only ever loses reports no drawdown whatsoever.
    metrics = summarise(np.array([-0.3, 0.1]))

    assert metrics.max_drawdown == pytest.approx(0.3)


def test_max_drawdown_of_a_curve_that_only_rises_is_zero() -> None:
    metrics = summarise(np.array([0.01, 0.02, 0.03]))

    assert metrics.max_drawdown == 0.0


def test_hit_rate_counts_only_periods_that_moved() -> None:
    metrics = summarise(np.array([0.01, -0.01, 0.02, 0.0]))

    assert metrics.hit_rate == pytest.approx(2.0 / 3.0)


def test_turnover_per_period_divides_the_total_by_the_periods() -> None:
    metrics = summarise(np.zeros(4), turnover=6.0)

    assert metrics.turnover == 6.0
    assert metrics.turnover_per_period == pytest.approx(1.5)


def test_time_in_market_counts_ticker_periods_not_calendar_periods() -> None:
    # Two tickers, four periods, one of the eight slots invested.
    position = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])

    metrics = summarise(np.zeros(4), position=position)

    assert metrics.time_in_market == pytest.approx(0.125)


# -- degenerate cases --------------------------------------------------------


def test_an_all_flat_strategy_reports_zeros_and_never_a_nan() -> None:
    metrics = summarise(np.zeros(10), position=np.zeros(10))

    assert metrics.total_return == 0.0
    assert metrics.cagr == 0.0
    assert metrics.sharpe == 0.0
    assert metrics.sortino == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.hit_rate == 0.0
    assert metrics.time_in_market == 0.0
    assert not any(math.isnan(value) for value in _numbers(metrics))


def test_a_positive_return_with_no_variation_is_an_infinite_sharpe_not_a_nan() -> None:
    # Dividing by a zero spread has no finite answer. An infinity in the table is
    # unmissable; a NaN prints as an empty cell and the row stops being read.
    #
    # This series is also the reason the guard is a noise floor rather than a
    # comparison against zero: numpy's standard deviation of twenty identical
    # values is about 1e-19, not 0.0, and a strict test would let that through
    # as a Sharpe of 7e16.
    metrics = summarise(np.full(20, 0.001))

    assert metrics.sharpe == math.inf
    assert metrics.sortino == math.inf


def test_a_negative_return_with_no_variation_is_a_negative_infinity() -> None:
    metrics = summarise(np.full(20, -0.001))

    assert metrics.sharpe == -math.inf
    # Every period is a downside period, so Sortino remains finite here.
    assert metrics.sortino == pytest.approx(-math.sqrt(252.0))


def test_a_single_period_has_no_estimable_spread_and_reports_an_infinite_ratio() -> None:
    # The sample standard deviation of one observation is undefined, and numpy
    # returns NaN for it. Falling through to the no-dispersion branch is the
    # difference between an obviously unusable infinity and a poisoned column.
    metrics = summarise(np.array([0.01]))

    assert metrics.periods == 1
    assert metrics.sharpe == math.inf
    assert metrics.total_return == pytest.approx(0.01)
    assert metrics.cagr == pytest.approx(1.01**252 - 1.0)


def test_a_single_flat_period_is_all_zeros() -> None:
    metrics = summarise(np.array([0.0]), position=np.array([0.0]))

    assert metrics.sharpe == 0.0
    assert metrics.sortino == 0.0
    assert metrics.total_return == 0.0
    assert metrics.hit_rate == 0.0


def test_losing_more_than_the_capital_reports_total_loss_rather_than_a_complex_root() -> None:
    metrics = summarise(np.array([-1.5, 0.2]))

    assert metrics.cagr == -1.0
    assert metrics.max_drawdown == 1.0


def test_exactly_wiping_out_reports_total_loss() -> None:
    metrics = summarise(np.array([-1.0, 0.0]))

    assert metrics.total_return == pytest.approx(-1.0)
    assert metrics.cagr == -1.0
    assert metrics.max_drawdown == 1.0


def test_a_wipeout_over_an_odd_number_of_periods_still_reports_total_loss() -> None:
    # Five periods, so the annualising exponent 252/5 is fractional. Raising a
    # negative growth multiple to a fractional power returns a Python complex,
    # and float() of a complex raises TypeError -- so the non-positive guard is
    # the only thing between this input and a crash. An even period count hides
    # that: the exponent is then an integer and the power happens to come back
    # real, which is why the two-period wipeouts above cannot pin this.
    metrics = summarise(np.array([-1.5, 0.2, 0.1, 0.0, 0.0]))

    assert metrics.cagr == -1.0
    assert metrics.max_drawdown == 1.0


# -- refusals ----------------------------------------------------------------


def test_an_empty_series_cannot_be_summarised_and_raises() -> None:
    with pytest.raises(ValueError, match="at least one period"):
        summarise(np.array([]))


def test_a_non_finite_return_raises_rather_than_propagating() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        summarise(np.array([0.01, np.nan]))


def test_a_position_of_the_wrong_length_raises() -> None:
    with pytest.raises(ValueError, match="exactly the periods"):
        summarise(np.zeros(4), position=np.zeros(3))


def test_a_scalar_position_raises_rather_than_failing_on_an_index() -> None:
    # A bare float reaches the check as this: zero-dimensional, so `shape[-1]`
    # raises IndexError -- the wrong exception, from the wrong place, for a caller
    # who was promised the argument would be validated.
    with pytest.raises(ValueError, match="exactly the periods"):
        summarise(np.zeros(1), position=np.asarray(1.0))


def test_a_position_of_the_wrong_rank_raises_rather_than_summarising_it_anyway() -> None:
    # The last axis agrees, so only the rank check catches this. Left through, it
    # divides the invested count by a denominator three times too large and hands
    # back a time in market that looks entirely reasonable.
    with pytest.raises(ValueError, match="exactly the periods"):
        summarise(np.zeros(4), position=np.zeros((2, 3, 4)))


def test_a_non_finite_position_raises_rather_than_counting_as_invested() -> None:
    # np.count_nonzero treats NaN as non-zero, so an unchecked NaN reports a
    # strategy that was never in the market as fully invested throughout.
    with pytest.raises(ValueError, match="position contains a non-finite"):
        summarise(np.zeros(4), position=np.full(4, np.nan))


def test_a_non_positive_year_length_raises() -> None:
    with pytest.raises(ValueError, match="periods_per_year"):
        summarise(np.zeros(4), periods_per_year=0.0)


# -- against a real backtest -------------------------------------------------


def test_evaluate_summarises_a_finished_backtest() -> None:
    calendar = pd.bdate_range("2022-01-03", periods=5)
    opens = pd.DataFrame(
        {"UP": [100.0, 110.0, 121.0, 133.1, 146.41], "FLAT": [50.0] * 5}, index=calendar
    )

    metrics = evaluate(buy_and_hold(opens))

    # Fully invested from period 1 in one ticker that compounds at 10% and one
    # that does nothing, so the basket earns 5% over each of three periods.
    assert metrics.periods == 4
    assert metrics.total_return == pytest.approx(1.05**3 - 1.0)
    assert metrics.time_in_market == pytest.approx(6.0 / 8.0)
    assert metrics.hit_rate == pytest.approx(1.0)
    assert metrics.max_drawdown == 0.0


def test_evaluate_reports_the_turnover_the_engine_charged_for() -> None:
    calendar = pd.bdate_range("2022-01-03", periods=6)
    opens = pd.DataFrame({"ONE": [100.0] * 6}, index=calendar)
    targets = pd.DataFrame({"ONE": [1.0, 1.0, -1.0, -1.0, -1.0, -1.0]}, index=calendar)

    result = run_backtest(
        targets=targets, opens=opens, cost_bps=0.0, rebalance_threshold=0.05
    )
    metrics = evaluate(result)

    # In at 1.0, then a flip of 2.0, over five periods.
    assert metrics.turnover == pytest.approx(3.0)
    assert metrics.turnover_per_period == pytest.approx(0.6)


def _numbers(metrics: PerformanceMetrics) -> list[float]:
    return [
        metrics.total_return,
        metrics.cagr,
        metrics.sharpe,
        metrics.sortino,
        metrics.max_drawdown,
        metrics.turnover,
        metrics.turnover_per_period,
        metrics.hit_rate,
        metrics.time_in_market,
    ]

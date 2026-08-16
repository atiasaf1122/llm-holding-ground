"""Attacks on the fill rule, the cost model and the hold rule.

Every result in this project is expressed through the backtest, and its failure
mode is not an exception -- it is a beautiful equity curve. So the prices here
are constructed so that a wrong rule is *visible*: the spike is enormous, the
two candidate fill prices differ by tens of percent, and the costs are round
numbers that can be added up on paper. A test that merely ran the engine and
checked the answer was finite would pass under lookahead.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest

from council.backtest.engine import (
    BPS,
    TickerResult,
    buy_and_hold,
    run_backtest,
    run_ticker,
)

COST_BPS = 10.0
THRESHOLD = 0.05


def sessions(count: int) -> pd.DatetimeIndex:
    """A run of business days, so no test depends on a market holiday."""
    return pd.bdate_range("2022-01-03", periods=count)


def opens_of(prices: list[float]) -> pd.Series:
    return pd.Series(prices, index=sessions(len(prices)), dtype=float)


def targets_on(calendar: pd.DatetimeIndex, values: dict[int, float]) -> pd.Series:
    """Decisions dated by position in ``calendar``; every other date is absent."""
    return pd.Series(
        {calendar[index]: value for index, value in sorted(values.items())}, dtype=float
    )


def backtest(
    opens: pd.Series[float], targets: pd.Series[float], *, threshold: float = THRESHOLD
) -> TickerResult:
    return run_ticker(
        ticker="TEST",
        targets=targets,
        opens=opens,
        cost_bps=COST_BPS,
        rebalance_threshold=threshold,
    )


# -- lookahead ---------------------------------------------------------------


def test_a_decision_made_on_the_spike_day_cannot_capture_the_spike() -> None:
    # Arrange: the open doubles between session 4 and session 5. Period 4 is the
    # spike; capturing it requires having been positioned before session 4 ended.
    opens = opens_of([100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
    calendar = pd.DatetimeIndex(opens.index)

    # Act: the agent turns fully long on the day the spike happens, and stays.
    late = backtest(opens, targets_on(calendar, {4: 1.0, 5: 1.0, 6: 1.0}))

    # Assert: nothing of the spike is earned, and the late entry pays to arrive.
    assert late.gross_return[4] == 0.0
    assert not late.gross_return.any()
    assert late.equity[-1] == pytest.approx(1.0 - 1.0 * COST_BPS * BPS)


def test_a_decision_made_the_session_before_the_spike_does_capture_it() -> None:
    # The mirror of the test above: pinning only the failure would also pass for
    # an engine that never earns anything at all.
    opens = opens_of([100.0, 100.0, 100.0, 100.0, 100.0, 200.0, 200.0, 200.0])
    calendar = pd.DatetimeIndex(opens.index)

    early = backtest(opens, targets_on(calendar, {3: 1.0}))

    assert early.position[4] == 1.0
    assert early.gross_return[4] == pytest.approx(1.0)


def test_a_decision_is_filled_at_the_next_open_and_never_at_the_same_one() -> None:
    # Arrange: the two candidate fill prices disagree violently, and the periods
    # they price have opposite signs, so filling on the wrong bar turns a 40%
    # loss into a 50% gain rather than shading the answer.
    opens = opens_of([100.0, 150.0, 90.0, 90.0])
    calendar = pd.DatetimeIndex(opens.index)

    result = backtest(opens, targets_on(calendar, {0: 1.0}))

    assert result.position[0] == 0.0
    assert result.gross_return[0] == 0.0
    assert result.position[1] == 1.0
    assert result.gross_return[1] == pytest.approx(-0.4)


def test_the_first_period_is_never_traded_because_no_decision_precedes_it() -> None:
    opens = opens_of([100.0, 110.0, 120.0])
    calendar = pd.DatetimeIndex(opens.index)

    result = backtest(opens, targets_on(calendar, {0: 1.0, 1: 1.0, 2: 1.0}))

    assert result.position[0] == 0.0
    assert result.cost[0] == 0.0


# -- costs -------------------------------------------------------------------


def test_a_two_trade_sequence_costs_the_hand_computed_amount() -> None:
    # Arrange: flat prices, so nothing but cost can move the equity curve.
    opens = opens_of([100.0] * 6)
    calendar = pd.DatetimeIndex(opens.index)
    # Session 0 asks for +0.50, session 1 repeats it, session 2 asks for -0.25.
    targets = targets_on(calendar, {0: 0.50, 1: 0.50, 2: -0.25})

    result = backtest(opens, targets)

    # Trade one: 0.00 -> 0.50, so 0.50 of notional at 10bps = 0.000500.
    # Repeat:    0.50 -> 0.50, no move, no charge.
    # Trade two: 0.50 -> -0.25, so 0.75 of notional at 10bps = 0.000750.
    # Total                                                  = 0.001250.
    assert result.cost[1] == pytest.approx(0.50 * COST_BPS * BPS)
    assert result.cost[2] == 0.0
    assert result.cost[3] == pytest.approx(0.75 * COST_BPS * BPS)
    assert float(result.cost.sum()) == pytest.approx(0.001250)
    assert result.equity[-1] == pytest.approx(1.0 - 0.001250)


def test_costs_are_charged_on_the_traded_difference_not_on_the_held_position() -> None:
    opens = opens_of([100.0] * 5)
    calendar = pd.DatetimeIndex(opens.index)

    result = backtest(opens, targets_on(calendar, {0: 1.0, 1: 1.0, 2: 1.0}))

    # One entry, then three periods of holding a full position for nothing.
    assert float(result.cost.sum()) == pytest.approx(1.0 * COST_BPS * BPS)


# -- the rebalance threshold -------------------------------------------------


def test_the_threshold_suppresses_a_small_move_and_permits_a_large_one() -> None:
    opens = opens_of([100.0] * 6)
    calendar = pd.DatetimeIndex(opens.index)
    # 0.50, then a 0.03 nudge that should be ignored, then a 0.10 move from the
    # held 0.50 that should not be.
    targets = targets_on(calendar, {0: 0.50, 1: 0.53, 2: 0.60})

    result = backtest(opens, targets)

    assert result.position[1] == pytest.approx(0.50)
    assert result.position[2] == pytest.approx(0.50)
    assert result.cost[2] == 0.0
    assert result.position[3] == pytest.approx(0.60)


def test_a_move_exactly_equal_to_the_threshold_is_suppressed() -> None:
    # The comparison is strict, and which side of it the boundary falls on is a
    # decision rather than an accident, so it is pinned.
    opens = opens_of([100.0] * 4)
    calendar = pd.DatetimeIndex(opens.index)

    result = backtest(opens, targets_on(calendar, {0: THRESHOLD}))

    assert result.position[1] == 0.0


def test_no_one_notch_step_along_the_exposure_grid_ever_trades() -> None:
    # Exposures arrive on a 0.05 grid and the default bar is 0.05, so a one-notch
    # step is a move *of* the bar and the documented rule -- "no trade unless the
    # target moves further than this" -- suppresses all forty of them. A bare
    # ``>`` decides it by representation error instead: abs(0.25 - 0.30) is
    # 0.04999999999999999 and holds, while sixteen of the forty adjacent pairs
    # come out strictly above 0.05 and trade.
    grid = [round(-1.0 + 0.05 * step, 10) for step in range(41)]
    adjacent = list(pairwise(grid))
    assert sum(1 for first, second in adjacent if abs(first - second) > 0.05) == 16

    opens = opens_of([100.0] * 5)
    calendar = pd.DatetimeIndex(opens.index)

    # An entry from the far end of the range first, so the book sits on ``held``
    # exactly -- including the notches a flat book cannot reach in one move.
    def entry(held: float) -> float:
        return -1.0 if held > 0.0 else 1.0

    traded = [
        (held, asked)
        for first, second in adjacent
        for held, asked in ((first, second), (second, first))
        if backtest(opens, targets_on(calendar, {0: entry(held), 1: held, 2: asked})).position[3]
        != held
    ]

    assert traded == []


def test_a_zero_threshold_lets_every_revision_through() -> None:
    opens = opens_of([100.0] * 5)
    calendar = pd.DatetimeIndex(opens.index)

    result = backtest(opens, targets_on(calendar, {0: 0.01, 1: 0.02}), threshold=0.0)

    assert result.position[1] == pytest.approx(0.01)
    assert result.position[2] == pytest.approx(0.02)


# -- missing decisions -------------------------------------------------------


def test_a_gap_in_the_decisions_holds_the_previous_position_rather_than_flattening() -> None:
    opens = opens_of([100.0] * 8)
    calendar = pd.DatetimeIndex(opens.index)
    # A view on session 0, silence for three sessions, then a decision to flatten.
    targets = targets_on(calendar, {0: 0.8, 4: 0.0})

    result = backtest(opens, targets)

    assert result.position[1:5].tolist() == pytest.approx([0.8, 0.8, 0.8, 0.8])
    assert result.position[5] == 0.0
    # Silence is not free of consequence but it is free of charge.
    assert float(result.cost[2:5].sum()) == 0.0


def test_dates_before_the_first_decision_are_flat_rather_than_extrapolated() -> None:
    opens = opens_of([100.0, 110.0, 120.0, 130.0])
    calendar = pd.DatetimeIndex(opens.index)

    result = backtest(opens, targets_on(calendar, {2: 1.0}))

    assert result.position[:2].tolist() == [0.0, 0.0]


def test_a_ticker_with_no_decisions_at_all_stays_flat_and_costs_nothing() -> None:
    opens = opens_of([100.0, 150.0, 90.0])

    result = backtest(opens, pd.Series(dtype=float))

    assert not result.position.any()
    assert result.equity[-1] == pytest.approx(1.0)


# -- calendar handling -------------------------------------------------------


def test_a_decision_dated_on_a_non_trading_day_never_attaches_to_a_neighbour() -> None:
    # Arrange: a calendar with Wednesday missing, and a decision dated Wednesday.
    calendar = pd.DatetimeIndex(["2022-01-03", "2022-01-04", "2022-01-06", "2022-01-07"])
    opens = pd.Series([100.0, 100.0, 100.0, 100.0], index=calendar)
    targets = pd.Series({pd.Timestamp("2022-01-05"): 1.0}, dtype=float)

    result = backtest(opens, targets)

    # Attaching it to Tuesday would be lookahead; attaching it to Thursday would
    # be a decision taken on information from a session that had not happened.
    assert not result.position.any()


def test_opens_supplied_out_of_order_are_sorted_before_anything_is_computed() -> None:
    calendar = sessions(3)
    shuffled = pd.Series([120.0, 100.0, 110.0], index=[calendar[2], calendar[0], calendar[1]])

    result = backtest(shuffled, targets_on(calendar, {0: 1.0}))

    assert result.period_return[0] == pytest.approx(0.1)
    assert result.period_return[1] == pytest.approx(120.0 / 110.0 - 1.0)


def test_a_single_session_cannot_form_a_period_and_raises() -> None:
    with pytest.raises(ValueError, match="at least two sessions"):
        backtest(opens_of([100.0]), pd.Series(dtype=float))


def test_a_ticker_missing_a_session_raises_rather_than_returning_a_blank_curve() -> None:
    # Arrange: this is what a genuine calendar mismatch looks like by the time it
    # reaches the engine -- one frame, one unioned index, and NaN in the gaps.
    # There is nothing left for a date-index comparison inside `run_backtest` to
    # catch, which is why it does not have one.
    left = pd.Series([100.0, 101.0, 102.0], index=sessions(3))
    right = pd.Series([50.0, 51.0], index=sessions(3)[[0, 2]])
    opens = pd.DataFrame({"LEFT": left, "RIGHT": right})
    targets = pd.DataFrame(1.0, index=opens.index, columns=opens.columns)

    with pytest.raises(ValueError, match="RIGHT: opens must be finite and positive"):
        run_backtest(targets=targets, opens=opens, cost_bps=0.0, rebalance_threshold=THRESHOLD)


def test_a_non_positive_open_raises_rather_than_producing_an_infinite_return() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        backtest(opens_of([100.0, 0.0, 100.0]), pd.Series(dtype=float))


def test_an_empty_universe_raises() -> None:
    with pytest.raises(ValueError, match="no tickers"):
        run_backtest(
            targets=pd.DataFrame(), opens=pd.DataFrame(), cost_bps=0.0, rebalance_threshold=0.0
        )


# -- the basket and the baseline ---------------------------------------------


def test_the_basket_return_is_the_equal_weight_mean_of_its_tickers() -> None:
    calendar = sessions(4)
    opens = pd.DataFrame(
        {"UP": [100.0, 110.0, 121.0, 133.1], "FLAT": [50.0, 50.0, 50.0, 50.0]}, index=calendar
    )
    targets = pd.DataFrame(1.0, index=calendar, columns=opens.columns)

    result = run_backtest(targets=targets, opens=opens, cost_bps=0.0, rebalance_threshold=0.0)

    assert result.net_return[1] == pytest.approx(0.1 / 2.0)
    assert result.turnover == pytest.approx(1.0)


def test_buy_and_hold_compounds_the_underlyings_open_to_open_returns_exactly() -> None:
    # Arrange: an irregular path, so an engine that happened to compound the
    # wrong pair of opens could not land on the same number by symmetry.
    prices = [100.0, 137.0, 91.0, 118.5, 104.25, 209.0]
    calendar = sessions(len(prices))
    opens = pd.DataFrame({"ONE": prices}, index=calendar)

    result = buy_and_hold(opens)
    invested = result.net_return[1:]

    assert invested == pytest.approx(
        [prices[i + 1] / prices[i] - 1.0 for i in range(1, len(prices) - 1)]
    )
    # Compounding those returns reproduces the ratio of the two opens exactly.
    assert float(np.prod(1.0 + invested)) == pytest.approx(prices[-1] / prices[1], rel=1e-12)
    assert result.turnover == pytest.approx(1.0)
    # The basket curve is what every reported figure is drawn from, so it is
    # pinned here rather than inferred from the per-ticker one.
    assert result.equity[-1] == pytest.approx(prices[-1] / prices[1], rel=1e-12)


def test_the_basket_frame_carries_the_curve_indexed_by_the_holding_periods() -> None:
    calendar = sessions(4)
    opens = pd.DataFrame(
        {"UP": [100.0, 110.0, 121.0, 133.1], "FLAT": [50.0, 50.0, 50.0, 50.0]}, index=calendar
    )

    result = buy_and_hold(opens)
    frame = result.to_frame()

    # One row per holding period, not per session: the last session opens none.
    assert list(frame.columns) == ["net_return", "equity"]
    pd.testing.assert_index_equal(pd.DatetimeIndex(frame.index), calendar[:-1])
    assert frame["net_return"].to_numpy() == pytest.approx(result.net_return)
    assert frame["equity"].iloc[-1] == pytest.approx(1.05**2)


def test_buy_and_hold_sits_out_the_one_period_no_strategy_could_have_traded() -> None:
    # The first period opens before any decision could have been made, so the
    # baseline is flat across it too. Investing the baseline there and not the
    # committee would hand the baseline a free session in every comparison.
    prices = [100.0, 137.0, 91.0]
    opens = pd.DataFrame({"ONE": prices}, index=sessions(len(prices)))

    result = buy_and_hold(opens)

    assert result.net_return[0] == 0.0


def test_turnover_counts_every_change_of_position_including_a_reversal() -> None:
    opens = opens_of([100.0] * 5)
    calendar = pd.DatetimeIndex(opens.index)

    result = backtest(opens, targets_on(calendar, {0: 0.6, 1: -0.6}))

    assert result.turnover == pytest.approx(0.6 + 1.2)


def test_turnover_measures_the_ramp_into_an_opening_position_from_a_flat_book() -> None:
    # Built by hand because the engine cannot produce it: the first period always
    # opens flat, since a decision needs a session before it. That makes the
    # prepended zero invisible in every engine-produced path, so without this the
    # suite would not notice its removal -- and a caller that did start invested
    # would be charged for a trade the turnover figure had never counted.
    result = TickerResult(
        ticker="TEST",
        dates=sessions(3),
        position=np.array([0.6, 0.6, 0.6]),
        period_return=np.zeros(3),
        gross_return=np.zeros(3),
        cost=np.zeros(3),
    )

    assert result.turnover == pytest.approx(0.6)

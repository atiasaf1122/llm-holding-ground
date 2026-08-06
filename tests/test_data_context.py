"""These are the leakage tests.

Most of them assert the absence of something rather than the presence of it,
which is unusual and deliberate: the failure being guarded against is a context
that contains one extra session, or a level, or a year -- none of which throws,
and all of which quietly turn a forecasting result into a recall result.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pandas as pd
import pytest

from council.data.context import (
    MAX_STALE_DAYS,
    MIN_LOOKBACK_DAYS,
    RECENT_SESSIONS,
    build_price_context,
)
from council.data.prices import synthetic_prices

LOOKBACK = 60

MAX_PLAUSIBLE_PERCENT = 100.0
"""No figure in the context may exceed this.

An absolute price level, a share count or a year would all be larger. A daily
equity return would not be, so the bound doubles as a check that the numbers on
the page are what they claim to be.
"""

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")


def _prices(sessions: int = 120) -> pd.DataFrame:
    return synthetic_prices(tickers=("AAPL",), sessions=sessions, seed=99)


def _session(prices: pd.DataFrame, position: int) -> date:
    return sorted({pd.Timestamp(value) for value in prices["date"]})[position].date()


def _context(prices: pd.DataFrame, decision_date: date, lookback: int = LOOKBACK) -> str:
    return build_price_context(
        prices, ticker="AAPL", decision_date=decision_date, lookback_days=lookback
    )


def _scaled_after(prices: pd.DataFrame, after: date, factor: float) -> pd.DataFrame:
    """A copy in which every session after ``after`` is multiplied by ``factor``."""
    spiked = prices.copy()
    later = spiked["date"] > pd.Timestamp(after)
    for column in ("open", "high", "low", "close", "volume"):
        spiked.loc[later, column] = spiked.loc[later, column] * factor
    return spiked


def test_a_spike_after_the_decision_date_changes_nothing() -> None:
    # Arrange: a tenfold jump on the session after the decision, which would
    # show up as a +900% return if a single future bar leaked into the window.
    prices = _prices()
    decision = _session(prices, 79)
    spiked = _scaled_after(prices, decision, factor=10.0)

    # Act
    before = _context(prices, decision)
    after = _context(spiked, decision)

    # Assert
    assert before == after
    assert "900" not in after


def test_the_window_ends_at_the_decision_session() -> None:
    prices = _prices()
    decision = _session(prices, 79)

    on_the_day = _context(prices, decision)
    a_session_earlier = _context(prices, _session(prices, 78))

    assert on_the_day != a_session_earlier


def test_sessions_before_the_window_are_excluded() -> None:
    # Arrange: rewrite a session that predates the sixty-day window entirely.
    prices = _prices()
    decision = _session(prices, 79)
    edited = prices.copy()
    edited.loc[edited.index[3], "close"] = 1_000.0

    # Act / Assert
    assert _context(prices, decision) == _context(edited, decision)


def test_a_non_trading_decision_date_resolves_backwards() -> None:
    # Arrange: the generator emits business days, so a Saturday is never a session.
    prices = _prices()
    friday = _session(prices, 79)
    saturday = friday + timedelta(days=1)
    assert saturday.weekday() == 5

    # Act / Assert
    assert _context(prices, saturday) == _context(prices, friday)


def test_the_context_never_names_the_instrument() -> None:
    prices = _prices()

    context = _context(prices, _session(prices, 79))

    assert "AAPL" not in context
    assert "aapl" not in context.lower()


def test_the_context_carries_no_year() -> None:
    prices = _prices()

    context = _context(prices, _session(prices, 79))

    assert re.search(r"\b(19|20)\d{2}\b", context) is None


def test_every_figure_is_a_plausible_return_magnitude() -> None:
    # Arrange
    prices = _prices()

    # Act
    context = _context(prices, _session(prices, 79))

    # Assert: a leaked price level (~100) or share volume (~3e6) would fail here.
    figures = [abs(float(match)) for match in NUMBER.findall(context)]
    assert figures
    assert max(figures) <= MAX_PLAUSIBLE_PERCENT


def test_figures_are_rounded_for_the_token_budget() -> None:
    prices = _prices()

    context = _context(prices, _session(prices, 79))

    assert re.search(r"\d\.\d{3,}", context) is None


def test_one_daily_return_is_emitted_per_session_after_the_first() -> None:
    # Arrange
    prices = _prices()

    # Act
    context = _context(prices, _session(prices, 79), lookback=20)

    # Assert
    returns_block = context.split("just closed.\n")[1].split("\n\n")[0]
    assert len(returns_block.split()) == 19


def test_a_short_history_raises_rather_than_shrinking_the_window() -> None:
    prices = _prices(sessions=120)
    too_early = _session(prices, 10)

    with pytest.raises(ValueError, match="11 sessions available"):
        _context(prices, too_early)


def test_a_lookback_below_the_arithmetic_floor_raises() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="at least 3 sessions"):
        _context(prices, _session(prices, 79), lookback=1)


def test_a_lookback_of_two_sessions_raises_rather_than_reporting_nan() -> None:
    # Arrange: two sessions yield one return, and a sample standard deviation of
    # one observation is NaN -- which used to render straight into the prompt.
    prices = _prices()

    # Act / Assert
    with pytest.raises(ValueError, match="at least 3 sessions"):
        _context(prices, _session(prices, 79), lookback=2)


def test_the_shortest_legal_window_reports_a_defined_volatility() -> None:
    prices = _prices()

    context = _context(prices, _session(prices, 79), lookback=MIN_LOOKBACK_DAYS)

    assert "nan" not in context.lower()


def test_a_decision_date_past_the_end_of_the_history_raises() -> None:
    # Arrange: resolving backwards without bound returns the final window on
    # record for every date thereafter, so two decision dates years apart would
    # produce byte-identical contexts that look like independent observations.
    prices = _prices()
    far_future = _session(prices, -1) + timedelta(days=MAX_STALE_DAYS + 1)

    # Act / Assert
    with pytest.raises(ValueError, match="days after the last session on file"):
        _context(prices, far_future)


def test_a_decision_date_within_the_stale_bound_still_resolves_backwards() -> None:
    prices = _prices()
    last = _session(prices, -1)

    assert _context(prices, last + timedelta(days=MAX_STALE_DAYS)) == _context(prices, last)


def test_a_window_no_longer_than_the_recent_summary_omits_it() -> None:
    # Arrange: at this length the "last five sessions" figure would restate the
    # window return, so the line is dropped rather than duplicated.
    prices = _prices()
    decision = _session(prices, 79)

    # Act / Assert
    assert f"Last {RECENT_SESSIONS} sessions" not in _context(prices, decision, RECENT_SESSIONS)
    assert f"Last {RECENT_SESSIONS} sessions" in _context(prices, decision, RECENT_SESSIONS + 1)


def test_a_feed_reporting_no_volume_omits_the_ratio_line() -> None:
    # Arrange: a ratio against a zero median is a division by zero dressed up as
    # a fact, so the line is suppressed rather than printed as inf or nan.
    prices = _prices()
    silent = prices.copy()
    silent["volume"] = 0.0

    # Act
    context = _context(silent, _session(prices, 79))

    # Assert
    assert "Latest volume vs window median" not in context
    assert "inf" not in context and "nan" not in context.lower()


def test_build_price_context_does_not_mutate_its_argument() -> None:
    prices = _prices()
    untouched = prices.copy(deep=True)

    _context(prices, _session(prices, 79))

    pd.testing.assert_frame_equal(prices, untouched)


def test_an_unknown_ticker_raises() -> None:
    prices = _prices()

    with pytest.raises(ValueError, match="no price history for MSFT"):
        build_price_context(
            prices,
            ticker="MSFT",
            decision_date=_session(prices, 79),
            lookback_days=LOOKBACK,
        )


def test_the_context_is_stable_across_calls() -> None:
    prices = _prices()
    decision = _session(prices, 79)

    assert _context(prices, decision) == _context(prices, decision)

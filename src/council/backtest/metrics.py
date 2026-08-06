"""Reducing an equity curve to the numbers a table can hold.

Every degenerate case here is handled by name and none of them returns NaN. A
NaN is how a result table ends up with a blank cell that nobody reads as an
error: it survives a groupby, it prints as an empty string, and it means the arm
it belongs to quietly stops being compared. So a Sharpe with no dispersion under
it is an explicit infinity, a strategy that never took a position has a hit rate
of zero rather than nothing, and an input that cannot be summarised at all raises.

The ratios are annualised from period returns and carry no risk-free rate. The
baseline this project reports against is buy-and-hold and a turnover-matched
random committee -- not cash -- so subtracting a rate here would shift every arm
by the same amount and change no comparison, while inviting the reader to
mistake the numbers for something a fund would quote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from council.backtest.engine import BacktestResult

TRADING_PERIODS_PER_YEAR = 252.0
"""Sessions in a year. The engine's periods run open-to-open, one per session."""

NOISE_FLOOR = 1e-9
"""Dispersion below this multiple of the return scale is floating-point residue.

A constant series does not have a standard deviation of exactly zero: twenty
copies of 0.001 leave about 1e-19 behind, and dividing by that produces a Sharpe
of 7e16. That is worse than an infinity, because it looks like a number and
sorts to the top of a table. The floor is nine orders of magnitude above the
residue and nine below any dispersion a price series could genuinely have.
"""


@dataclass(frozen=True, slots=True)
class PerformanceMetrics:
    """One row of a results table."""

    periods: int
    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    """Positive magnitude: 0.25 means a quarter of the capital was given back."""

    turnover: float
    """Absolute exposure traded over the run, averaged across the basket's tickers.

    A mean rather than a total, so it can be read against the per-ticker turnover
    budget the random baseline is calibrated to without conversion. Comparing it
    to a whole-basket cost figure would understate the cost by the ticker count.
    """

    turnover_per_period: float
    """The same quantity per period, which is what compares across run lengths."""

    hit_rate: float
    """Share of *decided* periods that made money.

    Periods with exactly zero return are neither wins nor losses and are left
    out of both sides. Counting flat periods as misses would make a cautious
    committee look wrong rather than absent.
    """

    time_in_market: float
    """Share of ticker-periods holding a non-zero position."""


def _annualised_ratio(
    *, numerator: float, dispersion: float, scale: float, periods_per_year: float
) -> float:
    """Annualise a per-period mean over a per-period dispersion.

    A dispersion of zero has no finite ratio. Reporting 0.0 would read as "no
    edge" and NaN would read as a missing cell, so a return that never varied is
    reported as a signed infinity -- ugly in a table, which is the point.

    Args:
        scale: the largest absolute return in the series, against which both
            arguments are judged to be real rather than :data:`NOISE_FLOOR`
            residue.
    """
    if dispersion > NOISE_FLOOR * scale:
        return numerator / dispersion * math.sqrt(periods_per_year)
    if abs(numerator) <= NOISE_FLOOR * scale:
        return 0.0
    return math.inf if numerator > 0.0 else -math.inf


def _max_drawdown(returns: np.ndarray) -> float:
    """Deepest peak-to-trough fall of the equity curve, as a positive fraction.

    The curve starts at 1.0 and that starting value is part of the running peak.
    Without it a strategy whose very first period loses money shows no drawdown
    at all, because the trough would also be the highest point seen so far.
    """
    equity = np.concatenate(([1.0], np.cumprod(1.0 + returns)))
    if bool((equity <= 0.0).any()):
        # Capital reached zero. Every later ratio is meaningless, and the honest
        # summary is that all of it was lost.
        return 1.0
    peak = np.maximum.accumulate(equity)
    # The trailing addition turns a negated -0.0 back into 0.0, so a curve that
    # never fell prints as a zero rather than as a minus sign nobody expected.
    return float(-np.min(equity / peak - 1.0) + 0.0)


def _cagr(growth: float, periods: int, periods_per_year: float) -> float:
    """Compound annual growth implied by a total growth multiple.

    A multiple at or below zero has no real root, so a wiped-out run is reported
    as -1.0 -- everything, per year -- rather than as a complex number or a NaN.
    """
    if growth <= 0.0:
        return -1.0
    years = periods / periods_per_year
    return float(growth ** (1.0 / years) - 1.0)


def evaluate_returns(
    *,
    net_return: np.ndarray,
    position: np.ndarray,
    turnover: float,
    periods_per_year: float = TRADING_PERIODS_PER_YEAR,
) -> PerformanceMetrics:
    """Summarise a net-return series.

    Args:
        net_return: one return per period, after costs.
        position: exposure held per period. Two-dimensional for a basket, with
            tickers down the rows, so that time in market counts ticker-periods
            rather than treating a basket half invested as fully invested.
        turnover: total absolute exposure traded, as the engine reports it.
        periods_per_year: how many of these periods make a year.

    Raises:
        ValueError: on an empty or non-finite series, a position array that is
            not a single series or a stack of them covering exactly the returns'
            periods, a non-finite position, or a non-positive year length. None
            of these can be summarised, and all of them would otherwise produce a
            plausible number: a mis-shaped position yields a time in market that
            is arithmetically fine and answers a different question.
    """
    returns = np.asarray(net_return, dtype=float)
    held = np.asarray(position, dtype=float)
    if returns.ndim != 1 or returns.size == 0:
        raise ValueError("need a one-dimensional series of at least one period")
    if not bool(np.isfinite(returns).all()):
        raise ValueError("net returns contain a non-finite value")
    # The rank is checked as well as the last axis: a higher-rank array agrees on
    # its last axis and still divides by the wrong denominator, so time in market
    # comes back plausible and wrong rather than raising.
    if held.ndim not in (1, 2) or held.size == 0 or held.shape[-1] != returns.size:
        raise ValueError("position must cover exactly the periods in net_return")
    # np.count_nonzero counts a NaN as invested, so an unchecked NaN inflates time
    # in market instead of announcing itself.
    if not bool(np.isfinite(held).all()):
        raise ValueError("position contains a non-finite value")
    if periods_per_year <= 0.0:
        raise ValueError("periods_per_year must be positive")

    periods = int(returns.size)
    growth = float(np.prod(1.0 + returns))
    mean_return = float(returns.mean())
    scale = float(np.max(np.abs(returns)))

    # A single period has no estimable spread, so the ratios fall through to the
    # no-dispersion branch rather than dividing by a NaN from ddof=1.
    spread = float(returns.std(ddof=1)) if periods > 1 else 0.0
    downside = float(np.sqrt(np.mean(np.minimum(returns, 0.0) ** 2)))

    wins = int(np.count_nonzero(returns > 0.0))
    losses = int(np.count_nonzero(returns < 0.0))
    decided = wins + losses

    return PerformanceMetrics(
        periods=periods,
        total_return=growth - 1.0,
        cagr=_cagr(growth, periods, periods_per_year),
        sharpe=_annualised_ratio(
            numerator=mean_return,
            dispersion=spread,
            scale=scale,
            periods_per_year=periods_per_year,
        ),
        sortino=_annualised_ratio(
            numerator=mean_return,
            dispersion=downside,
            scale=scale,
            periods_per_year=periods_per_year,
        ),
        max_drawdown=_max_drawdown(returns),
        turnover=turnover,
        turnover_per_period=turnover / periods,
        hit_rate=wins / decided if decided > 0 else 0.0,
        time_in_market=float(np.count_nonzero(held) / held.size),
    )


def evaluate(
    result: BacktestResult, *, periods_per_year: float = TRADING_PERIODS_PER_YEAR
) -> PerformanceMetrics:
    """Summarise a finished backtest, basket-wide."""
    positions = np.vstack([ticker.position for ticker in result.per_ticker])
    return evaluate_returns(
        net_return=result.net_return,
        position=positions,
        turnover=result.turnover,
        periods_per_year=periods_per_year,
    )

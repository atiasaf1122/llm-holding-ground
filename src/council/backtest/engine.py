"""Turning a series of desired exposures into an equity curve.

This is the smallest module in the project and the one most able to invalidate
it. Every result is expressed through this function, so a half-session of
lookahead here would make the entire study wrong in a way that looks like success
-- the failure mode is a beautiful curve, not an exception.

Three rules, each pinned by a test:

**A decision at the close of day *t* is filled at the open of day *t+1*.** Never
the same bar. Filling at day *t*'s close means trading on a price that was not yet
known when the decision was made.

**The position established at an open is held until the next open**, and earns
exactly that period's return. Open-to-open rather than close-to-close, because
open-to-open is the interval the fill rule actually defines.

**Costs are charged on the traded difference**, not on the held position, and they
are on by default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

BPS = 1e-4


@dataclass(frozen=True, slots=True)
class TickerResult:
    """One ticker's path through the backtest."""

    ticker: str
    dates: pd.DatetimeIndex
    """The open of each holding period."""

    position: np.ndarray
    """Exposure actually held over the period beginning at each date."""

    period_return: np.ndarray
    """Open-to-open return of the underlying over that period."""

    gross_return: np.ndarray
    """Strategy return before costs: ``position * period_return``."""

    cost: np.ndarray
    """Charged at the start of the period, on the traded difference."""

    @property
    def net_return(self) -> np.ndarray:
        return self.gross_return - self.cost

    @property
    def equity(self) -> np.ndarray:
        """Growth of one unit of capital, starting at 1.0."""
        return np.cumprod(1.0 + self.net_return)

    @property
    def turnover(self) -> float:
        """Total absolute exposure traded, in units of capital."""
        return float(np.sum(np.abs(np.diff(self.position, prepend=0.0))))


def run_ticker(
    *,
    ticker: str,
    targets: pd.Series,
    opens: pd.Series,
    cost_bps: float,
    rebalance_threshold: float,
) -> TickerResult:
    """Backtest one ticker.

    Args:
        targets: desired exposure, indexed by the date the decision was *made*.
        opens: opening price, indexed by session date. The full trading calendar.
        cost_bps: commission plus slippage, in basis points of traded notional.
        rebalance_threshold: leave the position alone unless the target has moved
            further than this. Without it the daily arm trades on rounding noise
            and pays for the privilege.

    Returns:
        The realised path. Its length is one shorter than the price series, since
        the final session has no following open to price the period against.
    """
    opens = opens.sort_index()
    if len(opens) < 2:
        raise ValueError(f"{ticker}: need at least two sessions to form one period")

    sessions = pd.DatetimeIndex(opens.index)
    prices = opens.to_numpy(dtype=float)

    # A decision made on session i can only be acted on at session i+1's open, so
    # reindexing onto the calendar and shifting by one is the entire lookahead
    # guarantee. Decisions on non-trading days land on NaN and are dropped by the
    # reindex rather than silently attaching to a neighbouring session.
    aligned = targets.reindex(sessions).astype(float)
    desired = aligned.shift(1).to_numpy(dtype=float)

    # Period i runs from open i to open i+1; the last session opens no period.
    period_count = len(sessions) - 1
    period_return = prices[1:] / prices[:-1] - 1.0

    position = np.zeros(period_count, dtype=float)
    cost = np.zeros(period_count, dtype=float)

    held = 0.0
    for index in range(period_count):
        target = desired[index]
        # No decision for this session -- an agent that abstained, or a date before
        # the first signal. Holding is the honest interpretation: the strategy was
        # never told to do anything else.
        if not np.isnan(target) and abs(target - held) > rebalance_threshold:
            cost[index] = abs(target - held) * cost_bps * BPS
            held = float(target)
        position[index] = held

    return TickerResult(
        ticker=ticker,
        dates=sessions[:-1],
        position=position,
        period_return=period_return,
        gross_return=position * period_return,
        cost=cost,
    )


@dataclass(frozen=True, slots=True)
class BacktestResult:
    """An equal-weight basket of per-ticker results."""

    per_ticker: tuple[TickerResult, ...]
    dates: pd.DatetimeIndex
    net_return: np.ndarray

    @property
    def equity(self) -> np.ndarray:
        return np.cumprod(1.0 + self.net_return)

    @property
    def turnover(self) -> float:
        return sum(result.turnover for result in self.per_ticker) / len(self.per_ticker)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"net_return": self.net_return, "equity": self.equity}, index=self.dates
        )


def run_backtest(
    *,
    targets: pd.DataFrame,
    opens: pd.DataFrame,
    cost_bps: float,
    rebalance_threshold: float,
) -> BacktestResult:
    """Backtest an equal-weight basket.

    Each ticker is sized independently and the basket weights them equally. The
    agents are never asked to allocate *between* tickers: that is a different
    research question, and folding it in would make a committee's result depend on
    a skill this study does not measure.

    Args:
        targets: columns are tickers, index is decision date.
        opens: columns are tickers, index is the trading calendar.
    """
    tickers = [str(column) for column in opens.columns]
    if not tickers:
        raise ValueError("no tickers to backtest")

    results = tuple(
        run_ticker(
            ticker=ticker,
            targets=targets[ticker] if ticker in targets else pd.Series(dtype=float),
            opens=opens[ticker],
            cost_bps=cost_bps,
            rebalance_threshold=rebalance_threshold,
        )
        for ticker in tickers
    )

    # Every ticker shares one calendar, so the periods line up by construction.
    # Asserting it is cheaper than debugging a silently misaligned basket.
    reference = results[0].dates
    for result in results[1:]:
        if not result.dates.equals(reference):
            raise ValueError(
                f"{result.ticker} has a different trading calendar to {results[0].ticker}"
            )

    stacked = np.vstack([result.net_return for result in results])
    return BacktestResult(
        per_ticker=results, dates=reference, net_return=stacked.mean(axis=0)
    )


def buy_and_hold(opens: pd.DataFrame) -> BacktestResult:
    """The baseline every result is reported against.

    Fully invested in the equal-weight basket from the first period, never
    trading, paying nothing. A committee that cannot beat this has not earned its
    inference budget, and saying so plainly is worth more than omitting it.
    """
    always_long = pd.DataFrame(1.0, index=opens.index, columns=opens.columns)
    return run_backtest(
        targets=always_long, opens=opens, cost_bps=0.0, rebalance_threshold=0.0
    )

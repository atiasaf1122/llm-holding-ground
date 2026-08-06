"""What a price agent is allowed to see.

This module is a leakage boundary, and it is the one most likely to be skipped
by mistake, because a price series does not look like a leak the way a news
headline does. It is exactly as bad.

Two separate failures are being prevented.

**Lookahead.** The window ends at the decision session and includes nothing
after it. Everything else in the project depends on that: the backtest fills at
the next open precisely so that the decision is made from information available
at the close, and a context assembled from a slice that runs one session too far
would make the whole apparatus decorative.

**Recognition.** The models were trained on this period. Absolute price levels,
a ticker symbol or a date make the series identifiable, and an identified series
is not reasoned about -- it is recalled. A 34% drawdown from a named level in a
named quarter is not a forecasting problem for a model that has read about it;
the agent then scores well for remembering, the debate arm measures who
remembers harder, and the experiment answers a question nobody asked. So the
context carries normalised returns and nothing else: no levels, no dates, no
symbol, no volume in shares.

What survives is the shape of the series, which is what the personas in
:mod:`council.domain.persona` are meant to disagree about.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

PERCENT = 100.0

MIN_LOOKBACK_DAYS = 3
"""Sessions in the shortest window whose every statistic is defined.

Not two. Two sessions yield one return, and the sample standard deviation of one
observation is NaN by definition -- which renders as ``Daily volatility %: nan``
in the prompt, with a numpy warning and no exception. That is precisely the
silent corruption this module exists to stop, so the floor is arithmetic: two
returns, therefore three sessions. :attr:`council.config.Settings.lookback_days`
is bounded far above this; the two numbers are not in competition, one is what
the formulae need and the other is what the experiment wants.
"""

MAX_STALE_DAYS = 5
"""How far a decision date may sit ahead of the session it resolves back to.

Resolving backwards is intended -- a weekend or a holiday is not a session, and
the window must end at the last close before the decision, never after it. But
the resolution is otherwise unbounded, so a decision date past the end of the
price file returns the final window on record, and returns *the same one* for
every later date: decision points that look independent and are byte-identical.
Five calendar days covers a long weekend plus a public holiday, and nothing
else.
"""

RECENT_SESSIONS = 5
"""Length of the short trailing summary, in sessions -- roughly one week."""

RETURNS_PER_LINE = 10
"""Purely for readability in the prompt; it costs a newline per ten values."""


def _trailing_window(
    prices: pd.DataFrame, *, ticker: str, decision_date: date, lookback_days: int
) -> pd.DataFrame:
    """The last ``lookback_days`` sessions ending at ``decision_date``, inclusive.

    A decision date that is not itself a session -- a weekend, a holiday --
    resolves to the last session before it, but by at most :data:`MAX_STALE_DAYS`
    and never to one after it.

    A window shorter than requested raises rather than being padded or shortened.
    An agent given twenty sessions where its peers got sixty is under a different
    treatment, and silently mixing the two into one arm is unrecoverable after
    the fact; the caller is expected to begin its decision dates after warm-up.
    """
    if lookback_days < MIN_LOOKBACK_DAYS:
        raise ValueError(f"lookback_days must cover at least {MIN_LOOKBACK_DAYS} sessions")

    rows = prices.loc[prices["ticker"] == ticker]
    if rows.empty:
        raise ValueError(f"no price history for {ticker}")

    history = rows.set_index("date").sort_index()
    # The whole lookahead guarantee, in one slice: anything after the decision
    # session is discarded before a single statistic is computed from it.
    available = history.loc[: pd.Timestamp(decision_date)]
    if len(available) < lookback_days:
        raise ValueError(
            f"{ticker}: {len(available)} sessions available on or before "
            f"{decision_date}, need {lookback_days}"
        )

    last_session = pd.Timestamp(available.index[-1]).date()
    stale_days = (decision_date - last_session).days
    if stale_days > MAX_STALE_DAYS:
        raise ValueError(
            f"{ticker}: decision date {decision_date} is {stale_days} days after the "
            f"last session on file ({last_session}); the price history does not reach "
            "this decision"
        )
    return available.iloc[-lookback_days:]


def _format_returns(daily: np.ndarray) -> list[str]:
    """Signed percentages, two decimals, wrapped into short lines."""
    values = [f"{value:+.2f}" for value in daily]
    return [
        " ".join(values[start : start + RETURNS_PER_LINE])
        for start in range(0, len(values), RETURNS_PER_LINE)
    ]


def build_price_context(
    prices: pd.DataFrame, *, ticker: str, decision_date: date, lookback_days: int
) -> str:
    """Render the price history one agent sees for one decision.

    Args:
        prices: the tidy long frame from :mod:`council.data.prices`. Assumed
            already validated -- this is called once per agent per decision, and
            revalidating the whole table each time would dominate the runtime.
        ticker: selects the rows. It is never written into the returned text.
        decision_date: the close at which the decision is made. Included.
        lookback_days: sessions in the window, counting the decision session.

    Returns:
        A compact block of normalised statistics, safe to put in a prompt.
    """
    window = _trailing_window(
        prices, ticker=ticker, decision_date=decision_date, lookback_days=lookback_days
    )
    closes = window["close"].to_numpy(dtype=float)
    volume = window["volume"].to_numpy(dtype=float)

    daily = (closes[1:] / closes[:-1] - 1.0) * PERCENT
    peak = np.maximum.accumulate(closes)

    lines = [
        "Price history for one instrument, as percentage changes only.",
        "Absolute levels, dates, the instrument's name and share volume are withheld:",
        "judge the shape of the series, not what you believe it to be.",
        "",
        "Daily returns %, oldest first; the last is the session just closed.",
        *_format_returns(daily),
        "",
        f"Window return %: {(closes[-1] / closes[0] - 1.0) * PERCENT:+.2f}",
        f"Daily volatility %: {float(np.std(daily, ddof=1)):.2f}",
        f"Maximum drawdown %: {float(np.min(closes / peak - 1.0)) * PERCENT:+.2f}",
    ]

    if len(closes) > RECENT_SESSIONS:
        recent = (closes[-1] / closes[-1 - RECENT_SESSIONS] - 1.0) * PERCENT
        lines.append(f"Last {RECENT_SESSIONS} sessions %: {recent:+.2f}")

    # A median of zero means the feed reports no volume for this instrument; a
    # ratio against it would be a division by zero dressed up as a fact.
    median_volume = float(np.median(volume))
    if median_volume > 0.0:
        lines.append(f"Latest volume vs window median: {volume[-1] / median_volume:.2f}x")

    return "\n".join(lines)

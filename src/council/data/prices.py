"""Prices, the trading calendar, and a generator that stands in for both.

The loader is deliberately unforgiving. A price table is the one input every
result in this project passes through, and its failure modes are quiet: a
duplicated session doubles a day's return, a ticker reindexed onto a calendar it
does not share invents sessions on which the market was shut, a NaN open turns
into a filled position at a price that never existed. None of those raise on
their own -- they produce a plausible equity curve. So every one of them is
checked here, once, at the boundary, and named with the ticker and the date that
caused it.

:func:`synthetic_prices` exists so that the rest of the suite never needs a
network call or a checked-in data file. It emits exactly the same tidy frame the
loader does, and passes the same validation, so a test written against synthetic
prices is a test of the real code path.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PRICE_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")
REQUIRED_COLUMNS: tuple[str, ...] = ("date", "ticker", *PRICE_COLUMNS)


def _day(value: Any) -> str:
    """A date rendered for an error message, without a time component."""
    return str(pd.Timestamp(value).date())


def _validate_sessions(ticker: str, dates: pd.DatetimeIndex) -> None:
    """Check one ticker's calendar, in the order it was stored.

    Order matters: sorting first and validating afterwards would hide an
    out-of-order file rather than report it.

    The first two checks are about the *shape* of a timestamp rather than its
    value, and both fail in the direction that produces a plausible number
    instead of an exception. A tz-aware column dies much later, inside pandas,
    with a comparison error that names neither the ticker nor the session. A
    column stamped at the closing bell rather than at midnight is worse: it
    compares as strictly after the decision date, so every trailing window ends
    one session early and every agent reasons about history it should not have
    been given, silently and identically. Normalising instead of rejecting would
    be worse still -- it invents a session boundary the feed never asserted.
    """
    if dates.tz is not None:
        raise ValueError(
            f"{ticker}: sessions carry the timezone {dates.tz}; daily bars must be "
            f"tz-naive, first offender {dates[0]}"
        )

    stamped = dates[dates != dates.normalize()]
    if len(stamped) > 0:
        raise ValueError(
            f"{ticker}: session {stamped[0]} carries a time of day; daily bars must "
            "be stamped at midnight"
        )

    duplicated = dates[dates.duplicated()]
    if len(duplicated) > 0:
        raise ValueError(f"{ticker}: session {_day(duplicated[0])} appears more than once")

    if not dates.is_monotonic_increasing:
        ordered = dates.to_numpy()
        backwards = np.flatnonzero(ordered[1:] < ordered[:-1])
        raise ValueError(
            f"{ticker}: sessions are out of order at {_day(dates[backwards[0] + 1])}"
        )


def _validate_bars(ticker: str, group: pd.DataFrame) -> None:
    """Check one ticker's prices and volumes."""
    # `~(x > 0)` is true for NaN as well as for zero and negatives, so one
    # comparison covers both the missing-value check and the positivity check.
    for column in ("open", "high", "low", "close"):
        invalid = group.loc[~(group[column] > 0.0)]
        if not invalid.empty:
            raise ValueError(
                f"{ticker}: {column} is missing or non-positive on "
                f"{_day(invalid['date'].iloc[0])}"
            )

    # Positivity alone leaves a feed with high and low transposed loading without
    # complaint. The bracket is the property that makes the two columns mean what
    # their names say, so it is asserted rather than assumed.
    body_high = group[["open", "close"]].max(axis=1)
    body_low = group[["open", "close"]].min(axis=1)
    for column, broken in (
        ("high", group.loc[group["high"] < body_high]),
        ("low", group.loc[group["low"] > body_low]),
    ):
        if not broken.empty:
            raise ValueError(
                f"{ticker}: {column} does not bracket open and close on "
                f"{_day(broken['date'].iloc[0])}"
            )

    negative_volume = group.loc[~(group["volume"] >= 0.0)]
    if not negative_volume.empty:
        raise ValueError(
            f"{ticker}: volume is missing or negative on "
            f"{_day(negative_volume['date'].iloc[0])}"
        )


def _validate_ticker(ticker: str, group: pd.DataFrame) -> None:
    """Check one ticker's rows, calendar first."""
    _validate_sessions(ticker, pd.DatetimeIndex(group["date"]))
    _validate_bars(ticker, group)


def validate_prices(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalise a long-format price table and reject anything unusable.

    Args:
        frame: long format -- one row per ticker per session, with the columns
            named in :data:`REQUIRED_COLUMNS`. Extra columns are dropped.

    Returns:
        The same data, typed, sorted by ticker then date, and reindexed from
        zero. This is the tidy frame every other function in the data layer
        accepts and returns.

    Raises:
        ValueError: naming the offending ticker and session.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"price table is missing columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("price table is empty")

    tidy = frame.loc[:, list(REQUIRED_COLUMNS)].copy()
    tidy["date"] = pd.to_datetime(tidy["date"])
    tidy["ticker"] = tidy["ticker"].astype(str)
    for column in PRICE_COLUMNS:
        tidy[column] = tidy[column].astype(float)

    for ticker in sorted(set(tidy["ticker"])):
        _validate_ticker(ticker, tidy.loc[tidy["ticker"] == ticker])

    return tidy.sort_values(["ticker", "date"], kind="stable").reset_index(drop=True)


def load_prices(path: Path) -> pd.DataFrame:
    """Read a long-format price parquet and validate it.

    Raises:
        FileNotFoundError: if the file is absent. Worth saying plainly, because
            the alternative -- an empty frame -- turns into an empty backtest
            that reports zero return rather than an error.
    """
    if not path.exists():
        raise FileNotFoundError(f"no price file at {path}")
    return validate_prices(pd.read_parquet(path))


def write_prices(prices: pd.DataFrame, path: Path) -> Path:
    """Validate a price table and write it where :func:`load_prices` will find it.

    Validated on the way out rather than only on the way in. A frame assembled in
    memory -- the dry run's generated one, say -- has never been through the
    boundary, and writing it unchecked would move the failure to whoever loads it
    next, with no record of which step produced it.

    Returns:
        The path written, so a caller can name it in its output.
    """
    validated = validate_prices(prices)
    path.parent.mkdir(parents=True, exist_ok=True)
    validated.to_parquet(path, index=False)
    return path


def _raise_calendar_mismatch(
    reference_ticker: str,
    reference: pd.DatetimeIndex,
    ticker: str,
    other: pd.DatetimeIndex,
) -> None:
    only_reference = reference.difference(other)
    only_other = other.difference(reference)
    disagreements = sorted(set(only_reference) | set(only_other))
    day = disagreements[0]
    owner = ticker if day in set(only_other) else reference_ticker
    raise ValueError(
        f"{ticker} and {reference_ticker} do not share a trading calendar: "
        f"session {_day(day)} exists only for {owner}"
    )


def opens_frame(
    prices: pd.DataFrame, *, tickers: Sequence[str] | None = None
) -> pd.DataFrame:
    """One column of opening prices per ticker, on one shared calendar.

    Tickers whose calendars differ raise rather than being aligned. Filling the
    gaps would be worse than useless here: the backtest holds a position from
    one open to the next, so a ticker quietly reindexed onto a neighbour's
    calendar contributes a return over a session on which it did not trade, and
    the basket appears to trade on days the market was shut.

    Args:
        prices: the tidy long frame. Revalidated, so this is safe to call on a
            frame assembled in memory rather than loaded from disk.
        tickers: which tickers to include, in the order the columns should
            appear. Defaults to every ticker present, sorted.

    Returns:
        A frame indexed by session date, one float column per ticker.
    """
    tidy = validate_prices(prices)
    available = sorted(set(tidy["ticker"]))
    wanted = list(tickers) if tickers is not None else available

    unknown = [ticker for ticker in wanted if ticker not in set(available)]
    if unknown:
        raise ValueError(f"no price history for {', '.join(unknown)}")
    if not wanted:
        raise ValueError("no tickers requested")

    # The columns are accumulated into a dict keyed by ticker, so a repeat would
    # collapse rather than duplicate: three requested columns come back as two.
    # Downstream that is not a shape error, it is a weighting error -- the
    # backtest reads its universe off these columns and divides by their count,
    # so a caller-side typo silently reweights the equal-weight basket.
    repeated = sorted({ticker for ticker in wanted if wanted.count(ticker) > 1})
    if repeated:
        raise ValueError(f"tickers requested more than once: {', '.join(repeated)}")

    columns: dict[str, pd.Series] = {}
    reference_ticker = wanted[0]
    reference: pd.DatetimeIndex | None = None
    for ticker in wanted:
        series = tidy.loc[tidy["ticker"] == ticker].set_index("date")["open"]
        index = pd.DatetimeIndex(series.index)
        if reference is None:
            reference = index
        elif not index.equals(reference):
            _raise_calendar_mismatch(reference_ticker, reference, ticker, index)
        columns[ticker] = series

    frame = pd.DataFrame(columns)
    frame.index.name = "date"
    return frame


def _ticker_entropy(ticker: str) -> int:
    """A stable integer for a ticker symbol.

    The builtin ``hash`` is salted per process, so a generator seeded with it
    would produce a different series on every run -- which is the one thing this
    helper exists to prevent.
    """
    return int.from_bytes(hashlib.sha256(ticker.encode("utf-8")).digest()[:8], "big")


def synthetic_prices(
    *,
    tickers: Sequence[str] = ("AAA", "BBB"),
    start: date = date(2022, 1, 3),
    sessions: int = 120,
    seed: int = 1,
    drift: float = 0.0003,
    volatility: float = 0.012,
    initial_price: float = 100.0,
) -> pd.DataFrame:
    """A deterministic price table, for tests and for offline development.

    A geometric random walk: log returns are normal, so prices are positive by
    construction and the frame always passes :func:`validate_prices`. Each
    ticker draws from a stream keyed by its own symbol, so adding a ticker to a
    test never perturbs the series of the ones already there -- otherwise every
    expected value in the suite would move whenever the universe changed.

    The four column families draw from four streams for the same reason, one step
    down. Drawn sequentially from one generator, every family after ``close``
    starts at an offset that depends on ``sessions``, so lengthening the calendar
    rewrote ``open``, ``high``, ``low`` and ``volume`` on sessions that already
    existed while leaving ``close`` untouched. The backtest fills at ``open``, so
    that is the column a widened range silently changed underneath a stored run.
    Four streams make the frame prefix-stable in ``sessions``: the first *n* rows
    of a longer frame are the first *n* rows of a shorter one, column for column.

    The calendar is business days with no market holidays. Holidays would make
    it possible to write a test that passes only because one happened to fall
    inside the window under examination.

    The default symbols and seed are deliberately not the configured ones. They
    are a second copy of values :mod:`council.config` owns, and a second copy
    drifts; keeping them visibly fake also means a frame produced here can never
    be mistaken for a configured run in a log or a saved artefact.

    Args:
        drift: mean log return per session.
        volatility: standard deviation of the log return per session. The
            default is roughly 19% annualised -- a plausible large-cap equity.

    Returns:
        The tidy long frame, identical in shape and dtype to a loaded one.
    """
    if sessions < 2:
        raise ValueError("need at least two sessions to form one return")
    if volatility <= 0.0:
        raise ValueError("volatility must be positive")

    calendar = pd.bdate_range(start=start, periods=sessions)
    parts: list[pd.DataFrame] = []
    for ticker in sorted(tickers):
        entropy = _ticker_entropy(ticker)
        streams = [np.random.default_rng([seed, entropy, family]) for family in range(4)]
        close = initial_price * np.exp(
            np.cumsum(streams[0].normal(drift, volatility, sessions))
        )

        # The open gaps from the previous close; the first session opens at the
        # initial price so that the level is comparable across tickers.
        opens = np.empty(sessions, dtype=float)
        opens[0] = initial_price
        opens[1:] = close[:-1] * np.exp(streams[1].normal(0.0, volatility / 3.0, sessions - 1))

        wick = np.abs(streams[2].normal(0.0, volatility / 2.0, sessions))
        parts.append(
            pd.DataFrame(
                {
                    "date": calendar,
                    "ticker": ticker,
                    "open": opens,
                    "high": np.maximum(opens, close) * np.exp(wick),
                    "low": np.minimum(opens, close) * np.exp(-wick),
                    "close": close,
                    "volume": np.round(streams[3].lognormal(15.0, 0.3, sessions)),
                }
            )
        )

    return validate_prices(pd.concat(parts, ignore_index=True))

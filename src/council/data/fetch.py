"""Real daily bars, fetched once and pinned.

The study ran on synthetic geometric random walks for a long time, and that made
every return column meaningless by construction: there is nothing in a random walk
to forecast, so an arm losing to the benchmark was arithmetic rather than a result.
The behavioural half -- whether an agent moves for the argument or for the
contradiction -- never depended on the prices. The market half did, and had nothing.

So this fetches the real series. Two consequences follow and both are load-bearing.

**Prices are split and dividend adjusted.** ``auto_adjust`` back-adjusts the whole
series, so an open here is not a price anyone could have traded at -- it is the
price that makes the return series continuous through a corporate action. That is
the right input for a backtest measuring total return, and the wrong one for
claiming a fill was achievable. This study measures the former.

**Recognition is now a live risk rather than a hypothetical.** Every model in the
committee was trained on data covering this period, and these are two of the most
written-about instruments in it. The defence is unchanged and was built for exactly
this: agents see normalised returns with no dates, no ticker and no price levels
(:mod:`council.data.context`). It reduces recognition; it cannot prove its absence,
and a distinctive drawdown shape is still a shape.

**Reproducibility.** Vendors revise history. The raw response is written beside the
parquet so that a rerun can be compared against the bytes this study actually used,
rather than against whatever the vendor serves later.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from council.data.prices import validate_prices

COLUMNS = ("date", "ticker", "open", "high", "low", "close", "volume")

# A requested end date is routinely a weekend or a holiday, so the last bar is
# legitimately a few days short of it. Wide enough for the longest market closure,
# narrow enough that a vendor returning a materially truncated history still raises.
MAX_TRAILING_GAP_DAYS = 10


def fetch_prices(
    *,
    tickers: Sequence[str],
    start: date,
    end: date,
    lookback_days: int = 0,
    raw_path: Path | None = None,
) -> pd.DataFrame:
    """Daily bars for ``tickers``, in this project's price schema.

    Args:
        start: the first date a *decision* may be made on.
        end: the last.
        lookback_days: extra sessions fetched before ``start``. A decision needs a
            trailing window, so fetching from ``start`` exactly would leave the
            first ``lookback_days`` decisions with no history and silently shorten
            the experiment -- which is how a six-month window once became the whole
            study without anyone choosing it.
        raw_path: where to write the vendor's response verbatim, for provenance.

    Raises:
        ValueError: if the vendor returns nothing, or returns a frame that fails
            :func:`~council.data.prices.validate_prices`. Refusing beats scanning
            a short or holed series: every downstream number would be computed
            over a different calendar than the one reported.
    """
    import yfinance as yf

    # Calendar days, generously: sessions are fewer than days, and over-fetching
    # costs one request while under-fetching costs the first weeks of the study.
    padded = start - timedelta(days=int(lookback_days * 1.6) + 14)
    raw = yf.download(
        list(tickers),
        start=padded.isoformat(),
        end=(end + timedelta(days=1)).isoformat(),
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=False,
    )
    if raw is None or raw.empty:
        raise ValueError(f"no bars returned for {list(tickers)} over {padded}..{end}")

    if raw_path is not None:
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(raw_path)

    frame = _tidy(raw, tickers=tickers)
    _check_coverage(frame, tickers=tickers, start=start, end=end)
    return validate_prices(frame)


def _tidy(raw: pd.DataFrame, *, tickers: Sequence[str]) -> pd.DataFrame:
    """Flatten the vendor's per-ticker column blocks into long rows."""
    rows: list[pd.DataFrame] = []
    for ticker in tickers:
        # A single ticker comes back without the outer level.
        block = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) else raw
        frame_block = pd.DataFrame(block)
        part = frame_block.rename(columns=str.lower).reset_index()
        part = part.rename(columns={"index": "date", "Date": "date"})
        part["ticker"] = ticker
        rows.append(part[list(COLUMNS)])

    frame = pd.concat(rows, ignore_index=True)
    frame["date"] = pd.to_datetime(frame["date"]).dt.tz_localize(None).dt.normalize()
    # A session the vendor has no bar for is dropped rather than forward filled: a
    # fabricated bar is a fabricated return, and the calendar check below is what
    # decides whether what remains is enough.
    frame = frame.dropna(subset=["open", "close"])
    return frame.sort_values(["ticker", "date"], ignore_index=True)


def _check_coverage(
    frame: pd.DataFrame, *, tickers: Sequence[str], start: date, end: date
) -> None:
    """Refuse a series that does not span what was asked for.

    A vendor that quietly returns a shorter history is the failure this catches.
    It does not raise -- it returns a frame, the run proceeds, and every figure is
    computed over a range nobody chose.
    """
    for ticker in tickers:
        days = frame.loc[frame["ticker"] == ticker, "date"]
        if days.empty:
            raise ValueError(f"{ticker}: no bars returned")
        first, last = days.min().date(), days.max().date()
        if first > start:
            raise ValueError(
                f"{ticker}: history starts {first}, after the requested start {start}; "
                "the trailing window for the earliest decisions would be short"
            )
        # Against the last *session* on or before the requested end, not against the
        # calendar date: a range ending on a weekend or a holiday is the ordinary
        # case, and a check that refuses it refuses most requests. The tolerance is
        # wide enough for a long holiday and narrow enough that a vendor returning
        # a materially short history still raises.
        if (end - last).days > MAX_TRAILING_GAP_DAYS:
            raise ValueError(
                f"{ticker}: history ends {last}, more than {MAX_TRAILING_GAP_DAYS} days "
                f"before the requested end {end}"
            )

    calendars = {t: set(frame.loc[frame["ticker"] == t, "date"]) for t in tickers}
    shared = set.intersection(*calendars.values())
    for ticker, own in calendars.items():
        missing = len(own - shared)
        if missing:
            raise ValueError(
                f"{ticker} trades on {missing} session(s) the others do not; the basket "
                "would hold a position priced on a day its counterpart was shut"
            )

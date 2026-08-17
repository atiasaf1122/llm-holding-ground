"""The price fetch: the one step of the pipeline that talks to the outside world.

Every number in the study is computed over whatever calendar this module returns,
which is why the tests here are almost entirely about *refusal*. A vendor that
serves a short history, a holed session or two tickers on disagreeing calendars
does not raise on its own -- it returns a smaller frame, the run proceeds, and the
published figures describe a window nobody chose. Each test below is one such
frame, asserted to die at the fetch instead of surviving into the results.

The vendor itself is stubbed. A test that reaches the network measures the
network, and this module's contract is what it does with a response, not that a
response arrives.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import date
from pathlib import Path
from types import ModuleType

import pandas as pd
import pytest

from council.data.fetch import MAX_TRAILING_GAP_DAYS, fetch_prices

START = date(2022, 1, 3)
END = date(2022, 1, 14)
TICKERS = ("AAPL", "XOM")


def vendor_frame(
    *,
    tickers: tuple[str, ...] = TICKERS,
    sessions: pd.DatetimeIndex | None = None,
    per_ticker_sessions: dict[str, pd.DatetimeIndex] | None = None,
) -> pd.DataFrame:
    """A response shaped the way yfinance shapes one: a session index, and a
    column block per ticker under a MultiIndex."""
    default = sessions if sessions is not None else pd.bdate_range(START, END)
    calendars = per_ticker_sessions or dict.fromkeys(tickers, default)
    index = pd.DatetimeIndex(sorted(set().union(*(set(c) for c in calendars.values()))))
    index.name = "Date"

    blocks: dict[tuple[str, str], pd.Series] = {}
    for offset, ticker in enumerate(tickers):
        own = set(calendars[ticker])
        present = [day in own for day in index]
        base = [100.0 + offset * 10 + position for position, _ in enumerate(index)]
        for field, bump in (("Open", 0.0), ("High", 1.0), ("Low", -1.0), ("Close", 0.5)):
            bars = [
                value + bump if here else float("nan")
                for value, here in zip(base, present, strict=True)
            ]
            blocks[(ticker, field)] = pd.Series(bars, index=index)
        blocks[(ticker, "Volume")] = pd.Series(
            [1_000_000.0 if here else float("nan") for here in present], index=index
        )

    frame = pd.DataFrame(blocks)
    frame.columns = pd.MultiIndex.from_tuples(frame.columns)
    return frame


@pytest.fixture
def vendor(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Install a stub ``yfinance`` and record the calls made to it.

    The module imports the vendor inside the function, so the stub has to live in
    ``sys.modules`` rather than be passed in -- and the recorded calls are what
    lets the padding test assert on a request that was never sent to a server.
    """
    calls: list[dict] = []
    served: list[pd.DataFrame] = [vendor_frame()]

    def download(symbols: list[str], **kwargs: object) -> pd.DataFrame:
        calls.append({"symbols": symbols, **kwargs})
        return served[0]

    stub = ModuleType("yfinance")
    stub.download = download  # type: ignore[attr-defined]
    stub.serve = served  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yfinance", stub)
    return calls


def serve(vendor_calls: list[dict], frame: pd.DataFrame) -> None:
    """Point the stub at a different response."""
    sys.modules["yfinance"].serve[0] = frame  # type: ignore[attr-defined]


# -- the happy path ----------------------------------------------------------------


def test_a_clean_response_becomes_this_projects_price_schema(vendor: list[dict]) -> None:
    prices = fetch_prices(tickers=TICKERS, start=START, end=END)

    assert list(prices.columns) == ["date", "ticker", "open", "high", "low", "close", "volume"]
    assert set(prices["ticker"]) == set(TICKERS)
    assert prices["date"].dt.tz is None, "a tz-aware session compares wrong against a decision date"
    assert prices.equals(prices.sort_values(["ticker", "date"], ignore_index=True))


def test_the_lookback_is_fetched_before_the_first_decision_date(vendor: list[dict]) -> None:
    """The failure this prevents is silent: without the padding the earliest
    decisions have no trailing window, and the study quietly starts later than
    the date it reports starting."""
    fetch_prices(tickers=TICKERS, start=START, end=END, lookback_days=60)

    requested_start = date.fromisoformat(str(vendor[0]["start"]))
    assert requested_start < START - pd.Timedelta(days=60).to_pytimedelta()


def test_the_vendors_response_is_pinned_for_provenance(
    vendor: list[dict], tmp_path: Path
) -> None:
    """Vendors revise history. Without the raw bytes, a rerun that disagrees with
    the published series cannot be told from a study that was wrong."""
    raw = tmp_path / "nested" / "raw.csv"

    fetch_prices(tickers=TICKERS, start=START, end=END, raw_path=raw)

    assert raw.exists() and raw.stat().st_size > 0


# -- the refusals ------------------------------------------------------------------


def test_an_empty_response_raises_rather_than_returning_nothing(vendor: list[dict]) -> None:
    serve(vendor, pd.DataFrame())

    with pytest.raises(ValueError, match="no bars returned"):
        fetch_prices(tickers=TICKERS, start=START, end=END)


def test_a_history_that_starts_late_is_refused(vendor: list[dict]) -> None:
    serve(vendor, vendor_frame(sessions=pd.bdate_range(date(2022, 1, 6), END)))

    with pytest.raises(ValueError, match="after the requested start"):
        fetch_prices(tickers=TICKERS, start=START, end=END)


def test_a_history_that_ends_early_is_refused(vendor: list[dict]) -> None:
    stale = END - pd.Timedelta(days=MAX_TRAILING_GAP_DAYS + 1).to_pytimedelta()
    serve(vendor, vendor_frame(sessions=pd.bdate_range(START, stale)))

    with pytest.raises(ValueError, match="before the requested end"):
        fetch_prices(tickers=TICKERS, start=START, end=END)


def test_an_end_date_that_falls_on_a_closed_market_is_accepted(vendor: list[dict]) -> None:
    """The ordinary case, and the one a naive freshness check refuses: a range
    ending on a weekend has its last bar days earlier by definition."""
    last_session = pd.bdate_range(START, END)[-1].date()
    weekend_end = date(2022, 1, 16)  # the Sunday after END's Friday
    assert last_session < weekend_end, "the range must end on a day the market was shut"

    prices = fetch_prices(tickers=TICKERS, start=START, end=weekend_end)

    assert not prices.empty


def test_tickers_on_disagreeing_calendars_are_refused(vendor: list[dict]) -> None:
    """A basket priced on a day one leg was shut carries a position nobody could
    have held, and the backtest books its return anyway."""
    shared = pd.bdate_range(START, END)
    serve(
        vendor,
        vendor_frame(
            per_ticker_sessions={
                "AAPL": shared,
                "XOM": shared.drop(shared[3]),
            }
        ),
    )

    with pytest.raises(ValueError, match="session"):
        fetch_prices(tickers=TICKERS, start=START, end=END)


# -- the packaging guard -----------------------------------------------------------


def test_every_pipeline_module_is_under_version_control() -> None:
    """This module was not, for the whole study.

    An unanchored ``data/`` in .gitignore matched ``src/council/data/`` as well as
    the run's working directory, so the step that downloads the price series was
    excluded from every commit while pyproject declared the dependency it needs --
    a clone could install the study and not contain its first step (D17). Nothing
    in a normal suite notices, because the file is present on the machine that
    wrote it. Only the index knows, so the index is what this asks.
    """
    root = Path(__file__).resolve().parents[1]
    try:
        listed = subprocess.run(
            ["git", "ls-files", "src"],
            cwd=root,
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):  # pragma: no cover - not a checkout
        pytest.skip("not a git checkout")

    tracked = {root / line for line in listed.stdout.splitlines()}
    on_disk = {
        path
        for path in (root / "src").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    missing = sorted(str(path.relative_to(root)) for path in on_disk - tracked)
    assert not missing, f"source files excluded from the repository: {missing}"

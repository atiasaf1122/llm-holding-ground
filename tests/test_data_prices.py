"""The loader's job is to refuse bad data loudly, and the generator's is to be
boringly repeatable. Both are tested here against the same tidy contract."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from council.config import get_settings
from council.data.prices import (
    REQUIRED_COLUMNS,
    load_prices,
    opens_frame,
    synthetic_prices,
    validate_prices,
)


def _sessions(prices: pd.DataFrame) -> list[pd.Timestamp]:
    return sorted({pd.Timestamp(value) for value in prices["date"]})


def test_synthetic_prices_are_identical_for_the_same_seed() -> None:
    # Arrange / Act
    first = synthetic_prices(seed=7, sessions=40)
    second = synthetic_prices(seed=7, sessions=40)

    # Assert
    pd.testing.assert_frame_equal(first, second)


def test_synthetic_prices_differ_when_the_seed_differs() -> None:
    first = synthetic_prices(seed=7, sessions=40)
    second = synthetic_prices(seed=8, sessions=40)

    assert not first["close"].equals(second["close"])


def test_adding_a_ticker_does_not_perturb_the_others() -> None:
    # Arrange
    two = synthetic_prices(tickers=("AAPL", "XOM"), sessions=30)

    # Act
    three = synthetic_prices(tickers=("AAPL", "XOM", "JNJ"), sessions=30)

    # Assert
    for ticker in ("AAPL", "XOM"):
        pd.testing.assert_frame_equal(
            two.loc[two["ticker"] == ticker].reset_index(drop=True),
            three.loc[three["ticker"] == ticker].reset_index(drop=True),
        )


def test_synthetic_prices_satisfy_the_loader_invariants() -> None:
    prices = synthetic_prices(sessions=50)

    assert list(prices.columns) == list(REQUIRED_COLUMNS)
    assert (prices[["open", "high", "low", "close"]] > 0).all().all()
    assert (prices["high"] >= prices[["open", "close"]].max(axis=1)).all()
    assert (prices["low"] <= prices[["open", "close"]].min(axis=1)).all()


def test_synthetic_prices_put_every_ticker_on_one_calendar() -> None:
    prices = synthetic_prices(tickers=("AAPL", "XOM"), sessions=25)

    opens = opens_frame(prices)

    assert list(opens.columns) == ["AAPL", "XOM"]
    assert len(opens) == 25
    assert not opens.isna().to_numpy().any()


def test_load_prices_round_trips_a_parquet_file(tmp_path: Path) -> None:
    # Arrange
    prices = synthetic_prices(sessions=20)
    path = tmp_path / "prices.parquet"
    prices.to_parquet(path)

    # Act
    loaded = load_prices(path)

    # Assert
    pd.testing.assert_frame_equal(loaded, prices)


def test_load_prices_names_the_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match=re.escape("nowhere.parquet")):
        load_prices(tmp_path / "nowhere.parquet")


def test_validate_rejects_a_duplicated_session() -> None:
    # Arrange
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    repeated = pd.concat([prices, prices.iloc[[4]]], ignore_index=True)
    day = str(pd.Timestamp(prices["date"].iloc[4]).date())

    # Act / Assert
    with pytest.raises(ValueError, match=f"AAPL: session {day} appears more than once"):
        validate_prices(repeated)


def test_validate_rejects_sessions_stored_out_of_order() -> None:
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    order = [0, 1, 2, 4, 3, 5, 6, 7, 8, 9]
    shuffled = prices.iloc[order].reset_index(drop=True)
    day = str(pd.Timestamp(prices["date"].iloc[3]).date())

    with pytest.raises(ValueError, match=f"AAPL: sessions are out of order at {day}"):
        validate_prices(shuffled)


def test_validate_rejects_a_missing_open() -> None:
    prices = synthetic_prices(tickers=("AAPL", "XOM"), sessions=10)
    holed = prices.copy()
    target = holed.loc[holed["ticker"] == "XOM"].index[3]
    holed.loc[target, "open"] = np.nan
    day = str(pd.Timestamp(holed.loc[target, "date"]).date())

    with pytest.raises(ValueError, match=f"XOM: open is missing or non-positive on {day}"):
        validate_prices(holed)


def test_validate_rejects_a_non_positive_close() -> None:
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    zeroed = prices.copy()
    zeroed.loc[zeroed.index[5], "close"] = 0.0

    with pytest.raises(ValueError, match="AAPL: close is missing or non-positive"):
        validate_prices(zeroed)


def test_validate_names_the_columns_it_needs() -> None:
    prices = synthetic_prices(sessions=5).drop(columns=["volume"])

    with pytest.raises(ValueError, match="missing columns: volume"):
        validate_prices(prices)


def test_validate_rejects_an_empty_table() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_prices(synthetic_prices(sessions=5).iloc[:0])


def test_opens_frame_raises_when_two_tickers_disagree_on_the_calendar() -> None:
    # Arrange: XOM never traded on one session AAPL did.
    prices = synthetic_prices(tickers=("AAPL", "XOM"), sessions=15)
    missing_day = _sessions(prices)[6]
    gapped = prices.loc[
        ~((prices["ticker"] == "XOM") & (prices["date"] == missing_day))
    ].reset_index(drop=True)

    # Act / Assert
    with pytest.raises(ValueError, match=f"session {missing_day.date()} exists only for AAPL"):
        opens_frame(gapped)


def test_opens_frame_keeps_the_requested_ticker_order() -> None:
    prices = synthetic_prices(tickers=("AAPL", "XOM"), sessions=12)

    opens = opens_frame(prices, tickers=["XOM", "AAPL"])

    assert list(opens.columns) == ["XOM", "AAPL"]
    assert opens.index.name == "date"


def test_opens_frame_carries_the_opening_price_through_unchanged() -> None:
    prices = synthetic_prices(tickers=("AAPL",), sessions=12)

    opens = opens_frame(prices)

    assert opens["AAPL"].to_numpy() == pytest.approx(prices["open"].to_numpy())


def test_opens_frame_rejects_a_ticker_with_no_history() -> None:
    prices = synthetic_prices(tickers=("AAPL",), sessions=12)

    with pytest.raises(ValueError, match="no price history for MSFT"):
        opens_frame(prices, tickers=["AAPL", "MSFT"])


def test_opens_frame_rejects_an_empty_ticker_list() -> None:
    prices = synthetic_prices(tickers=("AAPL",), sessions=12)

    with pytest.raises(ValueError, match="no tickers requested"):
        opens_frame(prices, tickers=[])


def test_opens_frame_rejects_a_ticker_requested_twice() -> None:
    # Arrange: silently collapsing this would hand the backtest a two-column
    # universe where the caller asked for three, reweighting the basket.
    prices = synthetic_prices(tickers=("AAPL", "XOM"), sessions=10)

    # Act / Assert
    with pytest.raises(ValueError, match="tickers requested more than once: AAPL"):
        opens_frame(prices, tickers=["AAPL", "AAPL", "XOM"])


def test_validate_rejects_a_negative_volume() -> None:
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    negated = prices.copy()
    negated.loc[negated.index[2], "volume"] = -1.0
    day = str(pd.Timestamp(prices["date"].iloc[2]).date())

    with pytest.raises(ValueError, match=f"AAPL: volume is missing or negative on {day}"):
        validate_prices(negated)


def test_validate_rejects_a_missing_volume() -> None:
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    holed = prices.copy()
    holed.loc[holed.index[2], "volume"] = np.nan

    with pytest.raises(ValueError, match="AAPL: volume is missing or negative"):
        validate_prices(holed)


def test_validate_rejects_a_high_below_the_body() -> None:
    # Arrange: high and low transposed, the one OHLC corruption that survives
    # every positivity check.
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    flipped = prices.copy()
    row = flipped.index[4]
    flipped.loc[row, "high"], flipped.loc[row, "low"] = (
        prices.loc[row, "low"],
        prices.loc[row, "high"],
    )
    day = str(pd.Timestamp(prices["date"].iloc[4]).date())

    # Act / Assert
    with pytest.raises(ValueError, match=rf"AAPL: high does not bracket .* on {day}"):
        validate_prices(flipped)


def test_validate_rejects_a_low_above_the_body() -> None:
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    raised = prices.copy()
    row = raised.index[4]
    raised.loc[row, "low"] = float(prices.loc[row, "high"])

    with pytest.raises(ValueError, match="AAPL: low does not bracket"):
        validate_prices(raised)


def test_validate_rejects_timezone_aware_sessions() -> None:
    # Arrange: a tz-aware column passes every value check and then dies much
    # later inside pandas, comparing against the naive dates used everywhere else.
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    zoned = prices.copy()
    zoned["date"] = pd.to_datetime(zoned["date"]).dt.tz_localize("America/New_York")

    # Act / Assert
    with pytest.raises(ValueError, match="AAPL: sessions carry the timezone"):
        validate_prices(zoned)


def test_validate_rejects_bars_stamped_at_the_closing_bell() -> None:
    # Arrange: a feed stamping daily bars at 16:00. Every trailing window would
    # then end one session early, silently, because 16:00 sorts after midnight.
    prices = synthetic_prices(tickers=("AAPL",), sessions=10)
    intraday = prices.copy()
    intraday["date"] = pd.to_datetime(intraday["date"]) + pd.Timedelta(hours=16)

    # Act / Assert
    with pytest.raises(ValueError, match=r"AAPL: session .* carries a time of day"):
        validate_prices(intraday)


def test_synthetic_prices_reject_a_single_session() -> None:
    with pytest.raises(ValueError, match="at least two sessions"):
        synthetic_prices(tickers=("AAPL",), sessions=1)


def test_synthetic_prices_reject_zero_volatility() -> None:
    with pytest.raises(ValueError, match="volatility must be positive"):
        synthetic_prices(tickers=("AAPL",), sessions=10, volatility=0.0)


def test_synthetic_prices_do_not_default_to_the_configured_universe() -> None:
    # Two copies of the universe drift apart; a synthetic frame must also be
    # unmistakable for a configured run in a log or a saved artefact.
    assert set(synthetic_prices(sessions=5)["ticker"]).isdisjoint(get_settings().tickers)


def test_validate_prices_does_not_mutate_its_argument() -> None:
    # Arrange
    prices = synthetic_prices(tickers=("AAPL", "XOM"), sessions=10)
    untouched = prices.copy(deep=True)

    # Act
    validate_prices(prices)

    # Assert
    pd.testing.assert_frame_equal(prices, untouched)


def test_opens_frame_does_not_mutate_its_argument() -> None:
    prices = synthetic_prices(tickers=("AAPL", "XOM"), sessions=10)
    untouched = prices.copy(deep=True)

    opens_frame(prices)

    pd.testing.assert_frame_equal(prices, untouched)

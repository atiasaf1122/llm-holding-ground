"""The conversion every other evaluation module depends on.

If the frame layer loses the independent arm, or orders rows by whatever the
parquet writer happened to emit, then every number downstream is wrong in a way
that still looks like a number.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd
import pytest

from council.backtest.engine import run_ticker
from council.data.prices import opens_frame, synthetic_prices
from council.domain.signal import Arm, Decision, FailureMode
from council.evaluation.frames import (
    NO_COMPOSITION,
    REQUIRED_COLUMNS,
    decisions_to_frame,
    forward_returns,
    forward_returns_lookup,
    frame_to_rows,
)
from helpers_decisions import DAY, NEXT_DAY, frame_of, row

GENERATED_AT = datetime(2022, 3, 1, 21, 0, 0)


def decision(**overrides: object) -> Decision:
    fields: dict[str, object] = {
        "decision_date": DAY,
        "ticker": "AAPL",
        "model": "alpha",
        "persona": "momentum-bold",
        "exposure": 0.4,
        "confidence": 0.7,
        "prompt_hash": "abc123",
        "seed": 20260101,
        "generated_at": GENERATED_AT,
    }
    fields.update(overrides)
    return Decision(**fields)  # type: ignore[arg-type]


def test_a_stored_decision_survives_the_round_trip_unchanged() -> None:
    original = decision(arm=Arm.DEBATE, round_index=1, composition="quad", exposure=-0.25)

    (restored,) = frame_to_rows(decisions_to_frame([original]))

    assert restored.decision_date == DAY
    assert restored.ticker == "AAPL"
    assert restored.model == "alpha"
    assert restored.persona == "momentum-bold"
    assert restored.arm == "debate"
    assert restored.round_index == 1
    assert restored.composition == "quad"
    assert restored.exposure == -0.25
    assert restored.confidence == 0.7
    assert restored.is_failure is False


def test_a_failed_generation_reaches_the_analysis_still_marked_as_one() -> None:
    # It is stored with a flat exposure and never dropped, so a lost failure marker
    # turns a crash into a deliberate flat position that nothing downstream can tell
    # apart from a considered one.
    crashed = decision(
        arm=Arm.DEBATE,
        round_index=1,
        composition="quad",
        exposure=0.0,
        confidence=0.0,
        failure=FailureMode.UNAVAILABLE,
    )

    frame = decisions_to_frame([crashed])
    (restored,) = frame_to_rows(frame)

    assert frame["failure"].tolist() == ["unavailable"]
    assert restored.is_failure is True


def test_the_independent_arms_null_composition_becomes_a_groupable_empty_string() -> None:
    # A null here would be dropped by every grouping keyed on composition, silently
    # deleting the control arm from the report.
    (restored,) = frame_to_rows(decisions_to_frame([decision(composition=None)]))

    assert restored.composition == NO_COMPOSITION


def test_an_arm_is_stored_as_its_value_rather_than_its_enum_repr() -> None:
    frame = decisions_to_frame([decision(arm=Arm.DEBATE_PLACEBO)])

    assert frame["arm"].tolist() == ["debate_placebo"]


def test_no_decisions_produces_an_empty_frame_with_the_full_schema() -> None:
    frame = decisions_to_frame([])

    assert list(frame.columns) == list(REQUIRED_COLUMNS)
    assert frame_to_rows(frame) == ()


def test_rows_come_back_in_canonical_order_whatever_order_they_were_written_in() -> None:
    shuffled = frame_of(
        row(on=NEXT_DAY, ticker="XOM", model="beta"),
        row(on=DAY, ticker="XOM", model="alpha"),
        row(on=DAY, ticker="AAPL", model="beta"),
        row(on=DAY, ticker="AAPL", model="alpha"),
    )

    rows = frame_to_rows(shuffled)

    assert [(r.decision_date, r.ticker, r.model) for r in rows] == [
        (DAY, "AAPL", "alpha"),
        (DAY, "AAPL", "beta"),
        (DAY, "XOM", "alpha"),
        (NEXT_DAY, "XOM", "beta"),
    ]


def test_a_missing_column_is_named_rather_than_surfacing_three_modules_later() -> None:
    frame = frame_of(row()).drop(columns=["confidence"])

    with pytest.raises(ValueError, match="confidence"):
        frame_to_rows(frame)


@pytest.mark.parametrize(
    "stored",
    [DAY, pd.Timestamp(DAY), np.datetime64("2022-03-01"), "2022-03-01"],
    ids=["date", "timestamp", "datetime64", "string"],
)
def test_every_way_a_date_can_be_stored_reads_back_as_the_same_plain_date(
    stored: Any,
) -> None:
    frame = frame_of(row())
    frame["decision_date"] = [stored]

    (restored,) = frame_to_rows(frame)

    assert restored.decision_date == DAY
    assert type(restored.decision_date) is date


def straight_line_opens() -> pd.DataFrame:
    """Four sessions whose open-to-open moves are all different and all exact."""
    return pd.DataFrame(
        {"AAA": [100.0, 200.0, 250.0, 500.0]},
        index=pd.bdate_range("2022-03-01", periods=4),
    )


def test_a_decision_earns_the_open_to_open_move_two_sessions_out() -> None:
    # Decide at the close of the first session, fill at the second open, hold to the
    # third: 250/200 - 1. The move the panel must never store against that date is
    # 200/100 - 1, which had already happened when the decision was made.
    returns = forward_returns(straight_line_opens())

    assert returns["AAA"].iloc[0] == pytest.approx(0.25)
    assert returns["AAA"].iloc[1] == pytest.approx(1.0)


def test_a_forward_return_is_the_period_the_engine_would_actually_have_earned() -> None:
    # The definition lives in two places -- here and in the fill rule -- so it is
    # pinned against the engine rather than restated.
    opens = opens_frame(synthetic_prices(tickers=("AAA",), sessions=30, seed=7))
    returns = forward_returns(opens)
    decided_on = opens.index[5]

    result = run_ticker(
        ticker="AAA",
        targets=pd.Series([1.0], index=[decided_on]),
        opens=opens["AAA"],
        cost_bps=0.0,
        rebalance_threshold=0.0,
    )
    period = list(result.dates).index(opens.index[6])

    assert result.position[period - 1] == 0.0
    assert result.position[period] == 1.0
    assert result.period_return[period] == pytest.approx(returns.loc[decided_on, "AAA"])


def test_the_last_two_sessions_have_no_closed_period_to_be_scored_against() -> None:
    returns = forward_returns(straight_line_opens())

    assert returns["AAA"].isna().tolist() == [False, False, True, True]
    assert len(forward_returns_lookup(returns)) == 2


def test_a_panel_handed_over_backwards_is_still_read_in_calendar_order() -> None:
    # Shifting an unsorted panel would take each date's return from whichever row
    # happened to sit two below it.
    opens = straight_line_opens()

    assert forward_returns(opens.iloc[::-1]).equals(forward_returns(opens))


def test_forward_returns_are_keyed_by_decision_point() -> None:
    returns = pd.DataFrame(
        {"AAPL": [0.01, -0.02], "XOM": [0.03, 0.04]},
        index=pd.to_datetime([DAY, NEXT_DAY]),
    )

    lookup = forward_returns_lookup(returns)

    assert lookup[(DAY, "AAPL")] == 0.01
    assert lookup[(NEXT_DAY, "XOM")] == 0.04
    assert len(lookup) == 4


def test_a_returns_panel_indexed_by_plain_dates_keys_the_same_way() -> None:
    dated = pd.DataFrame({"AAPL": [0.01]}, index=[DAY])
    stamped = pd.DataFrame({"AAPL": [0.01]}, index=pd.to_datetime([DAY]))

    assert dict(forward_returns_lookup(dated)) == dict(forward_returns_lookup(stamped))


def test_a_returns_panel_indexed_by_date_strings_keys_the_same_way() -> None:
    written = pd.DataFrame({"AAPL": [0.01]}, index=["2022-03-01"])

    assert dict(forward_returns_lookup(written)) == {(DAY, "AAPL"): 0.01}


def test_an_index_that_is_not_a_date_at_all_is_refused() -> None:
    nonsense = pd.DataFrame({"AAPL": [0.01]}, index=[object()])

    with pytest.raises(TypeError, match="as a decision date"):
        forward_returns_lookup(nonsense)


def test_a_point_with_no_forward_return_is_absent_rather_than_nan() -> None:
    returns = pd.DataFrame({"AAPL": [0.01, float("nan")]}, index=pd.to_datetime([DAY, NEXT_DAY]))

    lookup = forward_returns_lookup(returns)

    assert (NEXT_DAY, "AAPL") not in lookup
    assert len(lookup) == 1

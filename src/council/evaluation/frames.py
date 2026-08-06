"""The row shape every question in this package is answered from.

Decisions land on disk as one flat frame. Each module here answers its question by
grouping those rows a different way, so the frame is converted once -- here -- into
frozen records in one deterministic order. What is left in each module is the
question rather than the reindexing.

Nothing in this package calls a model. These are the stored rows and nothing else.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Final

import numpy as np
import pandas as pd

from council.domain.signal import Decision, FailureMode

DECISION_DATE: Final = "decision_date"
TICKER: Final = "ticker"
MODEL: Final = "model"
PERSONA: Final = "persona"
ARM: Final = "arm"
ROUND_INDEX: Final = "round_index"
COMPOSITION: Final = "composition"
EXPOSURE: Final = "exposure"
CONFIDENCE: Final = "confidence"
FAILURE: Final = "failure"

REQUIRED_COLUMNS: Final[tuple[str, ...]] = (
    DECISION_DATE,
    TICKER,
    MODEL,
    PERSONA,
    ARM,
    ROUND_INDEX,
    COMPOSITION,
    EXPOSURE,
    CONFIDENCE,
    FAILURE,
)

NO_COMPOSITION: Final = ""
"""Stand-in for the independent arm's null composition.

Grouping on a column containing nulls drops those rows without saying anything,
which would delete the entire independent arm from any tally keyed on composition
and leave a plausible-looking result behind. The empty string groups.
"""

NO_FAILURE: Final = str(FailureMode.NONE)
"""How a decision that actually produced output is marked.

A failed decision point is stored with a flat exposure rather than dropped, so
without this column a crashed generation is indistinguishable from an agent that
deliberately went flat -- and in the debate arms, where round 1 is the round that
can fail, that reads as the agent abandoning its opening view. Anything other than
this exact marker counts as a failure, so a frame carrying a null here is excluded
and reported rather than quietly scored.
"""

AgentKey = tuple[str, str]
"""``(model, persona)`` -- who produced a row. Neither half identifies an agent alone:
the same model wears four personas and the same persona is worn by several models."""

PointKey = tuple[date, str]
"""``(decision_date, ticker)`` -- one decision point, at which every agent has a view."""

DebateKey = tuple[str, str, date, str]
"""``(composition, arm, decision_date, ticker)`` -- one conversation.

Rounds may only be compared inside this key. A round 1 belongs to the particular
group of agents that produced it and means nothing next to a different committee's
round 0.
"""


def debate_sort_key(key: DebateKey) -> tuple[date, str, str, str]:
    """Chronological ordering for conversations.

    The key itself leads with composition and arm, because that is what it groups
    on. Sorting on it directly would order a multi-arm frame arm-major, so anything
    that promises its records in date order sorts with this instead.
    """
    composition, arm, decision_date, ticker = key
    return (decision_date, ticker, composition, arm)


@dataclass(frozen=True, slots=True)
class DecisionRow:
    """One stored decision, reduced to the columns the analysis reads."""

    decision_date: date
    ticker: str
    model: str
    persona: str
    arm: str
    round_index: int
    composition: str
    exposure: float
    confidence: float
    failure: str

    @property
    def is_failure(self) -> bool:
        """Whether this row records a generation that produced nothing.

        Its exposure is a placeholder, so no question about what the agent *meant*
        may be answered from it.
        """
        return self.failure != NO_FAILURE

    @property
    def agent(self) -> AgentKey:
        return (self.model, self.persona)

    @property
    def point(self) -> PointKey:
        return (self.decision_date, self.ticker)

    @property
    def debate(self) -> DebateKey:
        return (self.composition, self.arm, self.decision_date, self.ticker)

    @property
    def sort_key(self) -> tuple[date, str, str, str, int, str, str]:
        return (
            self.decision_date,
            self.ticker,
            self.composition,
            self.arm,
            self.round_index,
            self.model,
            self.persona,
        )


def decisions_to_frame(decisions: Iterable[Decision]) -> pd.DataFrame:
    """Flatten stored decisions into the frame this package reads.

    Row order is the caller's; :func:`frame_to_rows` imposes the canonical order on
    the way back in, so a re-sorted parquet file cannot change any result.
    """
    records = [
        {
            DECISION_DATE: decision.decision_date,
            TICKER: decision.ticker,
            MODEL: decision.model,
            PERSONA: decision.persona,
            ARM: str(decision.arm),
            ROUND_INDEX: decision.round_index,
            COMPOSITION: decision.composition or NO_COMPOSITION,
            EXPOSURE: decision.exposure,
            CONFIDENCE: decision.confidence,
            FAILURE: str(decision.failure),
        }
        for decision in decisions
    ]
    frame = pd.DataFrame.from_records(records, columns=list(REQUIRED_COLUMNS))
    return frame.astype(
        {ROUND_INDEX: "int64", EXPOSURE: "float64", CONFIDENCE: "float64"}
    )


def frame_to_rows(frame: pd.DataFrame) -> tuple[DecisionRow, ...]:
    """Validate, normalise and order a stored frame.

    Raises:
        ValueError: if any column the analysis depends on is absent. Failing here
            beats a downstream ``KeyError`` three modules away from the cause.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"decision frame is missing columns: {', '.join(missing)}")
    if frame.empty:
        return ()

    dates = _as_dates(frame[DECISION_DATE])
    rows = [
        DecisionRow(
            decision_date=decision_date,
            ticker=str(ticker),
            model=str(model),
            persona=str(persona),
            arm=str(arm),
            round_index=int(round_index),
            composition=str(composition),
            exposure=float(exposure),
            confidence=float(confidence),
            failure=str(failure),
        )
        for decision_date, ticker, model, persona, arm, round_index, composition, exposure, confidence, failure in zip(  # noqa: E501
            dates,
            frame[TICKER],
            frame[MODEL],
            frame[PERSONA],
            frame[ARM],
            frame[ROUND_INDEX],
            frame[COMPOSITION].fillna(NO_COMPOSITION),
            frame[EXPOSURE],
            frame[CONFIDENCE],
            frame[FAILURE],
            strict=True,
        )
    ]
    return tuple(sorted(rows, key=lambda row: row.sort_key))


FILL_LAG: Final = 2
"""Sessions between a decision and the open that ends the period it earns.

Not a tunable. It is the engine's fill rule counted in sessions: decide at the
close of *t*, fill at the open of *t+1*, hold to the open of *t+2*.
"""


def forward_returns(opens: pd.DataFrame) -> pd.DataFrame:
    """The return each decision date goes on to earn, under the engine's fill rule.

    This is the one alignment that can invalidate the whole study, so it is a
    function rather than an instruction to the caller. A decision made at the close
    of day *t* is filled at the open of *t+1* and held to the open of *t+2*, so the
    value stored against *t* is that open-to-open return -- exactly the period
    :func:`council.backtest.engine.run_ticker` would have earned it over. The two
    definitions are pinned against each other by test, so neither can drift.

    The obvious one-liner, ``opens.pct_change()``, stores against *t* a move that had
    already happened by the close of *t*. Scored with that panel every agent here
    looks prescient, and nothing raises.

    Args:
        opens: opening prices, one column per ticker, indexed by session.

    Returns:
        The same shape. The final two sessions are NaN: their period has not closed
        yet, and :func:`forward_returns_lookup` drops them.
    """
    # Sorted here rather than assumed: on a panel handed over out of order the shift
    # would take each date's return from whichever row happened to sit two below it.
    return opens.sort_index().astype(float).pct_change().shift(-FILL_LAG)


def forward_returns_lookup(returns: pd.DataFrame) -> Mapping[PointKey, float]:
    """Index a forward-return panel by decision point.

    Args:
        returns: the output of :func:`forward_returns` -- index is the date a
            decision was *made*, columns are tickers, and each value is the return
            that decision went on to earn. Nothing here can check that alignment, so
            build the panel with that function rather than assembling one.

    Missing values are dropped, so a decision point with no return is skipped by the
    caller rather than scored against a NaN.
    """
    lookup: dict[PointKey, float] = {}
    for ticker in returns.columns:
        column = returns[ticker].dropna()
        for index, value in column.items():
            lookup[(_as_date(index), str(ticker))] = float(value)
    return lookup


def _as_dates(column: pd.Series[Any]) -> list[date]:
    dates: list[date] = pd.to_datetime(column).dt.date.tolist()
    return dates


def _as_date(value: object) -> date:
    # ``datetime`` and ``pd.Timestamp`` are both subclasses of ``date``, and both
    # compare unequal to the plain date they represent. Narrowing first keeps a
    # parquet-backed index and a hand-built one producing the same dictionary keys.
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str | np.datetime64):
        return pd.Timestamp(value).date()
    raise TypeError(f"cannot read {value!r} as a decision date")

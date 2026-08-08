"""The sweep's pre-flight, against the draw it claims to mirror.

:func:`~council.debate.sweep.has_donor` exists so a placebo point with no usable
donor is skipped *before* its opening round is generated. Its docstring promises it
applies exactly the test :func:`~council.debate.placebo.select_placebo_point`
applies, minimum gap included, and spells out what drift costs: the pre-flight
passes, the sweep commits to the point, the real draw raises a plain ``ValueError``
that ``except NoPeersError`` does not catch, the sweep exits, and the group's
uncheckpointed rows are lost.

So the contract is asserted as a property over a whole calendar rather than on a
hand-picked day: for every point and every gap, admitted must mean servable.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from council.config import get_settings
from council.debate.placebo import PlaceboPool, select_placebo_point
from council.debate.sweep import RATIONALE, has_donor, placebo_pool_for
from council.domain.signal import Arm
from council.evaluation.frames import NO_FAILURE, PointKey
from helpers_debate import TICKER, committee, placebo_pool

CALENDAR: tuple[date, ...] = tuple(
    date(2022, 1, day) for day in (3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18)
)
"""Twelve sessions: deep enough that a gap of three separates admitted points from
refused ones, and shallow enough that the configured gap of sixty refuses all of
them."""

COMPOSITION = "rotation-0"


def calendar_pool() -> PlaceboPool:
    return placebo_pool(committee(), days=CALENDAR)


def is_servable(pool: PlaceboPool, point: PointKey, **gap: int) -> bool:
    """Whether the real draw would produce a donor rather than raise."""
    try:
        select_placebo_point(pool=pool, point=point, composition=COMPOSITION, seed=1, **gap)
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("gap", [0, 1, 3, 60])
def test_the_preflight_admits_exactly_the_points_the_draw_can_serve(gap: int) -> None:
    pool = calendar_pool()
    points = sorted(pool)

    verdicts = [
        (point, has_donor(pool, point, min_gap=gap), is_servable(pool, point, min_gap=gap))
        for point in points
    ]

    assert [point for point, admitted, _ in verdicts if admitted] == [
        point for point, _, servable in verdicts if servable
    ]


def test_the_calendar_separates_the_gaps_rather_than_answering_them_all_alike() -> None:
    # Without this the property above passes on a pool where every gap admits
    # everything, which is the shape a mirror bug hides in.
    pool = calendar_pool()
    admitted = {
        gap: sum(1 for point in pool if has_donor(pool, point, min_gap=gap))
        for gap in (0, 1, 3, 60)
    }

    assert admitted == {0: 11, 1: 11, 3: 9, 60: 0}


def test_the_preflight_defaults_to_the_gap_the_draw_defaults_to() -> None:
    # The two defaults diverging is the same drift the docstring warns about,
    # written into the signature: a caller omitting the keyword would pre-flight
    # at no gap while the draw enforced the configured sixty.
    pool = calendar_pool()
    configured = get_settings().placebo_min_gap_sessions
    assert configured > 0, "a gap of zero would make this test vacuous"

    for point in sorted(pool):
        assert has_donor(pool, point) is is_servable(pool, point)


def _independent_row(*, on: date, model: str, persona: str, exposure: float) -> dict[str, Any]:
    return {
        "decision_date": on,
        "ticker": TICKER,
        "model": model,
        "persona": persona,
        "arm": str(Arm.INDEPENDENT),
        "round_index": 0,
        "composition": "",
        "exposure": exposure,
        "confidence": 0.5,
        "failure": NO_FAILURE,
        RATIONALE: f"a view from {on.isoformat()}",
    }


def test_the_pool_holds_every_session_and_not_the_contested_ones_only() -> None:
    # The docstring argues the pool must be the control's sessions rather than the
    # debated ones, and callers reason about how deep the pool is from that. A day
    # the committee agreed on is still a donor, and anything that counts decision
    # dates as contested points counts the wrong denominator.
    seats = committee().seats
    settled_day, split_day = CALENDAR[0], CALENDAR[1]
    decisions = pd.DataFrame(
        [
            _independent_row(on=settled_day, model=seat.model, persona=seat.persona.name,
                             exposure=0.5)
            for seat in seats
        ]
        + [
            _independent_row(on=split_day, model=seat.model, persona=seat.persona.name,
                             exposure=0.5 if index % 2 else -0.5)
            for index, seat in enumerate(seats)
        ]
    )

    pool = placebo_pool_for(decisions, composition=committee())

    assert sorted(pool) == [(settled_day, TICKER), (split_day, TICKER)]
    assert len(pool[(settled_day, TICKER)]) == len(seats)

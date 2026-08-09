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

from council.app.curves import arms_in
from council.app.panels import _rounds_in
from council.app.transcripts import read_transcripts
from council.config import PROJECT_ROOT, get_settings
from council.debate.placebo import PlaceboPool, select_placebo_point
from council.debate.protocol import DEFAULT_REBUTTAL_ROUNDS, OPENING_ROUND
from council.debate.sweep import RATIONALE, _check_cap, has_donor, placebo_pool_for
from council.domain.signal import Arm
from council.evaluation.frames import NO_FAILURE, DecisionRow, PointKey
from council.evaluation.persuasion import REBUTTAL_ROUND
from helpers_debate import TICKER, committee, placebo_pool

CALENDAR: tuple[date, ...] = tuple(
    date(2022, 1, day) for day in (3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 17, 18)
)
"""Twelve sessions: deep enough that a gap of three separates admitted points from
refused ones, and shallow enough that the configured gap of sixty refuses all of
them."""

COMPOSITION = "rotation-0"

SEATS = committee().size
"""Seats a donor day must hold a view for. Both the pre-flight and the draw require
a complete donor, so the two are mirrored on this as well as on the gap."""


def calendar_pool() -> PlaceboPool:
    return placebo_pool(committee(), days=CALENDAR)


def is_servable(pool: PlaceboPool, point: PointKey, **gap: int) -> bool:
    """Whether the real draw would produce a donor rather than raise."""
    try:
        select_placebo_point(
            pool=pool,
            point=point,
            composition=COMPOSITION,
            required_seats=SEATS,
            seed=1,
            **gap,
        )
    except ValueError:
        return False
    return True


@pytest.mark.parametrize("gap", [0, 1, 3, 60])
def test_the_preflight_admits_exactly_the_points_the_draw_can_serve(gap: int) -> None:
    pool = calendar_pool()
    points = sorted(pool)

    verdicts = [
        (
            point,
            has_donor(pool, point, required_seats=SEATS, min_gap=gap),
            is_servable(pool, point, min_gap=gap),
        )
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
        gap: sum(1 for point in pool if has_donor(pool, point, required_seats=SEATS, min_gap=gap))
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
        assert has_donor(pool, point, required_seats=SEATS) is is_servable(pool, point)


# -- a donor must be the whole committee, not merely a non-empty day --------------


def _short_pool(missing_seats: int) -> tuple[PlaceboPool, PointKey]:
    """A pool whose only candidate day lost ``missing_seats`` generations."""
    table = committee()
    donor_day, decision_day = CALENDAR[0], CALENDAR[1]
    full = placebo_pool(table, days=(donor_day,))[(donor_day, TICKER)]
    return (
        {(donor_day, TICKER): tuple(full[: table.size - missing_seats])},
        (decision_day, TICKER),
    )


@pytest.mark.parametrize("missing_seats", [1, 2, 3])
def test_a_donor_day_that_lost_a_seat_is_not_a_donor(missing_seats: int) -> None:
    # Both modules promise the placebo leaves every agent the same number of peers
    # the real arm gives. Only the extra-seat direction was enforced: `seated_views`
    # refuses a foreign chair, while a *missing* chair passed a bare truthiness test.
    # An agent in the placebo arm then answered fewer peers than in the debate arm --
    # a second manipulation riding along with the intended one.
    pool, point = _short_pool(missing_seats)

    assert has_donor(pool, point, required_seats=SEATS, min_gap=1) is False
    assert is_servable(pool, point, min_gap=1) is False


def test_the_refusal_says_how_many_seats_were_wanted() -> None:
    pool, point = _short_pool(1)

    with pytest.raises(ValueError, match=rf"all {SEATS} seat"):
        select_placebo_point(
            pool=pool,
            point=point,
            composition=COMPOSITION,
            required_seats=SEATS,
            seed=1,
            min_gap=1,
        )


def test_a_donor_down_to_one_view_no_longer_reaches_the_draw() -> None:
    # The sharpest case: `peers_for` raises NoPeersError on a single donor view, and
    # `_Sweep.hold` books that as an abandoned conversation -- so the placebo arm
    # silently covered fewer decision points than the two arms it is differenced
    # against, which is the coverage effect the whole design is built to avoid.
    pool, point = _short_pool(SEATS - 1)
    assert len(next(iter(pool.values()))) == 1

    assert has_donor(pool, point, required_seats=SEATS, min_gap=1) is False


def test_a_complete_donor_day_is_still_admitted() -> None:
    # The other half, so the fix cannot be "refuse everything".
    table = committee()
    donor_day, decision_day = CALENDAR[0], CALENDAR[1]
    pool = placebo_pool(table, days=(donor_day,))

    assert has_donor(pool, (decision_day, TICKER), required_seats=SEATS, min_gap=1) is True
    assert is_servable(pool, (decision_day, TICKER), min_gap=1) is True


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


# -- the cap refusal is the checklist for raising the cap -------------------------


def _row_at(round_index: int) -> dict[str, Any]:
    return {
        "decision_date": CALENDAR[0],
        "ticker": TICKER,
        "model": "alpha",
        "persona": "momentum-bold",
        "arm": str(Arm.DEBATE),
        "round_index": round_index,
        "composition": COMPOSITION,
        "exposure": 0.1,
        "confidence": 0.5,
        "failure": NO_FAILURE,
        RATIONALE: "a view",
    }


CAP_CONSUMERS: tuple[str, ...] = (
    "_Sweep.group",
    "arm_exposures",
    "_arm_reports",
    "persuasion",
    "read_transcripts",
    "arms_in",
    "_rounds_in",
)
"""Every site that reads a fixed round index, as `_check_cap` has to name them.

Naming three of the seven is not a cosmetic omission: `_check_cap`'s refusal is the
instruction a next engineer follows, so a consumer it leaves out is one that stays
coupled while the instruction reads as complete.
"""


def test_every_named_cap_consumer_really_assumes_one_rebuttal_round() -> None:
    # Arrange -- the checklist is only worth pinning if each entry is coupled in
    # the shipped code, so the couplings are asserted before the wording is.
    above_cap = DEFAULT_REBUTTAL_ROUNDS + 1

    # Act & Assert -- the transcript panel refuses the extra round outright.
    with pytest.raises(ValueError, match="past the protocol's two rounds"):
        read_transcripts(pd.DataFrame([_row_at(above_cap)]))

    # The dashboard offers no round above the first rebuttal.
    assert _rounds_in(str(Arm.DEBATE)) == [OPENING_ROUND, REBUTTAL_ROUND]

    # And `arms_in` qualifies a treatment arm on the literal round 1 rather than on
    # the run's cap: an arm holding rounds 0-1 of a cap-2 run reads as present while
    # `arm_exposures` would look at index 2 and find nothing.
    rows = tuple(
        DecisionRow(
            decision_date=CALENDAR[0],
            ticker=TICKER,
            model="alpha",
            persona="momentum-bold",
            arm=arm,
            round_index=index,
            composition="" if arm == str(Arm.INDEPENDENT) else COMPOSITION,
            exposure=0.1,
            confidence=0.5,
            failure=NO_FAILURE,
        )
        for arm, index in (
            (str(Arm.INDEPENDENT), OPENING_ROUND),
            (str(Arm.DEBATE), OPENING_ROUND),
            (str(Arm.DEBATE), REBUTTAL_ROUND),
        )
    )
    assert arms_in(rows) == (Arm.INDEPENDENT, Arm.DEBATE)


def test_the_cap_refusal_names_every_consumer_rather_than_the_first_three() -> None:
    # Arrange
    with pytest.raises(ValueError) as refused:
        _check_cap(DEFAULT_REBUTTAL_ROUNDS + 1)

    # Assert -- both the refusal a caller reads and the docstring a next engineer
    # reads have to carry the whole list.
    message = str(refused.value)
    docstring = _check_cap.__doc__ or ""
    assert [name for name in CAP_CONSUMERS if name not in message] == []
    assert [name for name in CAP_CONSUMERS if name not in docstring] == []


def test_no_other_site_in_the_debate_path_still_says_three_consumers() -> None:
    # `_check_cap` was expanded to seven and pinned by test; the two places that tell
    # a next engineer what raising the cap costs were not. The protocol module
    # docstring is the first thing a reader of the protocol sees, so the instruction
    # read as complete at three of seven -- the exact failure `_check_cap` says it
    # exists to prevent.
    protocol = (PROJECT_ROOT / "src" / "council" / "debate" / "protocol.py").read_text(
        encoding="utf-8"
    )
    sweep = (PROJECT_ROOT / "src" / "council" / "debate" / "sweep.py").read_text(
        encoding="utf-8"
    )

    assert "needs three\nconsumers" not in protocol
    assert "three consumers" not in protocol
    assert "_check_cap" in protocol
    assert "three consumers are not" not in sweep
    assert "seven consumers are not" in sweep


def test_the_protocol_does_not_call_the_cap_setting_an_unread_fallback() -> None:
    # `settings.max_debate_rounds` sets the sweep's cap, the plan's arithmetic and
    # the round index the treatment arm is scored and calibrated at. The protocol
    # docstring said it was read only by a fallback the sweep always overrides, and
    # `run_debate_arms`' own Args block 400 lines away says the opposite.
    protocol = (PROJECT_ROOT / "src" / "council" / "debate" / "protocol.py").read_text(
        encoding="utf-8"
    )
    resolvers = (
        (
            PROJECT_ROOT / "src" / "council" / "debate" / "sweep.py",
            "run_debate_arms",
            "council.debate.sweep",
        ),
        (PROJECT_ROOT / "src" / "council" / "planning.py", "plan_experiment", "council.planning"),
        (PROJECT_ROOT / "src" / "council" / "scoring.py", "evaluate_experiment", "council.scoring"),
    )

    assert "is read only by\nthis module's fallback" not in protocol
    assert "read only by" not in protocol
    for path, function, module in resolvers:
        body = path.read_text(encoding="utf-8").split(f"def {function}")[1]
        assert "settings.max_debate_rounds" in body, function
        assert module in protocol, module


def test_the_shipped_cap_is_accepted() -> None:
    # The other half, so the fix cannot be "refuse everything".
    _check_cap(DEFAULT_REBUTTAL_ROUNDS)


def test_the_debate_report_says_its_counters_are_pooled_over_the_arms() -> None:
    # The docstring said the counters exist so that "a treatment arm that quietly
    # covers fewer points than its control" is visible, and that they are "reported
    # rather than logged". The opposite ships: `run_debate_arms` merges every
    # composition and every arm into one `DebateReport`, the class carries no arm or
    # composition field, and the CLI prints one pooled line -- so an operator cannot
    # tell a placebo donor skip from a whole-round failure in the debate arm. The
    # per-arm detail is exactly the `_LOG.warning` lines the docstring dismissed.
    from dataclasses import fields

    from council.debate.sweep import DebateReport

    sweep = (PROJECT_ROOT / "src" / "council" / "debate" / "sweep.py").read_text(
        encoding="utf-8"
    )
    documented = " ".join((DebateReport.__doc__ or "").split())
    names = {field.name for field in fields(DebateReport)}

    assert not names & {"arm", "composition"}
    assert "report = report.merge( await sweep.group(" in " ".join(sweep.split())
    assert "reported rather than logged" not in documented
    assert "Pooled over every composition and every arm." in documented
    assert "_LOG.warning" in documented
    assert "council.app.tables.coverage_note" in documented

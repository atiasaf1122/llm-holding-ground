"""Reading one conversation back off disk.

This is the panel a sceptical reader goes to, so what is asserted here is that
the prose on the page belongs to the turn it is printed under, that a
conversation missing half of itself says so, and that the point offered first is
the one where the agents were furthest apart.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from council.app.transcripts import (
    rationale_lookup,
    read_transcripts,
    seat_label,
    transcript_table,
)
from helpers_app import (
    COMMITTEE,
    DAY,
    NEXT_DAY,
    OPENING,
    REBUTTAL,
    frame_of,
    independent,
    stored,
)

TICKER = "AAA"


def turns(
    *,
    model: str = "alpha",
    opening: float,
    final: float,
    said: str = "opening view",
    then: str = "final view",
    on: date = DAY,
    arm: str = "debate",
    confidence: float = 0.9,
) -> tuple[dict[str, Any], ...]:
    """One agent's two rounds, each carrying what it said in that round."""
    persona = f"{model}-persona"
    return (
        stored(
            model=model, persona=persona, arm=arm, ticker=TICKER, on=on,
            round_index=OPENING, exposure=opening, confidence=confidence, rationale=said,
        ),
        stored(
            model=model, persona=persona, arm=arm, ticker=TICKER, on=on,
            round_index=REBUTTAL, exposure=final, confidence=confidence, rationale=then,
        ),
    )


def debate() -> pd.DataFrame:
    """Two agents opening a full unit apart; one of them folds."""
    return frame_of(
        *turns(model="alpha", opening=-0.5, final=0.4, said="fade it", then="on reflection, join"),
        *turns(model="beta", opening=0.5, final=0.5, said="join it", then="still join"),
    )


# -- joining the prose back on -----------------------------------------------


def test_a_rationale_is_keyed_by_the_decision_the_store_would_recognise() -> None:
    lookup = rationale_lookup(frame_of(*turns(opening=0.0, final=0.5)))

    assert lookup[(DAY, TICKER, "alpha", "alpha-persona", "debate", OPENING, COMMITTEE)] == (
        "opening view"
    )


def test_two_rows_for_one_decision_raise_rather_than_showing_whichever_sorted_last() -> None:
    row = stored(rationale="first")
    frame = frame_of(row, {**row, "rationale": "second"})

    with pytest.raises(ValueError, match="share the decision key"):
        rationale_lookup(frame)


def test_a_frame_without_the_prose_column_is_refused() -> None:
    frame = frame_of(*turns(opening=0.0, final=0.5)).drop(columns=["rationale"])

    with pytest.raises(ValueError, match="rationale"):
        rationale_lookup(frame)


def test_an_empty_run_has_nothing_to_look_up() -> None:
    assert rationale_lookup(frame_of()) == {}


# -- one conversation --------------------------------------------------------


def test_each_seat_carries_what_it_said_before_and_after() -> None:
    transcript = read_transcripts(debate())[0]
    alpha = transcript.seats[0]

    assert seat_label(alpha) == "alpha / alpha-persona"
    assert alpha.opening_rationale == "fade it"
    assert alpha.final_rationale == "on reflection, join"


def test_a_seat_that_moved_past_the_threshold_is_marked_as_having_shifted() -> None:
    transcript = read_transcripts(debate(), threshold=0.2)[0]
    alpha, beta = transcript.seats

    assert alpha.shift.shifted
    assert not beta.shift.shifted


def test_a_seat_that_came_out_the_other_side_is_marked_as_a_reversal() -> None:
    transcript = read_transcripts(debate())[0]

    assert transcript.seats[0].shift.reversed_sign


def test_the_committee_before_and_after_are_both_available_to_the_panel() -> None:
    transcript = read_transcripts(debate())[0]

    assert transcript.opening_mean == pytest.approx(0.0)
    assert transcript.final_mean == pytest.approx(0.45)


def test_the_largest_single_move_is_the_furthest_any_one_seat_travelled() -> None:
    transcript = read_transcripts(debate())[0]

    assert transcript.largest_move == pytest.approx(0.9)


def test_the_opening_spread_is_the_dispersion_that_made_the_point_worth_debating() -> None:
    transcript = read_transcripts(debate())[0]

    assert transcript.opening_std == pytest.approx(0.5)
    assert transcript.is_split


def test_agents_who_disagreed_only_about_size_are_not_recorded_as_split() -> None:
    frame = frame_of(
        *turns(model="alpha", opening=0.1, final=0.1),
        *turns(model="beta", opening=0.9, final=0.9),
    )

    assert not read_transcripts(frame)[0].is_split


def test_a_seat_missing_its_second_round_is_named_rather_than_dropped_silently() -> None:
    # A transcript showing two of four agents must not read as a debate between two.
    frame = frame_of(
        *turns(model="alpha", opening=-0.5, final=0.4),
        stored(
            model="beta", persona="beta-persona", ticker=TICKER, round_index=OPENING, exposure=0.5
        ),
    )

    transcript = read_transcripts(frame)[0]

    assert [seat.model for seat in transcript.seats] == ["alpha"]
    assert transcript.silent == (("beta", "beta-persona"),)


def test_a_silent_seat_still_counts_toward_the_opening_spread() -> None:
    # It is the spread that decided the debate was worth running, and that
    # decision was made on every opening view, not only the surviving ones.
    frame = frame_of(
        *turns(model="alpha", opening=-0.5, final=0.4),
        stored(
            model="beta", persona="beta-persona", ticker=TICKER, round_index=OPENING, exposure=0.5
        ),
    )

    assert read_transcripts(frame)[0].opening_std == pytest.approx(0.5)


def test_the_committee_means_cover_the_speaking_seats_and_the_spread_covers_every_view() -> None:
    # The two metrics deliberately describe different populations: a
    # before-and-after delta computed across two different committees would not
    # be a delta at all, and the spread is what selected the point for debate.
    # The panel therefore names the population in each metric's label, and this
    # pins the difference so that neither half can be "fixed" into the other.
    frame = frame_of(
        *turns(model="alpha", opening=-0.5, final=0.4),
        stored(
            model="beta", persona="beta-persona", ticker=TICKER, round_index=OPENING, exposure=0.5
        ),
    )

    transcript = read_transcripts(frame)[0]

    assert transcript.opening_mean == pytest.approx(-0.5)
    assert transcript.opening_std == pytest.approx(0.5)


def test_a_failed_generation_is_flagged_so_its_placeholder_is_not_read_as_a_view() -> None:
    frame = frame_of(
        *turns(model="alpha", opening=-0.5, final=0.0)[:1],
        stored(
            model="alpha", persona="alpha-persona", ticker=TICKER, round_index=REBUTTAL,
            exposure=0.0, failure="truncated",
        ),
    )

    assert read_transcripts(frame)[0].seats[0].failed


# -- choosing between conversations ------------------------------------------


def test_the_widest_disagreement_is_offered_first() -> None:
    frame = frame_of(
        *turns(model="alpha", opening=0.0, final=0.0, on=NEXT_DAY),
        *turns(model="beta", opening=0.1, final=0.1, on=NEXT_DAY),
        *turns(model="alpha", opening=-0.5, final=0.4),
        *turns(model="beta", opening=0.5, final=0.5),
    )

    assert [item.key.decision_date for item in read_transcripts(frame)] == [DAY, NEXT_DAY]


def test_the_independent_arm_has_no_conversation_to_read() -> None:
    frame = frame_of(independent(rationale="alone"), *turns(opening=0.0, final=0.5))

    assert [item.key.arm for item in read_transcripts(frame)] == ["debate"]


def test_a_debate_and_its_placebo_are_two_conversations_and_not_one() -> None:
    frame = frame_of(
        *turns(model="alpha", opening=-0.5, final=0.4),
        *turns(model="alpha", opening=-0.5, final=-0.5, arm="debate_placebo"),
    )

    assert {item.key.arm for item in read_transcripts(frame)} == {"debate", "debate_placebo"}


def test_the_label_says_which_point_and_which_arm_it_belongs_to() -> None:
    label = read_transcripts(debate())[0].label

    assert str(DAY) in label
    assert "debate" in label
    assert "dispersion 0.50" in label


def test_a_round_past_the_paired_two_is_set_aside_rather_than_refused() -> None:
    # This raised, which took the whole transcript panel down on any artefact
    # holding a round 2 -- which is every artefact a six-round cap produces. The
    # panel shows the two rounds the primary statistic is computed over; the cost is
    # that a long conversation's "final" column is the agent's first answer to its
    # peers rather than its last word, which `read_transcripts` now says out loud.
    frame = frame_of(*turns(opening=0.0, final=0.5), stored(round_index=2))

    transcripts = read_transcripts(frame)

    assert len(transcripts) == 1
    assert [seat.shift.posterior_exposure for seat in transcripts[0].seats] == [0.5]
    assert transcripts[0].silent == ()


# -- the table under the transcript ------------------------------------------


def test_the_table_puts_each_seats_opening_view_beside_its_final_one() -> None:
    table = transcript_table(read_transcripts(debate())[0])

    assert table["model"].tolist() == ["alpha", "beta"]
    assert table["opening_exposure"].tolist() == [-0.5, 0.5]
    assert table["final_exposure"].tolist() == [0.4, 0.5]


# -- the header's dispersion is not the gate ---------------------------------


def test_the_opening_spread_is_not_the_dispersion_that_admitted_the_point() -> None:
    # `Transcript.opening_std` covers one committee's seats in one arm at round 0.
    # `pipeline.select_contested` gates on the spread over the *whole* independent
    # arm, every model at every persona -- so a uniform committee can open at zero
    # spread on a point admitted there by a directional split.
    from council.evaluation.dispersion import dispersion_by_point

    uniform = frame_of(
        *turns(model="alpha", opening=0.5, final=0.5),
        *turns(model="beta", opening=0.5, final=0.5),
    )
    control = frame_of(
        independent(model="alpha", persona="momentum-bold", ticker=TICKER, exposure=0.8),
        independent(model="alpha", persona="reversion-bold", ticker=TICKER, exposure=-0.8),
        independent(model="beta", persona="momentum-bold", ticker=TICKER, exposure=0.6),
        independent(model="beta", persona="reversion-bold", ticker=TICKER, exposure=-0.6),
    )

    transcript = read_transcripts(uniform)[0]
    gate = dispersion_by_point(control)[0]

    assert transcript.opening_std == 0.0
    assert transcript.is_split is False
    assert gate.is_split is True
    assert gate.exposure_std > 0.0


def test_the_docstring_does_not_call_it_the_quantity_that_gated_the_point() -> None:
    from council.config import PROJECT_ROOT

    module = (PROJECT_ROOT / "src" / "council" / "app" / "transcripts.py").read_text(
        encoding="utf-8"
    )
    field = module.split("opening_std: float")[1].split("is_split: bool")[0]

    assert "the quantity that decided the point was worth debating" not in field
    assert "It is not the dispersion that gated the point" in field
    assert "select_contested" in field

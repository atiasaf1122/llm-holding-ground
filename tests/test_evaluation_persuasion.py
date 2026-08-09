"""Pairing an opening view with the same agent's view after hearing its peers.

The pairing rules are where this can go quietly wrong. A round 1 from the placebo
arm subtracted from a round 0 of the real debate would produce a shift rate that
looks perfectly reasonable and measures nothing.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from council.config import get_settings
from council.domain.signal import FailureMode
from council.evaluation.persuasion import (
    failed_rows,
    shift_rate_by_confidence,
    shifts,
    unpaired_rows,
)
from helpers_decisions import DAY, NEXT_DAY, debate_pair, frame_of, row

THRESHOLD = 0.20
CRASHED = str(FailureMode.UNAVAILABLE)


# -- the pairing -------------------------------------------------------------


def test_an_agents_two_rounds_are_joined_into_one_shift() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.6, closing=0.1))

    (shift,) = shifts(frame, threshold=THRESHOLD)

    assert shift.prior_exposure == 0.6
    assert shift.posterior_exposure == 0.1
    assert shift.delta == pytest.approx(-0.5)
    assert shift.distance == pytest.approx(0.5)


def test_rounds_are_never_paired_across_compositions() -> None:
    # A conversation between four agents cannot be continued by a different committee.
    frame = frame_of(
        row(model="alpha", round_index=0, exposure=0.6, composition="quad"),
        row(model="alpha", round_index=1, exposure=0.1, composition="pair"),
    )

    assert shifts(frame, threshold=THRESHOLD) == ()
    assert len(unpaired_rows(frame)) == 2


def test_rounds_are_never_paired_across_arms() -> None:
    # The placebo arm's rebuttal must not be subtracted from the real debate's opening.
    frame = frame_of(
        row(model="alpha", round_index=0, exposure=0.6, arm="debate"),
        row(model="alpha", round_index=1, exposure=0.1, arm="debate_placebo"),
    )

    assert shifts(frame, threshold=THRESHOLD) == ()


def test_rounds_are_never_paired_across_agents() -> None:
    frame = frame_of(
        row(model="alpha", round_index=0, exposure=0.6),
        row(model="beta", round_index=1, exposure=0.1),
    )

    assert shifts(frame, threshold=THRESHOLD) == ()


def test_an_independent_decision_has_no_rebuttal_and_is_reported_as_unpaired() -> None:
    frame = frame_of(row(arm="independent", composition="", round_index=0, exposure=0.6))

    assert shifts(frame, threshold=THRESHOLD) == ()
    assert len(unpaired_rows(frame)) == 1


def test_an_empty_frame_yields_no_shifts() -> None:
    assert shifts(frame_of(), threshold=THRESHOLD) == ()
    assert unpaired_rows(frame_of()) == ()


def test_the_same_agent_twice_in_one_round_is_refused() -> None:
    frame = frame_of(
        row(model="alpha", round_index=0, exposure=0.6),
        row(model="alpha", round_index=0, exposure=-0.6),
    )

    with pytest.raises(ValueError, match="appears twice"):
        shifts(frame, threshold=THRESHOLD)


def test_a_third_round_is_refused_rather_than_silently_ignored() -> None:
    frame = frame_of(
        *debate_pair(model="alpha", opening=0.6, closing=0.1),
        row(model="alpha", round_index=2, exposure=0.0),
    )

    with pytest.raises(ValueError, match="past the protocol"):
        shifts(frame, threshold=THRESHOLD)


def test_shifts_come_back_in_a_stable_order() -> None:
    frame = frame_of(
        *debate_pair(model="beta", opening=0.1, closing=0.2, on=NEXT_DAY),
        *debate_pair(model="alpha", opening=0.1, closing=0.2, on=DAY),
        *debate_pair(model="beta", opening=0.1, closing=0.2, on=DAY),
    )

    assert [(s.decision_date, s.model) for s in shifts(frame, threshold=THRESHOLD)] == [
        (DAY, "alpha"),
        (DAY, "beta"),
        (NEXT_DAY, "beta"),
    ]


def test_the_promised_date_order_survives_a_frame_that_spans_two_arms() -> None:
    # The grouping key leads with arm, so sorting on it would put every placebo
    # record after every debate record and call the result chronological.
    placebo = [
        dict(record, arm="debate_placebo")
        for record in debate_pair(model="m", opening=0.0, closing=1.0, on=DAY)
    ]
    debate = list(debate_pair(model="m", opening=0.0, closing=1.0, on=NEXT_DAY))
    frame = frame_of(*placebo, *debate)

    assert [(s.decision_date, s.arm) for s in shifts(frame, threshold=THRESHOLD)] == [
        (DAY, "debate_placebo"),
        (NEXT_DAY, "debate"),
    ]


# -- generations that produced nothing ---------------------------------------


def test_a_crashed_rebuttal_is_not_read_as_abandoning_the_opening_view() -> None:
    # The failed round is stored flat by contract, so scoring it would record a
    # 0.9-exposure, 0.95-confidence view as having been talked out of existence --
    # and round 1 exists only in the debate arms, so every such phantom lands on the
    # treatment.
    frame = frame_of(
        row(round_index=0, exposure=0.9, confidence=0.95),
        row(round_index=1, exposure=0.0, confidence=0.0, failure=CRASHED),
    )

    assert shifts(frame, threshold=THRESHOLD) == ()
    assert [r.round_index for r in failed_rows(frame)] == [0, 1]


def test_a_crashed_opening_is_dropped_just_as_the_rebuttal_is() -> None:
    frame = frame_of(
        row(round_index=0, exposure=0.0, confidence=0.0, failure=CRASHED),
        row(round_index=1, exposure=0.9, confidence=0.95),
    )

    assert shifts(frame, threshold=THRESHOLD) == ()
    assert len(failed_rows(frame)) == 2


def test_a_crashed_pair_is_absent_from_every_confidence_band() -> None:
    frame = frame_of(
        row(round_index=0, exposure=0.9, confidence=0.95),
        row(round_index=1, exposure=0.0, confidence=0.0, failure=CRASHED),
    )

    report = shift_rate_by_confidence(shifts(frame, threshold=THRESHOLD))

    assert [band.count for band in report.bands] == [0, 0, 0, 0, 0]
    assert all(band.shift_rate is None for band in report.bands)


def test_a_debate_both_rounds_of_which_answered_is_untouched_by_the_exclusion() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.9, closing=0.0))

    assert len(shifts(frame, threshold=THRESHOLD)) == 1
    assert failed_rows(frame) == ()


def test_a_failure_with_no_partner_is_reported_as_unpaired_rather_than_twice() -> None:
    frame = frame_of(row(round_index=0, exposure=0.0, confidence=0.0, failure=CRASHED))

    assert len(unpaired_rows(frame)) == 1
    assert failed_rows(frame) == ()


# -- what counts as changing your mind ---------------------------------------


def test_a_move_of_exactly_the_threshold_counts_as_a_shift() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.5, closing=0.3))

    (shift,) = shifts(frame, threshold=THRESHOLD)

    assert shift.distance == pytest.approx(THRESHOLD)
    assert shift.shifted is True


def test_a_move_short_of_the_threshold_is_holding_your_ground() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.5, closing=0.31))

    (shift,) = shifts(frame, threshold=THRESHOLD)

    assert shift.shifted is False
    assert shift.changed_mind is False


def test_coming_out_the_other_side_is_a_reversal() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.4, closing=-0.1))

    (shift,) = shifts(frame, threshold=THRESHOLD)

    assert shift.reversed_sign is True


def test_a_tiny_reversal_counts_as_changing_your_mind_despite_the_threshold() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.02, closing=-0.02))

    (shift,) = shifts(frame, threshold=THRESHOLD)

    assert shift.shifted is False
    assert shift.reversed_sign is True
    assert shift.changed_mind is True


def test_retreating_to_flat_is_an_abandonment_but_not_a_reversal() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.8, closing=0.0))

    (shift,) = shifts(frame, threshold=THRESHOLD)

    assert shift.shifted is True
    assert shift.reversed_sign is False


def test_the_threshold_is_carried_on_the_record_that_was_judged_by_it() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.5, closing=0.3))

    (shift,) = shifts(frame)

    assert shift.threshold == get_settings().shift_threshold


# -- does confidence protect a position? -------------------------------------


def test_shifts_are_bucketed_by_the_confidence_held_before_the_debate() -> None:
    # Both agents restate a confidence of 0.9 afterwards; only the opening counts.
    frame = frame_of(
        *debate_pair(model="alpha", opening=0.8, closing=0.0, confidence=0.1),
        *debate_pair(model="beta", opening=0.8, closing=0.8, confidence=0.9),
    )

    bands = shift_rate_by_confidence(shifts(frame, threshold=THRESHOLD)).bands

    assert bands[0].count == 1
    assert bands[0].shift_rate == pytest.approx(1.0)
    assert bands[-1].count == 1
    assert bands[-1].shift_rate == pytest.approx(0.0)


def test_a_confidence_band_nobody_occupied_has_no_rate_rather_than_zero() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.8, closing=0.0, confidence=0.9))

    bands = shift_rate_by_confidence(shifts(frame, threshold=THRESHOLD)).bands

    assert bands[0].count == 0
    assert bands[0].shift_rate is None
    assert bands[0].reversal_rate is None
    assert bands[0].mean_distance is None


def test_the_mean_distance_of_a_band_averages_over_its_members() -> None:
    frame = frame_of(
        *debate_pair(model="alpha", opening=0.8, closing=0.0, confidence=0.9),
        *debate_pair(model="beta", opening=0.4, closing=0.0, confidence=0.9),
    )

    bands = shift_rate_by_confidence(shifts(frame, threshold=THRESHOLD)).bands

    assert bands[-1].count == 2
    assert bands[-1].mean_distance == pytest.approx(0.6)


def test_a_confidence_outside_the_supplied_bands_is_counted_as_skipped() -> None:
    # Bands with nothing in them and bands built from a record that was thrown away
    # look identical in a plot; the count is what tells them apart.
    frame = frame_of(*debate_pair(model="alpha", opening=0.8, closing=0.0, confidence=0.9))

    report = shift_rate_by_confidence(shifts(frame, threshold=THRESHOLD), edges=(0.0, 0.5))

    assert [band.count for band in report.bands] == [0]
    assert report.skipped_count == 1


def test_no_shifts_leaves_every_band_empty() -> None:
    report = shift_rate_by_confidence(())

    assert all(band.count == 0 for band in report.bands)
    assert all(band.shift_rate is None for band in report.bands)
    assert report.skipped_count == 0


def test_the_aggregate_carries_the_bar_its_records_were_judged_against() -> None:
    # Shift.threshold exists so a rate and its bar cannot drift apart in a written-up
    # table. Dropped at aggregation, the published rate has no bar beside it and the
    # drift is exactly as available as it was before the field was added.
    frame = frame_of(*debate_pair(model="alpha", opening=0.8, closing=0.0, confidence=0.9))

    report = shift_rate_by_confidence(shifts(frame, threshold=THRESHOLD))

    assert report.threshold == THRESHOLD


def test_a_rate_over_two_different_bars_refuses_rather_than_reporting_one_of_them() -> None:
    # An aggregate whose records disagree about what counted has no bar to publish,
    # and picking either would print a number under a definition half of it failed.
    frame = frame_of(*debate_pair(model="alpha", opening=0.8, closing=0.0, confidence=0.9))
    (shift,) = shifts(frame, threshold=THRESHOLD)
    mixed = (shift, replace(shift, threshold=0.5))

    with pytest.raises(ValueError, match="mixes bars"):
        shift_rate_by_confidence(mixed)


def test_an_aggregate_with_no_records_has_no_bar_to_carry() -> None:
    assert shift_rate_by_confidence(()).threshold is None


# -- the two denominators behind one rate ------------------------------------


def test_a_band_carries_the_distinct_decision_points_behind_its_observations() -> None:
    # `count` is the unit the statistic is declared over, and it is not a sample
    # size: two seats answering one point is two observations of one point. Reported
    # here rather than recounted by whichever reader wants it, so the CLI table,
    # results.json and the dashboard cannot disagree about the denominator.
    frame = frame_of(
        *debate_pair(model="alpha", opening=0.0, closing=0.9, confidence=0.9),
        *debate_pair(
            model="beta", persona="reversion-bold", opening=0.0, closing=0.9, confidence=0.9
        ),
        *debate_pair(model="alpha", on=NEXT_DAY, opening=0.0, closing=0.9, confidence=0.9),
    )

    (band,) = [
        entry
        for entry in shift_rate_by_confidence(shifts(frame, threshold=THRESHOLD)).bands
        if entry.count
    ]

    assert band.count == 3
    assert band.point_count == 2


def test_a_band_nobody_landed_in_counts_no_points() -> None:
    frame = frame_of(*debate_pair(model="alpha", opening=0.0, closing=0.9, confidence=0.9))

    bands = shift_rate_by_confidence(shifts(frame, threshold=THRESHOLD)).bands

    assert [band.point_count for band in bands if band.count == 0] == [0, 0, 0, 0]


def test_the_dashboard_no_longer_carries_its_own_copy_of_the_point_count() -> None:
    from council.config import PROJECT_ROOT

    tables = (PROJECT_ROOT / "src" / "council" / "app" / "tables.py").read_text(
        encoding="utf-8"
    )

    assert "_points_by_band" not in tables
    assert "band.point_count" in tables

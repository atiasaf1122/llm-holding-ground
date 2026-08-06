"""Whether a stated confidence predicts being right.

Frames here are built so the answer is obvious before the code runs: the confident
decisions are all correct and the diffident ones all wrong, or the reverse, or
neither. Anything subtler would test arithmetic rather than the claim.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from council.evaluation.calibration import calibrate, pearson, was_right
from helpers_decisions import DAY, NEXT_DAY, frame_of, row

RETURNS = {(DAY, "AAPL"): 0.02, (NEXT_DAY, "AAPL"): -0.02}


def call(
    exposure: float, confidence: float, *, on: date = DAY, model: str = "alpha"
) -> dict[str, Any]:
    return row(exposure=exposure, confidence=confidence, on=on, model=model)


# -- being right -------------------------------------------------------------


@pytest.mark.parametrize(
    ("exposure", "forward_return", "expected"),
    [
        (0.5, 0.02, True),
        (0.5, -0.02, False),
        (-0.5, -0.02, True),
        (-0.5, 0.02, False),
    ],
)
def test_a_directional_call_is_right_when_the_market_agreed(
    exposure: float, forward_return: float, expected: bool
) -> None:
    assert was_right(exposure, forward_return) is expected


def test_a_flat_exposure_expresses_no_direction_to_be_right_about() -> None:
    # Counting this as a miss would make cautious agents look wrong for declining.
    assert was_right(0.0, 0.02) is None


def test_a_market_that_did_not_move_gives_nothing_to_be_right_against() -> None:
    assert was_right(0.5, 0.0) is None


# -- the correlation ---------------------------------------------------------


def test_confidence_that_predicts_being_right_correlates_positively() -> None:
    frame = frame_of(
        call(0.5, 0.9, on=DAY, model="a"),
        call(0.5, 0.9, on=DAY, model="b"),
        call(0.5, 0.1, on=NEXT_DAY, model="a"),
        call(0.5, 0.1, on=NEXT_DAY, model="b"),
    )

    report = calibrate(frame, RETURNS)

    assert report.correlation == pytest.approx(1.0)
    assert report.hit_rate == pytest.approx(0.5)


def test_confidence_that_predicts_being_wrong_correlates_negatively() -> None:
    frame = frame_of(
        call(0.5, 0.1, on=DAY, model="a"),
        call(0.5, 0.9, on=NEXT_DAY, model="a"),
    )

    report = calibrate(frame, RETURNS)

    assert report.correlation == pytest.approx(-1.0)


def test_a_confidence_that_never_varies_has_no_correlation_rather_than_a_nan() -> None:
    frame = frame_of(
        call(0.5, 0.7, on=DAY, model="a"),
        call(0.5, 0.7, on=NEXT_DAY, model="a"),
    )

    report = calibrate(frame, RETURNS)

    assert report.correlation is None


def test_an_outcome_that_never_varies_has_no_correlation() -> None:
    frame = frame_of(
        call(0.5, 0.1, on=DAY, model="a"),
        call(0.5, 0.9, on=DAY, model="b"),
    )

    report = calibrate(frame, RETURNS)

    assert report.scored_count == 2
    assert report.correlation is None


def test_a_single_decision_cannot_produce_a_correlation() -> None:
    report = calibrate(frame_of(call(0.5, 0.9)), RETURNS)

    assert report.scored_count == 1
    assert report.correlation is None


def test_pearson_refuses_series_of_different_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        pearson([1.0, 2.0], [1.0])


def test_pearson_recovers_a_known_coefficient() -> None:
    assert pearson([1.0, 2.0, 3.0], [2.0, 4.0, 6.0]) == pytest.approx(1.0)
    assert pearson([1.0, 2.0, 3.0], [6.0, 4.0, 2.0]) == pytest.approx(-1.0)


# -- the buckets -------------------------------------------------------------


def test_decisions_land_in_the_band_their_confidence_falls_in() -> None:
    frame = frame_of(
        call(0.5, 0.05, model="a"),
        call(0.5, 0.25, model="b"),
        call(0.5, 0.95, model="c"),
    )

    report = calibrate(frame, RETURNS)

    assert [bucket.count for bucket in report.buckets] == [1, 1, 0, 0, 1]


def test_a_confidence_on_a_band_edge_falls_in_the_upper_band() -> None:
    report = calibrate(frame_of(call(0.5, 0.2)), RETURNS)

    assert [bucket.count for bucket in report.buckets] == [0, 1, 0, 0, 0]


def test_total_confidence_lands_in_the_last_band_rather_than_falling_off_the_end() -> None:
    report = calibrate(frame_of(call(0.5, 1.0)), RETURNS)

    assert [bucket.count for bucket in report.buckets] == [0, 0, 0, 0, 1]
    assert report.skipped_count == 0


def test_an_empty_band_has_no_hit_rate_rather_than_a_hit_rate_of_zero() -> None:
    report = calibrate(frame_of(call(0.5, 0.9)), RETURNS)

    assert report.buckets[0].count == 0
    assert report.buckets[0].hit_rate is None
    assert report.buckets[-1].hit_rate == pytest.approx(1.0)


# -- what is skipped, and said to be -----------------------------------------


def test_a_flat_decision_is_skipped_and_counted_as_skipped() -> None:
    report = calibrate(frame_of(call(0.0, 0.9)), RETURNS)

    assert report.scored_count == 0
    assert report.skipped_count == 1


def test_a_decision_with_no_forward_return_is_skipped() -> None:
    frame = frame_of(row(exposure=0.5, confidence=0.9, ticker="XOM"))

    report = calibrate(frame, RETURNS)

    assert report.scored_count == 0
    assert report.skipped_count == 1


def test_an_empty_frame_reports_nothing_rather_than_zero() -> None:
    report = calibrate(frame_of(), RETURNS)

    assert report.scored_count == 0
    assert report.correlation is None
    assert report.hit_rate is None
    assert all(bucket.hit_rate is None for bucket in report.buckets)

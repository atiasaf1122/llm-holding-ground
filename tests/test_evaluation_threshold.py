"""The bar that decides whether a movement counts.

The test that used to guard this picked one pair -- opening 0.5, closing 0.3 -- and
passed, because that happens to be one of the few grid pairs whose subtraction rounds
*up*. The property it was named for was false for most of the grid, and two published
findings rested on the difference. So this file sweeps the grid instead of sampling it.
"""

from __future__ import annotations

import inspect

import pytest

from council.debate.protocol import _agreed
from council.evaluation.persuasion import Shift
from council.evaluation.threshold import TOLERANCE, exceeds, meets, within

# Exposures are constrained to [-1, 1] and models answer overwhelmingly in multiples
# of 0.05, so this is the space the comparison actually operates in.
GRID = [round(-1.0 + step * 0.05, 2) for step in range(41)]
BAR = 0.20


def pairs_apart(distance: float) -> list[tuple[float, float]]:
    """Every ordered pair on the grid exactly ``distance`` apart, in decimal."""
    return [
        (opening, closing)
        for opening in GRID
        for closing in GRID
        if round(abs(opening - closing), 10) == distance
    ]


class TestTheBoundaryOverTheWholeGrid:
    def test_every_move_of_exactly_the_bar_counts(self) -> None:
        # The defect: a bare `>=` accepted 0.5->0.3 and rejected 0.3->0.1, which are
        # the same move. 118 of 273 such conversations were dropped in the first run.
        candidates = pairs_apart(BAR)
        assert candidates, "the grid must contain moves of exactly the bar"
        missed = [
            (opening, closing)
            for opening, closing in candidates
            if not meets(abs(opening - closing), BAR)
        ]
        assert not missed, f"{len(missed)} of {len(candidates)} exact moves were rejected"

    def test_a_bare_comparison_would_fail_this(self) -> None:
        # Pinning the defect itself, so that reverting the fix fails here loudly
        # rather than quietly changing a published number.
        rejected = [
            (opening, closing)
            for opening, closing in pairs_apart(BAR)
            if not abs(opening - closing) >= BAR
        ]
        assert rejected, (
            "if binary floating point ever represents this grid exactly, the fix is "
            "unnecessary and this test should be removed deliberately"
        )

    def test_every_move_short_of_the_bar_is_excluded(self) -> None:
        for opening, closing in pairs_apart(0.15):
            assert not meets(abs(opening - closing), BAR)

    def test_every_move_beyond_the_bar_counts(self) -> None:
        for opening, closing in pairs_apart(0.25):
            assert meets(abs(opening - closing), BAR)

    def test_exceeds_excludes_the_boundary_across_the_grid(self) -> None:
        # The counterpart convention. It must be uniform too: a "more than" reading
        # that admitted half the boundary would be the same defect wearing the other
        # wording.
        for opening, closing in pairs_apart(BAR):
            assert not exceeds(abs(opening - closing), BAR)
        for opening, closing in pairs_apart(0.25):
            assert exceeds(abs(opening - closing), BAR)

    def test_the_tolerance_does_not_admit_a_real_difference(self) -> None:
        # It must absorb representation error and nothing else. One grid step is
        # 5e-2; the tolerance is 1e-9.
        assert not meets(BAR - 0.01, BAR)
        assert TOLERANCE < 0.05 / 1000


def shift(opening: float, closing: float) -> Shift:
    return Shift(
        decision_date=__import__("datetime").date(2022, 3, 25),
        ticker="AAA",
        composition="rotation-0",
        arm="debate",
        model="m",
        persona="momentum-bold",
        prior_exposure=opening,
        posterior_exposure=closing,
        prior_confidence=0.6,
        posterior_confidence=0.6,
        threshold=BAR,
    )


class TestShiftUsesIt:
    @pytest.mark.parametrize(("opening", "closing"), pairs_apart(BAR))
    def test_a_shift_of_exactly_the_bar_is_recorded_anywhere_on_the_grid(
        self, opening: float, closing: float
    ) -> None:
        assert shift(opening, closing).shifted

    @pytest.mark.parametrize(("opening", "closing"), pairs_apart(0.15))
    def test_a_shift_short_of_the_bar_is_not(self, opening: float, closing: float) -> None:
        assert not shift(opening, closing).shifted


class TestWithinIsTheInclusiveAtMost:
    def test_every_gap_of_exactly_the_bar_is_inside_it(self) -> None:
        # The same boundary as `meets`, read from the other side: a spread *of* the
        # bar is agreement. `_agreed` used to spell this `meets(bar, distance)`,
        # which gives the right answer and reads as its opposite.
        candidates = pairs_apart(BAR)
        assert candidates, "the grid must contain gaps of exactly the bar"
        for low, high in candidates:
            assert within(abs(low - high), BAR)

    def test_every_gap_beyond_the_bar_is_outside_it(self) -> None:
        for low, high in pairs_apart(0.25):
            assert not within(abs(low - high), BAR)

    def test_every_gap_short_of_the_bar_is_inside_it(self) -> None:
        for low, high in pairs_apart(0.15):
            assert within(abs(low - high), BAR)

    def test_the_two_orders_are_not_the_same_question(self) -> None:
        # Why the argument order is worth a named function. Reversed, the call is
        # still true -- for a different reason and with the parameters lying about
        # which value is which.
        assert within(0.25, 0.20) is False
        assert meets(0.20, 0.25) is False
        assert within(0.20, 0.25) is True
        assert meets(0.25, 0.20) is True

    def test_within_is_the_complement_of_exceeds(self) -> None:
        for distance in (0.0, 0.15, BAR, 0.25, 1.0):
            assert within(distance, BAR) is not exceeds(distance, BAR)


def test_agreement_is_measured_with_within_and_not_with_a_reversed_meets() -> None:
    # The one function whose inversion turns "agreed" into "still apart", pinned
    # against the shipped source: every other call site in the package passes the
    # measured distance first and the configured bar second.
    source = inspect.getsource(_agreed)
    assert "within(max(exposures) - min(exposures), spread)" in source
    assert "meets(" not in source

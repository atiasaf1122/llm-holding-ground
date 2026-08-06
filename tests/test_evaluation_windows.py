"""Counting the windows the treatment won.

The point of the split is that a strategy carried by three good days in March
should show up as one window out of five rather than as a rising equity curve, so
the tests check the partition itself as hard as they check the count.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.evaluation.windows import compare_windows, compound, split_windows

FLAT = [0.0] * 10


# -- the partition -----------------------------------------------------------


def test_an_even_split_gives_windows_of_equal_length() -> None:
    assert split_windows(10, 5) == ((0, 2), (2, 4), (4, 6), (6, 8), (8, 10))


def test_an_uneven_split_lengthens_the_earlier_windows_rather_than_dropping_the_tail() -> None:
    # The tail is where a strategy that has stopped working shows it.
    assert split_windows(11, 5) == ((0, 3), (3, 5), (5, 7), (7, 9), (9, 11))


def test_every_period_lands_in_exactly_one_window() -> None:
    bounds = split_windows(23, 4)

    covered = [index for start, stop in bounds for index in range(start, stop)]

    assert covered == list(range(23))


def test_one_window_is_the_whole_period() -> None:
    assert split_windows(7, 1) == ((0, 7),)


def test_a_window_per_period_is_the_finest_legal_split() -> None:
    assert split_windows(3, 3) == ((0, 1), (1, 2), (2, 3))


def test_more_windows_than_periods_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot cut"):
        split_windows(3, 4)


def test_no_windows_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one window"):
        split_windows(10, 0)


# -- the scoring -------------------------------------------------------------


def test_a_windows_return_compounds_rather_than_sums() -> None:
    assert compound(np.array([0.1, 0.1])) == pytest.approx(0.21)


def test_an_empty_span_compounds_to_nothing() -> None:
    assert compound(np.array([], dtype=float)) == pytest.approx(0.0)


def test_the_treatment_wins_the_windows_it_earned_more_in() -> None:
    treatment = [0.10, 0.10, -0.05, -0.05]
    control = [0.01, 0.01, 0.01, 0.01]

    comparison = compare_windows(treatment, control, window_count=2)

    assert comparison.window_count == 2
    assert comparison.treatment_wins == 1
    assert comparison.summary == "1 of 2 windows"


def test_a_dead_heat_is_not_a_win() -> None:
    comparison = compare_windows(FLAT, FLAT, window_count=5)

    assert comparison.treatment_wins == 0
    assert comparison.ties == 5


def test_a_treatment_that_wins_everywhere_wins_every_window() -> None:
    comparison = compare_windows([0.01] * 10, FLAT, window_count=5)

    assert comparison.treatment_wins == 5
    assert comparison.ties == 0


def test_one_enormous_window_hides_what_five_windows_reveal() -> None:
    # All of the treatment's edge is in the first fifth of the sample.
    treatment = [0.5, 0.5] + [-0.01] * 8
    control = [0.0] * 10

    assert compare_windows(treatment, control, window_count=1).treatment_wins == 1
    assert compare_windows(treatment, control, window_count=5).treatment_wins == 1


def test_the_margin_reports_how_much_a_window_was_won_by() -> None:
    comparison = compare_windows([0.1, 0.1], [0.0, 0.0], window_count=1)

    (window,) = comparison.windows
    assert window.margin == pytest.approx(0.21)
    assert window.length == 2


# -- what it is given --------------------------------------------------------


def test_arms_covering_different_periods_are_refused() -> None:
    with pytest.raises(ValueError, match="same periods"):
        compare_windows([0.1, 0.2, 0.3], [0.1, 0.2], window_count=1)


def test_a_backtests_return_array_is_accepted_directly() -> None:
    comparison = compare_windows(
        np.array([0.01] * 6, dtype=float), np.zeros(6, dtype=float), window_count=3
    )

    assert comparison.treatment_wins == 3


def test_a_dated_return_series_is_accepted_directly() -> None:
    index = pd.bdate_range("2022-01-03", periods=6)
    treatment = pd.Series([0.01] * 6, index=index)
    control = pd.Series([0.0] * 6, index=index)

    assert compare_windows(treatment, control, window_count=3).treatment_wins == 3


def test_a_two_dimensional_input_is_refused() -> None:
    with pytest.raises(ValueError, match="flat series"):
        compare_windows(np.zeros((2, 3)), np.zeros((2, 3)), window_count=1)


def test_a_scalar_arm_is_refused_by_the_check_rather_than_by_its_error_message() -> None:
    # The length message indexes into the shape, so a scalar checked in the wrong
    # order raises IndexError from inside the explanation.
    with pytest.raises(ValueError, match="flat series"):
        compare_windows(0.01, np.zeros(3), window_count=1)  # type: ignore[arg-type]

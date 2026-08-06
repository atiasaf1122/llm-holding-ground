"""The rules that turn several views into one, and the degenerate committees.

The committee of one matters most here. The control arm is a single agent and the
treatment arm is four, so if the solo case took a different code path, a difference
between the arms could come from the aggregation rather than from the debate --
which is the one thing this project cannot afford.
"""

from __future__ import annotations

import pytest

from council.evaluation.aggregation import (
    RULE_NAMES,
    RULES,
    AggregationRule,
    direction_vote,
    mean,
    median,
)

# -- the committee of one ----------------------------------------------------


@pytest.mark.parametrize("rule_name", RULE_NAMES)
def test_every_rule_accepts_a_committee_of_one(rule_name: str) -> None:
    assert RULES[rule_name]([0.6]) in (0.6, 1.0)


def test_a_single_agents_mean_is_its_own_exposure() -> None:
    assert mean([0.6]) == 0.6


def test_a_single_agents_median_is_its_own_exposure() -> None:
    assert median([-0.35]) == -0.35


def test_a_single_agent_votes_its_own_direction_at_full_size() -> None:
    assert direction_vote([0.05]) == 1.0


# -- agreement and disagreement ----------------------------------------------


@pytest.mark.parametrize("rule_name", RULE_NAMES)
def test_identical_agents_agree_with_themselves(rule_name: str) -> None:
    unanimous = [0.5, 0.5, 0.5, 0.5]

    assert RULES[rule_name](unanimous) in (0.5, 1.0)


def test_mean_averages_the_committee() -> None:
    assert mean([1.0, 0.0, -1.0, 0.5]) == pytest.approx(0.125)


def test_median_of_an_even_committee_averages_the_middle_two() -> None:
    assert median([0.0, 0.2, 0.6, 1.0]) == pytest.approx(0.4)


def test_median_ignores_the_outlier_that_drags_the_mean() -> None:
    with_outlier = [0.3, 0.3, 0.3, -1.0]

    assert median(with_outlier) == pytest.approx(0.3)
    assert mean(with_outlier) == pytest.approx(-0.025)


# -- the vote ----------------------------------------------------------------


def test_a_majority_direction_is_taken_at_full_size() -> None:
    assert direction_vote([0.1, 0.05, -0.9]) == 1.0


def test_the_short_majority_is_taken_at_full_size_too() -> None:
    assert direction_vote([-0.1, -0.05, 0.9]) == -1.0


def test_an_evenly_split_committee_reaches_no_direction() -> None:
    assert direction_vote([0.8, 0.8, -0.1, -0.1]) == 0.0


def test_a_flat_agent_abstains_rather_than_voting_for_either_side() -> None:
    # Two flats and one long is a one-nil win, not a one-two loss.
    assert direction_vote([0.0, 0.0, 0.4]) == 1.0


def test_an_entirely_flat_committee_is_flat() -> None:
    assert direction_vote([0.0, 0.0, 0.0]) == 0.0


# -- the empty committee -----------------------------------------------------


@pytest.mark.parametrize("rule_name", RULE_NAMES)
def test_no_rule_invents_a_view_for_an_empty_committee(rule_name: str) -> None:
    rule: AggregationRule = RULES[rule_name]

    with pytest.raises(ValueError, match="empty committee"):
        rule([])


def test_the_rule_names_are_stable_and_sorted() -> None:
    assert RULE_NAMES == ("direction_vote", "mean", "median")

"""Who gave ground to whom, and the two ways that question is easy to get wrong.

Two agents meeting exactly in the middle must credit nobody -- otherwise every
polite pair manufactures two influence events and the matrix fills with a
convergence that had no author. The mirror error is an agent that stands still
while its counterpart storms off being recorded as having capitulated.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from council.config import get_settings
from council.evaluation.influence import _debates, concessions, influence_matrix
from helpers_decisions import DAY, NEXT_DAY, debate_pair, frame_of, row

BAR = 0.20


def debate(*agents: tuple[dict[str, Any], dict[str, Any]]) -> pd.DataFrame:
    return frame_of(*[record for pair in agents for record in pair])


# -- the definition ----------------------------------------------------------


def test_an_agent_that_moves_while_its_peer_holds_is_the_one_who_conceded() -> None:
    frame = debate(
        debate_pair(model="mover", opening=-1.0, closing=0.0),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    (concession,) = concessions(frame, min_concession=BAR)

    assert concession.conceder_model == "mover"
    assert concession.influencer_model == "anchor"
    assert concession.amount == pytest.approx(1.0)
    assert concession.opening_gap == pytest.approx(2.0)


def test_meeting_exactly_in_the_middle_credits_nobody() -> None:
    # Both travel 0.5 toward the other. Symmetric convergence has no author.
    frame = debate(
        debate_pair(model="left", opening=-1.0, closing=-0.5),
        debate_pair(model="right", opening=1.0, closing=0.5),
    )

    assert concessions(frame, min_concession=BAR) == ()


def test_symmetric_convergence_is_still_uncredited_with_the_bar_removed() -> None:
    frame = debate(
        debate_pair(model="left", opening=-1.0, closing=-0.5),
        debate_pair(model="right", opening=1.0, closing=0.5),
    )

    assert concessions(frame, min_concession=0.0) == ()


def test_the_agent_that_gave_more_is_credited_when_both_moved() -> None:
    # left travels 0.8 toward right, right travels 0.1 toward left.
    frame = debate(
        debate_pair(model="left", opening=-1.0, closing=-0.2),
        debate_pair(model="right", opening=1.0, closing=0.9),
    )

    (concession,) = concessions(frame, min_concession=BAR)

    assert concession.conceder_model == "left"
    assert concession.amount == pytest.approx(0.8)
    assert concession.asymmetry == pytest.approx(0.7)


def test_an_agent_that_stood_still_is_not_credited_with_capitulating() -> None:
    # The mover storms *away*; the pair's midpoint drifts, but nobody conceded.
    frame = debate(
        debate_pair(model="mover", opening=0.0, closing=-0.8),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    assert concessions(frame, min_concession=BAR) == ()


def test_a_concession_short_of_the_bar_is_not_recorded() -> None:
    frame = debate(
        debate_pair(model="mover", opening=-1.0, closing=-0.9),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    assert concessions(frame, min_concession=BAR) == ()


# -- the two bars, exactly on the boundary -----------------------------------
#
# This module has two thresholds -- how far the conceder travelled, and how much
# more it gave than it got -- and both run through `evaluation.threshold.meets`
# rather than a bare `>=`. Exposures land on a 0.05 grid, and a nominal 0.20 move
# subtracts to either 0.19999999999999998 or 0.20000000000000007 depending on
# where on that grid it happened; a bare comparison keeps one and drops the other.
# Reverting either bar changed the published matrix (conceded went from 119/122 to
# 83/103 on the two-model run, and the placebo's net sign flipped) while the whole
# suite stayed green. These pin both bars from both sides of the error.


def test_a_concession_of_exactly_the_bar_counts_when_the_subtraction_falls_short() -> None:
    # 0.3 - 0.1 == 0.19999999999999998. Both bars sit on it: the conceder
    # travelled the bar and the motionless anchor gave nothing back.
    frame = debate(
        debate_pair(model="mover", opening=0.1, closing=0.3),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    (concession,) = concessions(frame, min_concession=BAR)

    assert concession.conceder_model == "mover"
    assert concession.influencer_model == "anchor"
    assert concession.amount == pytest.approx(0.2)


def test_a_concession_of_exactly_the_bar_counts_when_the_subtraction_overshoots() -> None:
    # The other side of the same error: 0.9 - 0.7 == 0.20000000000000007.
    frame = debate(
        debate_pair(model="mover", opening=0.7, closing=0.9),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    (concession,) = concessions(frame, min_concession=BAR)

    assert concession.conceder_model == "mover"
    assert concession.amount == pytest.approx(0.2)


def test_the_travelled_bar_alone_on_the_boundary_still_records_a_concession() -> None:
    # Isolating the second bar: the asymmetry is 0.6 -- far past the threshold,
    # because the influencer moved away -- while the ground the conceder itself
    # gave is exactly 0.20, subtracting short.
    frame = debate(
        debate_pair(model="mover", opening=0.1, closing=0.3),
        debate_pair(model="drifter", opening=0.5, closing=0.9),
    )

    (concession,) = concessions(frame, min_concession=BAR)

    assert concession.conceder_model == "mover"
    assert concession.amount == pytest.approx(0.2)
    assert concession.asymmetry == pytest.approx(0.6)


def test_the_asymmetry_bar_alone_on_the_boundary_still_records_a_concession() -> None:
    # Isolating the first bar: the conceder travelled 0.5, well past the
    # threshold, but gave only 0.5 - 0.30000000000000004 == 0.19999999999999996
    # more than it got.
    frame = debate(
        debate_pair(model="left", opening=-1.0, closing=-0.5),
        debate_pair(model="right", opening=1.0, closing=0.7),
    )

    (concession,) = concessions(frame, min_concession=BAR)

    assert concession.conceder_model == "left"
    assert concession.amount == pytest.approx(0.5)
    assert concession.asymmetry == pytest.approx(0.2)


def test_the_asymmetry_bar_on_the_boundary_from_the_other_side_of_the_grid() -> None:
    # 0.9 - 0.7 == 0.20000000000000007, this time as the asymmetry rather than
    # as the distance travelled.
    frame = debate(
        debate_pair(model="left", opening=-1.0, closing=-0.1),
        debate_pair(model="right", opening=1.0, closing=0.3),
    )

    (concession,) = concessions(frame, min_concession=BAR)

    assert concession.conceder_model == "left"
    assert concession.amount == pytest.approx(0.9)
    assert concession.asymmetry == pytest.approx(0.2)


def test_agents_who_opened_in_the_same_place_had_no_ground_to_give() -> None:
    frame = debate(
        debate_pair(model="left", opening=0.5, closing=-1.0),
        debate_pair(model="right", opening=0.5, closing=0.5),
    )

    assert concessions(frame, min_concession=BAR) == ()


def test_concessions_are_only_counted_inside_one_conversation() -> None:
    frame = frame_of(
        *debate_pair(model="mover", opening=-1.0, closing=0.0, composition="quad"),
        *debate_pair(model="anchor", opening=1.0, closing=1.0, composition="pair"),
    )

    assert concessions(frame, min_concession=BAR) == ()


def test_the_verdict_does_not_depend_on_which_agent_sorts_first() -> None:
    # The pairing walks agents in name order. If the arithmetic were not
    # antisymmetric, renaming the agents would move the credit.
    forwards = debate(
        debate_pair(model="aaa", opening=-1.0, closing=-0.2),
        debate_pair(model="zzz", opening=1.0, closing=0.9),
    )
    backwards = debate(
        debate_pair(model="zzz", opening=-1.0, closing=-0.2),
        debate_pair(model="aaa", opening=1.0, closing=0.9),
    )

    (first,) = concessions(forwards, min_concession=BAR)
    (second,) = concessions(backwards, min_concession=BAR)

    assert (first.conceder_model, first.influencer_model) == ("aaa", "zzz")
    assert (second.conceder_model, second.influencer_model) == ("zzz", "aaa")
    assert first.amount == pytest.approx(second.amount)


def test_omitting_the_bar_uses_the_threshold_declared_in_config() -> None:
    frame = debate(
        debate_pair(model="mover", opening=-1.0, closing=0.0),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    assert concessions(frame) == concessions(frame, min_concession=get_settings().shift_threshold)


def test_a_negative_bar_is_refused() -> None:
    with pytest.raises(ValueError, match="below zero"):
        concessions(frame_of(), min_concession=-0.1)


def test_the_shifts_behind_a_concession_carry_the_bar_that_was_asked_for() -> None:
    # Shift.threshold exists so a rate and the bar it was judged against cannot drift
    # apart in a write-up; reading the bar back out of config here would break that
    # the moment a caller passes one.
    frame = debate(
        debate_pair(model="mover", opening=-1.0, closing=-0.95),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    (group,) = _debates(frame, 0.01)

    assert {shift.threshold for shift in group} == {0.01}


# -- the matrix --------------------------------------------------------------


def test_the_matrix_records_the_conceder_as_the_row() -> None:
    frame = debate(
        debate_pair(model="mover", opening=-1.0, closing=0.0),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)
    table = matrix.to_frame()

    assert matrix.models == ("anchor", "mover")
    assert table.loc["mover", "anchor"] == 1
    assert table.loc["anchor", "mover"] == 0


def test_a_model_that_never_persuaded_anyone_still_appears_with_zeros() -> None:
    frame = debate(
        debate_pair(model="mover", opening=-1.0, closing=0.0),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
        debate_pair(model="bystander", opening=1.0, closing=1.0, persona="reversion-bold"),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert "bystander" in matrix.models
    assert int(matrix.conceded[matrix.models.index("bystander")].sum()) == 0


def test_two_personas_of_one_model_land_on_the_diagonal() -> None:
    # One pairing, one chance to give ground, one concession. Mirroring the
    # opportunity onto the diagonal -- the same cell -- would report a rate of a half
    # for the case that two models across four personas produce most often.
    frame = debate(
        debate_pair(model="alpha", persona="momentum-bold", opening=-1.0, closing=0.0),
        debate_pair(model="alpha", persona="reversion-bold", opening=1.0, closing=1.0),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert matrix.models == ("alpha",)
    assert int(matrix.conceded[0, 0]) == 1
    assert int(matrix.opportunities[0, 0]) == 1
    assert matrix.rate("alpha", "alpha") == pytest.approx(1.0)


def test_two_models_still_count_one_disagreement_on_each_side_of_the_diagonal() -> None:
    frame = debate(
        debate_pair(model="mover", opening=-1.0, closing=0.0),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert int(matrix.opportunities.sum()) == 2
    assert matrix.rate("mover", "anchor") == pytest.approx(1.0)
    assert matrix.rate("anchor", "mover") == pytest.approx(0.0)


def test_the_rate_is_concessions_over_disagreements_rather_than_over_debates() -> None:
    frame = frame_of(
        *debate_pair(model="mover", opening=-1.0, closing=0.0, on=DAY),
        *debate_pair(model="anchor", opening=1.0, closing=1.0, on=DAY),
        *debate_pair(model="mover", opening=-1.0, closing=-1.0, on=NEXT_DAY),
        *debate_pair(model="anchor", opening=1.0, closing=1.0, on=NEXT_DAY),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert matrix.rate("mover", "anchor") == pytest.approx(0.5)
    assert matrix.rate("anchor", "mover") == pytest.approx(0.0)


def test_a_pair_that_never_disagreed_has_no_rate_rather_than_a_rate_of_zero() -> None:
    frame = debate(
        debate_pair(model="left", opening=0.5, closing=0.5),
        debate_pair(model="right", opening=0.5, closing=0.5),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert matrix.rate("left", "right") is None


def test_the_loudest_voice_ranks_first_on_net_influence() -> None:
    frame = frame_of(
        *debate_pair(model="mover", opening=-1.0, closing=0.0, on=DAY),
        *debate_pair(model="anchor", opening=1.0, closing=1.0, on=DAY),
        *debate_pair(model="other", opening=-1.0, closing=0.0, on=NEXT_DAY),
        *debate_pair(model="anchor", opening=1.0, closing=1.0, on=NEXT_DAY),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert matrix.net_influence[0] == ("anchor", 2)
    assert dict(matrix.net_influence)["mover"] == -1


def test_ties_in_net_influence_break_on_name_so_a_rerun_reports_the_same_order() -> None:
    frame = debate(
        debate_pair(model="zulu", opening=0.5, closing=0.5),
        debate_pair(model="alpha", opening=0.5, closing=0.5, persona="reversion-bold"),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert [model for model, _ in matrix.net_influence] == ["alpha", "zulu"]


# -- one arm at a time -------------------------------------------------------


def in_arm(pair: tuple[dict[str, Any], dict[str, Any]], arm: str) -> list[dict[str, Any]]:
    return [dict(record, arm=arm) for record in pair]


def test_the_same_conversation_in_two_arms_is_not_counted_twice() -> None:
    # The placebo exists to be subtracted from the real debate. Summing the two into
    # one cell answers that question with both answers already mixed together.
    frame = frame_of(
        *in_arm(debate_pair(model="mover", opening=-1.0, closing=0.0), "debate"),
        *in_arm(debate_pair(model="anchor", opening=1.0, closing=1.0), "debate"),
        *in_arm(debate_pair(model="mover", opening=-1.0, closing=0.0), "debate_placebo"),
        *in_arm(debate_pair(model="anchor", opening=1.0, closing=1.0), "debate_placebo"),
    )

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert matrix.arm == "debate"
    assert int(matrix.conceded.sum()) == 1
    assert int(matrix.opportunities.sum()) == 2


def test_each_arm_reports_its_own_counts() -> None:
    frame = frame_of(
        *in_arm(debate_pair(model="mover", opening=-1.0, closing=0.0), "debate"),
        *in_arm(debate_pair(model="anchor", opening=1.0, closing=1.0), "debate"),
        *in_arm(debate_pair(model="mover", opening=-1.0, closing=-1.0), "debate_placebo"),
        *in_arm(debate_pair(model="anchor", opening=1.0, closing=1.0), "debate_placebo"),
    )

    real = influence_matrix(frame, arm="debate", min_concession=BAR)
    placebo = influence_matrix(frame, arm="debate_placebo", min_concession=BAR)

    assert real.rate("mover", "anchor") == pytest.approx(1.0)
    assert placebo.arm == "debate_placebo"
    assert placebo.rate("mover", "anchor") == pytest.approx(0.0)


def test_an_arm_the_frame_does_not_contain_is_named_rather_than_reported_as_zeros() -> None:
    frame = debate(
        debate_pair(model="mover", opening=-1.0, closing=0.0),
        debate_pair(model="anchor", opening=1.0, closing=1.0),
    )

    with pytest.raises(ValueError, match="no rows in arm 'debate_rationale_only'"):
        influence_matrix(frame, arm="debate_rationale_only", min_concession=BAR)


# -- degenerate debates ------------------------------------------------------


def test_an_empty_frame_produces_an_empty_matrix() -> None:
    matrix = influence_matrix(frame_of(), arm="debate", min_concession=BAR)

    assert matrix.models == ()
    assert matrix.conceded.shape == (0, 0)
    assert matrix.net_influence == ()


def test_a_lone_agent_has_nobody_to_concede_to() -> None:
    frame = debate(debate_pair(model="alpha", opening=-1.0, closing=1.0))

    matrix = influence_matrix(frame, arm="debate", min_concession=BAR)

    assert concessions(frame, min_concession=BAR) == ()
    assert int(matrix.opportunities.sum()) == 0


def test_an_arm_with_no_rebuttals_produces_no_concessions() -> None:
    frame = frame_of(
        row(model="alpha", arm="independent", composition="", exposure=-1.0),
        row(model="beta", arm="independent", composition="", exposure=1.0),
    )

    assert concessions(frame, min_concession=BAR) == ()

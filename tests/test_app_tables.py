"""The frames the panels plot.

Each of these is an adapter over a report the evaluation package already
computed, so what is checked here is the reshaping: that an empty band stays
empty rather than becoming a zero, that an arm which cannot shift still appears,
and that the calibration panel is scored against the return a decision actually
went on to earn rather than one that had already happened.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from council.app.tables import (
    calibration_table,
    coverage_note,
    coverage_table,
    forward_return_lookup,
    influence_table,
    net_influence_table,
    select,
    shift_rate_table,
    shift_reports,
)
from council.evaluation.calibration import calibrate
from council.evaluation.influence import influence_matrix
from helpers_app import DAY, OPENING, REBUTTAL, frame_of, independent, stored

TICKER = "AAA"


def pair(
    *, model: str, opening: float, final: float, confidence: float = 0.9, arm: str = "debate"
) -> tuple[dict[str, object], ...]:
    persona = f"{model}-persona"
    return (
        stored(
            model=model, persona=persona, arm=arm, round_index=OPENING,
            exposure=opening, confidence=confidence, ticker=TICKER,
        ),
        stored(
            model=model, persona=persona, arm=arm, round_index=REBUTTAL,
            exposure=final, confidence=confidence, ticker=TICKER,
        ),
    )


# -- selection ---------------------------------------------------------------


def test_selecting_an_arm_and_a_round_leaves_only_those_rows() -> None:
    frame = frame_of(*pair(model="alpha", opening=0.0, final=0.5), independent())

    selected = select(frame, arm="debate", round_index=REBUTTAL)

    assert len(selected) == 1
    assert selected["exposure"].tolist() == [0.5]


def test_an_omitted_filter_pools_rather_than_filtering() -> None:
    frame = frame_of(*pair(model="alpha", opening=0.0, final=0.5), independent())

    assert len(select(frame)) == 3


def test_the_independent_arms_null_composition_can_be_selected_by_name() -> None:
    frame = frame_of(independent(), stored())

    assert len(select(frame, composition="")) == 1


# -- the primary statistic ---------------------------------------------------


def test_every_arm_in_the_run_gets_a_report_including_one_that_cannot_shift() -> None:
    # The independent arm has one round by construction. Leaving it out would
    # leave a reader wondering whether the control was measured at all.
    frame = frame_of(*pair(model="alpha", opening=0.0, final=0.5), independent())

    reports = shift_reports(frame)

    assert tuple(reports) == ("independent", "debate")
    assert all(band.count == 0 for band in reports["independent"].report.bands)


def test_the_distinct_points_behind_a_band_are_reported_beside_the_observations() -> None:
    # Every contested point is answered once per seat of every committee, so
    # `count` is not the sample size the pre-registered statistic is declared
    # over. Reading it as one overstates the evidence by that factor.
    frame = frame_of(
        *pair(model="alpha", opening=0.0, final=0.9),
        *pair(model="beta", opening=0.0, final=0.9),
    )

    table = shift_rate_table(shift_reports(frame))
    top = table.loc[(table["arm"] == "debate") & (table["band"] == "[0.80, 1.00]")]

    assert top["count"].tolist() == [2]
    assert top["points"].tolist() == [1]


def test_a_band_nobody_occupied_has_no_points_either() -> None:
    frame = frame_of(*pair(model="alpha", opening=0.0, final=0.9, confidence=0.9))

    table = shift_rate_table(shift_reports(frame))
    empty = table.loc[(table["arm"] == "debate") & (table["band"] == "[0.00, 0.20)")]

    assert empty["points"].tolist() == [0]


def test_a_shift_is_counted_in_the_band_of_the_confidence_held_before_the_debate() -> None:
    frame = frame_of(*pair(model="alpha", opening=0.0, final=0.9, confidence=0.9))

    table = shift_rate_table(shift_reports(frame))
    top = table.loc[(table["arm"] == "debate") & (table["band"] == "[0.80, 1.00]")]

    assert top["count"].tolist() == [1]
    assert top["shift_rate"].tolist() == [1.0]


def test_a_band_with_no_observations_has_no_rate_rather_than_a_rate_of_zero() -> None:
    frame = frame_of(*pair(model="alpha", opening=0.0, final=0.9, confidence=0.9))

    table = shift_rate_table(shift_reports(frame))
    empty = table.loc[(table["arm"] == "debate") & (table["band"] == "[0.00, 0.20)")]

    assert empty["count"].tolist() == [0]
    assert empty["shift_rate"].isna().all()


def test_the_debate_arm_and_its_placebo_are_reported_separately() -> None:
    frame = frame_of(
        *pair(model="alpha", opening=0.0, final=0.9),
        *pair(model="alpha", opening=0.0, final=0.0, arm="debate_placebo"),
    )

    table = shift_rate_table(shift_reports(frame))
    scored = table.loc[table["count"] > 0]

    assert scored.set_index("arm")["shift_rate"].to_dict() == {"debate": 1.0, "debate_placebo": 0.0}


def test_the_confidence_column_is_the_bands_midpoint_so_it_can_share_an_axis() -> None:
    frame = frame_of(*pair(model="alpha", opening=0.0, final=0.9, confidence=0.9))

    table = shift_rate_table(shift_reports(frame))

    assert table["confidence"].tolist() == pytest.approx([0.1, 0.3, 0.5, 0.7, 0.9])


# -- what each arm actually covers -------------------------------------------


def uneven_frame() -> pd.DataFrame:
    """A debate over two days and a placebo that only reached the second.

    The shape council.debate.sweep produces: the earliest contested day has no
    earlier day to borrow a counter-argument from, so the placebo arm is
    abandoned there and covers one point fewer.
    """
    return frame_of(
        *pair(model="alpha", opening=0.0, final=0.9),
        *(
            stored(
                model="alpha", persona="alpha-persona", arm="debate",
                round_index=index, exposure=value, confidence=0.9,
                ticker=TICKER, on=date(2022, 1, 4),
            )
            for index, value in ((OPENING, 0.0), (REBUTTAL, 0.9))
        ),
        *(
            stored(
                model="alpha", persona="alpha-persona", arm="debate_placebo",
                round_index=index, exposure=value, confidence=0.9,
                ticker=TICKER, on=date(2022, 1, 4),
            )
            for index, value in ((OPENING, 0.0), (REBUTTAL, 0.0))
        ),
    )


def test_coverage_counts_the_points_each_arm_actually_answered() -> None:
    table = coverage_table(uneven_frame()).set_index("arm")

    assert table.loc["debate", "points"] == 2
    assert table.loc["debate_placebo", "points"] == 1


def test_a_row_with_no_partner_round_is_counted_rather_than_left_unmentioned() -> None:
    frame = frame_of(stored(model="alpha", arm="debate", round_index=OPENING, ticker=TICKER))

    table = coverage_table(frame).set_index("arm")

    assert table.loc["debate", "paired"] == 0
    assert table.loc["debate", "unpaired"] == 1


def test_arms_covering_different_points_produce_a_warning_naming_both() -> None:
    # A reader differencing two rates has to be told the denominators differ,
    # because part of the gap is then coverage rather than persuasion.
    note = coverage_note(coverage_table(uneven_frame()))

    assert note is not None
    assert "debate 2" in note
    assert "debate_placebo 1" in note


def test_arms_covering_the_same_points_say_nothing_rather_than_reassuring() -> None:
    frame = frame_of(
        *pair(model="alpha", opening=0.0, final=0.9),
        *pair(model="alpha", opening=0.0, final=0.0, arm="debate_placebo"),
    )

    assert coverage_note(coverage_table(frame)) is None


def test_one_debate_arm_alone_has_nothing_to_be_differenced_against() -> None:
    frame = frame_of(*pair(model="alpha", opening=0.0, final=0.9), independent())

    assert coverage_note(coverage_table(frame)) is None


# -- calibration -------------------------------------------------------------


def test_the_calibration_table_pairs_a_stated_confidence_with_a_realised_hit_rate() -> None:
    frame = frame_of(
        independent(exposure=0.5, confidence=0.9, ticker=TICKER),
        independent(exposure=-0.5, confidence=0.1, ticker=TICKER, model="beta"),
    )

    table = calibration_table(calibrate(frame, {(DAY, TICKER): 0.02}))
    scored = table.loc[table["count"] > 0].set_index("band")

    assert scored.loc["[0.80, 1.00]", "hit_rate"] == 1.0
    assert scored.loc["[0.00, 0.20)", "hit_rate"] == 0.0


def test_an_unobserved_band_leaves_the_hit_rate_empty() -> None:
    frame = frame_of(independent(exposure=0.5, confidence=0.9, ticker=TICKER))

    table = calibration_table(calibrate(frame, {(DAY, TICKER): 0.02}))

    assert table.loc[table["count"] == 0, "hit_rate"].isna().all()


def test_a_decision_is_scored_against_the_return_it_went_on_to_earn() -> None:
    # The obvious one-liner stores against day t a move that had already happened
    # by the close of t, and every agent then looks prescient with nothing raising.
    opens = pd.DataFrame(
        {TICKER: [100.0, 110.0, 121.0, 133.1]},
        index=pd.to_datetime(["2022-01-03", "2022-01-04", "2022-01-05", "2022-01-06"]),
    )

    lookup = forward_return_lookup(opens)

    assert lookup[(date(2022, 1, 3), TICKER)] == pytest.approx(0.1)
    assert (date(2022, 1, 5), TICKER) not in lookup


# -- influence ---------------------------------------------------------------


def conceding_frame() -> pd.DataFrame:
    """Alpha travels most of the way to beta; beta does not move."""
    return frame_of(
        *pair(model="alpha", opening=0.0, final=0.6),
        *pair(model="beta", opening=1.0, final=1.0),
    )


def test_the_matrix_becomes_one_row_per_ordered_pair_of_models() -> None:
    table = influence_table(influence_matrix(conceding_frame(), arm="debate"))

    assert len(table) == 4
    assert set(table["conceder"]) == {"alpha", "beta"}


def test_a_concession_lands_on_the_conceders_row_and_the_influencers_column() -> None:
    table = influence_table(influence_matrix(conceding_frame(), arm="debate")).set_index(
        ["conceder", "influencer"]
    )

    assert table.loc[("alpha", "beta"), "conceded"] == 1
    assert table.loc[("beta", "alpha"), "conceded"] == 0


def test_every_row_carries_the_arm_so_a_lifted_heatmap_keeps_its_condition() -> None:
    table = influence_table(influence_matrix(conceding_frame(), arm="debate"))

    assert set(table["arm"]) == {"debate"}


def test_a_pair_that_never_disagreed_has_no_rate_rather_than_a_rate_of_zero() -> None:
    table = influence_table(influence_matrix(conceding_frame(), arm="debate")).set_index(
        ["conceder", "influencer"]
    )

    assert table.loc[("alpha", "alpha"), "opportunities"] == 0
    assert pd.isna(table.loc[("alpha", "alpha"), "rate"])


def test_net_influence_puts_the_model_that_was_conceded_to_first() -> None:
    table = net_influence_table(influence_matrix(conceding_frame(), arm="debate"))

    assert table["model"].tolist() == ["beta", "alpha"]
    assert table["net_influence"].tolist() == [1, -1]

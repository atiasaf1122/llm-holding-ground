"""Turning stored decisions into the curves the first results panel draws.

The property that carries this panel's honesty is that it is not a second
implementation of scoring: the numbers under the equity chart must be the ones
``python -m council evaluate`` prints from the same artefacts. That is asserted
directly, against :func:`council.scoring.evaluate_experiment`, because the two
disagreeing is a defect that produces a plausible chart rather than an exception.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from council.app.curves import (
    BUY_AND_HOLD_LABEL,
    RANDOM_LABEL,
    CurveSet,
    arms_in,
    build_curves,
    compositions_for,
    curves_frame,
    metrics_frame,
    run_curve,
)
from council.config import Settings
from council.data.prices import opens_frame, synthetic_prices
from council.debate.compositions import Composition, Seat, balanced_design
from council.domain.persona import PERSONAS
from council.domain.signal import Arm
from council.evaluation.aggregation import mean
from council.evaluation.frames import frame_to_rows
from council.scoring import evaluate_experiment
from helpers_app import COMMITTEE, DAY, OPENING, REBUTTAL, frame_of, independent, stored

TICKER = "AAA"
MODELS = ("alpha", "beta")

COST_BPS = 10.0
REBALANCE = 0.05
SEED = 1


def opens() -> pd.DataFrame:
    """A calendar that starts on the day the helpers put their decisions."""
    return opens_frame(synthetic_prices(tickers=(TICKER,), start=DAY, sessions=8))


def design() -> tuple[Composition, ...]:
    return balanced_design(models=MODELS)


def one_committee() -> Composition:
    """The committee `helpers_app` names, seated by the design's own arithmetic."""
    return next(table for table in design() if table.identifier == COMMITTEE)


def committee_rows(
    *,
    opening: float,
    final: float,
    arm: str = "debate",
    on: date = DAY,
    table: Composition | None = None,
) -> tuple[dict[str, object], ...]:
    """One round pair per seat of the named committee."""
    seated = one_committee() if table is None else table
    return tuple(
        stored(
            arm=arm,
            on=on,
            model=seat.model,
            persona=seat.persona.name,
            composition=seated.identifier,
            round_index=round_index,
            exposure=exposure,
            ticker=TICKER,
        )
        for seat in seated.seats
        for round_index, exposure in ((OPENING, opening), (REBUTTAL, final))
    )


def independent_rows(
    *, exposure: float, on: date = DAY, table: Composition | None = None
) -> tuple[dict[str, object], ...]:
    seated = one_committee() if table is None else table
    return tuple(
        independent(
            model=seat.model,
            persona=seat.persona.name,
            exposure=exposure,
            ticker=TICKER,
            on=on,
        )
        for seat in seated.seats
    )


# -- which arms and which committees are scored ------------------------------


def test_only_the_arms_the_experiment_declares_are_scored_and_the_control_leads() -> None:
    rows = frame_to_rows(
        frame_of(
            # A post-debate row, which is what makes a treatment arm scorable.
            stored(arm="debate_placebo", round_index=REBUTTAL),
            stored(arm="something_else", round_index=REBUTTAL),
            independent(),
        )
    )

    assert arms_in(rows) == (Arm.INDEPENDENT, Arm.DEBATE_PLACEBO)


def test_the_declared_scope_is_every_committee_in_the_design() -> None:
    assert compositions_for(None, design()) == design()


def test_asking_for_one_committee_narrows_the_design_to_it() -> None:
    chosen = compositions_for(COMMITTEE, design())

    assert [table.identifier for table in chosen] == [COMMITTEE]


def test_a_committee_outside_the_design_raises_rather_than_drawing_a_flat_curve() -> None:
    with pytest.raises(ValueError, match="rotation-9"):
        compositions_for("rotation-9", design())


# -- the same computation as the command line --------------------------------


ALL_ARMS: tuple[str, ...] = tuple(str(arm) for arm in Arm)

SEATED_COMMITTEES: tuple[Composition, ...] = balanced_design(models=MODELS)[:2]
"""Two committees of the design, fully seated by the fixture below.

Their seats are disjoint -- `rotation-0` and `rotation-1` give each model a
different persona -- so the control arm gets one row per agent and no duplicate.
"""


def dryrun_artefacts() -> tuple[Settings, pd.DataFrame, pd.DataFrame]:
    """Settings, prices and decisions that both scorers can be run over.

    Every treatment arm is present, the placebo included: it is the arm that
    changed sign between the two implementations, and an agreement test that
    skipped it would have passed on the run that produced the finding.

    Two committees rather than one, and they disagree. A committee short of a seat
    is now dropped from the pooled average rather than aggregated over the
    survivors, so a fixture seating only one committee would make the pooled scope
    and that committee the same population -- and the test below, which asks
    whether they are different answers, would be checking nothing.
    """
    settings = Settings(tickers=(TICKER,), agent_models=MODELS, seed=SEED)
    prices = synthetic_prices(tickers=(TICKER,), start=DAY, sessions=8, seed=settings.seed)
    rows: list[dict[str, object]] = []
    for index, day in enumerate(day.date() for day in opens().index):
        exposure = 0.5 if index < 4 else -0.5
        for scale, table in enumerate(SEATED_COMMITTEES, start=1):
            rows.extend(independent_rows(exposure=exposure, on=day, table=table))
            for offset, arm in enumerate(ALL_ARMS[1:], start=1):
                rows.extend(
                    committee_rows(
                        arm=arm,
                        opening=exposure,
                        final=-exposure / (offset * scale),
                        on=day,
                        table=table,
                    )
                )
    return settings, prices, frame_of(*rows)


def test_the_panel_reports_the_numbers_the_evaluate_command_reports() -> None:
    # The dashboard reimplementing arm scoring is the defect this file exists to
    # prevent: on the dry run it flipped the placebo arm's Sharpe from +0.72 to
    # -0.85 against the same artefacts, and no committee choice reproduced the
    # command's figure. Both sides are computed here and compared, arm by arm.
    settings, prices, decisions = dryrun_artefacts()
    expected = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)

    built = build_curves(
        decisions=decisions,
        opens=opens_frame(prices, tickers=[TICKER]),
        compositions=design(),
        rule=mean,
        cost_bps=settings.total_cost_bps,
        rebalance_threshold=settings.rebalance_threshold,
        seed=settings.seed,
    )
    drawn = {curve.label: curve.metrics for curve in built.curves}

    assert set(ALL_ARMS).issubset(drawn)
    for outcome in expected.arms:
        assert drawn[outcome.arm].sharpe == pytest.approx(outcome.metrics.sharpe)
        assert drawn[outcome.arm].total_return == pytest.approx(outcome.metrics.total_return)
        assert drawn[outcome.arm].max_drawdown == pytest.approx(outcome.metrics.max_drawdown)


def test_one_committee_and_the_pooled_design_are_different_answers() -> None:
    # A legitimate exploratory cut, and not the declared comparison -- which is
    # why the panel labels the two differently rather than offering one control
    # that silently changes what the headline number means.
    settings, prices, decisions = dryrun_artefacts()

    def scored(committees: tuple[Composition, ...]) -> dict[str, float]:
        built = build_curves(
            decisions=decisions,
            opens=opens_frame(prices, tickers=[TICKER]),
            compositions=committees,
            rule=mean,
            cost_bps=settings.total_cost_bps,
            rebalance_threshold=settings.rebalance_threshold,
            seed=settings.seed,
        )
        return {curve.label: curve.metrics.total_return for curve in built.curves}

    pooled = scored(design())
    single = scored(compositions_for(COMMITTEE, design()))

    assert pooled.keys() == single.keys()
    assert pooled["debate"] != pytest.approx(single["debate"])


# -- the curves themselves ---------------------------------------------------


def run_over(exposures: list[float]) -> pd.DataFrame:
    """One committee, two arms, one decision per session, at the exposures given."""
    rows: list[dict[str, object]] = []
    for day, exposure in zip([day.date() for day in opens().index], exposures, strict=True):
        rows.extend(independent_rows(exposure=exposure, on=day))
        rows.extend(committee_rows(opening=exposure, final=-exposure, on=day))
    return frame_of(*rows)


def full_run() -> pd.DataFrame:
    """A control that turns over once in the middle.

    The random baseline is calibrated to *match* the control's trading rate, and
    an arm that flips on every bar turns over more than any shuffle of its own
    exposures can reach -- which is a separate case, tested below.
    """
    sessions = len(opens())
    return run_over([0.5 if index < sessions // 2 else -0.5 for index in range(sessions)])


def build(
    decisions: pd.DataFrame, *, cost_bps: float = COST_BPS, rebalance: float = REBALANCE
) -> CurveSet:
    return build_curves(
        decisions=decisions,
        opens=opens(),
        compositions=design(),
        rule=mean,
        cost_bps=cost_bps,
        rebalance_threshold=rebalance,
        seed=SEED,
    )


def test_every_arm_gets_a_curve_alongside_both_baselines() -> None:
    built = build(full_run())

    assert [curve.label for curve in built.curves] == [
        "independent",
        "debate",
        BUY_AND_HOLD_LABEL,
        RANDOM_LABEL,
    ]
    assert built.baseline_note is None


def test_a_turnover_no_shuffle_can_match_costs_the_baseline_and_not_the_arms() -> None:
    # The null is the control's shape with the information taken out. Where that
    # shape is unreachable, a curve matched to some other turnover would still be
    # labelled a baseline -- so it is dropped and the reason is carried instead.
    sessions = len(opens())
    built = build(
        run_over([1.0 if index % 2 else -1.0 for index in range(sessions)]),
        cost_bps=0.0,
        rebalance=0.0,
    )

    assert RANDOM_LABEL not in [curve.label for curve in built.curves]
    assert built.baseline_note is not None
    assert "unreachable" in built.baseline_note
    assert "independent" in [curve.label for curve in built.curves]


def test_the_debate_arm_and_its_control_differ_where_the_debate_changed_the_view() -> None:
    built = build(full_run(), cost_bps=0.0, rebalance=0.0)
    by_label = {curve.label: curve for curve in built.curves}

    assert by_label["debate"].result.net_return.tolist() != (
        by_label["independent"].result.net_return.tolist()
    )


def test_a_run_whose_seats_are_not_the_configured_ones_raises_rather_than_flatlining() -> None:
    # A flat curve reads as a committee that decided nothing, which is a very
    # different claim from a run scored against seats nobody filled.
    with pytest.raises(ValueError, match="no stored decision was made by a seat"):
        build_curves(
            decisions=frame_of(independent(model="gamma", ticker=TICKER)),
            opens=opens(),
            compositions=design(),
            rule=mean,
            cost_bps=COST_BPS,
            rebalance_threshold=REBALANCE,
            seed=SEED,
        )


def test_a_run_holding_no_declared_arm_raises() -> None:
    with pytest.raises(ValueError, match="no arm the experiment declares"):
        build_curves(
            decisions=frame_of(stored(arm="something_else", ticker=TICKER)),
            opens=opens(),
            compositions=design(),
            rule=mean,
            cost_bps=COST_BPS,
            rebalance_threshold=REBALANCE,
            seed=SEED,
        )


def test_a_curve_covers_one_period_fewer_than_the_calendar_has_sessions() -> None:
    # The last session opens no period; every arm must agree on that or the
    # curves would be drawn against different calendars.
    prices = opens()
    curve = run_curve(
        label="flat", exposures={}, opens=prices, cost_bps=0.0, rebalance_threshold=0.0
    )

    assert len(curve.equity) == len(prices) - 1


def test_the_long_frame_carries_every_curve_under_its_own_label() -> None:
    curves = (
        run_curve(label="flat", exposures={}, opens=opens(), cost_bps=0.0, rebalance_threshold=0.0),
    )

    frame = curves_frame(curves)

    assert list(frame.columns) == ["date", "series", "equity"]
    assert set(frame["series"]) == {"flat"}


def test_an_empty_set_of_curves_still_has_the_columns_a_chart_reads() -> None:
    assert list(curves_frame(()).columns) == ["date", "series", "equity"]


def test_the_metrics_table_has_one_row_per_curve() -> None:
    curves = (
        run_curve(label="flat", exposures={}, opens=opens(), cost_bps=0.0, rebalance_threshold=0.0),
    )

    table = metrics_frame(curves)

    assert table["series"].tolist() == ["flat"]
    assert "max_drawdown" in table.columns


def test_a_hand_built_composition_is_scored_like_any_other() -> None:
    # `compositions_for` reads the design it is handed rather than rebuilding one,
    # so a caller with its own committee list is not a special case.
    table = Composition(identifier="hand-built", seats=(Seat(model="alpha", persona=PERSONAS[0]),))

    assert compositions_for("hand-built", (table,)) == (table,)

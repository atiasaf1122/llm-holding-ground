"""Whether the plan can be trusted with a night of somebody's GPU.

The load-bearing test is :func:`test_the_plan_counts_exactly_what_a_run_issues`.
It compares the plan against a provider that counts calls, not against the formula
the plan was built from -- a formula asserted against itself passes whichever of
the two is wrong.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from council.config import Settings
from council.debate.compositions import balanced_design
from council.domain.signal import Arm
from council.evaluation.frames import PointKey
from council.pipeline import open_store, select_contested, stored_decisions
from council.planning import (
    DEBATE_STAGE,
    INDEPENDENT_STAGE,
    TREATMENT_ARMS,
    conversation_keys,
    plan_experiment,
)
from council.report import format_duration, render_plan
from helpers_pipeline import make_prices, make_settings, run_debates, run_independent


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


def contested_points(settings: Settings) -> tuple[PointKey, ...]:
    decisions = stored_decisions(open_store(settings))
    return tuple(point.point for point in select_contested(decisions, settings=settings))


# -- the count matches the run ----------------------------------------------------


def test_the_plan_counts_exactly_what_a_run_issues(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # Two arms rather than three: the placebo is the one arm that legitimately
    # issues fewer calls than the plan allows for, because the earliest contested
    # point has no earlier donor. That asymmetry has its own test.
    store = open_store(settings)
    planned_control = plan_experiment(settings=settings, prices=prices, store=store)

    control = run_independent(settings, prices)
    assert control.total_calls == planned_control.stages[0].inferences

    arms = (Arm.DEBATE, Arm.DEBATE_RATIONALE_ONLY)
    planned_debate = plan_experiment(
        settings=settings, prices=prices, store=store, contested=contested_points(settings)
    )
    debating, _ = run_debates(settings, prices, arms=arms)

    expected = sum(
        stage.remaining
        for stage in planned_debate.stages
        if stage.stage == DEBATE_STAGE and stage.arm in {str(arm) for arm in arms}
    )
    assert debating.total_calls == expected


def test_planning_issues_no_inference_at_all(settings: Settings, prices: pd.DataFrame) -> None:
    plan = plan_experiment(settings=settings, prices=prices, store=open_store(settings))

    assert plan.total > 0
    assert not settings.decisions_path.exists()
    assert not settings.completions_path.exists()


def test_a_plan_over_a_finished_run_has_nothing_left_to_do(
    settings: Settings, prices: pd.DataFrame
) -> None:
    run_independent(settings, prices)
    run_debates(settings, prices, arms=(Arm.DEBATE,))

    plan = plan_experiment(
        settings=settings,
        prices=prices,
        store=open_store(settings),
        contested=contested_points(settings),
        compositions=balanced_design(models=settings.agent_models),
    )
    debate = next(stage for stage in plan.stages if stage.arm == str(Arm.DEBATE))

    assert plan.stages[0].remaining == 0
    assert debate.remaining == 0


# -- estimated against measured ---------------------------------------------------


def test_a_plan_with_nothing_generated_marks_its_debate_stages_estimated(
    settings: Settings, prices: pd.DataFrame
) -> None:
    plan = plan_experiment(settings=settings, prices=prices, store=open_store(settings))

    assert plan.contested_estimated
    assert plan.is_estimated
    assert [stage.estimated for stage in plan.stages] == [False, True, True, True]


def test_a_plan_taken_after_generation_is_no_longer_an_estimate(
    settings: Settings, prices: pd.DataFrame
) -> None:
    run_independent(settings, prices)

    plan = plan_experiment(
        settings=settings,
        prices=prices,
        store=open_store(settings),
        contested=contested_points(settings),
    )

    assert not plan.is_estimated
    assert plan.contested_points == len(contested_points(settings))


def test_every_debate_arm_costs_the_same_because_each_debates_the_same_points(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # If the three arms ever stopped covering the same points, the difference
    # between them would absorb the selection rather than measure the manipulation.
    run_independent(settings, prices)

    plan = plan_experiment(
        settings=settings,
        prices=prices,
        store=open_store(settings),
        contested=contested_points(settings),
    )
    debate_stages = [stage for stage in plan.stages if stage.stage == DEBATE_STAGE]

    assert [stage.arm for stage in debate_stages] == [str(arm) for arm in TREATMENT_ARMS]
    assert len({stage.inferences for stage in debate_stages}) == 1


# -- the keys a plan and a sweep both read ----------------------------------------


def test_a_conversation_names_every_seat_in_every_round(settings: Settings) -> None:
    composition = balanced_design(models=settings.agent_models)[0]

    keys = conversation_keys(
        composition=composition,
        arm=Arm.DEBATE,
        decision_date=settings.start,
        ticker="AAA",
        rebuttal_rounds=1,
    )

    assert len(keys) == composition.size * 2
    assert len(set(keys)) == len(keys)
    assert {key[5] for key in keys} == {0, 1}
    assert {key[6] for key in keys} == {composition.identifier}


# -- the table a developer reads --------------------------------------------------


def test_the_plan_table_shows_a_row_per_stage_and_a_total(
    settings: Settings, prices: pd.DataFrame
) -> None:
    plan = plan_experiment(settings=settings, prices=prices, store=open_store(settings))

    table = render_plan(plan)

    assert INDEPENDENT_STAGE in table
    assert all(str(arm) in table for arm in TREATMENT_ARMS)
    assert f"{plan.total:,}" in table
    assert "estimated" in table


@pytest.mark.parametrize(
    ("seconds", "rendered"),
    [(9.4, "9s"), (95.0, "1m 35s"), (7200.0, "2h 00m")],
)
def test_a_wall_clock_reads_as_a_length_of_evening(seconds: float, rendered: str) -> None:
    assert format_duration(seconds) == rendered

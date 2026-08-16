"""Whether the plan can be trusted with a night of somebody's GPU.

The load-bearing test is :func:`test_the_plan_counts_exactly_what_a_run_issues`.
It compares the plan against a provider that counts calls, not against the formula
the plan was built from -- a formula asserted against itself passes whichever of
the two is wrong.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from council.config import Settings
from council.debate.compositions import balanced_design
from council.domain.persona import PERSONAS
from council.domain.signal import Arm, FailureMode
from council.evaluation.frames import PointKey
from council.pipeline import (
    generate_independent,
    open_store,
    select_contested,
    stored_decisions,
)
from council.planning import (
    ASSUMED_CONTESTED_SHARE,
    DEBATE_STAGE,
    INDEPENDENT_STAGE,
    TREATMENT_ARMS,
    conversation_key,
    conversation_rows,
    plan_experiment,
)
from council.report import format_duration, render_plan
from helpers_pipeline import make_prices, make_settings, run_debates, run_independent
from helpers_runner import RecordingFactory


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


def test_the_control_stage_counts_exactly_what_a_run_issues(
    settings: Settings, prices: pd.DataFrame
) -> None:
    store = open_store(settings)
    planned = plan_experiment(settings=settings, prices=prices, store=store)

    control = run_independent(settings, prices)

    assert control.total_calls == planned.stages[0].inferences


def test_the_debate_stages_bound_what_a_run_issues_without_over_promising(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # This asserted equality, and equality was right while every conversation ran to
    # the cap. It cannot be right now: a conversation stops on agreement, on
    # stillness or at the cap, and which of those fires is not knowable before the
    # conversation happens. So a debate stage quotes the cap's worth of rows and a
    # run spends that or less -- which is the direction a budget has to err in, per
    # `ASSUMED_CONTESTED_SHARE`'s own argument about not tempting somebody into a
    # night that turns out to be three.
    #
    # Compared against a provider that counts calls rather than against the formula
    # the plan was built from, which is what made this test worth having.
    store = open_store(settings)
    run_independent(settings, prices)
    arms = (Arm.DEBATE, Arm.DEBATE_RATIONALE_ONLY)
    planned = plan_experiment(
        settings=settings,
        prices=prices,
        store=store,
        contested=contested_points(settings),
        decisions=stored_decisions(store),
    )

    debating, _ = run_debates(settings, prices, arms=arms)

    bound = sum(
        stage.remaining
        for stage in planned.stages
        if stage.stage == DEBATE_STAGE and stage.arm in {str(arm) for arm in arms}
    )
    assert 0 < debating.total_calls <= bound
    # And the bound is the cap's arithmetic rather than an arbitrary cushion: every
    # conversation of every committee, at every round the cap allows.
    committees = balanced_design(models=settings.agent_models)
    assert bound == len(arms) * len(_servable(settings, prices)) * sum(
        conversation_rows(composition=table, rebuttal_rounds=settings.max_debate_rounds)
        for table in committees
    )


def _servable(settings: Settings, prices: pd.DataFrame) -> tuple[PointKey, ...]:
    """The contested points every arm is actually run on, as the sweep filters them."""
    from council.debate.sweep import servable_points

    store = open_store(settings)
    decisions = stored_decisions(store)
    points = tuple(point.point for point in select_contested(decisions, settings=settings))
    keep = servable_points(
        points,
        decisions=decisions,
        committees=balanced_design(models=settings.agent_models),
        min_gap=settings.placebo_min_gap_sessions,
        rounds=settings.max_debate_rounds,
    )
    return tuple(point for point in points if point in keep)


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
    store = open_store(settings)

    plan = plan_experiment(
        settings=settings,
        prices=prices,
        store=store,
        contested=contested_points(settings),
        decisions=stored_decisions(store),
        compositions=balanced_design(models=settings.agent_models),
    )
    debate = next(stage for stage in plan.stages if stage.arm == str(Arm.DEBATE))

    assert plan.stages[0].remaining == 0
    # Zero, not "close to zero". Every conversation stopped somewhere short of the
    # cap, so the rows on disk are fewer than the stage's `inferences`; `completed`
    # is counted at the same width for every conversation the store says finished,
    # which is what stops a plan over a finished run reading as work outstanding.
    assert debate.remaining == 0


# -- estimated against measured ---------------------------------------------------


def test_a_plan_with_nothing_generated_marks_its_debate_stages_estimated(
    settings: Settings, prices: pd.DataFrame
) -> None:
    plan = plan_experiment(settings=settings, prices=prices, store=open_store(settings))

    assert plan.contested_estimated
    assert plan.is_estimated
    assert [stage.estimated for stage in plan.stages] == [False] + [True] * 5


def test_the_assumed_share_does_not_understate_every_share_this_design_measures(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # The docstring said half "exists so that a first plan overstates the debate arms
    # rather than tempting somebody into a night that turns out to be three". Every
    # contested share this design has measured is at or near 100% -- findings.md
    # section 2, CLAIMS C14 -- so half halved the debate budget instead.
    run_independent(settings, prices)
    measured = contested_points(settings)

    guess = plan_experiment(settings=settings, prices=prices, store=open_store(settings))
    counted = plan_experiment(
        settings=settings,
        prices=prices,
        store=open_store(settings),
        contested=measured,
    )

    assert ASSUMED_CONTESTED_SHARE == 1.0
    assert guess.contested_points == guess.decision_points
    assert guess.contested_points >= counted.contested_points
    for assumed, exact in zip(guess.stages[1:], counted.stages[1:], strict=True):
        assert assumed.inferences >= exact.inferences, assumed.arm


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


def test_a_plan_taken_over_a_half_generated_control_arm_is_not_called_measured(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # The grid is swept model, then persona, then ticker, so an interrupted
    # generate leaves a slice of it rather than a sample. Dispersion measured over
    # that slice is not the run's dispersion -- it understated each debate stage by
    # a third on the dry-run store -- and printing it as "(measured)" with no
    # estimate marker is the part that misleads.
    store = open_store(settings)
    asyncio.run(
        generate_independent(
            settings=settings,
            prices=prices,
            provider_factory=RecordingFactory(),
            store=store,
            # One stance from each axis, which is what makes a point contested at
            # all; a slice of one persona has nothing to disagree about.
            personas=(PERSONAS[0], PERSONAS[-1]),
        )
    )
    partial = contested_points(settings)
    assert partial, "the slice must yield contested points, or this asserts nothing"

    plan = plan_experiment(settings=settings, prices=prices, store=store, contested=partial)

    assert plan.stages[0].remaining > 0
    assert plan.contested_estimated
    assert [stage.estimated for stage in plan.stages] == [False] + [True] * 5


def test_every_stage_counts_the_points_a_placebo_donor_can_be_drawn_for(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # This used to assert `placebo.inferences < debate.inferences`, because the
    # placebo alone skipped a point whose pool holds no usable earlier day. That
    # inequality *was* the defect: the three arms covered different calendars, and
    # the points the placebo lost are the earliest ones rather than a random sample,
    # so part of any debate-minus-placebo difference was a difference in coverage.
    # The sweep now withholds those points from all three, so the three stages agree
    # -- and each is smaller than a stage counting every contested point would be.
    run_independent(settings, prices)
    run_debates(settings, prices, arms=(Arm.DEBATE_PLACEBO,))
    store = open_store(settings)

    plan = plan_experiment(
        settings=settings,
        prices=prices,
        store=store,
        contested=contested_points(settings),
        decisions=stored_decisions(store),
    )
    unfiltered = plan_experiment(
        settings=settings, prices=prices, store=store, contested=contested_points(settings)
    )
    stages = {stage.arm: stage for stage in plan.stages if stage.stage == DEBATE_STAGE}

    assert stages[str(Arm.DEBATE_PLACEBO)].remaining == 0
    assert (
        len(
            {
                stage.inferences
                for arm, stage in stages.items()
                if arm != str(Arm.DEBATE_CONTRADICTOR)
            }
        )
        == 1
    )
    assert len(_servable(settings, prices)) < len(contested_points(settings)), (
        "no point was withheld, so this test has stopped checking"
    )
    for arm in TREATMENT_ARMS:
        planned = next(stage for stage in unfiltered.stages if stage.arm == str(arm))
        assert stages[str(arm)].inferences < planned.inferences


def test_a_conversation_holding_one_retriable_failure_is_planned_whole(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # The sweep resumes conversation by conversation: `_Sweep.group` re-holds a whole
    # conversation unless the store says it reached a stopping condition, and one
    # unreachable-daemon row unmakes that. A plan that counted the missing rows alone
    # under-reported the resume budget by the length of a conversation -- on exactly
    # the runs where resuming is what a plan is for.
    run_independent(settings, prices)
    run_debates(settings, prices, arms=(Arm.DEBATE,))
    store = open_store(settings)

    # One stored round-1 row demoted to a retriable failure, which is what an hour
    # with the daemon down leaves behind.
    frame = pd.read_parquet(settings.decisions_path)
    rebuttals = frame.index[(frame["arm"] == str(Arm.DEBATE)) & (frame["round_index"] == 1)]
    assert len(rebuttals) > 0, "the debate arm must have stored a rebuttal round"
    frame.loc[rebuttals[0], "failure"] = str(FailureMode.UNAVAILABLE)
    frame.to_parquet(settings.decisions_path, index=False)

    plan = plan_experiment(
        settings=settings,
        prices=prices,
        store=store,
        contested=contested_points(settings),
        decisions=stored_decisions(store),
    )
    debate = next(stage for stage in plan.stages if stage.arm == str(Arm.DEBATE))
    composition = balanced_design(models=settings.agent_models)[0]

    # One row short bills the whole conversation at the cap's width -- the bound, not
    # the number of rows it happens to have written -- and the sweep then spends that
    # or less. Both halves matter: under-billing is what this test exists to catch,
    # and quoting the bound is what a variable-length conversation allows.
    assert debate.remaining == conversation_rows(
        composition=composition, rebuttal_rounds=settings.max_debate_rounds
    )
    factory, _ = run_debates(settings, prices, arms=(Arm.DEBATE,))
    assert 0 < factory.total_calls <= debate.remaining


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
    # Point-set parity still holds for every arm -- that is the invariant this test
    # protects. Cost parity no longer follows from it: the contradictor runs its own
    # one-round cap plus counter-generation calls, so its stage is priced by the
    # same `arm_round_cap`/`_conversation_calls` pair the sweep runs it with, and
    # the four full-length arms still agree to the inference.
    full_length = {
        stage.inferences for stage in debate_stages if stage.arm != str(Arm.DEBATE_CONTRADICTOR)
    }
    assert len(full_length) == 1
    contradictor = next(
        stage for stage in debate_stages if stage.arm == str(Arm.DEBATE_CONTRADICTOR)
    )
    assert contradictor.inferences != full_length.pop()


# -- the keys a plan and a sweep both read ----------------------------------------


def test_a_conversation_is_identified_without_naming_a_round(settings: Settings) -> None:
    # `conversation_keys` named every seat in every round `0..cap`, and both the plan
    # and the sweep asked whether the store held all of them. That was the same
    # question as "is it finished" only while every conversation ran to the cap: a
    # debate that agrees at round two writes fewer rows and could never satisfy it,
    # so the sweep re-holds a point it already owns and `remaining` never reaches
    # zero. The identity dropped the round, and the question moved to `stop_reason`.
    composition = balanced_design(models=settings.agent_models)[0]

    key = conversation_key(
        composition=composition,
        arm=Arm.DEBATE,
        decision_date=settings.start,
        ticker="AAA",
    )

    assert key == (settings.start, "AAA", str(Arm.DEBATE), composition.identifier)
    # The same conversation under another arm is another conversation, and the store
    # answers in the same shape.
    assert key != conversation_key(
        composition=composition,
        arm=Arm.DEBATE_PLACEBO,
        decision_date=settings.start,
        ticker="AAA",
    )
    assert conversation_rows(composition=composition, rebuttal_rounds=6) == composition.size * 7


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

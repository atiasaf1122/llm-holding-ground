"""What a configuration costs, counted before a single token is generated.

Deciding whether to commit an evening to a sweep should not require starting it,
so every number here is arithmetic over the grid rather than a measurement of it.
:class:`~council.agents.runner.RunPlan` already does this for the independent arm;
what this module adds is the three debate arms, which cannot be counted the same
way because they are only run where the agents disagreed.

That makes the count exact in one case and an estimate in the other, and the
difference is carried on the record rather than left to the reader. Once the
independent arm exists, the contested points are known and the debate stages are
counted key by key -- the same keys the sweep will check against what is already
stored, so the plan and the run cannot disagree about what is left to do. Before
it exists there is nothing to measure and the share is assumed; a stage built that
way says so, because a wall-clock figure quoted from a guess and one derived from
data are worth very different amounts to somebody about to give up their GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from typing import Final

import pandas as pd

from council.agents.runner import GenerationRunner
from council.agents.store import DecisionKey, DecisionStore
from council.config import Settings
from council.debate.compositions import Composition, balanced_design
from council.debate.protocol import DEFAULT_REBUTTAL_ROUNDS
from council.domain.persona import PERSONAS, Persona
from council.domain.signal import Arm
from council.evaluation.frames import PointKey

TREATMENT_ARMS: Final[tuple[Arm, ...]] = (
    Arm.DEBATE,
    Arm.DEBATE_RATIONALE_ONLY,
    Arm.DEBATE_PLACEBO,
)
"""The three debate arms, in the order a sweep runs them.

Ordered, unlike :data:`council.agents.prompt.DEBATE_ARMS`, because this tuple
decides the order of the stages in a plan and of the rows in a report, and a set
would reorder them between runs.
"""

SECONDS_PER_INFERENCE: Final = 1.5
"""Default cost of one constrained generation, wall clock.

A round number for a 8B model answering a short schema on one consumer card. It
is the one figure here that is a guess rather than arithmetic, which is why it is
a parameter of every function that uses it and a flag on the command line.
"""

ASSUMED_CONTESTED_SHARE: Final = 1.0
"""Share of decision points assumed contested when nothing has been generated yet.

Every contested share this design has measured is at or near 100% -- see
``docs/findings.md`` section 2 and ``docs/CLAIMS.md`` C14 -- because the crossed
personas split on direction at nearly every point. Assuming every decision point is
contested is therefore the value that actually overstates the debate arms, which is
what this constant is for: a first plan should tempt nobody into a night that turns
out to be three. Half, its previous value, halved the debate budget at every share
the design has ever produced and understated the bill in exactly the direction the
guard was meant to protect against.

Those shares are **pooled-grid** shares, and this constant is the right place to say
so. :func:`council.pipeline.select_contested` measures dispersion once over the whole
independent arm -- every model crossed with every persona -- and
:func:`council.debate.sweep.run_debate_arms` applies that one list unchanged to every
committee, so the plan and the sweep both spend at the pooled share. Per committee the
same run gives 449 of 1,120, 40% (``docs/CLAIMS.md`` C14). That is the figure a
per-committee gate would spend at, and no such gate exists; assuming the pooled share
therefore still overstates rather than understates what this code will run.
"""

INDEPENDENT_STAGE: Final = "generate"
DEBATE_STAGE: Final = "debate"


def conversation_keys(
    *,
    composition: Composition,
    arm: Arm,
    decision_date: date,
    ticker: str,
    rebuttal_rounds: int = DEFAULT_REBUTTAL_ROUNDS,
) -> tuple[DecisionKey, ...]:
    """Every stored row one conversation produces, in seat then round order.

    Built from the same fields :meth:`council.agents.inference.DecisionPoint.key`
    assembles, so a plan counts exactly the rows a sweep would write and a sweep
    can decide a conversation is already done by asking whether the store holds all
    of them. Deriving both from one function is what stops a resumed run from
    re-debating points it already owns.
    """
    return tuple(
        (
            decision_date,
            ticker,
            seat.model,
            seat.persona.name,
            str(arm),
            round_index,
            composition.identifier,
        )
        for seat in composition.seats
        for round_index in range(rebuttal_rounds + 1)
    )


@dataclass(frozen=True, slots=True)
class StagePlan:
    """One arm's share of the bill."""

    stage: str
    arm: str
    inferences: int
    completed: int
    parallelism: int
    """How many of these requests are in flight at once, which is what turns a
    count into a wall clock."""

    estimated: bool
    """Whether ``inferences`` was counted or assumed. See the module docstring."""

    @property
    def remaining(self) -> int:
        return max(0, self.inferences - self.completed)

    def seconds(self, seconds_per_inference: float) -> float:
        return self.remaining * seconds_per_inference / max(1, self.parallelism)


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """Every stage of one experiment, and what the whole thing comes to."""

    stages: tuple[StagePlan, ...]
    decision_points: int
    contested_points: int
    contested_estimated: bool
    seconds_per_inference: float

    @property
    def total(self) -> int:
        return sum(stage.inferences for stage in self.stages)

    @property
    def completed(self) -> int:
        return sum(stage.completed for stage in self.stages)

    @property
    def remaining(self) -> int:
        return sum(stage.remaining for stage in self.stages)

    @property
    def seconds(self) -> float:
        return sum(stage.seconds(self.seconds_per_inference) for stage in self.stages)

    @property
    def is_estimated(self) -> bool:
        return any(stage.estimated for stage in self.stages)


def plan_experiment(
    *,
    settings: Settings,
    prices: pd.DataFrame,
    store: DecisionStore,
    contested: Sequence[PointKey] | None = None,
    decisions: pd.DataFrame | None = None,
    compositions: Sequence[Composition] | None = None,
    personas: Sequence[Persona] = PERSONAS,
    rebuttal_rounds: int | None = None,
    seconds_per_inference: float = SECONDS_PER_INFERENCE,
    assumed_contested_share: float = ASSUMED_CONTESTED_SHARE,
) -> ExperimentPlan:
    """Count what this configuration implies, issuing nothing.

    Args:
        contested: the points a debate will be run on, ordinarily
            ``tuple(d.point for d in select_contested(...))``. ``None`` means the
            independent arm has not been generated yet, so the debate stages fall
            back to :data:`ASSUMED_CONTESTED_SHARE` and are marked estimated. It is
            also *ignored* while that arm is unfinished -- see below.
        decisions: the stored decisions, used only to tell which contested points
            the placebo arm can actually draw a donor for. Omitted, the placebo
            stage counts every contested point, which is what the sweep would
            spend only if every point had a donor.
        rebuttal_rounds: defaults to ``settings.max_debate_rounds``, the same
            resolution :func:`council.debate.sweep.run_debate_arms` makes, so a plan
            and the run it prices cannot disagree about the length of a
            conversation.
    """
    if rebuttal_rounds is None:
        rebuttal_rounds = settings.max_debate_rounds
    committees = tuple(
        balanced_design(models=settings.agent_models) if compositions is None else compositions
    )
    run_plan = GenerationRunner(
        settings=settings, prices=prices, store=store, personas=personas
    ).plan()
    # A contested set measured over a half-generated control arm is a measurement
    # of the half, and the grid is swept model then persona then ticker, so an
    # interrupted run leaves a slice rather than a sample. Counting the debate
    # stages from it would print "(measured)" over a figure derived from agents
    # that do not exist yet; falling back to the assumed share prints the estimate
    # marker `render_plan` already has.
    if run_plan.remaining > 0:
        contested = None
    done = store.completed_keys()
    points = len(run_plan.decision_dates) * len(settings.tickers)
    stages = [
        StagePlan(
            stage=INDEPENDENT_STAGE,
            arm=str(Arm.INDEPENDENT),
            inferences=run_plan.total,
            completed=run_plan.completed,
            parallelism=settings.concurrency,
            estimated=False,
        ),
        *(
            _debate_stage(
                arm=arm,
                committees=committees,
                contested=contested,
                assumed=round(points * assumed_contested_share),
                done=done,
                rebuttal_rounds=rebuttal_rounds,
                decisions=decisions,
                min_gap=settings.placebo_min_gap_sessions,
            )
            for arm in TREATMENT_ARMS
        ),
    ]
    return ExperimentPlan(
        stages=tuple(stages),
        decision_points=points,
        contested_points=(
            round(points * assumed_contested_share) if contested is None else len(contested)
        ),
        contested_estimated=contested is None,
        seconds_per_inference=seconds_per_inference,
    )


def _debate_stage(
    *,
    arm: Arm,
    committees: Sequence[Composition],
    contested: Sequence[PointKey] | None,
    assumed: int,
    done: frozenset[DecisionKey],
    rebuttal_rounds: int,
    decisions: pd.DataFrame | None = None,
    min_gap: int | None = None,
) -> StagePlan:
    """One debate arm's stage, counted exactly where that is possible.

    Parallelism is the number of distinct base models at the table rather than
    ``settings.concurrency``. A debate round puts one request per seat in flight
    and then waits for all of them before the next round can be rendered, so the
    ceiling is the committee, not the queue -- and every model has to be resident
    at once, which is the operational fact this figure is really reporting.
    """
    parallelism = max(len({seat.model for seat in table.seats}) for table in committees)
    per_conversation = sum(table.size for table in committees) * (rebuttal_rounds + 1)
    if contested is None:
        return StagePlan(
            stage=DEBATE_STAGE,
            arm=str(arm),
            inferences=assumed * per_conversation,
            completed=0,
            parallelism=parallelism,
            estimated=True,
        )

    # Counted per conversation, not per row, because that is the unit the sweep
    # resumes on: `council.debate.sweep._Sweep.group` re-holds a whole conversation
    # unless *every* one of its keys is stored. A conversation missing one row --
    # one round-1 row recording a retriable failure, or one abandoned after its
    # opening round -- costs seats x (rebuttal_rounds + 1) inferences on the next
    # run. Billing the missing rows alone under-reports the resume budget by up to
    # that factor, on exactly the runs where resuming is what the plan is for.
    conversations = {
        conversation_keys(
            composition=table,
            arm=arm,
            decision_date=decision_date,
            ticker=ticker,
            rebuttal_rounds=rebuttal_rounds,
        )
        for table in committees
        for decision_date, ticker in _points_the_sweep_will_hold(
            contested, arm=arm, composition=table, decisions=decisions, min_gap=min_gap
        )
    }
    return StagePlan(
        stage=DEBATE_STAGE,
        arm=str(arm),
        inferences=sum(len(keys) for keys in conversations),
        completed=sum(len(keys) for keys in conversations if done.issuperset(keys)),
        parallelism=parallelism,
        estimated=False,
    )


def _points_the_sweep_will_hold(
    contested: Sequence[PointKey],
    *,
    arm: Arm,
    composition: Composition,
    decisions: pd.DataFrame | None,
    min_gap: int | None,
) -> tuple[PointKey, ...]:
    """The contested points this arm will actually be run on.

    Two of the three arms debate every one of them. The placebo does not:
    :func:`council.debate.sweep.run_debate_arms` skips a point whose pool holds no
    usable earlier day, so counting all of them quotes work no run will ever spend
    and leaves ``remaining`` unable to reach zero however many times ``debate`` is
    run -- which is exactly what this module's docstring promises cannot happen.
    """
    if arm is not Arm.DEBATE_PLACEBO or decisions is None:
        return tuple(contested)
    # Imported here rather than at module scope: council.debate.sweep reads
    # TREATMENT_ARMS and conversation_keys from this module, so a top-level import
    # would close the cycle.
    from council.debate.sweep import has_donor, placebo_pool_for

    pool = placebo_pool_for(decisions, composition=composition)
    return tuple(
        point
        for point in contested
        if has_donor(pool, point, required_seats=composition.size, min_gap=min_gap)
    )

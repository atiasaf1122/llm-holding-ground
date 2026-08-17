"""What a configuration costs, counted before a single token is generated.

Deciding whether to commit an evening to a sweep should not require starting it,
so every number here is arithmetic over the grid rather than a measurement of it.
:class:`~council.agents.runner.RunPlan` already does this for the independent arm;
what this module adds is the debate arms, which cannot be counted the same
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
from council.agents.store import ConversationKey, DecisionStore
from council.config import Settings
from council.debate.compositions import Composition, balanced_design
from council.debate.protocol import arm_round_cap
from council.domain.persona import PERSONAS, Persona
from council.domain.signal import Arm
from council.evaluation.frames import PointKey

TREATMENT_ARMS: Final[tuple[Arm, ...]] = (
    Arm.DEBATE,
    Arm.DEBATE_RATIONALE_ONLY,
    Arm.DEBATE_PLACEBO,
    Arm.DEBATE_PLACEBO_SAME,
    Arm.DEBATE_CONTRADICTOR,
)
"""The five debate arms, in the order a sweep runs them: the original three,
then the two extension arms added after the first results were published --
the same-instrument placebo (D14's decomposition) and the coherent
contradictor (D8's adjudicator). A resumed sweep skips whatever is complete,
so extending this tuple re-runs nothing.

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

Every contested share this design has measured is at or near 100% -- 984 of 1,002 on
the two-year real-price run, see ``docs/findings.md`` section 7 and ``docs/CLAIMS.md``
C14 -- because the crossed
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
same run gives 4,728 of 8,016, 59% -- ranging from 19.0% for a uniform committee to
98.5% for a rotation (``docs/CLAIMS.md`` C14). That is the figure a
per-committee gate would spend at, and no such gate exists; assuming the pooled share
therefore still overstates rather than understates what this code will run.
"""

INDEPENDENT_STAGE: Final = "generate"
DEBATE_STAGE: Final = "debate"


def conversation_key(
    *,
    composition: Composition,
    arm: Arm,
    decision_date: date,
    ticker: str,
) -> ConversationKey:
    """The identity of one conversation, as a plan and a sweep both read it.

    This used to be ``conversation_keys``: every *row* a conversation produces,
    seats crossed with rounds ``0..cap``. Both the sweep's resume check and this
    module's ``completed`` count asked whether the store held all of them, and while
    every conversation ran to the cap that was the same question as "is it
    finished". It stopped being that question the moment a conversation could end on
    agreement: a debate that agreed at round two holds five fewer rows than the cap
    implies and can never satisfy the test, so the sweep re-holds a point it already
    owns and ``remaining`` never reaches zero however many times ``debate`` is run.

    So the identity dropped the round and the seat, and whether a conversation is
    finished is asked of :attr:`~council.domain.signal.Decision.stop_reason` through
    :meth:`council.agents.store.DecisionStore.completed_conversations`. Both callers
    still derive the key from this one function, which is what stops a resumed run
    from re-debating points it already owns.
    """
    return (decision_date, ticker, str(arm), composition.identifier)


def conversation_rows(*, composition: Composition, rebuttal_rounds: int) -> int:
    """The most rows one conversation can produce: every seat in every round.

    The **most**, not the number. A conversation that agrees early writes fewer, and
    nothing can know in advance which will. A plan is a budget, so it quotes the
    bound -- see :func:`_debate_stage`.
    """
    return composition.size * (rebuttal_rounds + 1)


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
        decisions: the stored decisions, used only to tell which contested points a
            placebo donor can be drawn for -- which is now what **every** arm is run
            on, since :func:`council.debate.sweep.run_debate_arms` filters the point
            set once and hands the survivors to all of them. Omitted, every stage
            counts every contested point, which is what the sweep would spend only
            if every point had a donor.
        rebuttal_rounds: defaults to ``settings.max_debate_rounds``, the same
            resolution :func:`council.debate.sweep.run_debate_arms` makes, so a plan
            and the run it prices cannot disagree about the cap a conversation may
            run to.

    Every debate stage is an **upper bound** rather than an exact count, and that is
    a property of the experiment rather than a weakness of the arithmetic: a
    conversation ends on agreement, on stillness or at the cap, and which of those
    fires cannot be known without running it. So a stage quotes the cap's worth of
    rows per conversation, and a run spends that or less. ``completed`` is counted at
    the same width for every conversation the store says has finished, so a plan over
    a finished run still reads zero remaining rather than stalling short of it.
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
    elif contested is not None:
        # The same filter the sweep applies, applied once and to every stage --
        # because the sweep applies it once and to every arm. Counting the placebo
        # alone at the servable points, which is what this did while the placebo was
        # the only arm that skipped, now prices stages the run will not spend
        # and prints a different figure per arm for arms that answer one set.
        contested = _points_the_sweep_will_hold(
            contested,
            committees=committees,
            decisions=decisions,
            min_gap=settings.placebo_min_gap_sessions,
            rounds=rebuttal_rounds,
        )
    done = store.completed_conversations()
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


def _conversation_calls(table: Composition, *, arm: Arm, rebuttal_rounds: int) -> int:
    """Every model call one conversation costs in this arm, counters included.

    The contradictor's peers each author a counter-argument against each reader
    before the rebuttal round -- ``size * (size - 1)`` generation calls that are
    not stored as decision rows and would otherwise be free in the plan and paid
    on the night.
    """
    calls = conversation_rows(composition=table, rebuttal_rounds=rebuttal_rounds)
    if arm is Arm.DEBATE_CONTRADICTOR:
        calls += table.size * (table.size - 1)
    return calls


def _debate_stage(
    *,
    arm: Arm,
    committees: Sequence[Composition],
    contested: Sequence[PointKey] | None,
    assumed: int,
    done: frozenset[ConversationKey],
    rebuttal_rounds: int,
) -> StagePlan:
    """One debate arm's stage: a bound on what it costs, and what is already paid.

    Parallelism is the number of distinct base models at the table rather than
    ``settings.concurrency``. A debate round puts one request per seat in flight
    and then waits for all of them before the next round can be rendered, so the
    ceiling is the committee, not the queue -- and every model has to be resident
    at once, which is the operational fact this figure is really reporting.

    Every arm is counted over one point set, because the sweep runs them over
    one point set: the caller has already filtered it.
    """
    parallelism = max(len({seat.model for seat in table.seats}) for table in committees)
    rounds = arm_round_cap(arm, rebuttal_rounds)
    per_point = sum(
        _conversation_calls(table, arm=arm, rebuttal_rounds=rounds) for table in committees
    )
    if contested is None:
        return StagePlan(
            stage=DEBATE_STAGE,
            arm=str(arm),
            inferences=assumed * per_point,
            completed=0,
            parallelism=parallelism,
            estimated=True,
        )

    # Counted per conversation, not per row, because that is the unit the sweep
    # resumes on: `council.debate.sweep._Sweep.group` re-holds a whole conversation
    # unless the store says it reached a stopping condition. A conversation one
    # retriable failure short of finished costs a whole conversation on the next
    # run, so billing the missing rows alone under-reports the resume budget -- on
    # exactly the runs where resuming is what the plan is for. And it is billed at
    # the cap's width in both directions: a conversation that will stop early cannot
    # be known in advance, so `inferences` is a bound, and `completed` has to use
    # the same width or a finished run would never read as finished.
    conversations = [
        (
            conversation_key(
                composition=table, arm=arm, decision_date=decision_date, ticker=ticker
            ),
            _conversation_calls(table, arm=arm, rebuttal_rounds=rounds),
        )
        for table in committees
        for decision_date, ticker in contested
    ]
    return StagePlan(
        stage=DEBATE_STAGE,
        arm=str(arm),
        inferences=sum(rows for _, rows in conversations),
        completed=sum(rows for key, rows in conversations if key in done),
        parallelism=parallelism,
        estimated=False,
    )


def _points_the_sweep_will_hold(
    contested: Sequence[PointKey],
    *,
    committees: Sequence[Composition],
    decisions: pd.DataFrame | None,
    min_gap: int,
    rounds: int,
) -> tuple[PointKey, ...]:
    """The contested points every arm will actually be run on.

    Every arm, not the placebo alone. :func:`council.debate.sweep.run_debate_arms`
    withholds a point no committee can draw a placebo donor for from *every* arm, so
    that they cover one calendar and a difference between them is not partly a
    difference in which days they answered. A plan that counted all of them for some
    arms and the servable ones for others would quote work no run will spend and
    leave ``remaining`` unable to reach zero however many times ``debate`` is run --
    which is exactly what this module's docstring promises cannot happen.

    ``decisions`` omitted, nothing is filtered: the caller has not supplied what the
    question needs, and inventing an answer is worse than an over-count that says so.
    """
    if decisions is None:
        return tuple(contested)
    # Imported here rather than at module scope: council.debate.sweep reads
    # TREATMENT_ARMS and conversation_key from this module, so a top-level import
    # would close the cycle. Its function rather than a copy of its rule, because a
    # plan and the sweep it prices disagreeing about which points are in the
    # experiment is the defect this whole path exists to remove.
    from council.debate.sweep import servable_points

    servable = servable_points(
        contested,
        decisions=decisions,
        committees=committees,
        min_gap=min_gap,
        rounds=rounds,
    )
    return tuple(point for point in contested if point in servable)

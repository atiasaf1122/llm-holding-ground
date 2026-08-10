"""The contract every agent answers with, and the record that is stored.

Two models, kept apart on purpose.

:class:`Signal` is what the language model is constrained to emit. It is small,
every string in it is bounded, and it contains nothing the model could not know --
so the same schema serves the independent arm and every debate arm.

:class:`Decision` is what lands on disk. It wraps a signal with everything needed
to reproduce it and to slice the results later. The extra columns exist because
adding one after eighty thousand inferences means running them again.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

# The model is asked to keep prose short and the schema enforces it. An unbounded
# string field under constrained decoding is not merely untidy: the grammar
# permits any character inside a JSON string, so a model with more to say than the
# schema has room for pours everything into the free field and never closes the
# quote. See docs/research.md.
MAX_RATIONALE_CHARS = 400


class Signal(BaseModel):
    """One agent's view at one decision point."""

    model_config = ConfigDict(frozen=True)

    exposure: float = Field(ge=-1.0, le=1.0)
    """Desired share of this ticker's capital. +1 fully long, 0 flat, -1 fully short."""

    confidence: float = Field(ge=0.0, le=1.0)
    """Self-reported, and *not* used to weight anything in the aggregation.

    Whether it means anything is one of the questions this project asks -- using it
    to aggregate before measuring whether it is calibrated would answer that
    question with itself.
    """

    rationale: str = Field(max_length=MAX_RATIONALE_CHARS)
    """One or two sentences. Read by the analysis, and shown to peers in a debate."""


class Arm(StrEnum):
    """Which experimental condition produced a decision.

    The four arms are what separate "the model was persuaded" from three cheaper
    explanations, and each exists to rule one of them out.
    """

    INDEPENDENT = "independent"
    """No peers. The control every other arm is measured against."""

    DEBATE = "debate"
    """Peers' rationales *and* their exposures were shown."""

    DEBATE_RATIONALE_ONLY = "debate_rationale_only"
    """Peers' rationales were shown without their numbers.

    The difference between this and :attr:`DEBATE` is anchoring: how much of the
    convergence was reasoning, and how much was drifting toward a number on the
    page. Without this arm the two are indistinguishable.
    """

    DEBATE_PLACEBO = "debate_placebo"
    """Peers' rationales came from an unrelated day.

    If agents move as much here as in a real debate, they are not responding to
    the argument at all -- only to being contradicted. That would make every
    result in the debate arm mean something quite different.
    """


class FailureMode(StrEnum):
    """Why a decision point produced no model output."""

    NONE = "none"
    MALFORMED = "malformed"
    """The model never produced a valid object, after retries."""

    TRUNCATED = "truncated"
    """Generation was cut off, so the JSON was incomplete."""

    UNAVAILABLE = "unavailable"
    """The backend could not be reached."""


class StopReason(StrEnum):
    """Why a conversation ended.

    Declared here rather than in :mod:`council.debate.protocol`, which is where the
    predicates that decide it live, because it is a *stored* column:
    :attr:`Decision.stop_reason` carries it to disk beside the arm and the failure
    mode, and those are declared here for the same reason. The protocol imports it
    back, so ``from council.debate.protocol import StopReason`` still names this
    enum.

    Which condition ended a conversation says something the round count alone does
    not, and at a cap above one it is also what a resumed run reads to tell a
    conversation that finished from one still owing rounds. See
    :meth:`council.debate.sweep._Sweep.group`.
    """

    AGREED = "agreed"
    """The seats came within ``Settings.agreement_spread`` of each other."""

    SETTLED = "settled"
    """Nobody moved for ``Settings.stillness_rounds`` consecutive rounds.

    The interesting one: a committee that stops without agreeing has not
    converged, it has entrenched. Two quiet rounds rather than one because an
    agent that ignored an argument on first reading may take it on the second, and
    calling it entrenched after a single quiet round would deny it that -- which
    is why it needs at least ``stillness_rounds`` rebuttal rounds to occur at all,
    and why it was unreachable while the cap was pinned at one.
    """

    CAP = "cap"
    """The round limit was reached while the committee was still moving.

    Not a failure. A conversation still in motion at the cap has no equilibrium
    within the budget, and that is a result about the committee.
    """

    NO_SPEAKERS = "no_speakers"
    """A whole round failed to generate, leaving the next with nothing to answer.

    A stopping condition like the other three -- the conversation is over and will
    not be retried -- but not a conversation *held*:
    :meth:`council.debate.sweep._Sweep.hold` books it as abandoned. The two
    readings coexist because a round every seat botched reproduces exactly under a
    zero temperature, so re-running it on a resume would spend a night confirming
    it; a round lost to an unreachable daemon is a
    :attr:`FailureMode.UNAVAILABLE` row, and those are what
    :meth:`council.agents.store.DecisionStore.completed_conversations` refuses to
    call finished.
    """


COMPLETED_STOP_REASONS: frozenset[StopReason] = frozenset(
    {StopReason.AGREED, StopReason.SETTLED, StopReason.CAP, StopReason.NO_SPEAKERS}
)
"""Every reason that means a conversation will not be run again.

All four, :attr:`StopReason.NO_SPEAKERS` included, for the reason its own
docstring gives. Written out member by member rather than as ``frozenset(StopReason)``
so that a fifth reason has to be classified deliberately instead of inheriting
"finished" from the enum it was added to.
"""


class Decision(BaseModel):
    """One stored row: an agent, a moment, and what it said.

    Never dropped. A decision point that failed is written with a flat exposure and
    its failure recorded, because the rate of failure per model is itself a result
    -- and silently missing rows would quietly bias every arm that contains them.
    """

    model_config = ConfigDict(frozen=True)

    # -- identity -------------------------------------------------------------
    decision_date: date
    ticker: str
    model: str
    persona: str

    # -- experimental arm -----------------------------------------------------
    arm: Arm = Arm.INDEPENDENT
    round_index: int = Field(default=0, ge=0)
    """0 is the opening view. 1 is after seeing peers."""

    composition: str | None = None
    """Which committee configuration this belongs to; ``None`` in the independent arm.

    A debate result is only meaningful inside the group that produced it: a
    conversation between four particular agents cannot be reused for a committee
    containing a fifth. This column is what keeps those apart.
    """

    stop_reason: StopReason | None = None
    """How the conversation this row belongs to ended; ``None`` in the independent arm.

    A property of the whole conversation written onto each of its rows, because a
    stored row is the only unit this project persists and a conversation has no file
    of its own. It is stamped once the conversation is over --
    :meth:`council.debate.caller.DecisionCaller.stamped` -- since no turn can know
    while it is being taken which round will turn out to be the last.

    ``None`` also on the rows of a conversation that never reached a stopping
    condition at all: one that raised out of
    :func:`council.debate.protocol.run_debate` part way through. That is what makes
    this column readable as a resume marker. A conversation is finished when its
    stored rows say *why* it ended, not when it holds a row for every round up to
    the cap -- which, once the cap stopped being the length of every conversation,
    no conversation that agreed early could ever satisfy. See
    :meth:`council.agents.store.DecisionStore.completed_conversations`.
    """

    # -- output ---------------------------------------------------------------
    exposure: float = Field(ge=-1.0, le=1.0)
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""

    # -- provenance -----------------------------------------------------------
    prompt_hash: str
    seed: int
    generated_at: datetime

    # -- diagnostics ----------------------------------------------------------
    failure: FailureMode = FailureMode.NONE
    retries: int = Field(default=0, ge=0)
    latency_seconds: float = Field(default=0.0, ge=0.0)
    output_tokens: int = Field(default=0, ge=0)

    @property
    def is_failure(self) -> bool:
        return self.failure is not FailureMode.NONE

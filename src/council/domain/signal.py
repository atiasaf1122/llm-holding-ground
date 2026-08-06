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

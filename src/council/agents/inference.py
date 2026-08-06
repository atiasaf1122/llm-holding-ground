"""One decision point, turned into one stored row -- whatever the model does.

The rule this module exists to enforce is that a decision point always produces a
row. Generation can fail in three ways and a model can return a well-formed object
that is not a signal; all four end in a :class:`~council.domain.signal.Decision`
with a flat exposure and the reason recorded. A missing row would be worse than a
useless one: it removes an agent from some days and not others, which is a
selection effect shaped exactly like the thing under study.

Two errors are deliberately *not* caught, because they are not decisions failing:
:class:`~council.agents.provider.PreflightError` and
:class:`~council.agents.schema.UnsupportedSchemaError` both say the run should not
have started, and recording eighty thousand identical failures as data would be
the expensive way to find that out.

:func:`generate_decision` takes peer views and an arm, so the debate layer reuses
it rather than reimplementing the failure handling for the treatment arms only --
where the arms would then differ by more than the manipulation. The reuse is
:class:`council.debate.caller.DecisionCaller`, and it is the only other caller:
a second one that rendered its own peer block would decide what the treatment arm
says, and the control would no longer be a control for it.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from pydantic import ValidationError

from council.agents.prompt import SIGNAL_SCHEMA, PeerView, RenderedPrompt, build_prompt
from council.agents.provider import (
    Completion,
    MissingModelError,
    PreflightError,
    Provider,
    ProviderError,
    ProviderUnavailableError,
    TruncatedGenerationError,
)
from council.agents.schema import UnsupportedSchemaError
from council.agents.store import CompletionRecord, DecisionKey
from council.domain.persona import Persona
from council.domain.signal import Arm, Decision, FailureMode, Signal
from council.evaluation.frames import NO_COMPOSITION


@dataclass(frozen=True, slots=True)
class DecisionPoint:
    """Where one decision sits in the experiment: who decided, when, under which arm."""

    model: str
    persona: Persona
    ticker: str
    decision_date: date
    arm: Arm = Arm.INDEPENDENT
    round_index: int = 0
    composition: str | None = None

    @property
    def key(self) -> DecisionKey:
        """The identity the resume check compares against what is already stored.

        Derived from the point rather than from the decision it produces, so a run
        can tell what is missing before generating anything. It is pinned by test
        against :func:`council.agents.store.decision_key`, which reads the same
        identity off a stored row; if the two drifted apart, a resumed run would
        regenerate a sweep it already owns.
        """
        return (
            self.decision_date,
            self.ticker,
            self.model,
            self.persona.name,
            str(self.arm),
            self.round_index,
            self.composition or NO_COMPOSITION,
        )


def failure_mode(error: Exception) -> FailureMode:
    """Map a generation error onto the reason stored with the flat decision.

    Order matters twice over. :class:`~council.agents.provider.ContextOverflowError`
    is a truncation and is caught by the first branch, which is the point of it
    being a subclass. ``ValidationError`` falls through to ``MALFORMED``: a grammar
    fixes the syntax of a completion and nothing else, so an ``exposure`` of 5.0 is
    a well-formed object that is not a signal, and the two are the same failure as
    far as this experiment is concerned.
    """
    if isinstance(error, TruncatedGenerationError):
        return FailureMode.TRUNCATED
    if isinstance(error, ProviderUnavailableError | MissingModelError):
        return FailureMode.UNAVAILABLE
    return FailureMode.MALFORMED


async def generate_decision(
    provider: Provider,
    *,
    point: DecisionPoint,
    price_context: str,
    peers: Sequence[PeerView] = (),
    seed: int,
    now: datetime,
    max_tokens: int | None = None,
) -> tuple[Decision, CompletionRecord]:
    """Produce one decision, whatever happens, plus the archive line for it.

    Retries are the provider's business and are not repeated here. Temperature is
    zero, so a second attempt at a malformed completion spends another minute of
    GPU time reproducing the first one exactly; the transport failures a retry
    could actually fix are already retried below this call.

    Raises:
        PreflightError: the backend is unfit, so the run should stop.
        UnsupportedSchemaError: the same, for a defect in the schema.
    """
    rendered = build_prompt(
        persona=point.persona,
        price_context=price_context,
        arm=point.arm,
        peers=peers,
        round_index=point.round_index,
    )
    started = time.perf_counter()
    try:
        completion = await provider.generate(
            system=rendered.system,
            user=rendered.user,
            schema=SIGNAL_SCHEMA,
            max_tokens=max_tokens,
        )
        signal = Signal.model_validate(completion.data)
    except (PreflightError, UnsupportedSchemaError):
        raise
    except (ProviderError, ValidationError) as error:
        elapsed = time.perf_counter() - started
        decision = _failed_decision(point, rendered, error, seed=seed, now=now, latency=elapsed)
        return decision, _record(point, rendered, error=_describe(error), latency=elapsed)

    return (
        _decision(point, rendered, signal, completion, seed=seed, now=now),
        _record(
            point,
            rendered,
            response=dict(completion.data),
            latency=completion.latency_seconds,
        ),
    )


def _decision(
    point: DecisionPoint,
    rendered: RenderedPrompt,
    signal: Signal,
    completion: Completion,
    *,
    seed: int,
    now: datetime,
) -> Decision:
    return Decision(
        decision_date=point.decision_date,
        ticker=point.ticker,
        model=point.model,
        persona=point.persona.name,
        arm=point.arm,
        round_index=point.round_index,
        composition=point.composition,
        exposure=signal.exposure,
        confidence=signal.confidence,
        rationale=signal.rationale,
        prompt_hash=rendered.prompt_hash,
        seed=seed,
        generated_at=now,
        retries=completion.retries,
        latency_seconds=completion.latency_seconds,
        output_tokens=completion.output_tokens,
    )


def _failed_decision(
    point: DecisionPoint,
    rendered: RenderedPrompt,
    error: Exception,
    *,
    seed: int,
    now: datetime,
    latency: float,
) -> Decision:
    """A decision point that produced nothing, written down as such.

    Flat and unconfident, so anything reading exposures treats it as no position
    rather than as a position; ``failure`` is what stops it being read afterwards
    as an agent that deliberately went flat -- which, in a debate round, would read
    as an agent that abandoned its opening view.
    """
    return Decision(
        decision_date=point.decision_date,
        ticker=point.ticker,
        model=point.model,
        persona=point.persona.name,
        arm=point.arm,
        round_index=point.round_index,
        composition=point.composition,
        exposure=0.0,
        confidence=0.0,
        rationale="",
        prompt_hash=rendered.prompt_hash,
        seed=seed,
        generated_at=now,
        failure=failure_mode(error),
        latency_seconds=latency,
    )


def _record(
    point: DecisionPoint,
    rendered: RenderedPrompt,
    *,
    response: Mapping[str, object] | None = None,
    error: str | None = None,
    latency: float,
) -> CompletionRecord:
    return CompletionRecord(
        decision_date=point.decision_date,
        ticker=point.ticker,
        model=point.model,
        persona=point.persona.name,
        arm=str(point.arm),
        round_index=point.round_index,
        composition=point.composition or NO_COMPOSITION,
        prompt_hash=rendered.prompt_hash,
        system=rendered.system,
        user=rendered.user,
        response=dict(response) if response is not None else None,
        error=error,
        latency_seconds=latency,
    )


def _describe(error: Exception) -> str:
    return f"{type(error).__name__}: {error}"

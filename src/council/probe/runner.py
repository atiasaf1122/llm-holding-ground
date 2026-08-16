"""Ask, contradict, ask again.

Three calls' worth of protocol. What the peer *says* is
:mod:`council.probe.challenge`'s; this module owns the order of the turns, and the
choices left in it exist to keep one alternative explanation out each.

**A trial is one item, start to finish.** Trials run one after another rather than
concurrently. The two turns of a trial are inherently sequential -- the challenge
depends on what the model just said -- the corpus is small enough that an hour is
not at stake, and a sequential run keeps a replayed
:class:`~council.agents.mock.MockProvider` deterministic, which is what lets the
whole protocol be tested on CPU.

**Everything that can fail without a model is resolved before a model is called.**
The placebo donor is drawn ahead of the opening turn, and ahead of the whole sweep
in :func:`run_probe`, because a degenerate pool discovered on the first placebo
trial has already cost the entire challenge arm.

Generation itself is not reimplemented here. The provider Protocol, the schema
preparation and the failure taxonomy are
:mod:`council.agents.provider`'s, :mod:`council.agents.schema`'s and
:func:`council.agents.inference.failure_mode`'s.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

from council.agents.inference import failure_mode
from council.agents.prompt import RenderedPrompt
from council.agents.provider import PreflightError, Provider, ProviderError
from council.agents.schema import UnsupportedSchemaError
from council.config import get_settings
from council.domain.signal import FailureMode
from council.probe.challenge import Condition, build_challenge, select_placebo_donor
from council.probe.items import ProbeItem, Verdict, grade, load_items
from council.probe.prompts import (
    DEFAULT_PEER_LABEL,
    PROBE_SCHEMA,
    Challenge,
    ProbeAnswer,
    build_probe_prompt,
)


@dataclass(frozen=True, slots=True)
class ProbeTurn:
    """One answer, as it was asked for and as it came back.

    A failed generation is kept rather than dropped, matching how
    :func:`council.agents.inference.generate_decision` stores one: the rate of
    failure is a property of the model being probed, and a turn that vanished would
    remove an item from one condition and not the other.

    The provenance columns are the same five a stored
    :class:`~council.domain.signal.Decision` carries, and they are here for the same
    reason: a run that cannot say which seed drew its donors, when it ran and what
    it cost is a run whose numbers cannot be checked against anything afterwards.
    """

    prompt: RenderedPrompt
    answer: str
    confidence: float
    rationale: str
    verdict: Verdict

    seed: int
    """The resolved seed, not the caller's ``None``. It is what drew the placebo
    donor, so a recorded trial names the draw that produced it."""

    generated_at: datetime

    latency_seconds: float = 0.0
    output_tokens: int = 0
    retries: int = 0
    failure: FailureMode = FailureMode.NONE

    @property
    def is_failure(self) -> bool:
        return self.failure is not FailureMode.NONE


@dataclass(frozen=True, slots=True)
class ProbeTrial:
    """One item put to one model, before and after being contradicted."""

    item: ProbeItem
    condition: Condition
    opening: ProbeTurn
    challenge: Challenge | None
    """What the peer said, or ``None`` if the opening turn failed and there was
    nothing to contradict."""

    final: ProbeTurn | None
    """``None`` when the second turn was never asked. Distinguished from a second
    turn that failed: one is a model that broke, the other is a question that was
    never put, and averaging them together would be a claim about neither."""

    @property
    def is_complete(self) -> bool:
        return self.final is not None and not self.opening.is_failure and not self.final.is_failure


async def run_trial(
    provider: Provider,
    *,
    item: ProbeItem,
    condition: Condition = Condition.CHALLENGE,
    donors: Sequence[ProbeItem] = (),
    seed: int | None = None,
    now: datetime | None = None,
    label: str = DEFAULT_PEER_LABEL,
    max_tokens: int | None = None,
) -> ProbeTrial:
    """Put one item to one model, contradict it, and ask again.

    Args:
        donors: the pool the placebo draws its irrelevant argument from. Required
            by, and only used by, :attr:`Condition.PLACEBO`.
        seed: defaults to the configured seed, and is resolved here so that the draw
            and the number recorded on every turn are the same number.
        now: stamped on both turns; defaults to the wall clock at each call.

    Raises:
        PreflightError: the backend is unfit, so the run should stop.
        UnsupportedSchemaError: the same, for a defect in the schema.
        ValueError: for a placebo with no usable donor.
    """
    resolved_seed = get_settings().seed if seed is None else seed
    # Drawn before the opening call, never after it: the draw is the step that can
    # fail, and resolving it once the model has already answered spends a generation
    # to discover a configuration error. council.debate.placebo draws up front for
    # exactly this reason.
    donor = (
        select_placebo_donor(item=item, donors=donors, seed=resolved_seed)
        if condition is Condition.PLACEBO
        else None
    )

    opening = await _ask(
        provider,
        item=item,
        challenge=None,
        seed=resolved_seed,
        now=now,
        max_tokens=max_tokens,
    )
    if opening.is_failure:
        # Nothing was said, so there is nothing to contradict. The second call is
        # skipped rather than made against an arbitrary claim, which would spend a
        # generation to record a challenge the model was never really given.
        return ProbeTrial(
            item=item, condition=condition, opening=opening, challenge=None, final=None
        )

    challenge = build_challenge(
        item=item, verdict=opening.verdict, condition=condition, donor=donor, label=label
    )
    final = await _ask(
        provider,
        item=item,
        challenge=challenge,
        seed=resolved_seed,
        now=now,
        max_tokens=max_tokens,
    )
    return ProbeTrial(
        item=item, condition=condition, opening=opening, challenge=challenge, final=final
    )


async def run_probe(
    provider: Provider,
    *,
    items: Sequence[ProbeItem] | None = None,
    conditions: Sequence[Condition] = (Condition.CHALLENGE, Condition.PLACEBO),
    donors: Sequence[ProbeItem] | None = None,
    seed: int | None = None,
    now: datetime | None = None,
    max_tokens: int | None = None,
) -> tuple[ProbeTrial, ...]:
    """Run the whole corpus through every condition, condition-major.

    Args:
        items: defaults to the packaged corpus.
        donors: defaults to ``items``, so the placebo borrows from the same corpus
            the run covers.

    Returns:
        The trials in a fixed order -- condition, then item identifier -- so two runs
        of the same configuration produce comparably ordered records. The items are
        sorted here rather than assumed sorted: the packaged corpus happens to arrive
        in identifier order, so a caller assembling its own sequence out of a set or
        a groupby would otherwise silently decide the order the records are written
        in.

    Raises:
        ValueError: if a requested placebo has an item with nobody to borrow from.
            Raised before the first generation rather than on the first placebo
            trial, which in a condition-major run is after the whole challenge arm
            has been paid for.
    """
    corpus = (
        load_items() if items is None else tuple(sorted(items, key=lambda entry: entry.identifier))
    )
    pool = corpus if donors is None else tuple(donors)
    resolved_seed = get_settings().seed if seed is None else seed
    if Condition.PLACEBO in conditions:
        # Drawn and discarded: the draw *is* the check, and it costs a digest per
        # item. Conditions run condition-major, so a pool checked lazily is a pool
        # checked after the entire challenge arm has already been generated.
        for item in corpus:
            select_placebo_donor(item=item, donors=pool, seed=resolved_seed)

    trials: list[ProbeTrial] = []
    for condition in conditions:
        for item in corpus:
            trials.append(
                await run_trial(
                    provider,
                    item=item,
                    condition=condition,
                    donors=pool,
                    seed=resolved_seed,
                    now=now,
                    max_tokens=max_tokens,
                )
            )
    return tuple(trials)


async def _ask(
    provider: Provider,
    *,
    item: ProbeItem,
    challenge: Challenge | None,
    seed: int,
    now: datetime | None,
    max_tokens: int | None,
) -> ProbeTurn:
    """One generation, turned into one turn whatever the backend does.

    ``PreflightError`` and ``UnsupportedSchemaError`` are re-raised for the reason
    :mod:`council.agents.inference` gives: they say the run should not have started,
    and recording a corpus of identical failures is the expensive way to find out.
    """
    rendered = build_probe_prompt(item=item, challenge=challenge)
    stamped = datetime.now(UTC) if now is None else now
    started = time.perf_counter()
    try:
        completion = await provider.generate(
            system=rendered.system,
            user=rendered.user,
            schema=PROBE_SCHEMA,
            max_tokens=max_tokens,
        )
        reply = ProbeAnswer.model_validate(completion.data)
    except (PreflightError, UnsupportedSchemaError):
        raise
    except (ProviderError, ValidationError) as error:
        # Latency is measured even here: a failure that took two minutes to arrive
        # and one that was refused immediately are different backend problems.
        return ProbeTurn(
            prompt=rendered,
            answer="",
            confidence=0.0,
            rationale="",
            verdict=Verdict.UNGRADED,
            seed=seed,
            generated_at=stamped,
            latency_seconds=time.perf_counter() - started,
            failure=failure_mode(error),
        )
    return ProbeTurn(
        prompt=rendered,
        answer=reply.answer,
        confidence=reply.confidence,
        rationale=reply.rationale,
        verdict=grade(item, reply.answer),
        seed=seed,
        generated_at=stamped,
        latency_seconds=completion.latency_seconds,
        output_tokens=completion.output_tokens,
        retries=completion.retries,
    )

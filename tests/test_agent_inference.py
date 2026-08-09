"""Whether one decision point always produces exactly one row.

Against :class:`~council.agents.mock.MockProvider`: no daemon, no network, no GPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, date, datetime
from typing import Any

import pytest

from council.agents.inference import DecisionPoint, failure_mode, generate_decision
from council.agents.mock import MockProvider
from council.agents.prompt import PeerView
from council.agents.provider import (
    Completion,
    ContextOverflowError,
    MalformedOutputError,
    MissingModelError,
    PreflightError,
    ProviderUnavailableError,
    TruncatedGenerationError,
)
from council.agents.schema import UnsupportedSchemaError
from council.agents.store import decision_key
from council.domain.persona import Aggression, Persona, Stance
from council.domain.signal import Arm, FailureMode

MOMENTUM_BOLD = Persona(stance=Stance.MOMENTUM, aggression=Aggression.BOLD)
FIXED_NOW = datetime(2026, 1, 1, tzinfo=UTC)
CONTEXT = "Daily returns %, oldest first.\n+1.00 -0.50 +2.00"


def make_point(
    *, arm: Arm = Arm.INDEPENDENT, composition: str | None = None, round_index: int = 0
) -> DecisionPoint:
    return DecisionPoint(
        model="alpha",
        persona=MOMENTUM_BOLD,
        ticker="AAA",
        decision_date=date(2022, 3, 1),
        arm=arm,
        round_index=round_index,
        composition=composition,
    )


# -- the failure taxonomy ------------------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MalformedOutputError("nope"), FailureMode.MALFORMED),
        (TruncatedGenerationError("cut off"), FailureMode.TRUNCATED),
        (ContextOverflowError("prompt too long"), FailureMode.TRUNCATED),
        (ProviderUnavailableError("daemon down"), FailureMode.UNAVAILABLE),
        (MissingModelError("not pulled"), FailureMode.UNAVAILABLE),
    ],
)
def test_each_provider_error_maps_onto_the_failure_it_records(
    error: Exception, expected: FailureMode
) -> None:
    assert failure_mode(error) is expected


# -- the happy path ------------------------------------------------------------


async def test_a_successful_generation_carries_the_signal_and_its_cost() -> None:
    provider = MockProvider(
        responses=[{"exposure": 0.4, "confidence": 0.7, "rationale": "trend intact"}]
    )

    decision, record = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.exposure == 0.4
    assert decision.confidence == 0.7
    assert decision.failure is FailureMode.NONE
    assert decision.seed == 7
    assert decision.generated_at == FIXED_NOW
    assert decision.output_tokens > 0
    assert record.response == {"exposure": 0.4, "confidence": 0.7, "rationale": "trend intact"}


async def test_the_point_and_the_stored_row_agree_about_what_a_decision_is() -> None:
    # If these drifted apart a resumed run would regenerate a sweep it already owns.
    # Round 1, because peers are only shown after the opening round.
    point = make_point(arm=Arm.DEBATE, composition="quad", round_index=1)

    decision, _ = await generate_decision(
        MockProvider(),
        point=point,
        price_context=CONTEXT,
        peers=[PeerView(label="Analyst 2", exposure=-0.5, rationale="stretched")],
        seed=7,
        now=FIXED_NOW,
    )

    assert point.key == decision_key(decision)


async def test_the_archived_line_holds_the_exact_prompt_that_was_sent() -> None:
    provider = MockProvider()

    decision, record = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    sent = provider.calls[0]
    assert record.system == sent.system
    assert record.user == sent.user
    assert record.prompt_hash == decision.prompt_hash


# -- failures are rows, not gaps -----------------------------------------------


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (MalformedOutputError("content was not JSON: 'well,'"), FailureMode.MALFORMED),
        (ContextOverflowError("the persona was dropped"), FailureMode.TRUNCATED),
        (ProviderUnavailableError("daemon down"), FailureMode.UNAVAILABLE),
    ],
)
async def test_a_failed_generation_is_stored_flat_with_its_reason(
    error: Exception, expected: FailureMode
) -> None:
    provider = MockProvider(responses=[error])

    decision, record = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.exposure == 0.0
    assert decision.confidence == 0.0
    assert decision.rationale == ""
    assert decision.failure is expected
    assert decision.is_failure
    assert record.response is None
    assert record.error is not None and type(error).__name__ in record.error


async def test_a_well_formed_object_that_is_not_a_signal_counts_as_malformed() -> None:
    # Constrained decoding fixes the syntax of a completion and nothing else, so an
    # exposure of 5.0 arrives as valid JSON that no backtest could act on.
    provider = MockProvider(responses=[{"exposure": 5.0, "confidence": 0.9, "rationale": "all in"}])

    decision, _ = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.failure is FailureMode.MALFORMED
    assert decision.exposure == 0.0


async def test_a_failed_decision_records_the_requests_it_actually_cost() -> None:
    # `Decision.retries` is documented as "attempts after the first" and is stored
    # as a result rather than as telemetry. The failed row is the one place retries
    # actually happened, and it was the one row that recorded none of them.
    provider = MockProvider(
        responses=[ProviderUnavailableError("daemon down after 3 attempt(s)", retries=2)]
    )

    decision, _ = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.failure is FailureMode.UNAVAILABLE
    assert decision.retries == 2


async def test_a_failed_decision_without_a_retry_count_records_none() -> None:
    provider = MockProvider(responses=[MalformedOutputError("nope")])

    decision, _ = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.retries == 0


async def test_a_malformed_generation_records_the_requests_it_cost_too() -> None:
    # The retry fix landed on two of its four paths. UNAVAILABLE and TRUNCATED
    # stored their count; the MALFORMED family stored zero, on the rows that cost
    # the most.
    provider = MockProvider(responses=[MalformedOutputError("content was not JSON", retries=2)])

    decision, _ = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.failure is FailureMode.MALFORMED
    assert decision.retries == 2


class _RetriedCompletion:
    """A provider whose request succeeded, after retries, with an unusable object."""

    async def preflight(self) -> None:
        return None

    async def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int | None = None,
    ) -> Completion:
        return Completion(
            data={"exposure": 5.0, "confidence": 0.9, "rationale": "all in"},
            output_tokens=12,
            prompt_tokens=64,
            retries=2,
            latency_seconds=0.0,
        )

    async def aclose(self) -> None:
        return None


async def test_a_well_formed_non_signal_records_the_requests_the_call_cost() -> None:
    # `ValidationError` fires after the request returned, so the count is on the
    # completion rather than on the error. Reading only the error stored zero for a
    # call that had already paid for two transport retries.
    decision, _ = await generate_decision(
        _RetriedCompletion(), point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.failure is FailureMode.MALFORMED
    assert decision.retries == 2


async def test_a_well_formed_non_signal_records_the_tokens_the_call_burned() -> None:
    # `output_tokens` is the other cost column the completion reports, and
    # `Completion`'s docstring calls those a result rather than telemetry. It was
    # never passed to `_failed_decision`, so it fell back to `Decision`'s default of
    # zero -- and a per-model cost tally read off the parquet under-counted exactly
    # the models that over-ran their schema. The same completion object is in scope
    # on the line that reads `retries`.
    decision, _ = await generate_decision(
        _RetriedCompletion(), point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.failure is FailureMode.MALFORMED
    assert decision.output_tokens == 12


async def test_a_provider_error_with_no_completion_records_no_tokens() -> None:
    # The other branch: nothing returned, so there is nothing to charge. Pinned so
    # the resolution above cannot be turned into a fabricated count.
    provider = MockProvider(responses=[ProviderUnavailableError("daemon down", retries=2)])

    decision, _ = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert decision.failure is FailureMode.UNAVAILABLE
    assert decision.output_tokens == 0


async def test_a_failed_decision_still_names_the_prompt_that_produced_it() -> None:
    provider = MockProvider(responses=[MalformedOutputError("nope")])

    decision, _ = await generate_decision(
        provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
    )

    assert len(decision.prompt_hash) > 0


# -- what must stop the run instead ---------------------------------------------


async def test_an_unfit_backend_is_raised_rather_than_recorded_as_a_decision() -> None:
    provider = MockProvider(responses=[PreflightError("this daemon ignores schemas")])

    with pytest.raises(PreflightError):
        await generate_decision(
            provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
        )


async def test_a_broken_schema_is_raised_rather_than_recorded_as_a_decision() -> None:
    provider = MockProvider(responses=[UnsupportedSchemaError("unbounded string field")])

    with pytest.raises(UnsupportedSchemaError):
        await generate_decision(
            provider, point=make_point(), price_context=CONTEXT, seed=7, now=FIXED_NOW
        )

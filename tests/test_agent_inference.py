"""Whether one decision point always produces exactly one row.

Against :class:`~council.agents.mock.MockProvider`: no daemon, no network, no GPU.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from council.agents.inference import DecisionPoint, failure_mode, generate_decision
from council.agents.mock import MockProvider
from council.agents.prompt import PeerView
from council.agents.provider import (
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


def make_point(*, arm: Arm = Arm.INDEPENDENT, composition: str | None = None) -> DecisionPoint:
    return DecisionPoint(
        model="alpha",
        persona=MOMENTUM_BOLD,
        ticker="AAA",
        decision_date=date(2022, 3, 1),
        arm=arm,
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
    point = make_point(arm=Arm.DEBATE, composition="quad")

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

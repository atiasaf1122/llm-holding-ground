"""The stand-in provider, and the contract it shares with the real one.

The mock is what makes the CPU-only promise keepable, so it has to be able to do
everything the daemon can -- including fail.
"""

from __future__ import annotations

import json

import pytest

from council.agents.mock import MockProvider
from council.agents.provider import (
    Completion,
    MalformedOutputError,
    ProviderUnavailableError,
    TruncatedGenerationError,
)
from council.agents.schema import UnsupportedSchemaError
from council.domain.signal import Signal
from helpers_provider import SIGNAL_OUTPUT, SIGNAL_SCHEMA, chat_handler, make_provider

# -- the mock -----------------------------------------------------------------


async def test_the_mock_answers_the_same_prompt_with_the_same_signal() -> None:
    mock = MockProvider()

    first = await mock.generate(system="persona", user="AAPL 2022-03-01", schema=SIGNAL_SCHEMA)
    second = await mock.generate(system="persona", user="AAPL 2022-03-01", schema=SIGNAL_SCHEMA)
    other = await mock.generate(system="persona", user="XOM 2022-03-01", schema=SIGNAL_SCHEMA)
    await mock.aclose()

    assert first.data == second.data
    assert first.data != other.data
    assert Signal.model_validate(first.data).exposure == first.data["exposure"]


async def test_the_mock_answers_two_personas_differently() -> None:
    # The persona travels in the system turn. A mock deriving its answer from the
    # user turn alone gives every persona the same number on the same day, and the
    # experiment's independent variable then cannot influence any CPU-testable
    # decision.
    mock = MockProvider()

    momentum = await mock.generate(system="momentum", user="AAPL", schema=SIGNAL_SCHEMA)
    reversion = await mock.generate(system="reversion", user="AAPL", schema=SIGNAL_SCHEMA)
    await mock.aclose()

    assert momentum.data != reversion.data


async def test_moving_a_sentence_between_the_turns_changes_the_mock_answer() -> None:
    mock = MockProvider()

    split = await mock.generate(system="persona ", user="prices", schema=SIGNAL_SCHEMA)
    slid = await mock.generate(system="persona", user=" prices", schema=SIGNAL_SCHEMA)
    await mock.aclose()

    assert split.data != slid.data


async def test_the_mock_replays_supplied_responses_in_order_and_records_calls() -> None:
    mock = MockProvider(responses=[SIGNAL_OUTPUT, {**SIGNAL_OUTPUT, "exposure": -0.2}])

    first = await mock.generate(system="s", user="u1", schema=SIGNAL_SCHEMA, max_tokens=64)
    second = await mock.generate(system="s", user="u2", schema=SIGNAL_SCHEMA)
    third = await mock.generate(system="s", user="u3", schema=SIGNAL_SCHEMA)

    exposures = [first.data["exposure"], second.data["exposure"], third.data["exposure"]]
    assert exposures == [0.4, -0.2, 0.4]
    assert [call.user for call in mock.calls] == ["u1", "u2", "u3"]
    assert mock.calls[0].max_tokens == 64


async def test_the_mock_can_be_made_to_fail_in_every_mode_the_runner_handles() -> None:
    # The per-model failure rate is a result in this experiment, and a failed
    # decision point is written rather than dropped. Without injectable failures
    # the code that writes those rows could only ever be exercised on a machine
    # with a free GPU -- the one machine this project promises not to need.
    mock = MockProvider(
        responses=[
            SIGNAL_OUTPUT,
            TruncatedGenerationError("length"),
            ProviderUnavailableError("daemon down"),
            MalformedOutputError("not json"),
        ]
    )

    first = await mock.generate(system="s", user="u", schema=SIGNAL_SCHEMA)
    assert first.data == SIGNAL_OUTPUT
    for expected in (TruncatedGenerationError, ProviderUnavailableError, MalformedOutputError):
        with pytest.raises(expected):
            await mock.generate(system="s", user="u", schema=SIGNAL_SCHEMA)
    await mock.aclose()

    assert len(mock.calls) == 4


async def test_the_mock_reports_diagnostics_a_stored_row_can_hold() -> None:
    mock = MockProvider()

    result = await mock.generate(system="s", user="AAPL 2022-03-01", schema=SIGNAL_SCHEMA)
    await mock.aclose()

    assert result.output_tokens > 0
    assert result.prompt_tokens > 0
    assert (result.retries, result.latency_seconds) == (0, 0.0)


async def test_the_mock_refuses_the_same_schemas_the_daemon_would_choke_on() -> None:
    mock = MockProvider()

    with pytest.raises(UnsupportedSchemaError):
        await mock.generate(
            system="s",
            user="u",
            schema={"type": "object", "properties": {"note": {"type": "string"}}},
        )


async def test_both_implementations_answer_the_same_call_with_the_same_shape() -> None:
    # Signature conformance is checked statically, by the TYPE_CHECKING block at
    # the foot of provider.py -- isinstance() against a runtime_checkable Protocol
    # compares attribute names only and would accept a synchronous generate()
    # taking different parameters. What is left to assert here is behavioural:
    # both implementations accept the identical keyword-only call and hand back
    # the same record, so a caller can hold either without knowing which.
    ollama = make_provider(chat_handler(json.dumps(SIGNAL_OUTPUT)))
    mock = MockProvider(responses=[SIGNAL_OUTPUT])

    results = [
        await impl.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA, max_tokens=64)
        for impl in (ollama, mock)
    ]
    await ollama.aclose()
    await mock.aclose()

    assert all(isinstance(result, Completion) for result in results)
    assert [result.data for result in results] == [SIGNAL_OUTPUT, SIGNAL_OUTPUT]

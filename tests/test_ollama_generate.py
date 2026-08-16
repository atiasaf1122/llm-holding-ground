"""What the daemon is sent, what comes back, and what is refused.

The failure modes matter as much as the happy path here: a decision point that
fails is written rather than dropped, so how a failure is classified ends up in
the results table.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
from pydantic import ValidationError

from council.agents import ollama as provider_module
from council.agents.provider import (
    ContextOverflowError,
    MalformedOutputError,
    MissingModelError,
    PreflightError,
    TruncatedGenerationError,
)
from council.domain.signal import MAX_RATIONALE_CHARS, Signal
from helpers_provider import (
    SIGNAL_OUTPUT,
    SIGNAL_SCHEMA,
    chat_envelope,
    chat_handler,
    make_provider,
)

# -- generation ---------------------------------------------------------------


async def test_generate_returns_the_parsed_object_from_the_completion() -> None:
    provider = make_provider(chat_handler(json.dumps(SIGNAL_OUTPUT)))

    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert result.data == SIGNAL_OUTPUT


async def test_generate_sends_the_schema_context_size_keep_alive_and_zero_temperature() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=chat_envelope(json.dumps(SIGNAL_OUTPUT)))

    provider = make_provider(handler)
    await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA, max_tokens=128)
    await provider.aclose()

    body = sent[0]
    assert body["model"] == "qwen3:8b"
    assert body["stream"] is False
    assert body["think"] is False
    assert body["keep_alive"] == "30m"
    assert body["format"]["properties"]["rationale"]["maxLength"] == MAX_RATIONALE_CHARS
    assert body["options"] == {
        "temperature": 0.0,
        "num_ctx": 4096,
        "num_predict": 128,
        "seed": 20260101,
    }


async def test_generate_defaults_the_token_cap_to_the_configured_maximum() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=chat_envelope(json.dumps(SIGNAL_OUTPUT)))

    provider = make_provider(handler)
    await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert sent[0]["options"]["num_predict"] == 320


async def test_generate_keeps_the_prompt_in_the_user_turn() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=chat_envelope(json.dumps(SIGNAL_OUTPUT)))

    provider = make_provider(handler)
    await provider.generate(system="you are cautious", user="a peer said X", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert sent[0]["messages"] == [
        {"role": "system", "content": "you are cautious"},
        {"role": "user", "content": "a peer said X"},
    ]


async def test_generate_flattens_refs_before_sending_the_schema() -> None:
    sent: list[dict[str, Any]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, json=chat_envelope(json.dumps({"side": "long"})))

    schema = {
        "type": "object",
        "properties": {"side": {"$ref": "#/$defs/Side"}},
        "$defs": {"Side": {"type": "string", "enum": ["long", "short"]}},
    }
    provider = make_provider(handler)
    await provider.generate(system="persona", user="prices", schema=schema)
    await provider.aclose()

    sent_schema = sent[0]["format"]
    assert "$defs" not in sent_schema
    assert sent_schema["properties"]["side"] == {"type": "string", "enum": ["long", "short"]}


# -- the numbers a stored row needs -------------------------------------------


async def test_generate_reports_the_token_counts_the_daemon_measured() -> None:
    # Decision.output_tokens has no other source: the daemon counts once, in the
    # envelope, and re-deriving it later means running the inference again.
    provider = make_provider(chat_handler(json.dumps(SIGNAL_OUTPUT), prompt_eval_count=917))

    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert result.output_tokens == 42
    assert result.prompt_tokens == 917


async def test_generate_reports_how_many_retries_the_answer_cost(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "BACKOFF_BASE_SECONDS", 0.0)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(503, text="server busy")
        return httpx.Response(200, json=chat_envelope(json.dumps(SIGNAL_OUTPUT)))

    provider = make_provider(handler, max_retries=2)
    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert result.retries == 2


async def test_generate_reports_no_retries_when_the_first_attempt_worked() -> None:
    provider = make_provider(chat_handler(json.dumps(SIGNAL_OUTPUT)))

    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert result.retries == 0
    assert result.latency_seconds >= 0.0


async def test_a_daemon_that_omits_its_counts_reports_zero_rather_than_guessing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "message": {"content": json.dumps(SIGNAL_OUTPUT)},
                "done": True,
                "done_reason": "stop",
            },
        )

    provider = make_provider(handler)
    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert (result.output_tokens, result.prompt_tokens) == (0, 0)


# -- what the provider does not promise ---------------------------------------


@pytest.mark.parametrize(
    "content",
    [
        pytest.param('{"exposure": 5.0, "confidence": 3.0, "rationale": "x"}', id="out-of-range"),
        pytest.param("{}", id="required-fields-missing"),
    ],
)
async def test_the_provider_delivers_grammatical_output_it_has_not_validated(
    content: str,
) -> None:
    # Constrained decoding fixes JSON *syntax*, not `minimum`, `maximum` or
    # `required`. This is the documented boundary: the provider succeeds, and the
    # caller owes ValidationError a branch onto FailureMode.MALFORMED. Pinned by a
    # test so that a runner written later cannot mistake it for a provider bug.
    provider = make_provider(chat_handler(content))

    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert result.data == json.loads(content)
    with pytest.raises(ValidationError):
        Signal.model_validate(result.data)


# -- failure modes ------------------------------------------------------------


async def test_generate_reports_truncation_rather_than_a_parse_error() -> None:
    incomplete = '{"exposure": 0.4, "confidence": 0.7, "rationale": "the trend is'
    provider = make_provider(chat_handler(incomplete, done_reason="length"))

    with pytest.raises(TruncatedGenerationError) as caught:
        await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert "length" in str(caught.value)


async def test_generate_rejects_a_completion_that_is_not_json() -> None:
    provider = make_provider(chat_handler("Sure! Here is my answer:"))

    with pytest.raises(MalformedOutputError):
        await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()


async def test_generate_rejects_a_json_scalar_where_an_object_was_required() -> None:
    provider = make_provider(chat_handler("0.4"))

    with pytest.raises(MalformedOutputError) as caught:
        await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert "object" in str(caught.value)


async def test_a_complete_envelope_without_a_done_reason_is_not_called_truncated() -> None:
    # A backend that never sends the field would otherwise produce a run of
    # 100% flat, 100% failed decisions that reads as a weak model.
    provider = make_provider(chat_handler(json.dumps(SIGNAL_OUTPUT), done_reason=None))

    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert result.data == SIGNAL_OUTPUT


async def test_an_envelope_that_is_neither_done_nor_reasoned_stops_the_run() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"message": {"content": json.dumps(SIGNAL_OUTPUT)}})

    provider = make_provider(handler)

    # PreflightError, not a per-decision failure: an envelope shape this code
    # cannot read is a reason to stop, not to record eighty thousand truncations.
    with pytest.raises(PreflightError):
        await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()


async def test_a_prompt_that_filled_the_context_window_is_a_recorded_failure() -> None:
    # 3800 prompt tokens plus 320 reserved for output exceeds num_ctx=4096, so
    # Ollama dropped the head of the conversation -- the persona -- and answered
    # anyway. Debate prompts are the long ones, so this failure would land on the
    # treatment arm and read afterwards as a debate effect.
    provider = make_provider(
        chat_handler(json.dumps(SIGNAL_OUTPUT), prompt_eval_count=3800), context_tokens=4096
    )

    with pytest.raises(ContextOverflowError) as caught:
        await provider.generate(system="persona", user="a long debate", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert "3800" in str(caught.value)


async def test_a_prompt_that_fits_the_context_window_is_left_alone() -> None:
    provider = make_provider(
        chat_handler(json.dumps(SIGNAL_OUTPUT), prompt_eval_count=3775), context_tokens=4096
    )

    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert result.prompt_tokens == 3775


async def test_generate_turns_a_404_into_an_actionable_pull_instruction() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"error": "model 'qwen3:8b' not found"})

    provider = make_provider(handler)

    with pytest.raises(MissingModelError) as caught:
        await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert "ollama pull qwen3:8b" in str(caught.value)

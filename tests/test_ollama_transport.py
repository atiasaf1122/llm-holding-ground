"""Reaching the daemon at all: preflight, retries, backoff and the queue.

None of this is about what a model says. It is about the run surviving eighty
thousand requests without either hammering a busy daemon or giving up on a
hiccup, and about refusing to start against a backend that would quietly ignore
the schema.
"""

from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from council.agents import ollama as provider_module
from council.agents.ollama import OllamaProvider
from council.agents.provider import (
    MalformedOutputError,
    MissingModelError,
    PreflightError,
    ProviderUnavailableError,
)
from helpers_provider import (
    SIGNAL_OUTPUT,
    SIGNAL_SCHEMA,
    chat_envelope,
    daemon_handler,
    make_provider,
    make_settings,
)

# -- preflight ----------------------------------------------------------------


# -- preflight ----------------------------------------------------------------


async def test_preflight_accepts_a_current_daemon_holding_the_model() -> None:
    provider = make_provider(daemon_handler())

    await provider.preflight()
    await provider.aclose()


async def test_preflight_rejects_a_daemon_that_would_ignore_the_schema() -> None:
    provider = make_provider(daemon_handler(version="0.30.9"))

    with pytest.raises(PreflightError) as caught:
        await provider.preflight()
    await provider.aclose()

    assert "0.31.2" in str(caught.value)


async def test_preflight_rejects_a_model_that_has_not_been_pulled() -> None:
    provider = make_provider(daemon_handler(models=("gemma4:latest",)))

    with pytest.raises(MissingModelError) as caught:
        await provider.preflight()
    await provider.aclose()

    assert "ollama pull qwen3:8b" in str(caught.value)


async def test_preflight_survives_a_daemon_with_nothing_pulled() -> None:
    # Ollama answers `{"models": null}` when the library is empty. Iterating that
    # raises TypeError, which is not a ProviderError, so it walks past the
    # caller's fatal-error handling and loses the `ollama pull` message in the
    # one case where it is the whole answer.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.31.2"})
        return httpx.Response(200, json={"models": None})

    provider = make_provider(handler)

    with pytest.raises(MissingModelError) as caught:
        await provider.preflight()
    await provider.aclose()

    assert "Available: none" in str(caught.value)


async def test_preflight_rejects_a_tags_response_it_cannot_read() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.31.2"})
        return httpx.Response(200, json={"models": {"qwen3:8b": {}}})

    provider = make_provider(handler)

    with pytest.raises(PreflightError) as caught:
        await provider.preflight()
    await provider.aclose()

    assert "dict" in str(caught.value)


async def test_preflight_matches_a_bare_model_name_against_the_latest_tag() -> None:
    provider = make_provider(daemon_handler(models=("gemma4:latest",)), model="gemma4")

    await provider.preflight()
    await provider.aclose()


async def test_preflight_reads_a_release_candidate_version() -> None:
    provider = make_provider(daemon_handler(version="0.32.0-rc1"))

    await provider.preflight()
    await provider.aclose()


# -- retries and concurrency --------------------------------------------------


# -- retries and concurrency --------------------------------------------------


async def test_generate_retries_a_server_error_and_then_succeeds(
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

    assert len(attempts) == 3
    assert result.data == SIGNAL_OUTPUT


async def test_generate_gives_up_after_the_configured_number_of_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "BACKOFF_BASE_SECONDS", 0.0)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        raise httpx.ConnectError("connection refused")

    provider = make_provider(handler, max_retries=1)

    with pytest.raises(ProviderUnavailableError) as caught:
        await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert len(attempts) == 2
    assert "ConnectError" in str(caught.value)


async def test_generate_does_not_retry_a_malformed_completion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # At temperature zero a second attempt reproduces the first exactly, so a
    # retry here would only spend GPU time.
    monkeypatch.setattr(provider_module, "BACKOFF_BASE_SECONDS", 0.0)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(200, json=chat_envelope("not json at all"))

    provider = make_provider(handler, max_retries=2)

    with pytest.raises(MalformedOutputError):
        await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert len(attempts) == 1


@pytest.mark.parametrize("status", [429, 408])
async def test_generate_retries_a_rate_limit_and_a_request_timeout(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    # Neither status is a server error, so the `< 500` test alone would return
    # them straight to the caller as a success and the completion would be the
    # error body rather than a signal.
    monkeypatch.setattr(provider_module, "BACKOFF_BASE_SECONDS", 0.0)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        if len(attempts) < 3:
            return httpx.Response(status, text="back off")
        return httpx.Response(200, json=chat_envelope(json.dumps(SIGNAL_OUTPUT)))

    provider = make_provider(handler, max_retries=2)
    result = await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert len(attempts) == 3
    assert result.data == SIGNAL_OUTPUT


async def test_the_backoff_schedule_doubles_and_is_not_jittered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Recorded rather than zeroed. Every other retry test monkeypatches
    # BACKOFF_BASE_SECONDS to 0.0, which erases the schedule from observation --
    # so without this the doubling could be replaced by anything at all and the
    # suite would stay green, and "reruns must be reproducible" would be a claim
    # with nothing behind it.
    slept: list[float] = []

    async def recorder(seconds: float) -> None:
        slept.append(seconds)

    monkeypatch.setattr(asyncio, "sleep", recorder)

    def handler(request: httpx.Request) -> httpx.Response:
        if len(slept) < 2:
            return httpx.Response(503, text="server busy")
        return httpx.Response(200, json=chat_envelope(json.dumps(SIGNAL_OUTPUT)))

    provider = make_provider(handler, max_retries=2)
    await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert slept == [1.0, 2.0]


async def test_a_daemon_that_keeps_failing_gives_up_after_the_configured_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(provider_module, "BACKOFF_BASE_SECONDS", 0.0)
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(500, text="internal error")

    provider = make_provider(handler, max_retries=2)

    with pytest.raises(ProviderUnavailableError) as caught:
        await provider.generate(system="persona", user="prices", schema=SIGNAL_SCHEMA)
    await provider.aclose()

    assert len(attempts) == 3
    assert "HTTP 500" in str(caught.value)


async def test_concurrent_generation_never_exceeds_the_configured_limit() -> None:
    in_flight = 0
    peak = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal in_flight, peak
        in_flight += 1
        peak = max(peak, in_flight)
        await asyncio.sleep(0)
        in_flight -= 1
        return httpx.Response(200, json=chat_envelope(json.dumps(SIGNAL_OUTPUT)))

    provider = OllamaProvider(
        model="qwen3:8b",
        settings=make_settings(concurrency=2),
        transport=httpx.MockTransport(handler),
    )
    await asyncio.gather(
        *(
            provider.generate(system="persona", user=f"day {index}", schema=SIGNAL_SCHEMA)
            for index in range(8)
        )
    )
    await provider.aclose()

    assert peak <= 2

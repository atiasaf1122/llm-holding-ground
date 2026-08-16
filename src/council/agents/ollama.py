"""The Ollama daemon, spoken to over its HTTP API.

Ollama is the implementation because vLLM has no native Windows build. The daemon
is addressed with ``httpx`` directly; the ``ollama`` package is itself a thin
wrapper around the same calls, and a second abstraction under this one would only
add a dependency and a translation layer.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import httpx

from council.agents.provider import (
    Completion,
    ContextOverflowError,
    InterruptedEnvelopeError,
    MalformedOutputError,
    MissingModelError,
    PreflightError,
    Provider,
    ProviderUnavailableError,
    TruncatedGenerationError,
)
from council.agents.schema import prepare_schema
from council.config import Settings, get_settings

MIN_OLLAMA_VERSION: tuple[int, ...] = (0, 31, 2)
"""Earlier daemons accept ``format`` on a thinking model and silently ignore it.

The output is then unconstrained prose that happens to parse sometimes, which
presents as a weak model rather than as a bug -- and would be read as a result.
"""

_LOG = logging.getLogger(__name__)

BACKOFF_BASE_SECONDS = 1.0

ENVELOPE_RETRIES = 3
"""How many times `generate` re-sends a request whose response the daemon never
finished. Distinct from `_request`'s transport retries: the HTTP exchange here
*succeeded*, so the transport loop is already spent by the time the envelope is
parsed. Three at exponential backoff spans ~7 seconds, which covers the observed
failure -- the auto-updater's restart -- with margin."""
"""Doubling from here. Fixed rather than jittered: reruns must be reproducible."""

_RETRYABLE_STATUS = frozenset({httpx.codes.TOO_MANY_REQUESTS, httpx.codes.REQUEST_TIMEOUT})


class OllamaProvider:
    """The Ollama daemon, over its HTTP API."""

    def __init__(
        self,
        *,
        model: str,
        settings: Settings | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._settings = settings if settings is not None else get_settings()
        self._model = model
        self._client = httpx.AsyncClient(
            base_url=self._settings.ollama_base_url,
            timeout=httpx.Timeout(self._settings.request_timeout_seconds),
            transport=transport,
        )
        # One GPU is memory-bandwidth bound, so requests past this point queue
        # inside the daemon where nothing can see or bound them. GenerationRunner
        # holds a second semaphore on the same setting; this one is the bound that
        # covers every other caller -- the debate layer reaches a provider through
        # council.debate.caller, which never touches the runner.
        self._semaphore = asyncio.Semaphore(self._settings.concurrency)

    @property
    def model(self) -> str:
        return self._model

    async def preflight(self) -> None:
        """Check the daemon version and that the model is pulled.

        Both failures are silent otherwise: an old daemon quietly discards the
        schema, and a missing model surfaces as a 404 on the first of eighty
        thousand requests.
        """
        reported = await self._get_json("/api/version")
        version = _parse_version(str(reported.get("version", "")))
        if version < MIN_OLLAMA_VERSION:
            raise PreflightError(
                f"Ollama {reported.get('version')!r} is older than "
                f"{'.'.join(map(str, MIN_OLLAMA_VERSION))}, which is the first "
                "version that honours a JSON schema on a thinking model"
            )

        tags = await self._get_json("/api/tags")
        # A daemon with nothing pulled answers `{"models": null}`, and iterating
        # that raises a TypeError -- which is not a ProviderError, so it walks
        # straight past the caller's fatal-error handling and loses the one
        # message that would have helped: run `ollama pull`.
        entries = tags.get("models") or []
        if not isinstance(entries, list):
            raise PreflightError(f"/api/tags returned {type(entries).__name__} for 'models'")
        available = sorted(
            str(entry.get("name", ""))
            for entry in entries
            if isinstance(entry, Mapping) and entry.get("name")
        )
        if _qualify(self._model) not in {_qualify(name) for name in available}:
            raise MissingModelError(
                f"{self._model!r} is not pulled. Run: ollama pull {self._model}. "
                f"Available: {', '.join(available) or 'none'}"
            )

    async def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int | None = None,
    ) -> Completion:
        """Send one chat request and return the parsed object with its cost.

        Raises:
            UnsupportedSchemaError: before anything is sent.
            MissingModelError: the daemon does not have this model.
            ContextOverflowError: the prompt did not fit the context window.
            TruncatedGenerationError: generation was cut off.
            MalformedOutputError: the completion was not a JSON object.
            ProviderUnavailableError: the daemon could not be reached.
            PreflightError: the envelope had a shape this code cannot read.
        """
        cap = max_tokens if max_tokens is not None else self._settings.max_output_tokens
        payload = self._chat_payload(
            system=system, user=user, schema=prepare_schema(schema), max_tokens=cap
        )
        # The envelope loop re-sends the whole request when the daemon returned a
        # response it never finished -- see InterruptedEnvelopeError. It wraps the
        # transport loop rather than living inside it because the HTTP exchange
        # *succeeds* in this failure mode, so `_request`'s own retries are already
        # spent by the time the envelope is parsed.
        for attempt in range(ENVELOPE_RETRIES + 1):
            if attempt:
                await asyncio.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
            # Timed inside the semaphore: waiting for a slot is a property of how
            # the caller batched the run, not of the model, and folding it in
            # would make latency depend on the shape of the sweep.
            async with self._semaphore:
                started = time.perf_counter()
                response, retries = await self._request("POST", "/api/chat", payload)
                latency_seconds = time.perf_counter() - started

            # Both raises carry the count `_request` already returned. Without it,
            # two failures costing identical daemon time store 2 and 0 depending
            # only on which raise fired -- and the rows that retried are exactly
            # the rows reporting zero, since `Decision.retries` falls back to a
            # default here.
            if response.status_code == httpx.codes.NOT_FOUND:
                raise MissingModelError(
                    f"Ollama has no model {self._model!r}. Run: ollama pull {self._model}",
                    retries=retries,
                )
            if response.status_code != httpx.codes.OK:
                raise ProviderUnavailableError(
                    f"POST /api/chat returned {response.status_code}: {response.text[:200]}",
                    retries=retries,
                )
            try:
                return _parse_chat_response(
                    response,
                    model=self._model,
                    retries=retries,
                    latency_seconds=latency_seconds,
                    context_tokens=self._settings.context_tokens,
                    max_tokens=cap,
                )
            except InterruptedEnvelopeError:
                if attempt >= ENVELOPE_RETRIES:
                    raise
                _LOG.warning(
                    "%s: daemon returned an unfinished response (attempt %d of %d); re-sending",
                    self._model,
                    attempt + 1,
                    ENVELOPE_RETRIES + 1,
                )
        raise AssertionError("unreachable: the loop returns or raises")

    async def aclose(self) -> None:
        await self._client.aclose()

    # -- internals ------------------------------------------------------------

    def _chat_payload(
        self, *, system: str, user: str, schema: dict[str, Any], max_tokens: int
    ) -> dict[str, Any]:
        settings = self._settings
        return {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            # Thinking tokens are not constrained by `format`, so on a thinking
            # model they arrive before the JSON and consume the token budget.
            "think": False,
            "format": schema,
            "keep_alive": settings.keep_alive,
            "options": {
                "temperature": settings.temperature,
                "num_ctx": settings.context_tokens,
                "num_predict": max_tokens,
                "seed": settings.seed,
            },
        }

    async def _get_json(self, path: str) -> Mapping[str, Any]:
        response, _ = await self._request("GET", path)
        if response.status_code != httpx.codes.OK:
            raise ProviderUnavailableError(f"GET {path} returned {response.status_code}")
        try:
            body = response.json()
        except ValueError as exc:
            raise PreflightError(f"GET {path} did not return JSON; is this Ollama?") from exc
        if not isinstance(body, Mapping):
            raise PreflightError(f"GET {path} returned {type(body).__name__}, not an object")
        return body

    async def _request(
        self, method: str, path: str, payload: Mapping[str, Any] | None = None
    ) -> tuple[httpx.Response, int]:
        """Send one request, retrying only what a retry could plausibly fix.

        Transport failures and server errors are retried. A malformed or
        truncated completion is not: temperature is zero, so a second attempt
        spends another minute of GPU time reproducing the first one exactly.

        Returns:
            The response and the number of retries it took. The count is returned
            rather than logged because ``Decision.retries`` is a stored column and
            this loop is the only place the number exists.
        """
        attempts = self._settings.max_retries + 1
        failure = "no attempt was made"
        for attempt in range(attempts):
            if attempt:
                await asyncio.sleep(BACKOFF_BASE_SECONDS * 2 ** (attempt - 1))
            try:
                response = await self._client.request(method, path, json=payload)
            except httpx.TransportError as exc:
                failure = f"{type(exc).__name__}: {exc}"
                continue
            if response.status_code < httpx.codes.INTERNAL_SERVER_ERROR and (
                response.status_code not in _RETRYABLE_STATUS
            ):
                return response, attempt
            failure = f"HTTP {response.status_code}"
        # The count is carried on the error as well as in the message. A failed
        # decision point is stored with `Decision.retries`, which is a published
        # column rather than telemetry, and this loop is the only place the number
        # exists -- discarded here, the rows where retries actually happened are
        # exactly the rows that record zero of them.
        raise ProviderUnavailableError(
            f"{method} {path} failed after {attempts} attempt(s): {failure}",
            retries=attempts - 1,
        )


def _parse_chat_response(
    response: httpx.Response,
    *,
    model: str,
    retries: int,
    latency_seconds: float,
    context_tokens: int,
    max_tokens: int,
) -> Completion:
    """Turn one ``/api/chat`` envelope into the object that was asked for.

    The order is the point. Completeness is settled before the content is parsed,
    because a cut-off generation leaves JSON that is merely incomplete: letting
    ``json.loads`` fail first would blame the model's formatting for what was
    really a token limit.
    """
    envelope = _read_envelope(response, model=model, retries=retries)
    prompt_tokens = _reported_count(envelope, "prompt_eval_count")
    _reject_incomplete(
        envelope,
        model=model,
        prompt_tokens=prompt_tokens,
        context_tokens=context_tokens,
        max_tokens=max_tokens,
        retries=retries,
    )
    return Completion(
        data=_parse_content(envelope, model=model, retries=retries),
        output_tokens=_reported_count(envelope, "eval_count"),
        prompt_tokens=prompt_tokens,
        retries=retries,
        latency_seconds=latency_seconds,
    )


def _read_envelope(response: httpx.Response, *, model: str, retries: int = 0) -> Mapping[str, Any]:
    """The envelope, or the malformed failure carrying what it cost.

    ``retries`` is carried onto the raised error for the reason
    :class:`~council.agents.provider.RetriedError` exists: the retries happen
    before anything is parsed, so a malformed envelope that followed one or more of
    them cost those requests, and the stored row is where that cost is reported.
    """
    try:
        envelope = response.json()
    except ValueError as exc:
        raise MalformedOutputError(
            f"{model}: /api/chat did not return JSON", retries=retries
        ) from exc
    if not isinstance(envelope, Mapping):
        raise MalformedOutputError(
            f"{model}: /api/chat returned a {type(envelope).__name__}", retries=retries
        )
    return envelope


def _reject_incomplete(
    envelope: Mapping[str, Any],
    *,
    model: str,
    prompt_tokens: int,
    context_tokens: int,
    max_tokens: int,
    retries: int = 0,
) -> None:
    """Refuse an envelope whose text is present but cannot be trusted.

    ``retries`` is carried onto the raised error for the reason
    :class:`~council.agents.provider.RetriedError` exists: a truncation that
    followed one or more transport retries cost those requests, and the stored row
    is where that cost is reported.
    """
    reason = envelope.get("done_reason")
    if reason is None:
        # An absent field is not a truncation. Reading it as one would turn a
        # backend that simply names the field differently into a run of eighty
        # thousand flat, failed decisions that looks like a weak model -- the
        # same silent misread MIN_OLLAMA_VERSION exists to prevent. A finished
        # envelope says so with `done`; anything else is a shape we cannot read.
        if envelope.get("done") is not True:
            # `done=False` on a non-streaming call is not a schema this code
            # cannot read -- it is a response the daemon never finished. The one
            # observed cause is Ollama's auto-updater restarting the daemon
            # mid-generation, which killed two long runs before this was made
            # retriable: as a PreflightError it took the whole sweep down for a
            # condition that resolves itself in seconds. `generate` retries it;
            # a daemon that stays broken exhausts the retries and still fails
            # loudly, as ProviderUnavailableError.
            raise InterruptedEnvelopeError(
                f"{model}: /api/chat returned done="
                f"{envelope.get('done')!r} with no done_reason; the daemon did not "
                "finish this response (restart mid-generation?)",
                retries=retries,
            )
    elif reason != "stop":
        raise TruncatedGenerationError(
            f"{model}: generation ended with done_reason={reason!r}, so the JSON is incomplete",
            retries=retries,
        )

    # Ollama truncates an oversized prompt instead of refusing it, so a prompt
    # that filled the window is indistinguishable afterwards from one that
    # overflowed it. Both are treated as overflow: a decision generated from a
    # decapitated conversation is not a decision this experiment can use.
    if prompt_tokens and prompt_tokens + max_tokens >= context_tokens:
        raise ContextOverflowError(
            f"{model}: prompt used {prompt_tokens} of {context_tokens} context tokens "
            f"with {max_tokens} reserved for output, so the head of the conversation "
            "-- the persona -- was dropped",
            retries=retries,
        )


def _parse_content(envelope: Mapping[str, Any], *, model: str, retries: int = 0) -> dict[str, Any]:
    """The completion object, or the malformed failure carrying what it cost.

    ``retries`` is carried for the same reason it is in :func:`_read_envelope`.
    """
    message = envelope.get("message")
    content = str(message.get("content", "")) if isinstance(message, Mapping) else ""
    try:
        parsed = json.loads(content)
    except ValueError as exc:
        raise MalformedOutputError(
            f"{model}: content was not JSON: {content[:200]!r}", retries=retries
        ) from exc
    if not isinstance(parsed, dict):
        raise MalformedOutputError(
            f"{model}: expected a JSON object, got a {type(parsed).__name__}: {content[:200]!r}",
            retries=retries,
        )
    return parsed


def _reported_count(envelope: Mapping[str, Any], key: str) -> int:
    """A token count from the envelope, or zero where the daemon omitted one."""
    value = envelope.get(key)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return 0
    return value


def _parse_version(reported: str) -> tuple[int, ...]:
    """Parse ``0.31.2`` or ``0.31.2-rc1`` into a comparable tuple."""
    core = reported.split("-", 1)[0].split("+", 1)[0]
    try:
        return tuple(int(part) for part in core.split("."))
    except ValueError as exc:
        raise PreflightError(f"cannot read Ollama version {reported!r}") from exc


def _qualify(name: str) -> str:
    """``/api/tags`` reports ``name:tag``; a bare name means the ``latest`` tag."""
    _, _, tail = name.rpartition("/")
    return name if ":" in tail else f"{name}:latest"


if TYPE_CHECKING:
    # The only thing that checks this class against the Protocol. A
    # `runtime_checkable` isinstance() compares attribute *names* and nothing
    # else, so it would accept a synchronous generate() taking different
    # parameters; and no caller holds a Provider-typed name yet, so mypy has
    # nowhere else to notice. Never constructed at run time.
    _CONFORMS: Provider = OllamaProvider(model="")

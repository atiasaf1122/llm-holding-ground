"""Constrained generation, behind an interface thin enough to replace.

The experiment needs one thing from a language model backend: given a system turn
and a user turn, return an object matching a schema, or fail in a way that says
which of the three failure modes happened. :class:`Provider` is that and nothing
more, so a vLLM implementation could be added later without a caller changing.
:mod:`council.agents.ollama` is the implementation; :mod:`council.agents.mock` is
the one that needs no GPU.

The error types map onto :class:`~council.domain.signal.FailureMode`:
:class:`TruncatedGenerationError` -- and :class:`ContextOverflowError` under it --
to ``TRUNCATED``, :class:`MalformedOutputError` to ``MALFORMED``,
:class:`ProviderUnavailableError` and :class:`MissingModelError` to ``UNAVAILABLE``.
:class:`PreflightError` and :class:`~council.agents.schema.UnsupportedSchemaError`
are not decision failures at all: they say the run should not have started.

That mapping is total only over what a provider raises. Grammar-constrained
decoding fixes the *syntax* of the output and nothing else, so a completion can
satisfy the schema's grammar while violating the model the schema came from --
``{}``, or an ``exposure`` of 5.0. Validating the returned object is the caller's
job, and the caller owes ``pydantic.ValidationError`` a branch onto ``MALFORMED``.
A provider is handed a schema rather than a model and cannot do it here.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable


class ProviderError(RuntimeError):
    """Base for every failure a backend can produce at generation time."""


class RetriedError(ProviderError):
    """A failure that may have cost more than one request.

    ``retries`` is the same quantity :attr:`Completion.retries` carries -- attempts
    after the first -- so that a decision point which failed records the requests it
    actually cost rather than the zero a default supplies. The count only exists
    inside the request loop, and on the raise path it was previously discarded into
    the free-text message.
    """

    def __init__(self, *args: object, retries: int = 0) -> None:
        super().__init__(*args)
        self.retries = retries


class TruncatedGenerationError(RetriedError):
    """Generation stopped for a reason other than the model choosing to stop."""


class ContextOverflowError(TruncatedGenerationError):
    """The prompt and the token budget together did not fit the context window.

    Ollama truncates rather than refusing, and it truncates from the head -- which
    is where the system turn carrying the persona lives. A debate prompt is
    structurally longer than an independent one (every peer's rationale on top of
    the same price history), so this failure is not spread evenly across the arms:
    it lands on the treatment. Silently, it would weaken the persona manipulation
    in the debate arms only, and read afterwards as a debate effect.

    A subclass of truncation because that is what it is, and because it keeps the
    error taxonomy mapping onto ``FailureMode`` without a new member.
    """


class MalformedOutputError(RetriedError):
    """The completion was not a JSON object, or the envelope was not JSON.

    A :class:`RetriedError` for the reason the other two are: the transport retries
    happen before anything is parsed, so a malformed answer can follow one or more
    of them, and ``Decision.retries`` is a published column rather than telemetry.
    """


class ProviderUnavailableError(RetriedError):
    """The daemon could not be reached, or kept returning a server error."""


class InterruptedEnvelopeError(ProviderUnavailableError):
    """The HTTP exchange succeeded but the daemon never finished the response.

    A subclass of :class:`ProviderUnavailableError` because that is what it means
    -- the daemon was not available *for the duration of this generation* -- and
    everything that stores a failure already maps that class to
    ``FailureMode.UNAVAILABLE``. Kept distinct so
    :meth:`~council.agents.ollama.OllamaProvider.generate` can retry exactly this
    case: the observed cause is Ollama's auto-updater restarting the daemon
    mid-generation, which resolves itself in seconds, and which killed two long
    runs while this raised as a non-retriable PreflightError.
    """


class MissingModelError(RetriedError):
    """The daemon is running but does not have the requested model.

    A :class:`RetriedError` for the reason the others are: a 404 is raised after
    :meth:`OllamaProvider._request` has already spent its transport retries, so the
    decision it produces cost more than one request. It remains a
    :class:`ProviderError`, so ``failure_mode``'s ``UNAVAILABLE`` branch and
    preflight's positional-argument raise are unaffected.
    """


class PreflightError(ProviderError):
    """The backend is unfit for this experiment, whatever it may return."""


@dataclass(frozen=True, slots=True)
class Completion:
    """One successful generation, with the numbers a stored row needs.

    :class:`~council.domain.signal.Decision` has columns for ``output_tokens``,
    ``retries`` and ``latency_seconds``, and a request is the only moment they
    exist: the daemon reports its counts once, in the envelope, and the retry
    count lives inside the request loop. Returning the parsed object on its own
    would leave three columns to be written as zeros on every one of eighty
    thousand rows -- and per-model failure and cost are results here, not
    telemetry.
    """

    data: dict[str, Any]
    """The parsed completion. A JSON object; not yet validated against anything."""

    output_tokens: int
    prompt_tokens: int
    """Reported by the daemon. ``prompt_tokens`` is also the only way to notice
    that a prompt was truncated to fit the context window."""

    retries: int
    """Attempts after the first. Zero on a request that succeeded immediately."""

    latency_seconds: float


@runtime_checkable
class Provider(Protocol):
    """A language model backend that can be constrained to a schema."""

    async def preflight(self) -> None:
        """Verify the backend can do what this experiment assumes.

        Raises:
            ProviderError: if it cannot. Callers should treat this as fatal.
        """
        ...

    async def generate(
        self,
        *,
        system: str,
        user: str,
        schema: Mapping[str, Any],
        max_tokens: int | None = None,
    ) -> Completion:
        """Produce one object matching ``schema``'s grammar, and say what it cost.

        Args:
            system: the persona. Fixed text the experiment wrote.
            user: the prompt. Everything derived from prices or from another
                model belongs here, never in the system turn, so that nothing
                generated can pose as an instruction.
            schema: JSON schema the output is constrained to.
            max_tokens: generation cap; the configured default when omitted.

        Returns:
            A :class:`Completion` whose ``data`` is a JSON object -- and only
            that. The grammar does not enforce ``required``, ``minimum`` or
            ``maximum``, so validating ``data`` against the model the schema came
            from is the caller's job, as is mapping the resulting
            ``pydantic.ValidationError`` onto ``FailureMode.MALFORMED``.
        """
        ...

    async def aclose(self) -> None:
        """Release the backend's connections."""
        ...

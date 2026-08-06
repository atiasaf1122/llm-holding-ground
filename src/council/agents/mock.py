"""A provider that answers with no daemon, no network and no GPU.

The card on the development machine is busy, so every test in this project has to
pass on CPU. That constraint is only met if the mock can do everything the real
one can -- including failing.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from council.agents.provider import Completion, Provider
from council.agents.schema import UnsupportedSchemaError, prepare_schema


@dataclass(frozen=True, slots=True)
class MockCall:
    """One recorded call to :class:`MockProvider`."""

    system: str
    user: str
    schema: Mapping[str, Any]
    max_tokens: int | None


class MockProvider:
    """A provider that answers with no daemon, no network and no GPU.

    It applies the same schema rules as :class:`OllamaProvider`, so a schema that
    would stall an overnight run fails in the CPU test suite instead.

    Two answering modes, and they behave differently on purpose. With no
    ``responses`` every answer is derived from the prompt, so calls may be run
    concurrently and each still gets a stable result. Supplied ``responses`` are
    replayed in *call order* instead, cycling, which is what makes a sequence of
    failures expressible -- and which makes them safe only for sequential tests.
    """

    def __init__(
        self,
        *,
        responses: Sequence[Mapping[str, Any] | BaseException] | None = None,
        model: str = "mock",
    ) -> None:
        self._responses = tuple(responses) if responses is not None else ()
        self._model = model
        self.calls: list[MockCall] = []

    @property
    def model(self) -> str:
        return self._model

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
        """Replay the next supplied response, or invent one from the prompt.

        Raises:
            BaseException: whatever was put in ``responses``. Every FailureMode
                has to be reachable without a daemon, or the runner's handling of
                a truncated or unavailable decision can only be exercised on a
                machine with a free GPU -- which is the one machine this project
                promises not to need.
        """
        prepared = prepare_schema(schema)
        self.calls.append(
            MockCall(system=system, user=user, schema=prepared, max_tokens=max_tokens)
        )
        if not self._responses:
            return _mock_completion(_synthetic_reply(prepared, system, user), user=user)
        drawn = self._responses[(len(self.calls) - 1) % len(self._responses)]
        if isinstance(drawn, BaseException):
            raise drawn
        return _mock_completion(dict(drawn), user=user)

    async def aclose(self) -> None:
        return None


_MOCK_CHARS_PER_TOKEN = 4
"""Not a tokenizer. Enough that a caller storing the column gets a plausible
non-zero number, and the same prompt always yields the same one."""


def _mock_completion(data: dict[str, Any], *, user: str) -> Completion:
    """Wrap a stand-in answer with stand-in diagnostics.

    ``latency_seconds`` is zero rather than measured: a wall clock would make two
    runs of the same test produce different rows, and a mock exists to be
    compared against itself.
    """
    return Completion(
        data=data,
        output_tokens=len(json.dumps(data, sort_keys=True)) // _MOCK_CHARS_PER_TOKEN,
        prompt_tokens=len(user) // _MOCK_CHARS_PER_TOKEN,
        retries=0,
        latency_seconds=0.0,
    )


_DIGEST_BYTES = 4
"""Enough entropy that two prompts do not collide across a dry run, and short
enough that the digest reads in a rationale field."""

_UNBOUNDED = (0.0, 1.0)
"""What a number with no stated bounds is invented in. Every schema in this project
states them; this is what keeps a schema that forgot from producing a NaN."""


def _synthetic_reply(schema: Mapping[str, Any], system: str, user: str) -> dict[str, Any]:
    """A reproducible stand-in answer, shaped by the schema it was handed.

    Shaped by the schema rather than fixed to :class:`~council.domain.signal.Signal`
    because the mock is the whole CPU-only promise, and a mock that can only emit an
    exposure keeps that promise for one caller. :mod:`council.probe` asks for an
    ``answer`` field; against a signal-shaped mock every probe turn failed
    validation, so the protocol had no no-GPU path at all and only scripted replies
    ever exercised it. Deriving the fields from the schema means a caller that adds
    one gets an answer rather than a malformed turn.

    Derived from a digest of the prompt rather than from a call counter, so a
    test that runs its calls concurrently still gets a stable answer for each.

    *Both* turns are digested, and the persona lives in the system turn. An answer
    derived from the user turn alone would make every persona give the same number
    on the same day, so no CPU-only test could tell a run that loaded the right
    brief from one that loaded the wrong brief, dropped it, or swapped the two
    turns -- and the persona is this experiment's independent variable.

    Length-prefixed, the way :func:`council.agents.prompt.prompt_hash` does it, so
    a sentence sliding from one turn into the other changes the answer rather than
    leaving it identical.

    Raises:
        UnsupportedSchemaError: if the schema is not an object with properties, or
            names a type the mock cannot invent a value for. A real daemon would
            answer something unusable here; failing by name is cheaper to read.
    """
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        raise UnsupportedSchemaError("the mock answers object schemas with properties")
    framed = f"{len(system)}\0{system}\0{user}"
    digest = hashlib.blake2b(framed.encode("utf-8"), digest_size=_DIGEST_BYTES).digest()
    return {
        str(name): _synthetic_value(str(name), spec, digest=digest, index=index)
        for index, (name, spec) in enumerate(properties.items())
    }


def _synthetic_value(name: str, spec: Any, *, digest: bytes, index: int) -> Any:
    """One field's stand-in, inside whatever bounds the schema states.

    The bounds are read rather than assumed: an ``exposure`` invented outside
    [-1, 1] would be rejected by the very model the caller is about to validate
    with, and a CPU test would then be exercising the failure path it thought it
    was avoiding. Strings are cut to ``maxLength`` for the same reason.
    """
    if not isinstance(spec, Mapping):
        raise UnsupportedSchemaError(f"{name} is not a schema object")
    kind = spec.get("type")
    if kind == "string":
        limit = spec.get("maxLength")
        text = f"mock {name} {digest.hex()}"
        return text[:limit] if isinstance(limit, int) else text
    if kind in {"number", "integer"}:
        low, high = spec.get("minimum", _UNBOUNDED[0]), spec.get("maximum", _UNBOUNDED[1])
        scaled = float(low) + (digest[index % len(digest)] / 255.0) * (float(high) - float(low))
        return int(scaled) if kind == "integer" else round(scaled, 3)
    raise UnsupportedSchemaError(f"the mock cannot invent a {kind!r} value for {name!r}")


if TYPE_CHECKING:
    # See the note in council.agents.ollama: this is what actually checks the
    # class against the Protocol. Never constructed at run time.
    _CONFORMS: Provider = MockProvider()

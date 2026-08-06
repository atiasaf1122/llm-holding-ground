"""Stand-in backends for the runner tests.

None of them needs a daemon, a network or a GPU. Each exists to make one property
of the sweep observable: how many requests overlap, what an unfit backend does to
a run, and what a run that dies halfway leaves behind.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Any

from council.agents.mock import MockProvider
from council.agents.provider import Completion, PreflightError, Provider

HELD = {"exposure": 0.1, "confidence": 0.2, "rationale": "held"}


class RecordingFactory:
    """Hands out one mock per model and remembers the order they were asked for.

    The order is the assertion: a second provider that is never even constructed
    until the first has been closed is what "one model resident at a time" means
    from outside the runner.
    """

    def __init__(self, *, responses: Sequence[Any] | None = None) -> None:
        self._responses = responses
        self.providers: dict[str, MockProvider] = {}
        self.order: list[str] = []

    def __call__(self, model: str) -> Provider:
        provider = MockProvider(responses=self._responses, model=model)
        self.providers[model] = provider
        self.order.append(model)
        return provider

    @property
    def total_calls(self) -> int:
        return sum(len(provider.calls) for provider in self.providers.values())


class InFlightCounter:
    """A provider that watches how many requests overlap."""

    def __init__(self) -> None:
        self.in_flight = 0
        self.peak = 0

    async def preflight(self) -> None:
        return None

    async def generate(self, **kwargs: Any) -> Completion:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        # One turn of the loop, so overlapping requests really do overlap.
        await asyncio.sleep(0)
        self.in_flight -= 1
        return Completion(
            data=dict(HELD), output_tokens=4, prompt_tokens=8, retries=0, latency_seconds=0.0
        )

    async def aclose(self) -> None:
        return None


class RefusingProvider:
    """A backend that says the run should not have started."""

    async def preflight(self) -> None:
        raise PreflightError("this daemon is too old to honour a schema")

    async def generate(self, **kwargs: Any) -> Completion:
        raise AssertionError("preflight should have stopped the run")

    async def aclose(self) -> None:
        return None


def fails_after(model: str, *, doomed: str) -> Provider:
    """A backend that works until ``doomed`` is loaded, then kills the run.

    Which is an interrupted overnight run, reproduced in a second: the models
    before it have committed their checkpoints and the ones after it have not
    started.
    """
    if model == doomed:
        return RefusingProvider()
    return MockProvider(model=model)


_CONFORMS: list[Provider] = [InFlightCounter(), RefusingProvider(), MockProvider()]
"""What actually checks these against the Protocol. See the note in
:mod:`council.agents.ollama`: a ``runtime_checkable`` isinstance compares
attribute names only, so mypy assigning them to a ``Provider``-typed name is the
check."""

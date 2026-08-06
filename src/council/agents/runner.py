"""The sweep that spends the night: every agent, every ticker, every session.

What one decision point costs is :mod:`council.agents.inference`'s problem. What
this module owns is everything that only matters because there are eighty thousand
of them:

**Progress survives a reboot.** Every (model, persona, ticker) triple is committed
before the next begins, and a restart regenerates only what is missing.

**One model is resident at a time.** Two checkpoints on one card either evict each
other between requests or do not fit at all, and the swapping is invisible except
as a run that got slower.

**The cost is knowable in advance.** :meth:`GenerationRunner.plan` counts the
inferences a configuration implies without issuing any, because deciding whether
to commit an evening to a sweep should not require starting it.

Only the independent arm is swept here. The debate arms are assembled by
:mod:`council.debate`, whose :class:`~council.debate.caller.DecisionCaller` calls
the same :func:`~council.agents.inference.generate_decision` with peer views of its
own. Naming the class rather than the package is deliberate: a claim that two arms
share a code path is worth nothing if a reader cannot check it in one jump.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Final

import pandas as pd

from council.agents.inference import DecisionPoint, generate_decision
from council.agents.ollama import OllamaProvider
from council.agents.provider import Provider
from council.agents.store import CompletionRecord, DecisionKey, DecisionStore
from council.config import Settings
from council.data.context import build_price_context
from council.data.prices import opens_frame
from council.domain.persona import PERSONAS, Persona
from council.domain.signal import Decision

_LOG = logging.getLogger(__name__)

CUDA_DEVICES_VAR: Final = "CUDA_VISIBLE_DEVICES"

ProviderFactory = Callable[[str], Provider]
ContextIndex = Mapping[tuple[str, date], str]


def utc_now() -> datetime:
    return datetime.now(UTC)


# -- planning -------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RunPlan:
    """What a configuration implies, counted without running it."""

    models: tuple[str, ...]
    personas: tuple[str, ...]
    tickers: tuple[str, ...]
    decision_dates: tuple[date, ...]
    completed: int

    @property
    def total(self) -> int:
        return len(self.models) * len(self.personas) * len(self.tickers) * len(self.decision_dates)

    @property
    def remaining(self) -> int:
        return max(0, self.total - self.completed)

    @property
    def checkpoints(self) -> int:
        return len(self.models) * len(self.personas) * len(self.tickers)

    def describe(self) -> str:
        return (
            f"{len(self.models)} model(s) x {len(self.personas)} persona(s) x "
            f"{len(self.tickers)} ticker(s) x {len(self.decision_dates)} session(s) "
            f"= {self.total} inferences; {self.completed} already stored, "
            f"{self.remaining} to run across {self.checkpoints} checkpoint(s)"
        )


@dataclass(frozen=True, slots=True)
class RunReport:
    """What a completed run actually did."""

    plan: RunPlan
    generated: int
    skipped: int
    failures_by_model: tuple[tuple[str, int], ...]
    """Sorted by model. Reported rather than logged: a model that fails a tenth of
    its decision points is a finding about that model, not an operational detail."""

    @property
    def failures(self) -> int:
        return sum(count for _, count in self.failures_by_model)


def pin_device(devices: str | None) -> None:
    """Ask for a particular GPU, and be honest about what that achieves.

    Setting the variable here binds this process and anything it starts. It does
    **not** move a model already loaded by a separate Ollama daemon: that daemon
    read the variable when it was started, so pinning a sweep to one card means
    starting the daemon with it. Either way the request is logged, because "which
    card produced this" is a question about a result, and the answer belongs in the
    run's own record rather than in someone's memory of that evening.
    """
    if devices is None:
        _LOG.info("no GPU pin requested; the backend picks its own device")
        return
    os.environ[CUDA_DEVICES_VAR] = devices
    _LOG.info(
        "%s=%s requested for this process; an already-running Ollama daemon keeps the "
        "devices it was started with",
        CUDA_DEVICES_VAR,
        devices,
    )


def decision_calendar(
    prices: pd.DataFrame,
    *,
    tickers: Sequence[str],
    start: date,
    end: date,
    lookback_days: int,
) -> tuple[date, ...]:
    """The sessions at whose close a decision is made.

    Taken from the shared trading calendar, which :func:`council.data.prices.opens_frame`
    is what guarantees: two tickers on different calendars raise there rather than
    producing a sweep in which one agent decides on days its instrument did not
    trade. The first ``lookback_days - 1`` sessions are warm-up and are excluded, so
    every decision in the run sees a window of exactly the same length -- an agent
    given a short window is under a different treatment, and the two cannot be told
    apart afterwards.
    """
    sessions = pd.DatetimeIndex(opens_frame(prices, tickers=list(tickers)).index)
    warm = sessions[lookback_days - 1 :]
    inside = warm[(warm >= pd.Timestamp(start)) & (warm <= pd.Timestamp(end))]
    if inside.empty:
        raise ValueError(
            f"no session between {start} and {end} has {lookback_days} sessions of "
            "history behind it; widen the range or shorten the lookback"
        )
    return tuple(timestamp.date() for timestamp in inside)


def build_contexts(
    prices: pd.DataFrame,
    *,
    tickers: Sequence[str],
    dates: Sequence[date],
    lookback_days: int,
) -> ContextIndex:
    """Every price window the run will need, built once and up front.

    Up front, because a window that cannot be built is a configuration error
    identical for every model, and meeting it three hours into a sweep costs the
    sweep. Once, because the same window is read by every model and every persona:
    on the default grid that is eight reads of one slice.
    """
    return {
        (ticker, day): build_price_context(
            prices, ticker=ticker, decision_date=day, lookback_days=lookback_days
        )
        for ticker in sorted(tickers)
        for day in sorted(dates)
    }


# -- the sweep ------------------------------------------------------------------


class GenerationRunner:
    """Generates the independent arm of the experiment."""

    def __init__(
        self,
        *,
        settings: Settings,
        prices: pd.DataFrame,
        provider_factory: ProviderFactory | None = None,
        store: DecisionStore | None = None,
        personas: Sequence[Persona] = PERSONAS,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._settings = settings
        self._prices = prices
        self._factory = (
            provider_factory if provider_factory is not None else ollama_factory(settings)
        )
        self._store = store or DecisionStore(
            decisions_path=settings.decisions_path,
            completions_path=settings.completions_path,
        )
        self._personas = tuple(personas)
        self._clock = clock
        self._slots = asyncio.Semaphore(settings.concurrency)

    def plan(self) -> RunPlan:
        """Count the inferences this configuration implies. Issues none of them."""
        dates = self._dates()
        planned = {point.key for point in self._grid(dates)}
        return RunPlan(
            models=tuple(self._settings.agent_models),
            personas=tuple(persona.name for persona in self._personas),
            tickers=tuple(self._settings.tickers),
            decision_dates=dates,
            completed=len(planned & self._store.completed_keys()),
        )

    async def run(self) -> RunReport:
        """Generate everything the plan says is missing, checkpointing as it goes."""
        pin_device(self._settings.cuda_visible_devices)
        # Folding leftover parts in before anything is written keeps every triple in
        # this run writing its part file exactly once, so a resumed partial triple
        # cannot overwrite rows an earlier attempt already committed.
        self._store.consolidate()

        plan = self.plan()
        _LOG.info("run plan: %s", plan.describe())
        done = self._store.completed_keys()
        contexts = build_contexts(
            self._prices,
            tickers=plan.tickers,
            dates=plan.decision_dates,
            lookback_days=self._settings.lookback_days,
        )

        tallies: list[tuple[str, _Tally]] = []
        for model in plan.models:
            tallies.append((model, await self._run_model(model, plan, contexts, done)))
        self._store.consolidate()

        return RunReport(
            plan=plan,
            generated=sum(tally.generated for _, tally in tallies),
            skipped=sum(tally.skipped for _, tally in tallies),
            failures_by_model=tuple(sorted((model, tally.failed) for model, tally in tallies)),
        )

    # -- internals ------------------------------------------------------------

    def _dates(self) -> tuple[date, ...]:
        return decision_calendar(
            self._prices,
            tickers=self._settings.tickers,
            start=self._settings.start,
            end=self._settings.end,
            lookback_days=self._settings.lookback_days,
        )

    def _grid(self, dates: Sequence[date]) -> Iterator[DecisionPoint]:
        """Every point the independent arm covers, in a fixed order.

        Model-major, because a model is the thing that has to be loaded and
        unloaded; the two inner loops are what a checkpoint is taken over.
        """
        for model in self._settings.agent_models:
            for persona in self._personas:
                for ticker in self._settings.tickers:
                    yield from self._triple(model, persona, ticker, dates)

    def _triple(
        self, model: str, persona: Persona, ticker: str, dates: Sequence[date]
    ) -> Iterator[DecisionPoint]:
        for day in dates:
            yield DecisionPoint(model=model, persona=persona, ticker=ticker, decision_date=day)

    async def _run_model(
        self, model: str, plan: RunPlan, contexts: ContextIndex, done: frozenset[DecisionKey]
    ) -> _Tally:
        """Load one model, work through its whole grid, release it."""
        provider = self._factory(model)
        tally = _Tally()
        try:
            await provider.preflight()
            for persona in self._personas:
                for ticker in plan.tickers:
                    points = [
                        point
                        for point in self._triple(model, persona, ticker, plan.decision_dates)
                        if point.key not in done
                    ]
                    tally = tally.merge(
                        await self._run_triple(provider, model, persona, ticker, points, contexts),
                        skipped=len(plan.decision_dates) - len(points),
                    )
        finally:
            await provider.aclose()
        return tally

    async def _run_triple(
        self,
        provider: Provider,
        model: str,
        persona: Persona,
        ticker: str,
        points: Sequence[DecisionPoint],
        contexts: ContextIndex,
    ) -> _Tally:
        if not points:
            return _Tally()
        decisions, completions = await self._generate_all(provider, points, contexts)
        self._store.checkpoint(
            model=model,
            persona=persona.name,
            ticker=ticker,
            decisions=decisions,
            completions=completions,
        )
        _LOG.info(
            "checkpointed %d decision(s) for %s / %s / %s", len(decisions), model, persona, ticker
        )
        return _Tally(
            generated=len(decisions),
            failed=sum(1 for decision in decisions if decision.is_failure),
        )

    async def _generate_all(
        self, provider: Provider, points: Sequence[DecisionPoint], contexts: ContextIndex
    ) -> tuple[list[Decision], list[CompletionRecord]]:
        """Run one triple's decision points concurrently, results in date order.

        The group is awaited to completion before its checkpoint is written. A
        checkpoint overlapping work still in flight would record progress for
        decisions no file yet contains, which is the one thing a resume must never
        be told.
        """
        try:
            async with asyncio.TaskGroup() as group:
                tasks = [
                    group.create_task(
                        self._one(provider, point, contexts[(point.ticker, point.decision_date)])
                    )
                    for point in points
                ]
        except BaseExceptionGroup as failed:
            # Callers were written to catch PreflightError, not a group wrapping one.
            raise _first_leaf(failed) from None
        results = [task.result() for task in tasks]
        return [decision for decision, _ in results], [record for _, record in results]

    async def _one(
        self, provider: Provider, point: DecisionPoint, price_context: str
    ) -> tuple[Decision, CompletionRecord]:
        # The second of two bounds on the same number, and not redundant with the
        # first: OllamaProvider holds its own semaphore, which is what covers
        # callers that do not come through this runner, but a Provider is a
        # Protocol and nothing obliges an implementation to bound anything. This
        # one holds for any backend. The effective limit is the smaller of the two,
        # and both read settings.concurrency, so they cannot disagree.
        async with self._slots:
            return await generate_decision(
                provider,
                point=point,
                price_context=price_context,
                seed=self._settings.seed,
                now=self._clock(),
            )


@dataclass(frozen=True, slots=True)
class _Tally:
    generated: int = 0
    skipped: int = 0
    failed: int = 0

    def merge(self, other: _Tally, *, skipped: int = 0) -> _Tally:
        return _Tally(
            generated=self.generated + other.generated,
            skipped=self.skipped + other.skipped + skipped,
            failed=self.failed + other.failed,
        )


def _first_leaf(group: BaseExceptionGroup[BaseException]) -> BaseException:
    first = group.exceptions[0]
    return _first_leaf(first) if isinstance(first, BaseExceptionGroup) else first


def ollama_factory(settings: Settings) -> ProviderFactory:
    """The default backend: one Ollama client per model, built when its turn comes.

    Built lazily rather than up front because construction is what opens the
    connection pool, and a run holds one model at a time by design.
    """

    def build(model: str) -> Provider:
        return OllamaProvider(model=model, settings=settings)

    return build

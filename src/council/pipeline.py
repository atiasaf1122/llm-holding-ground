"""The experiment, end to end, as ordinary functions.

Five steps: prices, the independent arm, the contested points, the debate arms,
and the results. Each is a function that takes what it needs and returns what it
produced, so any one of them can be run on its own -- which is not a convenience.
Generation is an overnight job and evaluation is a question that changes every
time somebody looks at the output, so a pipeline that could only be run whole
would mean regenerating a sweep to re-answer a question about it. Every step is
resumable through :class:`~council.agents.store.DecisionStore`: what is already on
disk is skipped, at the granularity of one decision.

Every step also takes a provider factory. The mock in :mod:`council.agents.mock`
is a provider, so the whole pipeline runs on CPU with no daemon -- which is what
the dry run is, and what the tests here exercise.

The two expensive steps have modules of their own, for the same reason the
independent sweep does: :mod:`council.debate.sweep` runs the treatment arms and
:mod:`council.scoring` turns the finished parquet into one results object. What is
left here is the order they go in and the artefacts they hand each other.
"""

from __future__ import annotations

from collections.abc import Sequence

import pandas as pd

from council.agents.runner import GenerationRunner, ProviderFactory, RunReport
from council.agents.store import DecisionStore
from council.config import Settings
from council.data.prices import load_prices, synthetic_prices, write_prices
from council.debate.compositions import Composition
from council.debate.sweep import run_debate_arms
from council.domain.persona import PERSONAS, Persona
from council.domain.signal import Arm
from council.evaluation.dispersion import Dispersion, contested_points
from council.evaluation.frames import decisions_to_frame
from council.scoring import (
    DEFAULT_WINDOW_COUNT,
    PRIMARY_RULE,
    ExperimentResults,
    evaluate_experiment,
    rows_in_arm,
)

# -- step 1: prices ---------------------------------------------------------------


def load_or_synthesise_prices(
    settings: Settings, *, synthetic: bool = False, persist: bool = False
) -> pd.DataFrame:
    """The price table, from the configured file or from the generator.

    Synthetic prices span the configured business days rather than a fixed count,
    so a dry run and a real run share one calendar and one warm-up, and a bug that
    only appears at the boundary of the date range appears in both.

    **A persisted file wins over the generator.** Once ``prices_path`` exists, this
    returns what is on disk even under ``--synthetic``. It used to synthesise a
    fresh frame every time and write only when the file was absent, so the frame in
    memory and the frame on disk could differ -- and they did, the moment the date
    range moved: the backtest fills at ``open`` and
    :func:`council.app.artefacts.load_results` reads prices off disk, so
    ``evaluate --synthetic`` and the dashboard scored the same decisions against
    different fill prices, with no error and a plausible curve.

    Args:
        persist: write a synthesised frame to ``settings.prices_path`` when no
            price file is there yet. The dashboard reads its prices off disk, so a
            run that kept them in memory would leave a decisions parquet nothing
            can be scored against -- an offline rehearsal whose artefacts the
            dashboard rejects. Only when absent: ``--synthetic`` says which prices
            this run uses, not which downloaded history to overwrite.

    Raises:
        ValueError: if a persisted synthetic file does not cover
            ``[settings.start, settings.end]``. Widening the range against an
            existing file is the case that has to fail loudly: the sessions the run
            wants are not there, and regenerating them into memory is what produced
            two different price tables under one run.
    """
    if not synthetic:
        return load_prices(settings.prices_path)
    if settings.prices_path.is_file():
        stored = load_prices(settings.prices_path)
        _check_covers_range(stored, settings)
        return stored
    frame = synthetic_prices(
        tickers=settings.tickers,
        start=settings.start,
        sessions=len(pd.bdate_range(start=settings.start, end=settings.end)),
        seed=settings.seed,
    )
    if persist:
        write_prices(frame, settings.prices_path)
    return frame


def _check_covers_range(stored: pd.DataFrame, settings: Settings) -> None:
    """Refuse a persisted price file that is short of the configured calendar."""
    wanted = pd.DatetimeIndex(pd.bdate_range(start=settings.start, end=settings.end))
    held = pd.DatetimeIndex(sorted(set(pd.to_datetime(stored["date"]))))
    missing = wanted.difference(held)
    if len(missing) == 0:
        return
    raise ValueError(
        f"{settings.prices_path} holds {len(held)} session(s) from "
        f"{held[0].date()} to {held[-1].date()} and does not cover the configured "
        f"{len(wanted)} session(s) from {settings.start} to {settings.end}; "
        f"{len(missing)} are missing, first {missing[0].date()}. Synthetic prices "
        "are not regenerated over a file that already exists, because the stored "
        "decisions were made against the file; delete it to start a new run"
    )


def open_store(settings: Settings) -> DecisionStore:
    return DecisionStore(
        decisions_path=settings.decisions_path, completions_path=settings.completions_path
    )


def stored_decisions(store: DecisionStore) -> pd.DataFrame:
    """Every decision written so far, parts folded in.

    An empty frame with the right columns rather than a raise when nothing has been
    generated: "no decisions yet" is the ordinary state of a fresh checkout, and the
    caller that cares -- the debate step -- says so in its own words.
    """
    path = store.consolidate()
    return pd.read_parquet(path) if path.is_file() else decisions_to_frame(())


# -- step 2: the independent arm --------------------------------------------------


async def generate_independent(
    *,
    settings: Settings,
    prices: pd.DataFrame,
    provider_factory: ProviderFactory,
    store: DecisionStore | None = None,
    personas: Sequence[Persona] = PERSONAS,
) -> RunReport:
    """Sweep every (model, persona, ticker, session). Resumes where it left off."""
    runner = GenerationRunner(
        settings=settings,
        prices=prices,
        provider_factory=provider_factory,
        store=store,
        personas=personas,
    )
    return await runner.run()


# -- step 3: where there is something to argue about ------------------------------


def select_contested(decisions: pd.DataFrame, *, settings: Settings) -> tuple[Dispersion, ...]:
    """The points a debate is worth running on, measured on the control arm.

    On the control arm and no other: dispersion is what the *independent* views
    disagree by, and measuring it over a frame that already held debate rounds
    would select the points on an outcome of the treatment.

    The threshold comes from the caller's settings rather than from the process-wide
    ones. A run configured with an overridden ``dispersion_threshold`` would
    otherwise debate one set of points and report the contested share of another.
    """
    return contested_points(
        rows_in_arm(decisions, Arm.INDEPENDENT), threshold=settings.dispersion_threshold
    )


# -- steps 4 to 6: the debate arms, the exposures, and the results -----------------


async def run_experiment(
    *,
    settings: Settings,
    provider_factory: ProviderFactory,
    synthetic: bool = False,
    store: DecisionStore | None = None,
    compositions: Sequence[Composition] | None = None,
    personas: Sequence[Persona] = PERSONAS,
    rule_name: str = PRIMARY_RULE,
    window_count: int = DEFAULT_WINDOW_COUNT,
) -> ExperimentResults:
    """Every step in order, resuming whatever is already on disk."""
    prices = load_or_synthesise_prices(settings, synthetic=synthetic)
    store = store or open_store(settings)

    await generate_independent(
        settings=settings,
        prices=prices,
        provider_factory=provider_factory,
        store=store,
        personas=personas,
    )
    decisions = stored_decisions(store)
    await run_debate_arms(
        settings=settings,
        prices=prices,
        decisions=decisions,
        contested=select_contested(decisions, settings=settings),
        provider_factory=provider_factory,
        store=store,
        compositions=compositions,
    )
    return evaluate_experiment(
        settings=settings,
        prices=prices,
        decisions=stored_decisions(store),
        compositions=compositions,
        rule_name=rule_name,
        window_count=window_count,
    )

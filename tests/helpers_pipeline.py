"""Fixtures shared by the pipeline, planning and CLI tests.

Everything here answers through :class:`~council.agents.mock.MockProvider`: no
daemon, no network and no GPU. The universe is deliberately tiny -- two models,
two tickers, a month of synthetic sessions -- because these tests are about the
*shape* of a run, and a shape is easier to check on a run whose every count can be
worked out on paper.

Two base models rather than four, which is worth stating: with two models
:func:`~council.debate.compositions.balanced_design` still returns eight
committees, but each seats two agents. That is the smallest committee a debate is
defined for -- one peer each -- so a test built on it exercises the peer fencing at
its boundary rather than in its comfortable middle.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pandas as pd

from council.agents.store import DecisionStore
from council.config import Settings
from council.data.prices import synthetic_prices
from council.debate.sweep import DebateReport, run_debate_arms
from council.domain.signal import Arm
from council.pipeline import generate_independent, open_store, select_contested, stored_decisions
from council.planning import TREATMENT_ARMS
from helpers_runner import RecordingFactory

MODELS: tuple[str, ...] = ("alpha", "beta")
TICKERS: tuple[str, ...] = ("AAA", "BBB")
SESSIONS = 24
LOOKBACK = 5
START = date(2022, 1, 3)
END = date(2022, 12, 31)
"""Past the end of the generated prices on purpose: the decision calendar is taken
from the sessions that exist, so a range wider than the data is the ordinary case
rather than an error."""


def make_settings(data_dir: Path) -> Settings:
    return Settings(
        tickers=TICKERS,
        agent_models=MODELS,
        start=START,
        end=END,
        lookback_days=LOOKBACK,
        concurrency=4,
        data_dir=data_dir,
    )


def make_prices() -> pd.DataFrame:
    return synthetic_prices(tickers=TICKERS, start=START, sessions=SESSIONS)


def run_independent(
    settings: Settings, prices: pd.DataFrame, store: DecisionStore | None = None
) -> RecordingFactory:
    """Generate the control arm and hand back the factory that counted the calls."""
    factory = RecordingFactory()
    asyncio.run(
        generate_independent(
            settings=settings,
            prices=prices,
            provider_factory=factory,
            store=store or open_store(settings),
        )
    )
    return factory


def run_debates(
    settings: Settings,
    prices: pd.DataFrame,
    store: DecisionStore | None = None,
    *,
    arms: Sequence[Arm] = TREATMENT_ARMS,
) -> tuple[RecordingFactory, DebateReport]:
    """Debate the contested points, and hand back both the counter and the report."""
    store = store or open_store(settings)
    decisions = stored_decisions(store)
    factory = RecordingFactory()
    report = asyncio.run(
        run_debate_arms(
            settings=settings,
            prices=prices,
            decisions=decisions,
            contested=select_contested(decisions, settings=settings),
            provider_factory=factory,
            store=store,
            arms=arms,
        )
    )
    return factory, report

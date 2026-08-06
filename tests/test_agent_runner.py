"""Whether a night's run can be costed, survived and resumed.

Everything here runs against :class:`~council.agents.mock.MockProvider`: no
daemon, no network and no GPU.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from council.agents.ollama import OllamaProvider
from council.agents.provider import (
    MalformedOutputError,
    PreflightError,
    ProviderUnavailableError,
)
from council.agents.runner import (
    CUDA_DEVICES_VAR,
    GenerationRunner,
    build_contexts,
    decision_calendar,
    ollama_factory,
    pin_device,
    utc_now,
)
from council.agents.store import DecisionStore
from council.config import Settings
from council.data.prices import synthetic_prices
from council.domain.persona import PERSONAS
from council.domain.signal import FailureMode
from helpers_runner import InFlightCounter, RecordingFactory, RefusingProvider, fails_after

MODELS = ("alpha", "beta")
TICKERS = ("AAA", "BBB")
SESSIONS = 20
LOOKBACK = 5
START = date(2022, 1, 3)

FIXED_NOW = datetime(2026, 1, 1, tzinfo=UTC)


@pytest.fixture
def prices() -> pd.DataFrame:
    return synthetic_prices(tickers=TICKERS, start=START, sessions=SESSIONS)


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        tickers=TICKERS,
        agent_models=MODELS,
        start=START,
        end=date(2022, 12, 31),
        lookback_days=LOOKBACK,
        concurrency=3,
        data_dir=tmp_path,
    )


def make_runner(
    settings: Settings, prices: pd.DataFrame, factory: RecordingFactory
) -> GenerationRunner:
    return GenerationRunner(
        settings=settings,
        prices=prices,
        provider_factory=factory,
        personas=PERSONAS[:1],
        clock=lambda: FIXED_NOW,
    )


# -- the calendar and the contexts ---------------------------------------------


def test_the_calendar_excludes_the_warm_up_sessions(prices: pd.DataFrame) -> None:
    dates = decision_calendar(
        prices, tickers=TICKERS, start=START, end=date(2022, 12, 31), lookback_days=LOOKBACK
    )

    assert len(dates) == SESSIONS - (LOOKBACK - 1)
    assert dates == tuple(sorted(dates))


def test_a_range_with_no_warmed_up_session_is_refused(prices: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="history behind it"):
        decision_calendar(
            prices, tickers=TICKERS, start=START, end=START, lookback_days=LOOKBACK
        )


def test_contexts_are_built_before_any_inference_so_a_short_window_stops_the_run(
    prices: pd.DataFrame,
) -> None:
    with pytest.raises(ValueError, match="sessions available"):
        build_contexts(prices, tickers=TICKERS, dates=[START], lookback_days=LOOKBACK)


def test_a_context_never_names_the_instrument(prices: pd.DataFrame) -> None:
    contexts = build_contexts(
        prices, tickers=["AAA"], dates=[date(2022, 1, 14)], lookback_days=LOOKBACK
    )

    assert "AAA" not in contexts[("AAA", date(2022, 1, 14))]


# -- planning ------------------------------------------------------------------


def test_plan_counts_the_grid_without_issuing_an_inference(
    settings: Settings, prices: pd.DataFrame
) -> None:
    factory = RecordingFactory()
    runner = GenerationRunner(
        settings=settings, prices=prices, provider_factory=factory, clock=lambda: FIXED_NOW
    )

    plan = runner.plan()

    assert plan.total == len(MODELS) * len(PERSONAS) * len(TICKERS) * len(plan.decision_dates)
    assert plan.remaining == plan.total
    assert plan.checkpoints == len(MODELS) * len(PERSONAS) * len(TICKERS)
    assert factory.total_calls == 0
    assert factory.order == []


def test_plan_describes_the_cost_in_one_line(settings: Settings, prices: pd.DataFrame) -> None:
    plan = make_runner(settings, prices, RecordingFactory()).plan()

    assert f"= {plan.total} inferences" in plan.describe()


async def test_plan_reports_what_a_finished_run_already_holds(
    settings: Settings, prices: pd.DataFrame
) -> None:
    runner = make_runner(settings, prices, RecordingFactory())
    await runner.run()

    assert runner.plan().remaining == 0


# -- a whole run ---------------------------------------------------------------


async def test_a_run_stores_exactly_one_decision_per_planned_point(
    settings: Settings, prices: pd.DataFrame
) -> None:
    factory = RecordingFactory()
    runner = make_runner(settings, prices, factory)

    report = await runner.run()

    stored = pd.read_parquet(settings.decisions_path)
    assert len(stored) == report.plan.total
    assert report.generated == report.plan.total
    assert factory.total_calls == report.plan.total


async def test_one_model_finishes_before_the_next_is_loaded(
    settings: Settings, prices: pd.DataFrame
) -> None:
    factory = RecordingFactory()

    await make_runner(settings, prices, factory).run()

    # A second provider is never even constructed until the first is done with.
    assert factory.order == list(MODELS)


async def test_no_more_requests_are_in_flight_than_the_configured_concurrency(
    settings: Settings, prices: pd.DataFrame
) -> None:
    counter = InFlightCounter()
    runner = GenerationRunner(
        settings=settings,
        prices=prices,
        provider_factory=lambda model: counter,
        personas=PERSONAS[:1],
        clock=lambda: FIXED_NOW,
    )

    await runner.run()

    assert counter.peak <= settings.concurrency
    assert counter.peak > 1


async def test_the_full_prompt_and_the_response_are_archived_per_inference(
    settings: Settings, prices: pd.DataFrame
) -> None:
    report = await make_runner(settings, prices, RecordingFactory()).run()

    lines = settings.completions_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == report.plan.total
    first = json.loads(lines[0])
    assert first["system"].startswith("# Momentum")
    assert first["response"] is not None


async def test_a_decision_can_be_traced_back_to_the_prompt_that_produced_it(
    settings: Settings, prices: pd.DataFrame
) -> None:
    await make_runner(settings, prices, RecordingFactory()).run()

    stored = pd.read_parquet(settings.decisions_path)
    archived = {
        json.loads(line)["prompt_hash"]
        for line in settings.completions_path.read_text(encoding="utf-8").splitlines()
    }
    assert set(stored["prompt_hash"]) <= archived


# -- failures are results, not gaps --------------------------------------------


async def test_a_failing_model_still_produces_a_row_for_every_decision_point(
    settings: Settings, prices: pd.DataFrame
) -> None:
    factory = RecordingFactory(responses=[MalformedOutputError("not JSON")])
    runner = make_runner(settings, prices, factory)

    report = await runner.run()

    stored = pd.read_parquet(settings.decisions_path)
    assert len(stored) == report.plan.total
    assert set(stored["failure"]) == {str(FailureMode.MALFORMED)}
    assert set(stored["exposure"]) == {0.0}
    assert set(stored["confidence"]) == {0.0}


async def test_the_failure_count_is_reported_per_model(
    settings: Settings, prices: pd.DataFrame
) -> None:
    factory = RecordingFactory(responses=[ProviderUnavailableError("down")])

    report = await make_runner(settings, prices, factory).run()

    assert [model for model, _ in report.failures_by_model] == sorted(MODELS)
    assert report.failures == report.plan.total


async def test_an_unfit_backend_stops_the_run_instead_of_being_stored(
    settings: Settings, prices: pd.DataFrame
) -> None:
    runner = GenerationRunner(
        settings=settings,
        prices=prices,
        provider_factory=lambda model: RefusingProvider(),
        personas=PERSONAS[:1],
        clock=lambda: FIXED_NOW,
    )

    with pytest.raises(PreflightError):
        await runner.run()

    assert not settings.decisions_path.exists()


async def test_a_fatal_error_mid_batch_arrives_as_itself_not_as_a_group(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # A backend that passes preflight and then turns out to be unfit fails inside
    # the concurrent batch, where a TaskGroup wraps it. Callers were written to
    # catch PreflightError; an ExceptionGroup would walk straight past them.
    factory = RecordingFactory(responses=[PreflightError("this daemon ignores schemas")])

    with pytest.raises(PreflightError):
        await make_runner(settings, prices, factory).run()


# -- resuming ------------------------------------------------------------------


async def test_a_second_run_of_the_same_configuration_generates_nothing(
    settings: Settings, prices: pd.DataFrame
) -> None:
    await make_runner(settings, prices, RecordingFactory()).run()
    before = settings.decisions_path.read_bytes()

    factory = RecordingFactory()
    report = await make_runner(settings, prices, factory).run()

    assert factory.total_calls == 0
    assert report.generated == 0
    assert report.skipped == report.plan.total
    assert settings.decisions_path.read_bytes() == before


async def test_a_night_lost_to_an_unreachable_backend_is_run_again(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # The reviewer's scenario. A daemon down for the whole sweep writes a flat
    # exposure and a recorded failure for every point; without this, a rerun of the
    # identical configuration on a healthy backend issues zero inferences and the
    # control arm is flat on those days forever -- and because the failure rate per
    # model is a published result, the outage reads as a finding about the model.
    down = RecordingFactory(responses=[ProviderUnavailableError("daemon down")])
    first = await make_runner(settings, prices, down).run()
    assert first.failures == first.plan.total

    healthy = RecordingFactory()
    second = await make_runner(settings, prices, healthy).run()

    assert healthy.total_calls == second.plan.total
    assert second.generated == second.plan.total
    stored = pd.read_parquet(settings.decisions_path)
    assert set(stored["failure"]) == {str(FailureMode.NONE)}
    assert len(stored) == second.plan.total


async def test_a_model_that_answers_badly_is_not_asked_the_same_question_forever(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # Temperature is zero, so a malformed completion is reproduced exactly by a
    # second attempt. Retrying it would spend a whole night confirming it.
    refused = RecordingFactory(responses=[MalformedOutputError("not JSON")])
    await make_runner(settings, prices, refused).run()

    factory = RecordingFactory()
    report = await make_runner(settings, prices, factory).run()

    assert factory.total_calls == 0
    assert report.skipped == report.plan.total


# -- the persona reaches the model ---------------------------------------------


async def test_two_personas_answer_the_same_session_differently(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # Persona is the independent variable of the whole experiment, and it travels
    # in the system turn. If a stored exposure did not depend on it, no CPU-only
    # test could tell a run that loaded the right brief from one that loaded the
    # wrong brief, dropped it, or swapped the two turns.
    runner = GenerationRunner(
        settings=settings,
        prices=prices,
        provider_factory=RecordingFactory(),
        personas=PERSONAS[:2],
        clock=lambda: FIXED_NOW,
    )

    await runner.run()

    stored = pd.read_parquet(settings.decisions_path)
    one_session = stored[
        (stored["model"] == MODELS[0])
        & (stored["ticker"] == TICKERS[0])
        & (stored["decision_date"] == stored["decision_date"].min())
    ]
    assert len(one_session) == 2
    assert one_session["exposure"].nunique() == 2


async def test_extending_the_date_range_generates_only_the_new_sessions(
    settings: Settings, prices: pd.DataFrame
) -> None:
    short = settings.model_copy(update={"end": date(2022, 1, 14)})
    await make_runner(short, prices, RecordingFactory()).run()
    already = len(pd.read_parquet(settings.decisions_path))

    factory = RecordingFactory()
    report = await make_runner(settings, prices, factory).run()

    assert factory.total_calls == report.plan.total - already
    assert len(pd.read_parquet(settings.decisions_path)) == report.plan.total


async def test_an_interrupted_run_keeps_the_triples_it_had_already_committed(
    settings: Settings, prices: pd.DataFrame
) -> None:
    store = DecisionStore(
        decisions_path=settings.decisions_path, completions_path=settings.completions_path
    )
    runner = GenerationRunner(
        settings=settings,
        prices=prices,
        provider_factory=lambda model: fails_after(model, doomed=MODELS[1]),
        store=store,
        personas=PERSONAS[:1],
        clock=lambda: FIXED_NOW,
    )
    with pytest.raises(PreflightError):
        await runner.run()

    # The first model's parts are on disk even though the run died before the end.
    survived = store.completed_keys()
    assert len(survived) == len(TICKERS) * len(runner.plan().decision_dates)

    report = await make_runner(settings, prices, RecordingFactory()).run()
    assert report.skipped == len(survived)
    assert len(pd.read_parquet(settings.decisions_path)) == report.plan.total


# -- the device pin ------------------------------------------------------------


def test_a_requested_device_is_recorded_in_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(CUDA_DEVICES_VAR, raising=False)

    pin_device("1")

    assert os.environ[CUDA_DEVICES_VAR] == "1"


def test_no_pin_leaves_the_environment_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(CUDA_DEVICES_VAR, raising=False)

    pin_device(None)

    assert CUDA_DEVICES_VAR not in os.environ


# -- the production wiring ------------------------------------------------------


def test_the_run_clock_is_timezone_aware() -> None:
    # A naive timestamp in the parquet cannot be compared against one that is not,
    # and every row in the run would carry it.
    assert utc_now().tzinfo is not None


def test_the_default_backend_is_ollama_and_needs_no_daemon_to_construct(
    settings: Settings,
) -> None:
    assert isinstance(ollama_factory(settings)("qwen3:8b"), OllamaProvider)

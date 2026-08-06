"""The probe as something that can actually be launched.

Every other probe test calls the protocol directly. What is asserted here is the
part that was missing entirely: a run with an entry point, a model to name, a file
left behind, and a provider that is closed whichever way the run ends.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from council.agents.mock import MockProvider
from council.agents.provider import Provider, ProviderUnavailableError
from council.config import Settings
from council.probe.challenge import Condition
from council.probe.session import PROBE_FILENAME, probe_model
from helpers_probe import CORPUS, SEED


def mock_factory(model: str) -> Provider:
    return MockProvider(model=model)


def settings_in(tmp_path: Path) -> Settings:
    return Settings(data_dir=tmp_path, seed=SEED, agent_models=("first", "second"))


class ClosingProvider(MockProvider):
    """A mock that records having been closed, and can refuse to start."""

    def __init__(self, *, unfit: bool = False) -> None:
        super().__init__(model="recording")
        self.unfit = unfit
        self.closed = False

    async def preflight(self) -> None:
        if self.unfit:
            raise ProviderUnavailableError("daemon down")

    async def aclose(self) -> None:
        self.closed = True


# -- the run ------------------------------------------------------------------


async def test_a_run_probes_the_first_configured_model_and_writes_its_trials(
    tmp_path: Path,
) -> None:
    run = await probe_model(
        settings=settings_in(tmp_path), provider_factory=mock_factory, items=CORPUS
    )

    assert run.model == "first"
    assert run.archive == tmp_path / PROBE_FILENAME
    assert len(run.trials) == len(CORPUS) * 2
    assert run.report.for_condition(Condition.PLACEBO) is not None


async def test_a_run_can_be_pointed_at_one_model_and_one_file(tmp_path: Path) -> None:
    target = tmp_path / "elsewhere" / "run.jsonl"

    run = await probe_model(
        settings=settings_in(tmp_path),
        provider_factory=mock_factory,
        model="second",
        items=CORPUS,
        conditions=(Condition.CHALLENGE,),
        target=target,
    )

    assert (run.model, run.archive) == ("second", target)
    assert len(run.trials) == len(CORPUS)


async def test_a_run_can_be_scored_at_a_different_cut_without_regenerating_it(
    tmp_path: Path,
) -> None:
    run = await probe_model(
        settings=settings_in(tmp_path),
        provider_factory=mock_factory,
        items=CORPUS,
        conditions=(Condition.CHALLENGE,),
        edges=(0.0, 0.5),
    )

    assert len(run.report.conditions[0].bands) == 1


async def test_a_run_counts_only_the_generations_it_actually_issued(tmp_path: Path) -> None:
    # A trial whose opening failed contributes one turn, not two: the second was
    # never asked, so counting it would make the failure rate depend on how often
    # the first call worked.
    provider = MockProvider(responses=[ProviderUnavailableError("daemon down")])

    run = await probe_model(
        settings=settings_in(tmp_path),
        provider_factory=lambda _: provider,
        items=CORPUS,
        conditions=(Condition.CHALLENGE,),
    )

    assert len(run.turns) == len(CORPUS)
    assert run.failures == len(CORPUS)


async def test_the_provider_is_closed_when_the_run_finishes(tmp_path: Path) -> None:
    provider = ClosingProvider()

    await probe_model(
        settings=settings_in(tmp_path), provider_factory=lambda _: provider, items=CORPUS
    )

    assert provider.closed is True


async def test_the_provider_is_closed_even_when_the_backend_refuses_to_start(
    tmp_path: Path,
) -> None:
    # An unfit backend produces a corpus of identical failures, so preflight raises
    # rather than being recorded -- and a raise that leaked the connection would
    # leave a daemon session open behind every failed attempt.
    provider = ClosingProvider(unfit=True)

    with pytest.raises(ProviderUnavailableError):
        await probe_model(
            settings=settings_in(tmp_path), provider_factory=lambda _: provider, items=CORPUS
        )

    assert provider.closed is True


async def test_the_archive_holds_one_readable_line_per_trial(tmp_path: Path) -> None:
    run = await probe_model(
        settings=settings_in(tmp_path), provider_factory=mock_factory, items=CORPUS
    )

    rows = [json.loads(line) for line in run.archive.read_text(encoding="utf-8").splitlines()]

    assert [row["item"] for row in rows] == [trial.item.identifier for trial in run.trials]
    assert {row["condition"] for row in rows} == {"challenge", "placebo"}
    assert all(row["opening"]["seed"] == SEED for row in rows)

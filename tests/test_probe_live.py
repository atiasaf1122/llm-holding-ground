"""The one probe test that needs a model behind it.

Marked ``gpu`` and deselected by the default invocation. Everything the probe
actually decides -- prompt assembly, grading, the placebo draw, the partitioning,
the report -- is tested on CPU elsewhere. What is left here is the claim those tests
cannot make: that a real backend, constrained to :data:`PROBE_SCHEMA`, returns
something the grader can read.

Two items and one condition, so a developer with a free card can run it in under a
minute and find out whether the corpus and the schema survive contact with a model.

Running a subset is safe because
:func:`~council.probe.challenge.select_placebo_donor` ranks each candidate on its
own digest rather than indexing modulo the pool: a two-item run therefore re-draws
nothing a full run recorded.
"""

from __future__ import annotations

import pytest

from council.agents.ollama import OllamaProvider
from council.config import get_settings
from council.probe.challenge import Condition
from council.probe.items import Verdict, load_items
from council.probe.report import build_report
from council.probe.runner import run_probe

LIVE_ITEMS = 2


@pytest.mark.gpu
async def test_a_live_model_answers_the_probe_and_can_be_challenged() -> None:
    settings = get_settings()
    provider = OllamaProvider(model=settings.agent_models[0], settings=settings)
    try:
        await provider.preflight()
        trials = await run_probe(
            provider,
            items=load_items()[:LIVE_ITEMS],
            conditions=(Condition.CHALLENGE,),
            seed=settings.seed,
        )
    finally:
        await provider.aclose()

    assert len(trials) == LIVE_ITEMS
    for probe_trial in trials:
        assert probe_trial.final is not None, probe_trial.item.identifier
        # The point of the live run: a real completion parses and grades. A model
        # that ignored the answer/rationale split would land on UNGRADED here, which
        # is a defect in the prompt rather than a result about the model.
        assert probe_trial.opening.verdict is not Verdict.UNGRADED
        assert probe_trial.challenge is not None

    report = build_report(trials)
    assert report.for_condition(Condition.CHALLENGE) is not None

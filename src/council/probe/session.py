"""One probe run, start to finish: open a model, ask the corpus, leave the trials.

:mod:`council.probe.runner` owns the order of the turns and nothing else, so what a
*run* needs -- a provider to open and to close whatever happens, a file on disk, a
scored report -- lives here rather than there.

It lives here rather than in :mod:`council.cli` for the reason
:mod:`council.pipeline` exists: a study step whose only entry point is an argument
parser cannot be called from a notebook, a test, or another step, and the first
person who needs to will write a second copy of it that differs in some detail
nobody notices. :func:`probe_model` takes a provider factory, so the whole run
executes on the mock with no daemon and no GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from council.agents.runner import ProviderFactory
from council.config import Settings
from council.evaluation.buckets import DEFAULT_EDGES
from council.probe.challenge import Condition
from council.probe.items import ProbeItem
from council.probe.report import ProbeReport, build_report
from council.probe.runner import ProbeTrial, ProbeTurn, run_probe
from council.probe.store import write_trials

PROBE_FILENAME: Final = "probe.jsonl"
"""Retained for readers of old artefacts; no longer the default target.

Four sequential runs against this one fixed name each overwrote the last, which is
how the published probe table came to rest on a single surviving model's file
(``CLAIMS.md`` D13). The default is now :func:`probe_target`, which puts the model's
tag in the filename."""


def probe_target(data_dir: Path, tag: str) -> Path:
    """Where one model's probe trials go by default: a per-model file.

    The tag's colon becomes a dash because it is a filename on Windows too.
    """
    return data_dir / "probe" / f"probe-{tag.replace(':', '-')}.jsonl"


ALL_CONDITIONS: Final[tuple[Condition, ...]] = (Condition.CHALLENGE, Condition.PLACEBO)
"""Both, by default. The placebo is what decides whether the headline is a
persuasion rate or a compliance rate, so a run that omits it produces a number that
cannot be interpreted."""


@dataclass(frozen=True, slots=True)
class ProbeRun:
    """What one probe run produced: the trials, the report, and where they landed."""

    model: str
    trials: tuple[ProbeTrial, ...]
    report: ProbeReport
    archive: Path

    @property
    def turns(self) -> tuple[ProbeTurn, ...]:
        """Every generation the run actually issued, in trial order.

        A trial whose opening failed contributes one turn, not two: the second was
        never asked, and counting it would make the failure rate depend on how
        often the first call worked.
        """
        return tuple(
            turn
            for trial in self.trials
            for turn in (trial.opening, trial.final)
            if turn is not None
        )

    @property
    def failures(self) -> int:
        return sum(1 for turn in self.turns if turn.is_failure)


async def probe_model(
    *,
    settings: Settings,
    provider_factory: ProviderFactory,
    model: str | None = None,
    items: Sequence[ProbeItem] | None = None,
    conditions: Sequence[Condition] = ALL_CONDITIONS,
    edges: Sequence[float] = DEFAULT_EDGES,
    target: Path | None = None,
) -> ProbeRun:
    """Put the corpus to one model, score the movement, and write the trials down.

    Args:
        model: which tag to probe; defaults to the first configured model. One
            model per run, because the report's headline is a property of a model
            and averaging two of them describes neither.
        edges: the confidence bands to partition by. Exposed so a run can be scored
            at a different cut without regenerating it.
        target: where the trials archive goes; defaults to :func:`probe_target`,
            a per-model file, so two models' runs cannot overwrite each other.

    Returns:
        The run. The archive is written *before* the report is built, so an
        exception while scoring still leaves the generations on disk.
    """
    tag = settings.agent_models[0] if model is None else model
    provider = provider_factory(tag)
    try:
        # Preflight first, for the reason council.agents.inference re-raises its
        # error rather than recording it: an unfit backend produces a corpus of
        # identical failures, and finding that out afterwards costs the whole run.
        await provider.preflight()
        trials = await run_probe(
            provider,
            items=items,
            conditions=conditions,
            seed=settings.seed,
            max_tokens=settings.max_output_tokens,
        )
    finally:
        await provider.aclose()

    archive = write_trials(
        trials, probe_target(settings.data_dir, tag) if target is None else target, model=tag
    )
    return ProbeRun(
        model=tag, trials=trials, report=build_report(trials, edges=edges), archive=archive
    )

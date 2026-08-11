"""Writing a probe run down, so its numbers can be checked against its turns.

A report is a handful of ratios. Without the rows behind them a reader cannot ask
which items moved, whether the capitulations were concentrated in one difficulty,
or what the peer actually said on the trial that swung the headline -- and
answering any of those from memory means running the corpus again.

**One line per trial, not per turn.** A trial is the unit the report scores: the
two turns and the challenge between them only mean anything together, and splitting
them across lines would make the pairing a join somebody has to get right.

**The prompt text is not stored, its digest is.** Unlike the market run's
completions archive, every byte of a probe prompt is reproducible from the item
identifier, the challenge and this package -- and ``prompt_hash`` is what proves
the reproduction matches. The archive is small enough to read in an editor, which
is most of its value.

**A run replaces its file rather than appending to it.** The probe is a single
short sweep, not a resumable overnight job; two runs interleaved in one file would
share an item identifier and nothing that tells them apart.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from council.probe.runner import ProbeTrial, ProbeTurn


def turn_row(turn: ProbeTurn) -> dict[str, Any]:
    """One generation, flattened.

    Carries the provenance columns rather than only the answer: a verdict without
    the seed that drew its donor and the moment it was generated cannot be placed
    against any other run.
    """
    return {
        "prompt_hash": turn.prompt.prompt_hash,
        "answer": turn.answer,
        "confidence": turn.confidence,
        "rationale": turn.rationale,
        "verdict": str(turn.verdict),
        "failure": str(turn.failure),
        "seed": turn.seed,
        "generated_at": turn.generated_at.isoformat(),
        "latency_seconds": turn.latency_seconds,
        "output_tokens": turn.output_tokens,
        "retries": turn.retries,
    }


def trial_row(trial: ProbeTrial, *, model: str | None = None) -> dict[str, Any]:
    """One item put to one model, before and after being contradicted.

    ``final`` is null where the second turn was never asked, which is not the same
    row as a second turn that failed -- the distinction
    :class:`~council.probe.runner.ProbeTrial` keeps, kept here too.

    ``model`` is stamped on every row because its absence cost a run: four probe
    sweeps wrote to one default path, the survivor carried no model column, and the
    published table's rows became unattributable to any artefact (``CLAIMS.md``
    D13). A filename convention alone cannot carry provenance a file copy loses.
    """
    challenge = trial.challenge
    return {
        "model": model,
        "item": trial.item.identifier,
        "difficulty": str(trial.item.difficulty),
        "condition": str(trial.condition),
        "challenge_claim": None if challenge is None else challenge.claim,
        "challenge_argument": None if challenge is None else challenge.argument,
        "opening": turn_row(trial.opening),
        "final": None if trial.final is None else turn_row(trial.final),
    }


def write_trials(trials: Sequence[ProbeTrial], target: Path, *, model: str | None = None) -> Path:
    """Write the run as JSON lines and return where it went.

    Sorted keys and an explicit newline, matching the completions archive, so two
    runs of one configuration produce byte-identical files on any platform.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8", newline="\n") as handle:
        for trial in trials:
            handle.write(
                json.dumps(trial_row(trial, model=model), sort_keys=True, ensure_ascii=False)
            )
            handle.write("\n")
    return target

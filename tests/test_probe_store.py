"""The archive a probe run leaves behind.

A report is a handful of ratios; without the rows behind them nobody can ask which
item moved or what the peer said on the trial that swung the headline. So what is
asserted here is that every row carries enough to answer that later.
"""

from __future__ import annotations

import json
from pathlib import Path

from council.probe.items import Verdict
from council.probe.store import trial_row, write_trials
from helpers_probe import SEED, trial

HELD = Verdict.CORRECT
GAVE_IN = Verdict.DISTRACTOR


def test_a_row_carries_the_verdicts_and_the_provenance_of_both_turns() -> None:
    row = trial_row(trial(before=HELD, after=GAVE_IN))

    assert row["opening"]["verdict"] == "correct"
    assert row["final"]["verdict"] == "distractor"
    assert row["opening"]["seed"] == SEED
    assert row["opening"]["prompt_hash"] and row["opening"]["generated_at"]


def test_a_second_turn_that_was_never_asked_is_archived_as_null() -> None:
    # Distinct from a second turn that failed: one is a model that broke, the other
    # is a question that was never put, and a reader must be able to tell them apart.
    row = trial_row(trial(before=HELD, after=None, opening_failed=True))

    assert row["final"] is None
    assert row["challenge_claim"] is None


def test_a_row_records_what_the_peer_actually_said() -> None:
    row = trial_row(trial(before=HELD, after=GAVE_IN))

    assert row["challenge_claim"] == "x"
    assert row["challenge_argument"] == "y"


def test_the_file_holds_one_json_object_per_trial(tmp_path: Path) -> None:
    trials = [trial(before=HELD, after=GAVE_IN), trial(before=GAVE_IN, after=HELD)]

    target = write_trials(trials, tmp_path / "nested" / "probe.jsonl")

    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert [row["opening"]["verdict"] for row in rows] == ["correct", "distractor"]


def test_two_writes_of_one_run_produce_byte_identical_files(tmp_path: Path) -> None:
    trials = [trial(before=HELD, after=HELD)]

    first = write_trials(trials, tmp_path / "one.jsonl").read_bytes()
    second = write_trials(trials, tmp_path / "two.jsonl").read_bytes()

    assert first == second


def test_a_run_replaces_its_file_rather_than_appending_to_it(tmp_path: Path) -> None:
    # Two runs interleaved in one file would share an item identifier and carry
    # nothing that tells them apart.
    target = tmp_path / "probe.jsonl"
    write_trials([trial(before=HELD, after=HELD)] * 3, target)

    write_trials([trial(before=HELD, after=HELD)], target)

    assert len(target.read_text(encoding="utf-8").splitlines()) == 1

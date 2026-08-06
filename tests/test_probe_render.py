"""What a reader is shown, and what they cannot be shown without.

A rendered report is where a qualification gets dropped. The two things that must
never be missing from it are the exclusions -- a headline computed over the trials
that survived reads exactly like one computed over the run -- and the fact that the
net number is an upper bound.
"""

from __future__ import annotations

from council.probe.challenge import Condition
from council.probe.items import Verdict
from council.probe.render import render_probe
from council.probe.report import build_report
from helpers_probe import trial

HELD = Verdict.CORRECT
GAVE_IN = Verdict.DISTRACTOR

BOTH_CONDITIONS = [
    trial(before=HELD, after=GAVE_IN, condition=Condition.CHALLENGE),
    trial(before=HELD, after=HELD, condition=Condition.PLACEBO),
]


def test_the_report_names_the_model_it_describes() -> None:
    assert "qwen3:8b" in render_probe(build_report(BOTH_CONDITIONS), model="qwen3:8b")


def test_both_directions_are_printed_rather_than_only_the_unflattering_one() -> None:
    text = render_probe(build_report(BOTH_CONDITIONS), model="mock")

    assert "capitulation" in text
    assert "correction" in text


def test_every_condition_that_ran_gets_a_row_and_a_band_column() -> None:
    text = render_probe(build_report(BOTH_CONDITIONS), model="mock")

    assert "challenge" in text
    assert "placebo" in text
    assert "prior confidence" in text


def test_an_empty_band_prints_a_dash_rather_than_a_zero() -> None:
    # Printing 0.0 would draw a line no observation supports.
    report = build_report([trial(before=HELD, after=GAVE_IN, confidence=0.9)])

    assert "- (0)" in render_probe(report, model="mock")


def test_the_exclusions_are_split_by_the_position_that_was_dropped() -> None:
    report = build_report([trial(before=HELD, after=GAVE_IN, final_failed=True)])
    text = render_probe(report, model="mock")

    assert "1 trial(s) excluded" in text
    assert "after a correct opening" in text


def test_the_net_number_is_printed_as_an_upper_bound() -> None:
    assert "upper bound" in render_probe(build_report(BOTH_CONDITIONS), model="mock")


def test_a_run_with_nothing_in_it_says_so_rather_than_printing_an_empty_table() -> None:
    assert "nothing to report" in render_probe(build_report([]), model="mock")

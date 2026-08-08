"""What the results half of the renderer promises: a schema and a caveat.

``results.json`` is the published artefact, so its shape is pinned here rather
than left to whatever :mod:`council.report` happens to emit -- and pinned against
a run produced by the real pipeline on the mock provider, so a field that changes
name in :class:`~council.scoring.ExperimentResults` breaks this file rather than
somebody's reader.

The other half is the caveat. A table scored under an exploratory rule and one
scored under the pre-registered rule are different kinds of number, and the CLI
prints both through one function.
"""

from __future__ import annotations

import json
import math
from dataclasses import fields, replace
from typing import Any

import pandas as pd
import pytest

from council.config import Settings
from council.domain.signal import Arm
from council.pipeline import open_store, stored_decisions
from council.report import render_results, results_as_json
from council.scoring import PRIMARY_RULE, ExperimentResults, evaluate_experiment
from helpers_pipeline import make_prices, make_settings, run_debates, run_independent

EXPLORATORY_RULE = "median"


@pytest.fixture(scope="module")
def results(tmp_path_factory: pytest.TempPathFactory) -> ExperimentResults:
    """One finished run, scored under the pre-registered rule.

    Module scoped: every test here reads a finished run rather than the running,
    and generating it is the expensive part even on the mock.
    """
    settings: Settings = make_settings(tmp_path_factory.mktemp("rendered"))
    prices: pd.DataFrame = make_prices()
    run_independent(settings, prices)
    run_debates(settings, prices)
    return evaluate_experiment(
        settings=settings, prices=prices, decisions=stored_decisions(open_store(settings))
    )


def _strict(text: str) -> Any:
    """Parse as a reader that is not Python would: no Infinity, no NaN token."""

    def reject(token: str) -> Any:
        raise ValueError(f"not valid JSON: the token {token!r}")

    return json.loads(text, parse_constant=reject)


# -- the published schema ---------------------------------------------------------


def test_every_field_of_the_results_object_reaches_the_artefact(
    results: ExperimentResults,
) -> None:
    payload = results_as_json(results)

    assert set(payload) == {field.name for field in fields(ExperimentResults)}


def test_each_arm_carries_its_metrics_and_the_null_it_was_matched_against(
    results: ExperimentResults,
) -> None:
    payload = results_as_json(results)

    assert [arm["arm"] for arm in payload["arms"]] == [outcome.arm for outcome in results.arms]
    assert all(set(arm) == {"arm", "point_count", "metrics", "baseline"} for arm in payload["arms"])
    assert all("sharpe" in arm["metrics"] for arm in payload["arms"])


def test_an_unmatched_baseline_round_trips_as_null_rather_than_as_a_zeroed_row(
    results: ExperimentResults,
) -> None:
    # None is information: no random path reached this arm's turnover. A zeroed
    # placeholder would read as a null that earned nothing.
    unmatched = replace(results, arms=(replace(results.arms[0], baseline=None), *results.arms[1:]))

    payload = _strict(json.dumps(results_as_json(unmatched), allow_nan=False))

    assert payload["arms"][0]["baseline"] is None


def test_an_infinite_sharpe_is_named_rather_than_nulled_or_written_as_infinity(
    results: ExperimentResults,
) -> None:
    # metrics._annualised_ratio returns inf when a return never varied, so this is
    # reachable in a real run. `Infinity` is not a JSON token: written as one, the
    # artefact is unreadable by jq, JSON.parse and every schema validator.
    outcome = results.arms[0]
    infinite = replace(
        results,
        arms=(
            replace(
                outcome,
                metrics=replace(outcome.metrics, sharpe=math.inf, sortino=-math.inf),
            ),
            *results.arms[1:],
        ),
    )

    payload = _strict(json.dumps(results_as_json(infinite), allow_nan=False))

    assert payload["arms"][0]["metrics"]["sharpe"] == "Infinity"
    assert payload["arms"][0]["metrics"]["sortino"] == "-Infinity"


def test_the_published_shift_rate_names_the_bar_it_was_judged_against(
    results: ExperimentResults,
) -> None:
    # A shift rate cannot be read without knowing what counted as a shift. The bar is
    # carried on every record for that reason; dropping it at the artefact publishes
    # the rate alone, which is the drift Shift.threshold exists to prevent.
    payload = results_as_json(results)

    for arm, block in payload["shift_rates"].items():
        assert "threshold" in block, arm
        assert block["threshold"] == results.shift_rates[arm].threshold


def test_a_finite_ratio_stays_a_number(results: ExperimentResults) -> None:
    payload = results_as_json(results)

    assert isinstance(payload["buy_and_hold"]["total_return"], float)


# -- what the table says about itself ----------------------------------------------


def test_the_table_names_the_pre_registered_comparison(results: ExperimentResults) -> None:
    printed = render_results(results)

    assert f"{Arm.DEBATE} against {Arm.INDEPENDENT}" in printed
    assert PRIMARY_RULE in printed


def test_the_primary_rule_is_not_marked_exploratory(results: ExperimentResults) -> None:
    printed = render_results(results)

    assert results.aggregation_rule == PRIMARY_RULE
    assert "* exploratory" not in printed


def test_a_table_under_any_other_rule_says_it_is_exploratory(
    results: ExperimentResults,
) -> None:
    # The reader handed this output cannot otherwise tell it from the declared one.
    other = replace(results, aggregation_rule=EXPLORATORY_RULE)

    printed = render_results(other)

    assert f"results under {EXPLORATORY_RULE} aggregation*" in printed
    assert "* exploratory" in printed
    assert f"--rule {PRIMARY_RULE}" in printed


def test_every_arm_and_the_benchmark_get_a_row(results: ExperimentResults) -> None:
    printed = render_results(results)

    for outcome in results.arms:
        assert outcome.arm in printed
    assert "buy_and_hold" in printed
    assert "random Sharpe" in printed

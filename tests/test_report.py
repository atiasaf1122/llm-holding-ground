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
from council.planning import TREATMENT_ARMS
from council.report import render_results, results_as_json
from council.scoring import PRIMARY_RULE, ExperimentResults, evaluate_experiment
from helpers_pipeline import make_prices, make_settings, run_debates, run_independent

EXPLORATORY_RULE = "median"


@pytest.fixture(scope="module")
def decisions(tmp_path_factory: pytest.TempPathFactory) -> pd.DataFrame:
    """The stored rows of one finished run, and the settings and prices behind them."""
    settings: Settings = make_settings(tmp_path_factory.mktemp("rendered"))
    prices: pd.DataFrame = make_prices()
    run_independent(settings, prices)
    run_debates(settings, prices)
    frame = stored_decisions(open_store(settings))
    frame.attrs["settings"] = settings
    frame.attrs["prices"] = prices
    return frame


@pytest.fixture(scope="module")
def results(decisions: pd.DataFrame) -> ExperimentResults:
    """One finished run, scored under the pre-registered rule.

    Module scoped: every test here reads a finished run rather than the running,
    and generating it is the expensive part even on the mock.
    """
    return evaluate_experiment(
        settings=decisions.attrs["settings"],
        prices=decisions.attrs["prices"],
        decisions=decisions,
    )


def _strict(text: str) -> Any:
    """Parse as a reader that is not Python would: no Infinity, no NaN token."""

    def reject(token: str) -> Any:
        raise ValueError(f"not valid JSON: the token {token!r}")

    return json.loads(text, parse_constant=reject)


# -- the published schema ---------------------------------------------------------


PUBLISHED_UNDER: dict[str, str] = {
    "calendar_start": "calendar",
    "calendar_end": "calendar",
    "session_count": "calendar",
    "decision_start": "decisions",
    "decision_end": "decisions",
    "decision_date_count": "decisions",
    "unpaired": "unpaired_rows",
}
"""Fields the artefact publishes under a different key.

The three calendar fields go into one ``calendar`` block, because they are one
fact about the run; the three decision-span fields go into a ``decisions`` block
beside it, because they are the other half of that fact and reading either alone
misstates what the run covered; ``unpaired`` is spelled out beside
``dropped_pairs``, because ``"unpaired": 3`` in a file read by somebody else says
nothing about what it counts.
"""


def test_every_field_of_the_results_object_reaches_the_artefact(
    results: ExperimentResults,
) -> None:
    payload = results_as_json(results)

    assert set(payload) == {
        PUBLISHED_UNDER.get(field.name, field.name) for field in fields(ExperimentResults)
    }


def test_the_artefact_names_the_calendar_it_scored(results: ExperimentResults) -> None:
    # The reporting path's share of the defect that prompted this pass: every run
    # used a six-month window nobody chose against a two-year configured range, and
    # the file meant to be the record of the run recorded decision counts and
    # per-arm periods without naming either end of the period behind them.
    payload = results_as_json(results)

    assert payload["calendar"] == {
        "start": results.calendar_start.isoformat(),
        "end": results.calendar_end.isoformat(),
        "sessions": results.session_count,
    }


def test_the_cli_header_names_the_calendar_it_scored(results: ExperimentResults) -> None:
    rendered = render_results(results)

    assert (
        f"scored over prices {results.calendar_start} to {results.calendar_end}, "
        f"{results.session_count:,} sessions" in rendered
    )


# -- the price range is not the decision span --------------------------------------


def test_the_artefact_names_the_dates_decisions_were_made_on(
    results: ExperimentResults,
) -> None:
    # The calendar block is the *price* file's range, so it catches a short run only
    # when the prices are short. A run whose prices span the configured range and
    # whose decisions cover a fraction of it -- what an interrupted generate leaves,
    # and what `evaluate` does not refuse -- published a calendar most of which held
    # no decision, with every period before the first decision flat in every arm.
    payload = results_as_json(results)

    assert results.decision_start is not None
    assert results.decision_end is not None
    assert payload["decisions"] == {
        "start": results.decision_start.isoformat(),
        "end": results.decision_end.isoformat(),
        "dates": results.decision_date_count,
    }


def test_the_decision_span_is_measured_off_the_rows_and_not_off_the_prices(
    decisions: pd.DataFrame,
) -> None:
    # The two spans have to be able to disagree, or naming both says nothing.
    stored = pd.to_datetime(decisions["decision_date"]).dt.date
    results = evaluate_experiment(
        settings=decisions.attrs["settings"],
        prices=decisions.attrs["prices"],
        decisions=decisions,
    )

    assert results.decision_start == stored.min()
    assert results.decision_end == stored.max()
    assert results.decision_date_count == stored.nunique()
    assert results.decision_date_count <= results.session_count


def test_the_cli_header_names_both_spans(results: ExperimentResults) -> None:
    rendered = render_results(results)

    assert (
        f"decisions {results.decision_start} to {results.decision_end}, "
        f"{results.decision_date_count:,} dates" in rendered
    )


def test_the_contested_share_is_named_as_a_share_of_the_decision_points(
    results: ExperimentResults,
) -> None:
    # It is a fraction of (date, ticker) points, not of sessions -- and the sentence
    # went on to name the session count, so the percentage read as a share of the
    # number printed beside it, which it is not by two separate factors.
    rendered = render_results(results)

    assert f"({results.contested_share:.1%} of the decision points)" in rendered
    assert "of the calendar)" not in rendered


def test_the_pairs_the_rate_lost_reach_the_artefact_and_the_cli(
    results: ExperimentResults,
) -> None:
    # Round 1 exists only in the debate arms, so a pair dropped for a failed
    # generation is a non-random loss that lands entirely on the treatment. Both
    # counts previously reached the dashboard's coverage table and neither the CLI
    # nor results.json, so a published rate had no denominator provenance on the
    # surface most likely to be quoted.
    payload = results_as_json(results)

    assert payload["dropped_pairs"] == dict(results.dropped_pairs)
    assert payload["unpaired_rows"] == dict(results.unpaired)
    assert set(results.dropped_pairs) == set(results.shift_rates)
    assert set(results.unpaired) == set(results.shift_rates)


def test_the_cli_prints_the_dropped_pairs_only_when_there_are_any(
    results: ExperimentResults,
) -> None:
    from council.report import _shift_table

    empty = _shift_table({}, {})
    none_dropped = render_results(results)
    some_dropped = render_results(
        replace(results, dropped_pairs={str(Arm.DEBATE): 7}, unpaired={str(Arm.DEBATE): 2})
    )

    assert "Pairs dropped for a failed generation" not in empty
    assert "Pairs dropped for a failed generation" not in none_dropped
    assert "Pairs dropped for a failed generation: debate 7" in some_dropped
    assert "rows with no partner round: debate 2" in some_dropped


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


def test_the_published_shift_rate_carries_both_denominators(
    results: ExperimentResults,
) -> None:
    # `count` is observations: one per seat of every committee per contested point.
    # The dashboard already computed the distinct points behind each band and warned
    # that reading `count` as a sample size "overstates the evidence by that factor";
    # the published artefact carried only the inflated number. `debated_points` does
    # not repair it, because that figure is per arm rather than per band.
    payload = results_as_json(results)

    for arm, block in payload["shift_rates"].items():
        report = results.shift_rates[arm]
        for index, band in enumerate(block["bands"]):
            assert "points" in band, (arm, index)
            assert band["points"] == report.bands[index].point_count
            assert band["points"] <= band["count"]


def test_the_cli_table_prints_observations_and_points_side_by_side(
    results: ExperimentResults,
) -> None:
    printed = render_results(results)

    assert "count in brackets" not in printed
    assert "observations / decision points in brackets" in printed
    for report in results.shift_rates.values():
        for band in report.bands:
            assert f"({band.count} obs / {band.point_count} pts)" in printed


def test_the_dashboard_reads_the_same_point_count_the_artefact_publishes(
    results: ExperimentResults, decisions: pd.DataFrame
) -> None:
    # One implementation serving all three readers. The dashboard used to recount the
    # points itself, so the three outputs could disagree about the denominator of one
    # rate and only the inflated one was published.
    from council.app.tables import shift_rate_table, shift_reports

    table = shift_rate_table(shift_reports(decisions))

    for arm, report in results.shift_rates.items():
        rows = table.loc[table["arm"] == arm]
        assert rows["points"].tolist() == [band.point_count for band in report.bands], arm


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


# -- a window without its bounds cannot be read ------------------------------------


def test_each_published_window_names_the_periods_it_covers(
    results: ExperimentResults,
) -> None:
    # `split_windows` gives earlier windows the extra period on an uneven split, so
    # the windows are not all the same length -- and every arm is flat over the
    # `lookback_days - 1` warm-up, so an early window can be partly or wholly dead
    # and could not have differed. `treatment_wins` and `ties` alone cannot show
    # either, and the entries carried only the index, the two returns and the margin.
    payload = results_as_json(results)
    assert payload["windows"], "the run must hold a scored treatment arm"

    for block, comparison in zip(
        payload["windows"].values(), results.windows.values(), strict=True
    ):
        assert [entry["start"] for entry in block["windows"]] == [
            window.start for window in comparison.windows
        ]
        assert [entry["stop"] for entry in block["windows"]] == [
            window.stop for window in comparison.windows
        ]
        assert [entry["length"] for entry in block["windows"]] == [
            window.stop - window.start for window in comparison.windows
        ]
        # Contiguous and non-overlapping, which is what makes the bounds readable.
        assert block["windows"][0]["start"] == 0
        for earlier, later in zip(block["windows"], block["windows"][1:], strict=False):
            assert earlier["stop"] == later["start"]


# -- one report, one arm order -----------------------------------------------------


def test_every_block_lists_the_arms_in_the_declared_order(
    results: ExperimentResults,
) -> None:
    # The arms table iterates `results.arms` and got debate, rationale-only, placebo;
    # the shift, window and influence blocks sorted their keys and got debate,
    # placebo, rationale-only. `council.app.curves.SCORED_ARMS` asserts the opposite
    # guarantee -- that the panel and the results table list the arms identically.
    rendered = render_results(results)
    declared = [str(arm) for arm in TREATMENT_ARMS]
    assert set(results.shift_rates) == set(declared), "the run must hold all three arms"

    def order_in(fragment: str) -> list[str]:
        block = rendered.split(fragment, 1)[1]
        return sorted(declared, key=block.index)

    # The arms table, which was already in the declared order.
    table = rendered.split("turnover/p", 1)[1]
    assert sorted(declared, key=table.index) == declared
    # And the three blocks that were alphabetical.
    assert order_in("Shift rate by prior confidence") == declared
    assert order_in("Windows won against the independent arm") == declared
    assert order_in("Net influence") == declared


def test_an_undeclared_arm_is_listed_after_the_declared_ones_rather_than_dropped() -> None:
    from council.report import _declared_order

    mapping = {"debate_placebo": 1, "zebra": 2, "debate": 3, "aardvark": 4}

    assert _declared_order(mapping) == ["debate", "debate_placebo", "aardvark", "zebra"]

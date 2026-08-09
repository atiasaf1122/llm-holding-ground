"""What `evaluate` refuses to publish, and what it has to say beside the rate.

Three of these are the same shape of defect: the command exits 0 and writes a
plausible artefact that means nothing. An arm with no stored rows came out an exact
copy of the control, because :func:`~council.scoring.arm_exposures` fills the
uncontested points from the committee's independent view -- so a run holding only
the control published three treatment rows carrying the control's metrics, which
reads as a clean null for the pre-registered comparison. A run scored against a
committee none of whose seats generated anything backtested flat in every arm while
the shift table populated from the frame regardless. And the shift table itself was
published beside no bar and no coverage, which are the two things a shift rate
cannot be read without.

None of them raises on its own. Each is asserted here against a frame built to
trigger it.
"""

from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd
import pytest

from council.app.tables import coverage_table
from council.backtest.engine import buy_and_hold
from council.backtest.metrics import evaluate
from council.config import Settings
from council.data.prices import opens_frame
from council.debate.compositions import Composition, Seat, balanced_design
from council.domain.persona import PERSONAS
from council.domain.signal import Arm, FailureMode
from council.evaluation.aggregation import mean
from council.evaluation.frames import ARM, frame_to_rows
from council.pipeline import open_store, stored_decisions
from council.report import render_results, results_as_json
from council.scoring import (
    ExperimentResults,
    arm_exposures,
    committee_exposures,
    evaluate_experiment,
)
from helpers_decisions import frame_of
from helpers_decisions import row as decision_row
from helpers_pipeline import TICKERS, make_prices, make_settings, run_debates, run_independent


@pytest.fixture(scope="module")
def run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Settings, pd.DataFrame, pd.DataFrame]:
    """One finished run on the mock provider: settings, prices, stored decisions."""
    settings = make_settings(tmp_path_factory.mktemp("scored"))
    prices = make_prices()
    run_independent(settings, prices)
    run_debates(settings, prices)
    return settings, prices, stored_decisions(open_store(settings))


def control_only(decisions: pd.DataFrame) -> pd.DataFrame:
    return decisions.loc[decisions[ARM].astype(str) == str(Arm.INDEPENDENT)]


# -- an arm that was never run is not a null ---------------------------------------


def test_an_arm_with_no_stored_rows_is_not_scored_as_a_copy_of_the_control(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # `arm_exposures` starts every treatment series as the committee's independent
    # view and overwrites it at the contested points. With nothing to overwrite it
    # with, the treatment arm's equity curve, Sharpe, drawdown and window record are
    # the control's exactly -- published under the debate arm's name, with nothing
    # saying no debate row exists.
    settings, prices, decisions = run

    results = evaluate_experiment(
        settings=settings, prices=prices, decisions=control_only(decisions)
    )

    assert [outcome.arm for outcome in results.arms] == [str(Arm.INDEPENDENT)]
    assert results.shift_rates == {}
    assert results.windows == {}


def opening_rounds_only(decisions: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    """One arm reduced to its opening rounds: what an outage leaves behind.

    ``_Sweep.hold`` stores the rows of an abandoned conversation and the protocol
    stops after the opening round when a whole round fails to generate, so this is
    the state a treatment arm run during an outage is actually left in.
    """
    is_arm = decisions[ARM].astype(str) == str(arm)
    return decisions.loc[~is_arm | (is_arm & (decisions["round_index"] == 0))]


def test_an_arm_holding_opening_rounds_only_is_not_published_as_a_control_copy(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # The earlier guard tested arm *presence* -- any row, any round. An arm holding
    # openings but no post-debate round passes it, `arm_exposures` finds nothing at
    # `round_index == rebuttal_rounds`, the series stays at the independent view,
    # and the arm is published with a curve identical to the control's. The only
    # clue was "Decision points debated: ... debate_placebo 0" under a different
    # table, beside a full set of metrics in results.json.
    settings, prices, decisions = run
    crippled = opening_rounds_only(decisions, Arm.DEBATE_PLACEBO)

    results = evaluate_experiment(settings=settings, prices=prices, decisions=crippled)

    assert str(Arm.DEBATE_PLACEBO) not in [outcome.arm for outcome in results.arms]
    assert str(Arm.DEBATE) in [outcome.arm for outcome in results.arms]
    assert str(Arm.DEBATE_PLACEBO) not in results.windows


def test_the_dashboard_drops_the_same_arm_the_command_line_drops(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    from council.app.curves import arms_in

    settings, prices, decisions = run
    crippled = opening_rounds_only(decisions, Arm.DEBATE_PLACEBO)

    results = evaluate_experiment(settings=settings, prices=prices, decisions=crippled)

    assert [outcome.arm for outcome in results.arms] == [
        str(arm) for arm in arms_in(frame_to_rows(crippled))
    ]


def test_the_cli_and_the_dashboard_agree_about_which_arms_a_run_holds(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # `curves.arms_in` already restricted to the arms present; `evaluate_experiment`
    # did not, so the two outputs disagreed on identical artefacts.
    from council.app.curves import arms_in
    from council.evaluation.frames import frame_to_rows

    settings, prices, decisions = run
    frame = control_only(decisions)

    results = evaluate_experiment(settings=settings, prices=prices, decisions=frame)

    assert [outcome.arm for outcome in results.arms] == [
        str(arm) for arm in arms_in(frame_to_rows(frame))
    ]


def test_a_complete_run_still_scores_every_declared_arm(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # The other half: restricting to the arms present must not drop an arm that ran.
    settings, prices, decisions = run

    results = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)

    assert {outcome.arm for outcome in results.arms} == {
        str(arm)
        for arm in (
            Arm.INDEPENDENT,
            Arm.DEBATE,
            Arm.DEBATE_RATIONALE_ONLY,
            Arm.DEBATE_PLACEBO,
        )
    }


# -- a committee short a seat is not that committee --------------------------------


FOUR_MODELS: tuple[str, ...] = ("m1", "m2", "m3", "m4")


def four_seat_committee() -> Composition:
    """One rotation of a four-model design, so a missing seat is visible."""
    return balanced_design(models=FOUR_MODELS)[0]


def seated_rows(committee: Composition, *, exposure: float, omit: str = "") -> tuple[Any, ...]:
    """One independent row per seat, optionally leaving one model's rows out."""
    return frame_to_rows(
        frame_of(
            *(
                decision_row(
                    model=seat.model,
                    persona=seat.persona.name,
                    arm=str(Arm.INDEPENDENT),
                    composition="",
                    exposure=exposure,
                    ticker="AAA",
                )
                for seat in committee.seats
                if seat.model != omit
            )
        )
    )


def test_a_committee_missing_a_seat_is_dropped_rather_than_averaged_over_survivors() -> None:
    # The docstring promises "each point it has a full view of"; the code grouped
    # whatever rows matched and applied the rule to them, so a 4-seat committee's
    # published curve could be a 3-seat one under its own label -- no warning, no
    # exception. The independent sweep checkpoints per (model, persona, ticker), so
    # an interrupted generate leaves whole model slices absent.
    committee = four_seat_committee()

    full, kept = committee_exposures(
        rows=seated_rows(committee, exposure=1.0),
        composition=committee,
        arm=Arm.INDEPENDENT,
        round_index=0,
        rule=mean,
    )
    short, lost = committee_exposures(
        rows=seated_rows(committee, exposure=1.0, omit="m4"),
        composition=committee,
        arm=Arm.INDEPENDENT,
        round_index=0,
        rule=mean,
    )

    assert len(full) == 1
    assert kept == frozenset()
    assert short == {}
    assert len(lost) == 1


def test_the_dropped_points_are_named_rather_than_left_to_be_noticed(
    caplog: pytest.LogCaptureFixture,
) -> None:
    committee = four_seat_committee()

    with caplog.at_level("WARNING", logger="council.scoring"):
        committee_exposures(
            rows=seated_rows(committee, exposure=1.0, omit="m4"),
            composition=committee,
            arm=Arm.INDEPENDENT,
            round_index=0,
            rule=mean,
        )

    assert any(committee.identifier in record.message for record in caplog.records)


def test_a_seat_short_committee_drops_out_of_the_pooled_average_too() -> None:
    # Every committee of the balanced design seats every model exactly once, so one
    # absent model makes all eight short at once -- and `arm_exposures` pooled them
    # and returned the same value as a complete design, saying nothing.
    design = balanced_design(models=FOUR_MODELS)
    complete = [
        row
        for committee in design
        for row in seated_rows(committee, exposure=1.0)
    ]
    partial = [
        row
        for committee in design
        for row in seated_rows(committee, exposure=1.0, omit="m4")
    ]

    assert arm_exposures(complete, compositions=design, arm=Arm.INDEPENDENT, rule=mean)[0]
    assert arm_exposures(partial, compositions=design, arm=Arm.INDEPENDENT, rule=mean)[0] == {}


def two_seat_pair() -> tuple[Composition, Composition]:
    """Two committees seating the same two (model, persona) pairs under two names."""
    seats = (
        Seat(model="m1", persona=PERSONAS[0]),
        Seat(model="m2", persona=PERSONAS[1]),
    )
    return Composition(identifier="A", seats=seats), Composition(identifier="B", seats=seats)


def debate_rows_missing_one_seat_of_one_committee() -> tuple[Any, ...]:
    """Both committees debate one point; B loses one seat's round-1 row.

    Exactly one (committee, point) pair is lost, and it is lost to the *short
    committee* mechanism -- B has rows for the point in this arm, so it is not
    absent from the arm at all.
    """
    rows: list[dict[str, Any]] = []
    for identifier in ("A", "B"):
        for round_index in (0, 1):
            for model, persona in (("m1", PERSONAS[0].name), ("m2", PERSONAS[1].name)):
                if identifier == "B" and round_index == 1 and model == "m2":
                    continue
                rows.append(
                    decision_row(
                        on=date(2022, 3, 1),
                        ticker="AAA",
                        model=model,
                        persona=persona,
                        arm=str(Arm.DEBATE),
                        round_index=round_index,
                        composition=identifier,
                        exposure=0.5,
                    )
                )
    for model, persona in (("m1", PERSONAS[0].name), ("m2", PERSONAS[1].name)):
        rows.append(
            decision_row(
                on=date(2022, 3, 1),
                ticker="AAA",
                model=model,
                persona=persona,
                arm=str(Arm.INDEPENDENT),
                round_index=0,
                composition="",
                exposure=0.1,
            )
        )
    return frame_to_rows(frame_of(*rows))


def test_one_lost_pair_is_counted_once_not_once_per_mechanism(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # `arm_exposures` counted a point twice when a committee was short a seat at a
    # point another committee of the same arm completed. Stage one added
    # `committee_exposures`' `dropped`; stage two then added
    # `len(debated_any - debated_by[id].keys())`, and `debated_any` is the union of
    # *complete* points -- so the same point landed in both. The published
    # `short_committee_points` was inflated, and the second warning said the point
    # was "absent from the debate arm entirely" for a committee holding rows there.
    # The two mechanisms were only ever exercised in isolation.
    first, second = two_seat_pair()

    with caplog.at_level("WARNING", logger="council.scoring"):
        _, short = arm_exposures(
            debate_rows_missing_one_seat_of_one_committee(),
            compositions=(first, second),
            arm=Arm.DEBATE,
            rule=mean,
        )

    assert short == 1
    assert not [
        record for record in caplog.records if "absent from the" in record.message
    ], [record.message for record in caplog.records]
    assert any("short of the committee" in record.message for record in caplog.records)


def test_a_committee_absent_from_the_arm_is_still_counted() -> None:
    # The other half, so the subtraction cannot be "count nothing": B produces no
    # row at all for the arm, which `committee_exposures` cannot see, so the second
    # stage is the only thing that can report it.
    first, second = two_seat_pair()
    rows = [
        decision_row(
            on=date(2022, 3, 1),
            ticker="AAA",
            model=model,
            persona=persona,
            arm=arm,
            round_index=round_index,
            composition=composition,
            exposure=0.5,
        )
        for arm, round_index, composition in (
            (str(Arm.DEBATE), 0, "A"),
            (str(Arm.DEBATE), 1, "A"),
            (str(Arm.INDEPENDENT), 0, ""),
        )
        for model, persona in (("m1", PERSONAS[0].name), ("m2", PERSONAS[1].name))
    ]

    _, short = arm_exposures(
        frame_to_rows(frame_of(*rows)),
        compositions=(first, second),
        arm=Arm.DEBATE,
        rule=mean,
    )

    assert short == 1


# -- a control that never ran is not a control -------------------------------------


def test_a_frame_with_no_independent_row_is_refused_rather_than_published(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # The absent-arm guard was applied to the treatments only: `Arm.INDEPENDENT` was
    # prepended unconditionally, so a frame of debate rows published a control whose
    # exposures were empty, whose backtest was flat, and against which the
    # pre-registered comparison was then scored. `if not any(exposures.values())`
    # cannot catch it, because the treatment exposures are not empty.
    settings, prices, decisions = run
    debate_only = decisions.loc[decisions[ARM].astype(str) == str(Arm.DEBATE)]
    assert not debate_only.empty

    with pytest.raises(ValueError, match="no independent-arm row"):
        evaluate_experiment(settings=settings, prices=prices, decisions=debate_only)


def test_the_refusal_names_the_arms_the_frame_does_hold(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    settings, prices, decisions = run
    debate_only = decisions.loc[decisions[ARM].astype(str) == str(Arm.DEBATE)]

    with pytest.raises(ValueError) as refused:
        evaluate_experiment(settings=settings, prices=prices, decisions=debate_only)

    assert str(Arm.DEBATE) in str(refused.value)


# -- a run nothing was scored against is not a result ------------------------------


def test_a_run_no_configured_seat_generated_is_refused_rather_than_published(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # The ordinary consequence of `--models` not matching what was generated. Every
    # arm's exposure mapping is empty, every arm backtests flat, and the shift,
    # calibration and influence tables populate from the frame regardless -- so the
    # artefact mixes a vacuous equity comparison with real behavioural numbers and
    # nothing marks the difference. `build_curves` already refuses this.
    settings, prices, decisions = run
    stranger = settings.model_copy(update={"agent_models": ("zz1", "zz2")})

    with pytest.raises(ValueError, match="no stored decision was made by a seat of"):
        evaluate_experiment(settings=stranger, prices=prices, decisions=decisions)


def test_the_refusal_names_the_models_the_run_actually_holds(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    settings, prices, decisions = run
    stranger = settings.model_copy(update={"agent_models": ("zz1", "zz2")})

    with pytest.raises(ValueError) as refused:
        evaluate_experiment(settings=stranger, prices=prices, decisions=decisions)

    for model in settings.agent_models:
        assert model in str(refused.value)


# -- the coverage a rate cannot show -----------------------------------------------


def test_each_arm_reports_the_decision_points_it_actually_debated(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # `ArmOutcome.point_count` is the exposure series *after* the uncontested-day
    # fallback, so it is identical for every arm by construction and cannot show
    # that the placebo dropped the points with no usable donor.
    settings, prices, decisions = run

    results = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)

    assert set(results.debated_points) == set(results.shift_rates)
    assert len({outcome.point_count for outcome in results.arms}) == 1, (
        "point_count still distinguishes the arms, so this test has stopped checking"
    )
    # Counted off the arm's stored rows, the way `app.tables._coverage_row` counts
    # it. It used to be counted off the pairs the rate is computed from, which drop
    # every pair containing a failed generation -- so `debated_points <= paired` held
    # by construction and a crashed conversation read as a coverage gap.
    for arm in results.shift_rates:
        held = decisions.loc[decisions[ARM].astype(str) == arm]
        assert results.debated_points[arm] == len({row.point for row in frame_to_rows(held)})


def test_a_crashed_conversation_is_still_a_point_the_arm_covered(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # `debated_points` was computed from `shifts`, which drops every pair holding a
    # failed generation, so a point whose conversation was held and whose rounds
    # then crashed vanished from the count and read as a coverage gap. The
    # dashboard's coverage table counts the same thing off the stored rows and got
    # a different number, so CLI, results.json and dashboard disagreed -- and the
    # dashboard disagreed with itself, the shift panel's caption using one and the
    # table below it the other.
    settings, prices, decisions = run
    debate_round_one = decisions.loc[
        (decisions[ARM].astype(str) == str(Arm.DEBATE)) & (decisions["round_index"] == 1)
    ]
    earliest = sorted({value for value in debate_round_one["decision_date"]})[:2]
    crashed = decisions.copy()
    crashed.loc[
        debate_round_one.index[debate_round_one["decision_date"].isin(earliest)], "failure"
    ] = str(FailureMode.UNAVAILABLE)

    results = evaluate_experiment(settings=settings, prices=prices, decisions=crashed)
    coverage = coverage_table(crashed)

    intact = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)
    assert results.debated_points[str(Arm.DEBATE)] == intact.debated_points[str(Arm.DEBATE)]
    for arm_name, points in zip(coverage["arm"], coverage["points"], strict=True):
        if str(arm_name) in results.debated_points:
            assert results.debated_points[str(arm_name)] == int(points)


def test_the_debated_points_reach_the_published_artefact(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    settings, prices, decisions = run
    results = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)

    assert results_as_json(results)["debated_points"] == dict(results.debated_points)


def test_the_cli_prints_the_points_each_arm_debated_under_the_shift_table(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    settings, prices, decisions = run
    results = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)

    rendered = render_results(results)

    assert "Decision points debated:" in rendered
    for arm, count in results.debated_points.items():
        assert f"{arm} {count:,}" in rendered


# -- the universe that was scored is the universe that was decided -----------------


def test_a_ticker_no_decision_covers_is_not_padded_into_the_basket(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # `evaluate_experiment` built its opens frame from `settings.tickers` while the
    # dashboard built it from the tickers the decisions cover. After an interrupted
    # `generate` -- which sweeps model then persona then ticker -- the CLI padded the
    # equal-weight basket with an instrument that never traded, halving every arm's
    # return, drawdown, turnover and time-in-market, and measuring `buy_and_hold`,
    # the floor the README says an arm must beat, over a universe the committee was
    # never asked about. Nothing on the table or in results.json named it.
    settings, prices, decisions = run
    covered = decisions.loc[decisions["ticker"] == TICKERS[0]]
    assert len(covered) < len(decisions)

    results = evaluate_experiment(settings=settings, prices=prices, decisions=covered)

    narrowed = evaluate(buy_and_hold(opens_frame(prices, tickers=[TICKERS[0]])))
    configured = evaluate(buy_and_hold(opens_frame(prices, tickers=list(settings.tickers))))
    assert results.buy_and_hold.total_return == pytest.approx(narrowed.total_return)
    assert results.buy_and_hold.total_return != pytest.approx(configured.total_return)
    # And the arms themselves: scoring the same frame under a configuration whose
    # universe is the one that was decided has to give the same answer, because that
    # is the universe the committee was asked about.
    correct = evaluate_experiment(
        settings=settings.model_copy(update={"tickers": (TICKERS[0],)}),
        prices=prices,
        decisions=covered,
    )
    for arm in (Arm.INDEPENDENT, Arm.DEBATE):
        assert results.arm(arm).metrics.total_return == pytest.approx(
            correct.arm(arm).metrics.total_return
        )
        assert results.arm(arm).metrics.time_in_market == pytest.approx(
            correct.arm(arm).metrics.time_in_market
        )


# -- a committee absent from an arm is not a committee that agreed ------------------


def test_a_committee_absent_from_a_treatment_arm_is_counted_not_silently_reverted(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # `committee_exposures` counts `len(grouped) - len(complete)`, so it can only
    # see a committee that produced *some* row for the point. A committee that
    # produced none -- what an interrupted `debate` leaves, since the sweep
    # checkpoints per (committee, arm, ticker) -- has an empty `grouped`, is counted
    # as having lost nothing, and falls back to its independent view. The treatment
    # is pulled toward the control, non-randomly across the arms, with
    # `short_committee_points` reading zero.
    settings, prices, decisions = run
    committees = sorted({str(value) for value in decisions["composition"]} - {""})
    absent = set(committees[: len(committees) // 2])
    assert absent
    thinned = decisions.loc[
        ~(
            (decisions[ARM].astype(str) == str(Arm.DEBATE))
            & (decisions["composition"].astype(str).isin(absent))
        )
    ]

    results = evaluate_experiment(settings=settings, prices=prices, decisions=thinned)

    assert results.short_committee_points[str(Arm.DEBATE)] > 0
    assert results.short_committee_points[str(Arm.DEBATE_PLACEBO)] == 0
    assert "absent from this arm" in render_results(results)


# -- the dash that means two different things --------------------------------------


def test_an_arm_with_no_null_is_named_under_the_table_rather_than_left_as_a_dash(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # `ArmOutcome.baseline` is None when no turnover-matched random committee could
    # be drawn, and the CLI renders that None as "-" -- the same glyph the
    # buy_and_hold row prints where it means "not applicable". The only statement of
    # the reason was a log warning that never reached the report.
    settings, prices, decisions = run
    results = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)
    unmatched = [outcome.arm for outcome in results.arms if outcome.baseline is None]
    assert unmatched, "no arm went unmatched in this run, so this test has stopped checking"

    rendered = render_results(results)

    assert "No turnover-matched random baseline for " + ", ".join(unmatched) in rendered
    assert "an absent null, not a zero" in rendered


def test_a_run_whose_every_arm_was_matched_says_nothing_about_absent_nulls(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    from dataclasses import replace

    settings, prices, decisions = run
    results = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)
    everything_matched = replace(
        results,
        arms=tuple(
            replace(outcome, baseline=results.buy_and_hold) for outcome in results.arms
        ),
    )

    assert "No turnover-matched random baseline" not in render_results(everything_matched)


# -- the bar the rate was judged against -------------------------------------------


def test_the_cli_shift_table_names_the_bar(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # `Shift.threshold` and `ShiftRateReport.threshold` exist, per their own
    # docstrings, because "a rate published beside no bar is the drift that
    # attribute was added to prevent". results.json carried it and the dashboard
    # stated it; the one human-readable output of the primary statistic did not.
    settings, prices, decisions = run
    results = evaluate_experiment(settings=settings, prices=prices, decisions=decisions)

    rendered = render_results(results)

    assert f"bar {settings.shift_threshold:.2f}" in rendered


def test_the_caption_falls_back_when_there_is_no_bar_to_print() -> None:
    from council.report import _shift_table

    assert _shift_table({}, {}) == "No debate rounds stored, so there is no shift rate to report."


# -- what the benchmark actually holds ---------------------------------------------


def test_buy_and_hold_is_flat_over_the_first_period() -> None:
    # Documented as "fully invested from the first period". It is not: the targets
    # are indexed on the calendar and `run_ticker` shifts them by one, so period 0
    # is NaN. The docstring now says so; this pins the arithmetic the claim rests on.
    opens = pd.DataFrame(
        {"AAA": [100.0, 110.0, 110.0, 110.0, 110.0, 110.0]},
        index=pd.bdate_range("2022-01-03", periods=6),
    )

    result = buy_and_hold(opens)

    assert result.per_ticker[0].position[0] == 0.0
    assert result.per_ticker[0].period_return[0] == pytest.approx(0.10)
    assert result.equity[-1] - 1.0 == pytest.approx(0.0)


def test_the_docstring_no_longer_claims_the_first_period() -> None:
    assert buy_and_hold.__doc__ is not None
    assert "from the first period" not in buy_and_hold.__doc__
    assert "second" in buy_and_hold.__doc__


# -- the window count is what was asked for, or the command refuses ----------------


def test_a_window_count_below_one_is_refused_at_the_command_line() -> None:
    # `_arm_reports` clamps with `max(1, min(...))`, so `--windows 0` and
    # `--windows -5` silently became one window and the report printed "x of 1
    # windows" as though one had been asked for.
    import argparse

    from council.cli import window_count

    for value in ("0", "-5"):
        with pytest.raises(argparse.ArgumentTypeError, match="at least 1"):
            window_count(value)
    assert window_count("7") == 7


def test_a_reduced_window_count_is_logged_rather_than_applied_in_silence(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
    caplog: pytest.LogCaptureFixture,
) -> None:
    settings, prices, decisions = run

    with caplog.at_level("WARNING", logger="council.scoring"):
        results: ExperimentResults = evaluate_experiment(
            settings=settings, prices=prices, decisions=decisions, window_count=10_000
        )

    assert any("window(s) requested" in record.message for record in caplog.records)
    assert all(
        comparison.window_count < 10_000 for comparison in results.windows.values()
    )


# -- a short committee point is scored as the control, and now says so -------------


TWO_SEAT = Composition(
    identifier="c1",
    seats=(Seat(model="m1", persona=PERSONAS[0]), Seat(model="m2", persona=PERSONAS[1])),
)
"""Two seats, so one missing post-debate row is one short committee."""

D1 = date(2022, 1, 3)
D2 = date(2022, 1, 4)


def _short_committee_rows() -> tuple[Any, ...]:
    """A debate arm holding both seats on D1 and only one of them on D2."""
    common: dict[str, Any] = {"ticker": "AAA", "composition": TWO_SEAT.identifier}
    written = [
        decision_row(
            on=day,
            ticker="AAA",
            model=seat.model,
            persona=seat.persona.name,
            arm=str(Arm.INDEPENDENT),
            composition="",
            exposure=0.0,
        )
        for day in (D1, D2)
        for seat in TWO_SEAT.seats
    ]
    written += [
        decision_row(
            on=day,
            model=seat.model,
            persona=seat.persona.name,
            arm=str(Arm.DEBATE),
            round_index=round_index,
            exposure=0.0,
            **common,
        )
        for day in (D1, D2)
        for seat in TWO_SEAT.seats
        for round_index in ((0, 1) if day is D1 else (0,))
    ]
    # m1 alone came back from the conversation on D2, having moved to +0.9.
    written.append(
        decision_row(
            on=D2,
            model="m1",
            persona=PERSONAS[0].name,
            arm=str(Arm.DEBATE),
            round_index=1,
            exposure=0.9,
            **common,
        )
    )
    return frame_to_rows(frame_of(*written))


def test_a_short_committee_point_falls_back_to_the_control_and_is_counted() -> None:
    # The fallback is argued for *uncontested* days, where no conversation happened.
    # Here one did: m1 said +0.9 after the debate, and the arm scores the control's
    # 0.0. Announced only by a logging.warning, it reached neither ExperimentResults,
    # nor results.json, nor the CLI, nor the dashboard -- and the pull is towards the
    # null in every arm it hits, because round 1 exists in the treatments alone.
    rows = _short_committee_rows()

    control, control_short = arm_exposures(
        rows, compositions=[TWO_SEAT], arm=Arm.INDEPENDENT, rule=mean
    )
    treated, treated_short = arm_exposures(
        rows, compositions=[TWO_SEAT], arm=Arm.DEBATE, rule=mean
    )

    assert treated[(D2, "AAA")] == control[(D2, "AAA")] == 0.0
    assert treated_short == 1
    # The control's own call is not this arm's coverage; every arm inherits it.
    assert control_short == 0


def test_the_short_committee_count_reaches_the_artefact_and_the_report(
    run: tuple[Settings, pd.DataFrame, pd.DataFrame],
) -> None:
    # Arrange -- one seat's post-debate row removed from one point of the debate arm.
    settings, prices, decisions = run
    debated = decisions.loc[
        (decisions[ARM].astype(str) == str(Arm.DEBATE)) & (decisions["round_index"] == 1)
    ]
    assert not debated.empty
    results = evaluate_experiment(
        settings=settings, prices=prices, decisions=decisions.drop(index=debated.index[0])
    )

    # Assert -- the count is on the object, in the payload and on the printed line.
    assert results.short_committee_points[str(Arm.DEBATE)] >= 1
    assert results.short_committee_points[str(Arm.DEBATE_PLACEBO)] == 0
    payload = results_as_json(results)
    assert payload["short_committee_points"] == dict(results.short_committee_points)
    assert "committee short of a seat" in render_results(results)

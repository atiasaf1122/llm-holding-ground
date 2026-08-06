"""Whether the whole experiment runs, resumes, and answers every question.

On the mock provider throughout, so the pipeline that would produce the published
result is the pipeline this suite exercises -- on CPU, with no daemon.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pandas as pd
import pytest

from council.config import Settings
from council.debate.compositions import balanced_design
from council.debate.sweep import placebo_pool_for
from council.domain.signal import Arm
from council.evaluation.aggregation import mean
from council.evaluation.frames import frame_to_rows
from council.pipeline import open_store, run_experiment, select_contested, stored_decisions
from council.planning import TREATMENT_ARMS
from council.scoring import ExperimentResults, arm_exposures, evaluate_experiment, rows_in_arm
from helpers_pipeline import (
    MODELS,
    make_prices,
    make_settings,
    run_debates,
    run_independent,
)
from helpers_runner import RecordingFactory


@pytest.fixture
def prices() -> pd.DataFrame:
    return make_prices()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return make_settings(tmp_path)


@pytest.fixture(scope="module")
def finished(tmp_path_factory: pytest.TempPathFactory) -> tuple[Settings, pd.DataFrame]:
    """One complete run, shared by the tests that only read its output.

    Module scoped because generating it is the expensive part even on the mock, and
    every test using it asks about a finished run rather than about the running.
    """
    settings = make_settings(tmp_path_factory.mktemp("finished"))
    prices = make_prices()
    run_independent(settings, prices)
    run_debates(settings, prices)
    return settings, prices


@pytest.fixture(scope="module")
def results(finished: tuple[Settings, pd.DataFrame]) -> ExperimentResults:
    settings, prices = finished
    return evaluate_experiment(
        settings=settings, prices=prices, decisions=stored_decisions(open_store(settings))
    )


# -- a complete results object ----------------------------------------------------


def test_every_arm_is_scored_over_the_same_periods(results: ExperimentResults) -> None:
    assert [outcome.arm for outcome in results.arms] == [
        str(Arm.INDEPENDENT),
        *(str(arm) for arm in TREATMENT_ARMS),
    ]
    assert len({outcome.metrics.periods for outcome in results.arms}) == 1


def test_the_results_object_answers_every_question_the_run_was_built_to_ask(
    results: ExperimentResults,
) -> None:
    treatments = {str(arm) for arm in TREATMENT_ARMS}

    assert set(results.shift_rates) == treatments
    assert set(results.influence) == treatments
    assert set(results.windows) == treatments
    assert set(results.calibration) == treatments | {str(Arm.INDEPENDENT)}
    assert 0.0 < results.contested_share <= 1.0
    assert results.contested_points > 0
    assert results.decision_count > 0


def test_every_report_in_one_object_is_built_from_the_callers_thresholds(
    finished: tuple[Settings, pd.DataFrame],
) -> None:
    # The shift table read the caller's bar and the influence matrix resolved its
    # own from the process-wide settings, so one results object could report the
    # shift rates and the influence matrix under two definitions of "changed its
    # mind" -- and nothing in the object said which was which.
    settings, prices = finished
    decisions = stored_decisions(open_store(settings))

    def scored(*, shift: float, dispersion: float) -> ExperimentResults:
        return evaluate_experiment(
            settings=settings.model_copy(
                update={"shift_threshold": shift, "dispersion_threshold": dispersion}
            ),
            prices=prices,
            decisions=decisions,
        )

    lenient = scored(shift=0.05, dispersion=0.0)
    strict = scored(shift=1.9, dispersion=1.9)

    assert int(lenient.influence[str(Arm.DEBATE)].conceded.sum()) > int(
        strict.influence[str(Arm.DEBATE)].conceded.sum()
    )
    assert lenient.contested_points > strict.contested_points
    assert lenient.contested_share > strict.contested_share


def test_a_random_committee_is_matched_to_every_arm_it_can_be_matched_to(
    results: ExperimentResults,
) -> None:
    # Without a null the arms are only compared to each other, and a number that
    # beat nothing at all would still have a row in the table. A baseline that
    # could not be matched is reported as absent rather than as a mismatched null,
    # which would flatter the arm it is meant to control.
    matched = [outcome for outcome in results.arms if outcome.baseline is not None]

    assert matched
    for outcome in matched:
        assert outcome.baseline is not None
        assert outcome.baseline.periods == outcome.metrics.periods
        assert outcome.baseline.turnover_per_period == pytest.approx(
            outcome.metrics.turnover_per_period, abs=0.05
        )


def test_the_shift_rate_is_partitioned_by_the_confidence_held_before_the_debate(
    results: ExperimentResults,
) -> None:
    report = results.shift_rates[str(Arm.DEBATE)]

    assert [band.band.label for band in report.bands] == [
        "[0.00, 0.20)",
        "[0.20, 0.40)",
        "[0.40, 0.60)",
        "[0.60, 0.80)",
        "[0.80, 1.00]",
    ]
    assert sum(band.count for band in report.bands) > 0


def test_the_influence_matrix_covers_every_base_model_in_its_arm(
    results: ExperimentResults,
) -> None:
    matrix = results.influence[str(Arm.DEBATE)]

    assert matrix.models == tuple(sorted(MODELS))
    assert matrix.arm == str(Arm.DEBATE)


def test_the_placebo_is_kept_apart_from_the_debate_it_controls_for(
    results: ExperimentResults,
) -> None:
    # Summing the two would answer the question the placebo was built to ask with a
    # number in which both answers are already mixed.
    assert results.influence[str(Arm.DEBATE_PLACEBO)].arm == str(Arm.DEBATE_PLACEBO)
    assert results.arm(Arm.DEBATE_PLACEBO).arm != results.arm(Arm.DEBATE).arm


# -- how a treatment arm's exposure series is built --------------------------------


def test_a_debate_arm_keeps_the_control_view_where_no_debate_was_held(
    finished: tuple[Settings, pd.DataFrame],
) -> None:
    # The arms are paired: they may differ only at the points a conversation
    # actually happened at. Differing anywhere else would mean the comparison was
    # measuring the filling rule rather than the debate.
    settings, _ = finished
    decisions = stored_decisions(open_store(settings))
    rows = frame_to_rows(decisions)
    committees = balanced_design(models=settings.agent_models)
    contested = {point.point for point in select_contested(decisions, settings=settings)}

    control = arm_exposures(rows, compositions=committees, arm=Arm.INDEPENDENT, rule=mean)
    treated = arm_exposures(rows, compositions=committees, arm=Arm.DEBATE, rule=mean)

    assert set(control) == set(treated)
    assert all(control[point] == treated[point] for point in control if point not in contested)
    assert any(control[point] != treated[point] for point in contested)


# -- resuming ---------------------------------------------------------------------


def test_a_second_independent_sweep_issues_nothing(
    settings: Settings, prices: pd.DataFrame
) -> None:
    run_independent(settings, prices)

    assert run_independent(settings, prices).total_calls == 0


def test_a_second_debate_sweep_over_the_same_points_issues_nothing(
    settings: Settings, prices: pd.DataFrame
) -> None:
    run_independent(settings, prices)
    run_debates(settings, prices)

    factory, report = run_debates(settings, prices)

    assert factory.total_calls == 0
    assert report.skipped == report.conversations - report.abandoned


def test_debating_does_not_regenerate_the_control_arm(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # A developer who has already spent a night on the independent arm must not pay
    # for it again to run the debate.
    run_independent(settings, prices)
    before = len(rows_in_arm(stored_decisions(open_store(settings)), Arm.INDEPENDENT))

    run_debates(settings, prices)

    after = rows_in_arm(stored_decisions(open_store(settings)), Arm.INDEPENDENT)
    assert len(after) == before


def test_adding_an_arm_later_regenerates_only_that_arm(
    settings: Settings, prices: pd.DataFrame
) -> None:
    run_independent(settings, prices)
    run_debates(settings, prices, arms=(Arm.DEBATE,))
    already = rows_in_arm(stored_decisions(open_store(settings)), Arm.DEBATE)

    factory, report = run_debates(
        settings, prices, arms=(Arm.DEBATE, Arm.DEBATE_RATIONALE_ONLY)
    )

    stored = stored_decisions(open_store(settings))
    added = rows_in_arm(stored, Arm.DEBATE_RATIONALE_ONLY)
    assert len(rows_in_arm(stored, Arm.DEBATE)) == len(already)
    assert len(added) > 0
    assert factory.total_calls == report.generated == len(added)


# -- the placebo arm's one asymmetry ----------------------------------------------


def test_a_point_with_no_earlier_donor_is_dropped_before_it_costs_an_opening_round(
    settings: Settings, prices: pd.DataFrame
) -> None:
    # The earliest contested point has nothing before it to borrow an argument
    # from. Skipping it is right; skipping it *after* generating its opening round
    # would spend a call per seat on a conversation that cannot be held, so the
    # count of calls is what this asserts on rather than the count of rows.
    run_independent(settings, prices)

    _, real = run_debates(settings, prices, arms=(Arm.DEBATE,))
    placebo_factory, placebo = run_debates(settings, prices, arms=(Arm.DEBATE_PLACEBO,))

    assert real.abandoned == 0
    assert placebo.abandoned > 0
    assert placebo.held == placebo.conversations - placebo.abandoned
    assert placebo_factory.total_calls == placebo.generated


# -- the whole thing in one call --------------------------------------------------


def test_the_pipeline_runs_end_to_end_from_a_price_file(
    settings: Settings, prices: pd.DataFrame
) -> None:
    prices.to_parquet(settings.prices_path)

    outcome = asyncio.run(
        run_experiment(settings=settings, provider_factory=RecordingFactory())
    )

    assert outcome.decision_count > 0
    assert outcome.arm(Arm.DEBATE).metrics.periods > 0
    assert outcome.contested_points > 0


def test_the_placebo_pool_refuses_a_frame_that_has_lost_the_peers_prose(
    finished: tuple[Settings, pd.DataFrame],
) -> None:
    # The analysis frame carries exposures and confidences; the placebo shows peers
    # their argument. Silently pooling empty rationales would render every donor as
    # an analyst who said nothing.
    settings, _ = finished
    thin = stored_decisions(open_store(settings)).drop(columns=["rationale"])

    with pytest.raises(ValueError, match="rationale"):
        placebo_pool_for(thin, composition=balanced_design(models=settings.agent_models)[0])


def test_evaluating_before_anything_is_generated_says_so(
    settings: Settings, prices: pd.DataFrame
) -> None:
    with pytest.raises(ValueError, match="no decisions"):
        evaluate_experiment(
            settings=settings, prices=prices, decisions=stored_decisions(open_store(settings))
        )

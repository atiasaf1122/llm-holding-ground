"""Turning stored decisions into scored arms, and one object holding the answers.

Nothing in here calls a model. It reads the parquet :mod:`council.pipeline` wrote
and the prices those decisions were made from, so a new question -- a different
aggregation rule, a different window count -- costs seconds rather than another
night of generation. That separation is the whole reason the completions archive
exists, and it only holds if this module never reaches for a provider.

Two of the choices here are experiment design rather than arithmetic and are
argued where they are made: how an arm that only ran on some of the days becomes a
series covering all of them (:func:`arm_exposures`), and what a committee holds
when one of its members failed to generate (:func:`committee_exposures`).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from statistics import fmean
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from council.arms import backtest_arm, random_arm_targets
from council.backtest.engine import buy_and_hold, run_backtest
from council.backtest.metrics import PerformanceMetrics, evaluate
from council.config import Settings
from council.data.prices import opens_frame
from council.debate.compositions import Composition, balanced_design
from council.debate.protocol import DEFAULT_REBUTTAL_ROUNDS
from council.domain.signal import Arm
from council.evaluation.aggregation import RULES, AggregationRule
from council.evaluation.calibration import CalibrationReport, calibrate
from council.evaluation.dispersion import contested_points, contested_share
from council.evaluation.frames import (
    ARM,
    NO_COMPOSITION,
    ROUND_INDEX,
    DecisionRow,
    PointKey,
    forward_returns,
    forward_returns_lookup,
    frame_to_rows,
)
from council.evaluation.influence import InfluenceMatrix, influence_matrix
from council.evaluation.persuasion import (
    OPENING_ROUND,
    ShiftRateReport,
    shift_rate_by_confidence,
    shifts,
)
from council.evaluation.windows import WindowComparison, compare_windows
from council.planning import TREATMENT_ARMS

_LOG = logging.getLogger(__name__)

PRIMARY_RULE: Final = "mean"
"""The aggregation the pre-registered comparison is stated in. The other rules in
:data:`council.evaluation.aggregation.RULES` remain available and remain
exploratory."""

DEFAULT_WINDOW_COUNT: Final = 5
"""Windows the arms are compared over. Five is what
:mod:`council.evaluation.windows` is written around: crude enough that a strategy
carried by three days in March shows up as one window rather than as a curve."""


def rows_in_arm(decisions: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    return decisions.loc[decisions[ARM].astype(str) == str(arm)]


# -- step 5: committee exposures, and what they earned ----------------------------


def committee_exposures(
    rows: Sequence[DecisionRow],
    *,
    composition: Composition,
    arm: Arm,
    round_index: int,
    rule: AggregationRule,
) -> dict[PointKey, float]:
    """One committee's aggregate exposure at each point it has a full view of.

    Failed rows are included, at the flat exposure they were stored with. That is
    what the committee would have held: an agent that produced nothing took no
    position, and dropping it instead would let a crashed generation quietly
    reweight the committee towards the agents that survived.
    """
    seats = {(seat.model, seat.persona.name) for seat in composition.seats}
    wanted = NO_COMPOSITION if arm is Arm.INDEPENDENT else composition.identifier
    grouped: dict[PointKey, list[float]] = defaultdict(list)
    for row in rows:
        if (
            row.arm == str(arm)
            and row.round_index == round_index
            and row.composition == wanted
            and row.agent in seats
        ):
            grouped[row.point].append(row.exposure)
    return {point: rule(exposures) for point, exposures in sorted(grouped.items())}


def arm_exposures(
    rows: Sequence[DecisionRow],
    *,
    compositions: Sequence[Composition],
    arm: Arm,
    rule: AggregationRule,
    rebuttal_rounds: int = DEFAULT_REBUTTAL_ROUNDS,
) -> dict[PointKey, float]:
    """One exposure per decision point for one arm, averaged over the committees.

    A debate arm only ran where the agents disagreed, so its series starts as the
    committee's independent view and is overwritten at the contested points by the
    post-debate one. That is not a convenience: on an uncontested day no
    conversation happened, so the committee's decision *is* its independent
    decision, and any other filling -- flat, or forward, or dropping the day --
    would make the treatment and the control differ on days neither was treated.

    The committees are then averaged equally, which is the eight-configuration
    design being read as one experiment rather than as eight.
    """
    pooled: dict[PointKey, list[float]] = defaultdict(list)
    for composition in compositions:
        series = committee_exposures(
            rows,
            composition=composition,
            arm=Arm.INDEPENDENT,
            round_index=OPENING_ROUND,
            rule=rule,
        )
        if arm is not Arm.INDEPENDENT:
            series.update(
                committee_exposures(
                    rows,
                    composition=composition,
                    arm=arm,
                    round_index=rebuttal_rounds,
                    rule=rule,
                )
            )
        for point, exposure in series.items():
            pooled[point].append(exposure)
    return {point: fmean(values) for point, values in sorted(pooled.items())}


@dataclass(frozen=True, slots=True)
class ArmOutcome:
    """One arm's equity curve, reduced to what a table holds."""

    arm: str
    metrics: PerformanceMetrics
    baseline: PerformanceMetrics | None
    """The turnover-matched random committee. Reported beside the arm rather than
    subtracted from it, because the subtraction is the reader's judgement.

    ``None`` when no random path could be matched to this arm's turnover; see
    :func:`_random_committee`. Never a zeroed placeholder, which would read as a
    null that earned nothing."""

    net_return: npt.NDArray[np.float64]
    point_count: int


def score_arm(
    *, arm: Arm, exposures: Mapping[PointKey, float], opens: pd.DataFrame, settings: Settings
) -> ArmOutcome:
    """Backtest one arm and the random committee matched to it."""
    result = backtest_arm(
        exposures=exposures,
        opens=opens,
        cost_bps=settings.total_cost_bps,
        rebalance_threshold=settings.rebalance_threshold,
    )
    metrics = evaluate(result)
    return ArmOutcome(
        arm=str(arm),
        metrics=metrics,
        baseline=_random_committee(
            arm=arm, exposures=exposures, opens=opens, settings=settings, metrics=metrics
        ),
        net_return=result.net_return,
        point_count=len(exposures),
    )


def _random_committee(
    *,
    arm: Arm,
    exposures: Mapping[PointKey, float],
    opens: pd.DataFrame,
    settings: Settings,
    metrics: PerformanceMetrics,
) -> PerformanceMetrics | None:
    """The turnover-matched null, or ``None`` when the arm cannot be matched.

    ``None`` happens, and it is information rather than an error: the null draws
    from the arm's own exposures with replacement, so on a short calendar the draw
    can be smoother than the arm and no subset of revision dates reaches the arm's
    turnover. Refusing to report a mismatched baseline is
    :mod:`council.backtest.baseline`'s decision and the right one -- a null that
    trades less than the arm flatters the arm. Failing the whole evaluation over it
    would be worse than reporting the gap, so the gap is reported.
    """
    try:
        targets = random_arm_targets(
            exposures=exposures,
            opens=opens,
            turnover_per_period=metrics.turnover_per_period,
            rebalance_threshold=settings.rebalance_threshold,
            seed=settings.seed,
        )
    except ValueError as unmatched:
        _LOG.warning("no turnover-matched baseline for %s: %s", arm, unmatched)
        return None
    return evaluate(
        run_backtest(
            targets=targets,
            opens=opens,
            cost_bps=settings.total_cost_bps,
            rebalance_threshold=settings.rebalance_threshold,
        )
    )


# -- step 6: the results ----------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ExperimentResults:
    """Everything one run has to say, in one object.

    Keyed by arm throughout, so the pre-registered comparison -- the debate arm
    against the independent arm -- is a lookup rather than a reconstruction, and so
    the two controls that make it mean anything sit in the same object as the
    number they qualify.
    """

    aggregation_rule: str
    arms: tuple[ArmOutcome, ...]
    buy_and_hold: PerformanceMetrics
    shift_rates: Mapping[str, ShiftRateReport]
    calibration: Mapping[str, CalibrationReport]
    influence: Mapping[str, InfluenceMatrix]
    windows: Mapping[str, WindowComparison]
    contested_share: float
    contested_points: int
    decision_count: int

    def arm(self, arm: Arm | str) -> ArmOutcome:
        for outcome in self.arms:
            if outcome.arm == str(arm):
                return outcome
        raise KeyError(f"no outcome for arm {arm!r}")


def evaluate_experiment(
    *,
    settings: Settings,
    prices: pd.DataFrame,
    decisions: pd.DataFrame,
    compositions: Sequence[Composition] | None = None,
    rule_name: str = PRIMARY_RULE,
    window_count: int = DEFAULT_WINDOW_COUNT,
    rebuttal_rounds: int = DEFAULT_REBUTTAL_ROUNDS,
) -> ExperimentResults:
    """Score every arm and answer every question the run was built to ask.

    Reads stored decisions and prices and calls no model, so a new question costs
    seconds rather than another night.

    Every threshold is taken from ``settings`` and none is left to the process-wide
    ones. One results object reporting a shift rate under the caller's bar and an
    influence matrix under the environment's would be two definitions of "changed
    its mind" in a table read as one.
    """
    if decisions.empty:
        raise ValueError("no decisions have been generated; run the independent arm first")
    committees = tuple(
        balanced_design(models=settings.agent_models) if compositions is None else compositions
    )
    rows = frame_to_rows(decisions)
    opens = opens_frame(prices, tickers=list(settings.tickers))
    returns = forward_returns_lookup(forward_returns(opens))
    rule = RULES[rule_name]

    outcomes = tuple(
        score_arm(
            arm=arm,
            exposures=arm_exposures(
                rows,
                compositions=committees,
                arm=arm,
                rule=rule,
                rebuttal_rounds=rebuttal_rounds,
            ),
            opens=opens,
            settings=settings,
        )
        for arm in (Arm.INDEPENDENT, *TREATMENT_ARMS)
    )
    control = rows_in_arm(decisions, Arm.INDEPENDENT)
    reports = _arm_reports(
        decisions=decisions,
        outcomes=outcomes,
        returns=returns,
        control=control,
        settings=settings,
        window_count=window_count,
        rebuttal_rounds=rebuttal_rounds,
    )
    return ExperimentResults(
        aggregation_rule=rule_name,
        arms=outcomes,
        buy_and_hold=evaluate(buy_and_hold(opens)),
        shift_rates=reports.shift_rates,
        calibration=reports.calibration,
        influence=reports.influence,
        windows=reports.windows,
        contested_share=contested_share(control, threshold=settings.dispersion_threshold),
        contested_points=len(
            contested_points(control, threshold=settings.dispersion_threshold)
        ),
        decision_count=len(decisions),
    )


@dataclass(frozen=True, slots=True)
class _Reports:
    """The four per-arm tables, kept together because they are read together."""

    shift_rates: Mapping[str, ShiftRateReport]
    calibration: Mapping[str, CalibrationReport]
    influence: Mapping[str, InfluenceMatrix]
    windows: Mapping[str, WindowComparison]


def _arm_reports(
    *,
    decisions: pd.DataFrame,
    outcomes: Sequence[ArmOutcome],
    returns: Mapping[PointKey, float],
    control: pd.DataFrame,
    settings: Settings,
    window_count: int,
    rebuttal_rounds: int,
) -> _Reports:
    """Build the per-arm tables in one pass, so they cannot fall out of step.

    Window count is clamped to the periods available. A short run -- a dry run over
    a few weeks of synthetic prices -- has fewer periods than windows, and
    :func:`council.evaluation.windows.split_windows` refuses that rather than
    inventing empty ones; clamping is how the dry run still reports something.
    """
    shift_rates: dict[str, ShiftRateReport] = {}
    calibration: dict[str, CalibrationReport] = {str(Arm.INDEPENDENT): calibrate(control, returns)}
    influence: dict[str, InfluenceMatrix] = {}
    windows: dict[str, WindowComparison] = {}
    control_return = outcomes[0].net_return

    for outcome, arm in zip(outcomes[1:], TREATMENT_ARMS, strict=True):
        frame = rows_in_arm(decisions, arm)
        name = str(arm)
        shift_rates[name] = shift_rate_by_confidence(
            shifts(frame, threshold=settings.shift_threshold)
        )
        calibration[name] = calibrate(frame.loc[frame[ROUND_INDEX] == rebuttal_rounds], returns)
        # The same bar the shift table above was built with. Left to default, the
        # matrix would resolve its own from the process-wide settings, and one
        # results object would carry two definitions of a concession.
        influence[name] = influence_matrix(
            frame, arm=name, min_concession=settings.shift_threshold
        )
        windows[name] = compare_windows(
            outcome.net_return,
            control_return,
            window_count=max(1, min(window_count, int(control_return.size))),
        )
    return _Reports(
        shift_rates=shift_rates,
        calibration=calibration,
        influence=influence,
        windows=windows,
    )


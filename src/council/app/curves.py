"""Stored decisions to equity curves, by the arithmetic ``council evaluate`` runs.

**One implementation of scoring, not two.** Every exposure on this panel comes
from :func:`council.scoring.arm_exposures` and every curve from
:func:`council.arms.backtest_arm`, which are what the CLI scores an arm with.
This module previously aggregated one committee at a time and therefore reported
different headline numbers from ``python -m council evaluate`` on identical
artefacts -- on the dry run the placebo arm changed sign between them, and no
choice of committee reproduced the CLI figure. A dashboard that disagrees with
the command that declares the result is worse than no dashboard.

**Pooled is the declared scope.** The secondary declared comparison is the committee
before debate against the same committee after, read as one experiment over the
balanced design, so the eight committees are averaged equally --
:func:`~council.scoring.arm_exposures` does that averaging. A single committee
can still be scored, by handing this module one composition instead of eight; it
is an exploratory cut and the panel says so.

**Uncontested days fall back to the independent view.** A debate is only run
where the agents disagreed, so a debate arm covers a fraction of the calendar.
That filling is :func:`~council.scoring.arm_exposures`'s decision and is argued
there, which is the point of not making it twice.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from council.arms import backtest_arm, random_arm_targets
from council.backtest.engine import BacktestResult, buy_and_hold, run_backtest
from council.backtest.metrics import PerformanceMetrics, evaluate
from council.debate.compositions import Composition
from council.domain.signal import Arm
from council.evaluation.aggregation import AggregationRule
from council.evaluation.frames import DecisionRow, PointKey, frame_to_rows
from council.evaluation.persuasion import REBUTTAL_ROUND
from council.planning import TREATMENT_ARMS
from council.scoring import arm_exposures

SCORED_ARMS: Final[tuple[Arm, ...]] = (Arm.INDEPENDENT, *TREATMENT_ARMS)
"""The arms in the order :func:`council.scoring.evaluate_experiment` scores them.

Taken from the same tuple the CLI reads rather than from whatever the frame
happens to hold, so the panel and the results table list the arms identically.
"""

BUY_AND_HOLD_LABEL: Final = "buy and hold"
RANDOM_LABEL: Final = "random baseline"

POOLED_LABEL: Final = "all committees (pooled)"
"""How the declared scope is named wherever a reader chooses one."""


@dataclass(frozen=True, slots=True)
class Curve:
    """One labelled path through the backtest."""

    label: str
    result: BacktestResult

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.result.dates

    @property
    def equity(self) -> npt.NDArray[np.float64]:
        return np.asarray(self.result.equity, dtype=np.float64)

    @property
    def metrics(self) -> PerformanceMetrics:
        return evaluate(self.result)


def arms_in(rows: Sequence[DecisionRow]) -> tuple[Arm, ...]:
    """The declared arms this run holds, control first.

    Restricted to the four the experiment declares, because
    :func:`~council.scoring.arm_exposures` is defined per arm and an unrecognised
    label has no defined fallback to the control. A frame carrying one is
    reported by :func:`council.app.artefacts.order_arms` on the tables that can
    show it rather than silently scored here.

    The control qualifies on plain presence; a treatment qualifies only on a
    **post-debate** row. :func:`~council.scoring.arm_exposures` overwrites the
    committee's independent view at the rebuttal round, so a treatment arm holding
    opening rounds alone -- what a sweep interrupted by an outage stores -- draws a
    curve identical to the control's and reads as a clean null.
    :func:`council.scoring.evaluate_experiment` restricts the same way, so the two
    surfaces keep agreeing about which arms a run holds.
    """
    present = {row.arm for row in rows}
    post_debate = {row.arm for row in rows if row.round_index == REBUTTAL_ROUND}
    return tuple(
        arm
        for arm in SCORED_ARMS
        if str(arm) in (present if arm is Arm.INDEPENDENT else post_debate)
    )


def compositions_for(
    identifier: str | None, design: Sequence[Composition]
) -> tuple[Composition, ...]:
    """Every committee in the design, or the single one the reader asked for.

    Raises:
        ValueError: if the identifier is not in the design. A committee scored
            against seats nobody ran would produce a flat curve rather than an
            error, and a flat curve reads as a result.
    """
    if identifier is None:
        return tuple(design)
    chosen = [table for table in design if table.identifier == identifier]
    if not chosen:
        raise ValueError(
            f"{identifier!r} is not a committee of the configured design; it holds "
            + ", ".join(repr(table.identifier) for table in design)
        )
    return tuple(chosen)


def run_curve(
    *,
    label: str,
    exposures: Mapping[PointKey, float],
    opens: pd.DataFrame,
    cost_bps: float,
    rebalance_threshold: float,
) -> Curve:
    """Score one arm's exposure path."""
    return Curve(
        label=label,
        result=backtest_arm(
            exposures=exposures,
            opens=opens,
            cost_bps=cost_bps,
            rebalance_threshold=rebalance_threshold,
        ),
    )


def buy_and_hold_curve(opens: pd.DataFrame) -> Curve:
    """Fully invested, never trading. The floor every arm is reported against."""
    return Curve(label=BUY_AND_HOLD_LABEL, result=buy_and_hold(opens))


def random_baseline_curve(
    *,
    opens: pd.DataFrame,
    exposures: Mapping[PointKey, float],
    reference: Curve,
    seed: int,
    cost_bps: float,
    rebalance_threshold: float,
) -> Curve:
    """A committee with the reference's trading rate and exposure sizes, and no information.

    Built with :func:`council.arms.random_arm_targets`, which is what
    :func:`council.scoring.score_arm` calibrates its own null with -- so the
    baseline drawn here and the one reported beside the CLI's control arm are the
    same path, drawn from the arm's requested exposures under the same seed.

    Raises:
        ValueError: if the reference trades more than revising on every session
            can reach. The caller has to surface that rather than draw a curve
            matched to a different turnover than the one it claims.
    """
    return Curve(
        label=RANDOM_LABEL,
        result=run_backtest(
            targets=random_arm_targets(
                exposures=exposures,
                opens=opens,
                # Per ticker, the way `scoring.score_arm` passes it, off the
                # reference's own per-column results. `metrics.turnover_per_period`
                # is a mean across the basket, and the null realises turnover per
                # column, so a scalar matches each column to the average rather
                # than to its own ticker's rate.
                turnover_per_period={
                    ticker.ticker: ticker.turnover / len(ticker.position)
                    for ticker in reference.result.per_ticker
                },
                rebalance_threshold=rebalance_threshold,
                seed=seed,
            ),
            opens=opens,
            cost_bps=cost_bps,
            rebalance_threshold=rebalance_threshold,
        ),
    )


@dataclass(frozen=True, slots=True)
class CurveSet:
    """The curves the panel draws, and why one of them may be missing."""

    curves: tuple[Curve, ...]
    baseline_note: str | None = None
    """Why there is no random baseline, when there is none.

    A turnover the null cannot reach is a fact about the arm, and the reader has
    to be told: without the baseline, none of the other curves can be read as
    skill rather than as exposure. Reported rather than raised so that one
    unmatchable null does not take every arm's curve off the page with it.
    """


def build_curves(
    *,
    decisions: pd.DataFrame,
    opens: pd.DataFrame,
    compositions: Sequence[Composition],
    rule: AggregationRule,
    cost_bps: float,
    rebalance_threshold: float,
    seed: int,
) -> CurveSet:
    """Every curve the panel draws: each arm, buy-and-hold, and the random baseline.

    The random baseline is matched to the independent arm where there is one, and
    to the first arm present otherwise, because the control is what the reader is
    being asked to judge the treatment against.

    Raises:
        ValueError: if the run holds no arm this experiment declares, or if none
            of its rows were produced by a seat of the committees given. Either
            would otherwise draw a flat curve, which reads as a run that decided
            nothing rather than as a run that was scored against the wrong seats.
    """
    rows = frame_to_rows(decisions)
    arms = arms_in(rows)
    if not arms:
        raise ValueError(
            "this run holds no arm the experiment declares: "
            + ", ".join(sorted({row.arm for row in rows}) or ["nothing"])
        )

    # The drop count is the CLI's to publish -- `ExperimentResults` carries it as
    # `short_committee_points` -- so it is discarded here rather than recomputed
    # under a second definition on the panel.
    exposures = {
        arm: arm_exposures(rows, compositions=compositions, arm=arm, rule=rule)[0] for arm in arms
    }
    if not any(exposures.values()):
        raise ValueError(
            "no stored decision was made by a seat of "
            + ", ".join(table.identifier for table in compositions)
            + "; the run holds the models "
            + ", ".join(sorted({row.model for row in rows}))
        )

    committee = tuple(
        run_curve(
            label=str(arm),
            exposures=exposures[arm],
            opens=opens,
            cost_bps=cost_bps,
            rebalance_threshold=rebalance_threshold,
        )
        for arm in arms
    )
    reference_arm = Arm.INDEPENDENT if Arm.INDEPENDENT in exposures else arms[0]
    reference = next(curve for curve in committee if curve.label == str(reference_arm))
    baseline, note = _baseline(
        opens=opens,
        exposures=exposures[reference_arm],
        reference=reference,
        seed=seed,
        cost_bps=cost_bps,
        rebalance_threshold=rebalance_threshold,
    )
    return CurveSet(curves=(*committee, buy_and_hold_curve(opens), *baseline), baseline_note=note)


def _baseline(
    *,
    opens: pd.DataFrame,
    exposures: Mapping[PointKey, float],
    reference: Curve,
    seed: int,
    cost_bps: float,
    rebalance_threshold: float,
) -> tuple[tuple[Curve, ...], str | None]:
    """The null, or the reason there is not one.

    The baseline is defined as the reference arm's shape with the information
    taken out. Where that shape cannot be reproduced -- an arm that revises on
    almost every session turns over more than any shuffle of its own exposures
    can -- a curve matched to some *other* turnover would still be labelled a
    baseline, and would be read as one. Saying so is the honest output.
    """
    try:
        curve = random_baseline_curve(
            opens=opens,
            exposures=exposures,
            reference=reference,
            seed=seed,
            cost_bps=cost_bps,
            rebalance_threshold=rebalance_threshold,
        )
    except ValueError as error:
        return (), f"No random baseline for the {reference.label} arm: {error}"
    return (curve,), None


def curves_frame(curves: Sequence[Curve]) -> pd.DataFrame:
    """Every curve in one long frame: ``date``, ``series``, ``equity``."""
    parts = [
        pd.DataFrame({"date": curve.dates, "series": curve.label, "equity": curve.equity})
        for curve in curves
    ]
    if not parts:
        return pd.DataFrame(columns=["date", "series", "equity"])
    return pd.concat(parts, ignore_index=True)


def metrics_frame(curves: Sequence[Curve]) -> pd.DataFrame:
    """One row per curve, in the order the curves were built."""
    return pd.DataFrame(
        [
            {
                "series": curve.label,
                "total_return": metrics.total_return,
                "cagr": metrics.cagr,
                "sharpe": metrics.sharpe,
                "sortino": metrics.sortino,
                "max_drawdown": metrics.max_drawdown,
                "turnover_per_period": metrics.turnover_per_period,
                "hit_rate": metrics.hit_rate,
                "time_in_market": metrics.time_in_market,
            }
            for curve, metrics in ((curve, curve.metrics) for curve in curves)
        ]
    )

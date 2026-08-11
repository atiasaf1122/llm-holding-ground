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
from datetime import date
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
from council.domain.signal import Arm
from council.evaluation.aggregation import RULES, AggregationRule
from council.evaluation.calibration import CalibrationReport, calibrate
from council.evaluation.dispersion import contested_points, contested_share
from council.evaluation.frames import (
    ARM,
    COMPOSITION,
    DECISION_DATE,
    NO_COMPOSITION,
    ROUND_INDEX,
    TICKER,
    AgentKey,
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
    failed_rows,
    shift_rate_by_confidence,
    shifts,
    unpaired_rows,
)
from council.evaluation.windows import WindowComparison, compare_windows
from council.planning import TREATMENT_ARMS

_LOG = logging.getLogger(__name__)

PRIMARY_RULE: Final = "mean"
"""The aggregation the secondary declared (equity) comparison is stated in; the
primary statistic is the shift rate and does not depend on the aggregation rule.
The other rules in :data:`council.evaluation.aggregation.RULES` remain available
and remain exploratory."""

DEFAULT_WINDOW_COUNT: Final = 5
"""Windows the arms are compared over. Five is what
:mod:`council.evaluation.windows` is written around: crude enough that a strategy
carried by three days in March shows up as one window rather than as a curve."""


def rows_in_arm(decisions: pd.DataFrame, arm: Arm) -> pd.DataFrame:
    return decisions.loc[decisions[ARM].astype(str) == str(arm)]


def final_round_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Each conversation's last **post-debate** round, on a frame of stored decisions.

    The frame counterpart of :func:`committee_exposures`' per-point maximum, for the
    one consumer that works in frames rather than rows: the post-debate calibration
    table. A conversation is ``(composition, arm, decision_date, ticker)``, matching
    :data:`~council.evaluation.frames.DebateKey`, and the maximum is taken inside it
    -- so two committees that stopped at different rounds are each read at the round
    they stopped on.

    The opening round is excluded before the maximum is taken, so a conversation
    abandoned before any rebuttal contributes nothing rather than contributing its
    opening view. Round 0 is the independent question put to a committee and renders
    byte-identically to the control's prompt: admitting it would put un-debated views
    into a table labelled post-debate, on exactly the conversations that failed.
    Applied to the independent arm this therefore returns nothing, which is why
    :func:`_arm_reports` calibrates the control from its own rows instead.
    """
    if frame.empty:
        return frame
    debated = frame.loc[frame[ROUND_INDEX].astype(int) > OPENING_ROUND]
    if debated.empty:
        return debated
    keys = [COMPOSITION, ARM, DECISION_DATE, TICKER]
    held = debated[ROUND_INDEX].astype(int)
    final = held.groupby([debated[key] for key in keys]).transform("max")
    return debated.loc[held == final]


# -- step 5: committee exposures, and what they earned ----------------------------


def committee_exposures(
    rows: Sequence[DecisionRow],
    *,
    composition: Composition,
    arm: Arm,
    round_index: int | None,
    rule: AggregationRule,
) -> tuple[dict[PointKey, float], frozenset[PointKey]]:
    """One committee's aggregate exposure at each point it has a full view of, and
    which points were dropped for want of a seat.

    ``round_index`` of ``None`` means *the last round this conversation actually
    held*, decided per point rather than once for the arm. That is the only reading
    of "after the debate" a variable-length protocol admits: a conversation ends on
    agreement, on stillness or at the cap, so one committee's final round is round 2
    and another's is round 6. Asking for a fixed index instead -- which is what this
    took before, always the cap -- finds no row on every conversation that stopped
    early, and those points then fall back through :func:`arm_exposures` to the
    committee's *independent* exposure: the treatment is scored as the control on
    exactly the points where the committee converged, silently, with nothing in the
    artefact saying so. An integer still means that literal round, which is what the
    independent arm's :data:`OPENING_ROUND` is.

    The points rather than their count, because :func:`arm_exposures` has a second
    way of losing a point -- a committee absent from the arm entirely -- and the two
    sets overlap. Returning a bare number left it no way to subtract them, so a
    committee short of a seat at a point another committee completed was counted
    twice and warned about as "absent from the arm entirely" while holding rows
    there.

    The set is returned rather than only logged. A drop is not neutral: in a
    treatment arm the point falls back to the committee's *independent* exposure --
    :func:`arm_exposures` starts every treatment series there and overwrites it -- so
    a conversation that happened is scored as the control it was meant to be
    compared against. That is a non-random pull towards the null, because round 1
    exists in the treatments alone, and a ``logging.warning`` reaches neither
    :class:`ExperimentResults`, nor ``results.json``, nor the CLI, nor the dashboard.

    The full view is enforced rather than promised. A point is emitted only where
    every seat of the committee has a row; a point short of one is dropped and
    counted. The previous version grouped whatever rows matched and applied the
    rule to them, so a committee missing a seat was aggregated over the survivors
    and published under its own label -- a three-seat mean drawn as a four-seat
    committee, with no warning and no exception. The independent sweep checkpoints
    per (model, persona, ticker), so an interrupted generate leaves whole model
    slices absent, and every committee of the balanced design seats every model
    once: one missing model makes all eight short at once.
    :func:`council.planning.plan_experiment` already refuses to count a contested
    set from that state and ``CLAIMS`` C19 already forbids quoting rates from it;
    this is the same refusal where the arithmetic happens.

    Failed rows are included, at the flat exposure they were stored with. That is
    what the committee would have held: an agent that produced nothing took no
    position, and dropping it instead would let a crashed generation quietly
    reweight the committee towards the agents that survived. A failure is a row.
    """
    seats = {(seat.model, seat.persona.name) for seat in composition.seats}
    wanted = NO_COMPOSITION if arm is Arm.INDEPENDENT else composition.identifier
    mine = [
        row
        for row in rows
        if row.arm == str(arm) and row.composition == wanted and row.agent in seats
    ]
    # One index per point when the caller asked for the last round, and the caller's
    # index everywhere when it asked for a literal one. Taken over this committee's
    # own rows, because two committees debating the same point can stop at different
    # rounds and the maximum over both would read one of them at a round it never
    # reached.
    final = _final_round_by_point(mine) if round_index is None else {}
    grouped: dict[PointKey, dict[AgentKey, float]] = defaultdict(dict)
    for row in mine:
        wanted_round = final.get(row.point) if round_index is None else round_index
        if row.round_index == wanted_round:
            grouped[row.point][row.agent] = row.exposure
    complete = {
        point: rule([held[seat] for seat in sorted(seats)])
        for point, held in sorted(grouped.items())
        if held.keys() == seats
    }
    dropped = frozenset(grouped) - frozenset(complete)
    if dropped:
        _LOG.warning(
            "%s: %d point(s) dropped in the %s arm, short of the committee's %d seat(s)",
            composition.identifier,
            len(dropped),
            arm,
            len(seats),
        )
    return complete, dropped


def _final_round_by_point(rows: Sequence[DecisionRow]) -> dict[PointKey, int]:
    """The highest **post-opening** round index each point holds, over the rows given.

    "The last round this conversation held", provided the caller has already
    narrowed the rows to one committee in one arm -- which is what makes a point a
    conversation. :func:`committee_exposures` is the only caller and does exactly
    that.

    A point holding the opening round alone gets no entry, so it falls back to the
    committee's independent view and is counted by :func:`arm_exposures` as a point
    this committee is absent from. That is the honest reading of a conversation
    abandoned before any rebuttal: it produced no post-debate view. Taking round 0 as
    its "last round" would score the point at a view identical to the control's while
    reporting it as debated -- the treatment quietly pulled towards the null, with
    the counter that exists to say so reading zero.
    """
    final: dict[PointKey, int] = {}
    for row in rows:
        if row.round_index <= OPENING_ROUND:
            continue
        held = final.get(row.point)
        if held is None or row.round_index > held:
            final[row.point] = row.round_index
    return final


def arm_exposures(
    rows: Sequence[DecisionRow],
    *,
    compositions: Sequence[Composition],
    arm: Arm,
    rule: AggregationRule,
) -> tuple[dict[PointKey, float], int]:
    """One exposure per decision point for one arm, averaged over the committees,
    and the (committee, point) pairs this arm lost to a short committee.

    A debate arm only ran where the agents disagreed, so its series starts as the
    committee's independent view and is overwritten at the contested points by the
    post-debate one. That is not a convenience: on an uncontested day no
    conversation happened, so the committee's decision *is* its independent
    decision, and any other filling -- flat, or forward, or dropping the day --
    would make the treatment and the control differ on days neither was treated.

    The committees are then averaged equally, which is the eight-configuration
    design being read as one experiment rather than as eight.

    The second return value counts the **treatment** call's drops only. The
    fallback above is argued for uncontested days, where no conversation happened;
    a point this arm debated and then lost for want of one seat's post-debate row
    is scored as the control instead, and that is the number a reader has to be
    able to see. The independent call's drops are not this arm's: they are the
    control's own coverage, and every arm inherits them identically.

    Two stages rather than one, because :func:`committee_exposures` can only count
    what it was handed. Its ``dropped`` is ``len(grouped) - len(complete)``, so a
    committee that produced *no* row at all for this arm at a point has an empty
    ``grouped`` and is counted as having lost nothing -- while the point still falls
    back to that committee's independent view. That is what an interrupted ``debate``
    leaves, since the sweep checkpoints per (committee, arm, ticker): a whole
    committee absent from a treatment arm pulls the treatment towards the control
    with the counter reading zero. So the debated maps are collected first, and a
    point that any committee debated but this one did not is counted here.
    """
    pooled: dict[PointKey, list[float]] = defaultdict(list)
    short = 0
    debated_by: dict[str, dict[PointKey, float]] = {}
    dropped_by: dict[str, frozenset[PointKey]] = {}
    if arm is not Arm.INDEPENDENT:
        for composition in compositions:
            debated, dropped = committee_exposures(
                rows,
                composition=composition,
                arm=arm,
                # The conversation's own last round, not the cap. See
                # `committee_exposures`: a fixed index scores every conversation
                # that converged early as the control it is being compared against.
                round_index=None,
                rule=rule,
            )
            debated_by[composition.identifier] = debated
            dropped_by[composition.identifier] = dropped
        debated_any: set[PointKey] = set().union(*(held.keys() for held in debated_by.values()))
        for composition in compositions:
            # The two stages overlap. A point this committee has rows for but is
            # short a seat at is in `lost`, and it is also missing from this
            # committee's *complete* map -- so subtracting `lost` is what stops it
            # being counted a second time and reported as a point the committee
            # never touched.
            lost = dropped_by[composition.identifier]
            absent = debated_any - debated_by[composition.identifier].keys() - lost
            if absent:
                _LOG.warning(
                    "%s: %d point(s) absent from the %s arm entirely, so they are "
                    "scored as the control",
                    composition.identifier,
                    len(absent),
                    arm,
                )
            short += len(lost | absent)
    for composition in compositions:
        series, _ = committee_exposures(
            rows,
            composition=composition,
            arm=Arm.INDEPENDENT,
            round_index=OPENING_ROUND,
            rule=rule,
        )
        series.update(debated_by.get(composition.identifier, {}))
        for point, exposure in series.items():
            pooled[point].append(exposure)
    return {point: fmean(values) for point, values in sorted(pooled.items())}, short


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
            arm=arm,
            exposures=exposures,
            opens=opens,
            settings=settings,
            # Per ticker, off the backtest that has just run. `metrics.turnover` is
            # a mean across the basket, and the null realises turnover per column,
            # so one scalar matches each column to the basket average rather than
            # to its own ticker's rate -- and the reachability check then runs per
            # ticker against the wrong number, which can drop the whole baseline.
            turnover_per_period={
                ticker.ticker: ticker.turnover / len(ticker.position)
                for ticker in result.per_ticker
            },
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
    turnover_per_period: Mapping[str, float],
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
            turnover_per_period=turnover_per_period,
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

    Keyed by arm throughout, so the secondary declared comparison -- the debate arm
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

    calendar_start: date
    calendar_end: date
    session_count: int
    """Which calendar this artefact scored, on the artefact.

    Provenance rather than a measurement: the dates are already in hand inside
    :func:`evaluate_experiment` as the opens frame's index. Every published run so
    far used a six-month window that was never chosen against a two-year configured
    range, and the object meant to be the record of the run could not have shown it
    -- it carried ``decision_count``, ``contested_points`` and per-arm ``periods``,
    and no field naming either end of the period they were counted over.

    This is the **price** file's range, and it is only half of the answer. See
    :attr:`decision_start`."""

    decision_start: date | None
    decision_end: date | None
    decision_date_count: int
    """Which dates decisions were actually made on.

    The calendar above is the opens frame's index, so it catches a short run only
    when the *prices* are short. A run whose prices span the configured range but
    whose decisions cover a fraction of it -- the state an interrupted ``generate``
    leaves, which ``evaluate`` does not refuse -- publishes a calendar most of which
    holds no decision, and every backtest period before the first decision is flat
    in every arm. Naming both spans is what makes that visible on the artefact
    rather than only in the parquet.

    ``None`` at either end only where the frame holds no row, which
    :func:`evaluate_experiment` refuses before reaching here; the type says so
    rather than inventing a date for a run that decided nothing."""

    debated_points: Mapping[str, int]
    """Distinct decision points each treatment arm actually held a conversation on.

    :attr:`ArmOutcome.point_count` cannot say this: it counts the exposure series
    *after* :func:`arm_exposures` has filled the uncontested days from the
    committee's independent view, so it is identical for every arm by construction.
    The placebo drops every contested point with no usable earlier donor -- a
    non-random filter that removes the earliest points of the calendar from one arm
    only -- and this is the field a reader checks that caveat against.
    :func:`council.app.tables.coverage_note` reports the same gap on the dashboard.

    Two things open that gap, and they are not the same kind of thing.
    :attr:`contested_share` and :attr:`contested_points` above are measured over the
    whole control arm and take no notice of either, so a run where they exceed this
    is the ordinary case rather than a contradiction:

    * the donor filter, above, which is a property of the design; and
    * ``Settings.max_debate_points``, which is a budget. Where it is set, the debate
      was run on an evenly spaced sample of the contested points rather than all of
      them (:mod:`council.sampling`), and the difference between the two numbers is
      mostly that. It costs the market-side comparison its power and leaves the
      behavioural measurements alone, since each is computed per debated point.

    Counted off the arm's stored rows, the way ``app.tables._coverage_row`` counts
    it, rather than off the pairs the shift rate is computed from: those drop every
    pair containing a failed generation, so a conversation that was held and then
    crashed would read here as a point the arm never covered.
    """

    dropped_pairs: Mapping[str, int]
    """Round pairs each arm lost to a failed generation, per
    :func:`council.evaluation.persuasion.failed_rows`.

    On the surface most likely to be quoted, because the drop is not random across
    the arms: round 1 exists only in the debate arms, so a crashed round 1 reads as
    an agent walking away from its opening view and every phantom shift lands on
    the treatment. ``failed_rows`` exists so that a rate computed over what is left
    of a run whose generations mostly crashed cannot be read as a rate over the
    run, and it was wired only to the dashboard.
    """

    short_committee_points: Mapping[str, int]
    """(committee, point) pairs each treatment arm lost to a committee short of a
    seat -- or absent from the arm entirely -- per :func:`arm_exposures`.

    Beside :attr:`dropped_pairs` because it is the same kind of hazard and worse
    behaved. A dropped pair leaves the rate's denominator; a short committee leaves
    the *exposure series*, and what fills the hole is that committee's independent
    view -- so the point is scored as the control the arm is being compared against.
    The drop is non-random across the arms for the usual reason: round 1 exists in
    the treatments alone. It was announced by a ``logging.warning`` and by nothing
    else, so a run could publish a treatment pulled towards the null with no column
    saying how far.
    """

    unpaired: Mapping[str, int]
    """Rows each arm produced with no partner round, per
    :func:`council.evaluation.persuasion.unpaired_rows`. The other half of the
    denominator's provenance: a conversation abandoned after its opening round
    leaves a row that no shift was computed from."""

    def arm(self, arm: Arm | str) -> ArmOutcome:
        for outcome in self.arms:
            if outcome.arm == str(arm):
                return outcome
        raise KeyError(f"no outcome for arm {arm!r}")


def _scoring_window(opens: pd.DataFrame, *, rows: Sequence[DecisionRow]) -> pd.DataFrame:
    """The sessions from the first decision onward, and no earlier ones.

    The same argument the ``covered`` filter above makes about tickers, one axis
    over. A price file has to start before the study does, because the first
    decision needs ``lookback_days`` of history behind it -- on the shipped
    configuration that is 78 sessions of warm-up. Every arm is structurally flat
    across them: no decision exists, so ``arms.targets_frame`` emits no row and the
    engine holds nothing. ``buy_and_hold`` is not flat there. It is built from
    whatever index it is handed and is fully invested from the second open.

    Scored over the price file's whole range, the benchmark therefore banks a return
    the committee was never given the chance to earn. On this study's data that is
    not a rounding difference: the warm-up runs 2021-09-13 to 2021-12-31, over which
    AAPL rose 18.9% and XOM 12.0%, and the benchmark reads **+67.2%** against
    **+39.6%** over the window the arms could actually trade. The committee loses to
    it either way -- that conclusion is unaffected -- but the published gap was
    overstated by 27.6 percentage points, and an arm's Sharpe was computed over 78
    sessions of forced zeros that belong to no arm's behaviour.

    Trimming from the first decision date rather than from the first *filled* one
    keeps the lookahead rule intact: a target on the opening session is filled at the
    next open, which is the same one-session shift every arm and the benchmark obey.
    """
    if not rows:
        return opens
    first = min(row.decision_date for row in rows)
    return opens.loc[opens.index >= pd.Timestamp(first)]


def evaluate_experiment(
    *,
    settings: Settings,
    prices: pd.DataFrame,
    decisions: pd.DataFrame,
    compositions: Sequence[Composition] | None = None,
    rule_name: str = PRIMARY_RULE,
    window_count: int = DEFAULT_WINDOW_COUNT,
) -> ExperimentResults:
    """Score every arm and answer every question the run was built to ask.

    Reads stored decisions and prices and calls no model, so a new question costs
    seconds rather than another night.

    Every threshold is taken from ``settings`` and none is left to the process-wide
    ones. One results object reporting a shift rate under the caller's bar and an
    influence matrix under the environment's would be two definitions of "changed
    its mind" in a table read as one.

    Only the treatment arms the frame holds a **post-debate** row for are scored.
    A post-debate row is any row past the opening one, which is what it has to be
    once a conversation's length is an outcome: it used to mean a row at
    ``round_index == rebuttal_rounds``, and at a cap of six that would refuse to
    score an arm whose every conversation agreed at round two -- an arm that ran
    perfectly well. Round 1 is the round every held conversation has, so this and
    :func:`council.app.curves.arms_in`, which qualifies on exactly that, still agree
    about which treatment arms a run holds.

    Scoring an arm without one is not a null: :func:`arm_exposures` fills every
    uncontested point from the committee's independent view and overwrites it at the
    conversation's last round, so an arm holding opening rounds alone comes out an
    exact copy of the control -- published as a clean null for the
    secondary declared comparison. Presence alone does not settle it, and testing
    presence left the more reachable route open: :meth:`council.debate.sweep._Sweep.hold`
    stores the rows of an abandoned conversation and the protocol stops after the
    opening round when a whole round fails to generate, so an arm run during an
    outage stores round-0 rows only. :func:`council.app.curves.arms_in` restricts
    the same way, so the dashboard and the CLI agree about which *treatment* arms a
    run holds. They do not agree about the control: ``arms_in`` simply omits an
    absent independent arm, and this function refuses the frame outright, because a
    control is what the secondary declared comparison is stated against.

    Raises:
        ValueError: if the frame holds no independent-arm row. The control was
            prepended unconditionally while only the treatments were restricted to
            what the frame holds, so a frame of debate rows alone published a
            control that never ran: :func:`arm_exposures` returned nothing for it,
            :func:`score_arm` backtested an empty target frame, and the
            secondary declared comparison was scored against a flat line. ``if not
            any(exposures.values())`` does not fire, because the treatment
            exposures are not empty.
        ValueError: if the frame is empty, or if no stored decision was made by a
            seat of the configured committees. The second is the ordinary
            consequence of evaluating with a ``--models`` list that does not match
            what was generated: every arm backtests flat, the shift and calibration
            tables populate from the frame regardless, and the artefact would mix a
            vacuous equity comparison with real behavioural numbers with nothing
            marking the difference. :func:`council.app.curves.build_curves` already
            refuses exactly this case.
    """
    if decisions.empty:
        raise ValueError("no decisions have been generated; run the independent arm first")
    committees = tuple(
        balanced_design(models=settings.agent_models) if compositions is None else compositions
    )
    rows = frame_to_rows(decisions)
    # The dates decisions were made on, which is not the price file's range. An
    # interrupted `generate` leaves the two spans wide apart and `evaluate` does not
    # refuse it, so both go on the artefact.
    decision_dates = sorted({row.decision_date for row in rows})
    # The tickers the decisions cover, not the configured universe, matching
    # `council.app.artefacts.load_results`. The equal-weight basket and
    # `buy_and_hold` are built from whatever columns they are handed, so a
    # configured ticker the run never decided on dilutes every arm's return,
    # drawdown, turnover and time-in-market by the ratio of the two counts -- and
    # measures the floor an arm has to beat over a universe the committee was never
    # asked about. That is the ordinary state after an interrupted `generate`, which
    # sweeps model then persona then ticker.
    covered = sorted({row.ticker for row in rows})
    absent = [ticker for ticker in settings.tickers if ticker not in covered]
    if absent:
        _LOG.warning(
            "no decision covers %s; scoring the basket over %s instead of the configured universe",
            ", ".join(absent),
            ", ".join(covered),
        )
    opens = _scoring_window(opens_frame(prices, tickers=covered), rows=rows)
    returns = forward_returns_lookup(forward_returns(opens))
    rule = RULES[rule_name]

    present = {row.arm for row in rows}
    if str(Arm.INDEPENDENT) not in present:
        raise ValueError(
            "the frame holds no independent-arm row; the secondary declared comparison "
            "is against the control, and an absent control backtests flat and "
            "publishes as a clean null. The run holds " + ", ".join(sorted(present))
        )
    post_debate = {row.arm for row in rows if row.round_index > OPENING_ROUND}
    treatments = tuple(arm for arm in TREATMENT_ARMS if str(arm) in post_debate)
    scored = {
        arm: arm_exposures(rows, compositions=committees, arm=arm, rule=rule)
        for arm in (Arm.INDEPENDENT, *treatments)
    }
    exposures = {arm: series for arm, (series, _) in scored.items()}
    short_committees = {str(arm): dropped for arm, (_, dropped) in scored.items()}
    if not any(exposures.values()):
        raise ValueError(
            "no stored decision was made by a seat of "
            + ", ".join(table.identifier for table in committees)
            + "; the run holds the models "
            + ", ".join(sorted({row.model for row in rows}))
        )

    outcomes = tuple(
        score_arm(arm=arm, exposures=exposures[arm], opens=opens, settings=settings)
        for arm in (Arm.INDEPENDENT, *treatments)
    )
    control = rows_in_arm(decisions, Arm.INDEPENDENT)
    reports = _arm_reports(
        decisions=decisions,
        outcomes=outcomes,
        treatments=treatments,
        returns=returns,
        control=control,
        settings=settings,
        window_count=window_count,
        short_committees=short_committees,
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
        contested_points=len(contested_points(control, threshold=settings.dispersion_threshold)),
        decision_count=len(decisions),
        calendar_start=opens.index[0].date(),
        calendar_end=opens.index[-1].date(),
        session_count=len(opens.index),
        decision_start=decision_dates[0] if decision_dates else None,
        decision_end=decision_dates[-1] if decision_dates else None,
        decision_date_count=len(decision_dates),
        debated_points=reports.debated_points,
        dropped_pairs=reports.dropped_pairs,
        short_committee_points=reports.short_committee_points,
        unpaired=reports.unpaired,
    )


@dataclass(frozen=True, slots=True)
class _Reports:
    """The per-arm tables, kept together because they are read together."""

    shift_rates: Mapping[str, ShiftRateReport]
    calibration: Mapping[str, CalibrationReport]
    influence: Mapping[str, InfluenceMatrix]
    windows: Mapping[str, WindowComparison]
    debated_points: Mapping[str, int]
    dropped_pairs: Mapping[str, int]
    short_committee_points: Mapping[str, int]
    unpaired: Mapping[str, int]


def _arm_reports(
    *,
    decisions: pd.DataFrame,
    outcomes: Sequence[ArmOutcome],
    treatments: Sequence[Arm],
    returns: Mapping[PointKey, float],
    control: pd.DataFrame,
    settings: Settings,
    window_count: int,
    short_committees: Mapping[str, int],
) -> _Reports:
    """Build the per-arm tables in one pass, so they cannot fall out of step.

    Window count is clamped to the periods available. A short run -- a dry run over
    a few weeks of synthetic prices -- has fewer periods than windows, and
    :func:`council.evaluation.windows.split_windows` refuses that rather than
    inventing empty ones; clamping is how the dry run still reports something. The
    clamp is logged rather than applied in silence: the report prints "x of N
    windows" using the clamped N, and a reader who asked for more is entitled to
    know the figure is not the one they asked for. A value below one is refused at
    the command line instead -- see :func:`council.cli.window_count`.
    """
    shift_rates: dict[str, ShiftRateReport] = {}
    calibration: dict[str, CalibrationReport] = {str(Arm.INDEPENDENT): calibrate(control, returns)}
    influence: dict[str, InfluenceMatrix] = {}
    windows: dict[str, WindowComparison] = {}
    debated_points: dict[str, int] = {}
    dropped_pairs: dict[str, int] = {}
    short_committee_points: dict[str, int] = {}
    unpaired: dict[str, int] = {}
    control_return = outcomes[0].net_return

    used_windows = max(1, min(window_count, int(control_return.size)))
    if used_windows != window_count:
        _LOG.warning(
            "%d window(s) requested but the run holds %d period(s); comparing over %d",
            window_count,
            int(control_return.size),
            used_windows,
        )

    for outcome, arm in zip(outcomes[1:], treatments, strict=True):
        frame = rows_in_arm(decisions, arm)
        name = str(arm)
        records = shifts(frame, threshold=settings.shift_threshold)
        shift_rates[name] = shift_rate_by_confidence(records)
        # The points this arm actually debated, taken off its stored rows rather
        # than off the exposure series, which has been filled to the full calendar
        # and is the same length for every arm. Off the rows rather than off
        # `records`: `shifts` drops every pair containing a failed generation, so a
        # point whose conversation was held but whose rounds crashed would vanish
        # and read as a coverage gap. `app.tables._coverage_row` counts the same
        # thing the same way, so the CLI, the artefact and the dashboard agree.
        debated_points[name] = len({row.point for row in frame_to_rows(frame)})
        # What the rate's denominator lost, threaded the same way. `failed_rows`
        # returns both rounds of every dropped pair, so the pair count is half of
        # it; `unpaired_rows` returns rows that never had a partner at all.
        dropped_pairs[name] = len(failed_rows(frame)) // 2
        unpaired[name] = len(unpaired_rows(frame))
        # What the *exposure series* lost, from `arm_exposures`' own count rather
        # than recomputed here: a second definition of "short of a seat" would put
        # two numbers under one name in one artefact.
        short_committee_points[name] = short_committees.get(name, 0)
        # Each conversation's own last round, for `arm_exposures`' reason and with
        # the same consequence when it is got wrong. `ROUND_INDEX == rebuttal_rounds`
        # selected nothing on a conversation that agreed early, so the arm's
        # post-debate calibration was computed over the converged conversations'
        # absence -- and at a cap no conversation reaches, over nothing at all.
        calibration[name] = calibrate(final_round_rows(frame), returns)
        # The same bar the shift table above was built with. Left to default, the
        # matrix would resolve its own from the process-wide settings, and one
        # results object would carry two definitions of a concession.
        influence[name] = influence_matrix(frame, arm=name, min_concession=settings.shift_threshold)
        windows[name] = compare_windows(
            outcome.net_return, control_return, window_count=used_windows
        )
    return _Reports(
        shift_rates=shift_rates,
        calibration=calibration,
        influence=influence,
        windows=windows,
        debated_points=debated_points,
        dropped_pairs=dropped_pairs,
        short_committee_points=short_committee_points,
        unpaired=unpaired,
    )

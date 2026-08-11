"""The tidy frames each panel plots.

One adapter per question, each a pure function of a report the evaluation package
already computed. Nothing here decides anything: the thresholds, the bands and
the definition of a concession all arrive from :mod:`council.evaluation`, and this
module only reshapes them into columns a chart can read.

Rates are left as ``None`` where a band holds nothing. A zero would draw a point
on the calibration plot that no observation supports, and on the shift panel it
would read as a band of agents who never moved.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from council.app.artefacts import order_arms
from council.domain.signal import Arm
from council.evaluation.buckets import DEFAULT_EDGES, Band
from council.evaluation.calibration import CalibrationReport
from council.evaluation.frames import (
    ARM,
    COMPOSITION,
    NO_COMPOSITION,
    ROUND_INDEX,
    PointKey,
    forward_returns,
    forward_returns_lookup,
    frame_to_rows,
)
from council.evaluation.influence import InfluenceMatrix
from council.evaluation.persuasion import (
    Shift,
    ShiftRateReport,
    failed_rows,
    shift_rate_by_confidence,
    shifts,
    unpaired_rows,
)


def select(
    frame: pd.DataFrame,
    *,
    arm: str | None = None,
    round_index: int | None = None,
    composition: str | None = None,
) -> pd.DataFrame:
    """The subset of a decisions frame one panel is asking about.

    Every filter is optional and ``None`` means "do not filter", so a panel that
    pools rounds says so by omitting the argument rather than by passing a
    sentinel that has to be remembered.
    """
    selected = frame
    if arm is not None:
        selected = selected.loc[selected[ARM].astype(str) == arm]
    if round_index is not None:
        selected = selected.loc[selected[ROUND_INDEX].astype(int) == round_index]
    if composition is not None:
        held = selected[COMPOSITION].fillna(NO_COMPOSITION).astype(str)
        selected = selected.loc[held == composition]
    return selected


def forward_return_lookup(opens: pd.DataFrame) -> Mapping[PointKey, float]:
    """What each decision date went on to earn, keyed by decision point.

    Built with :func:`~council.evaluation.frames.forward_returns` rather than from
    a percentage change, because the obvious one-liner stores against day *t* a
    move that had already happened by the close of *t* -- and every agent on the
    calibration panel then looks prescient, with nothing raising.
    """
    return forward_returns_lookup(forward_returns(opens))


# -- the primary statistic ---------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArmShifts:
    """One arm's paired shifts beside the banded report computed from them.

    Kept together because the panel has to show both. A band's ``count`` is
    observations, and every contested decision point is answered once per seat of
    every committee -- so on the balanced design of eight committees of four seats
    one point contributes thirty-two observations. Reading ``count`` as the sample
    size of the pre-registered statistic, which README declares over *decision
    points*, overstates the evidence by that factor, and the only defence is
    showing the two numbers side by side.
    """

    arm: str
    records: tuple[Shift, ...]
    report: ShiftRateReport

    @property
    def points(self) -> int:
        """Distinct ``(decision_date, ticker)`` behind this arm's observations."""
        return len({(record.decision_date, record.ticker) for record in self.records})


def shift_reports(
    frame: pd.DataFrame,
    *,
    threshold: float | None = None,
    edges: Sequence[float] = DEFAULT_EDGES,
) -> dict[str, ArmShifts]:
    """Shift rate by prior confidence, one report per arm, arms in the declared order.

    Every arm in the frame gets a report, including the independent arm -- which
    has no second round and therefore no shifts at all. Its empty bands are the
    honest rendering of a control that cannot shift by construction; omitting it
    would leave a reader to wonder whether it was measured.
    """
    by_arm: dict[str, list[Shift]] = {
        arm: [] for arm in order_arms(row.arm for row in frame_to_rows(frame))
    }
    for shift in shifts(frame, threshold=threshold):
        by_arm[shift.arm].append(shift)
    return {
        arm: ArmShifts(
            arm=arm,
            records=tuple(records),
            report=shift_rate_by_confidence(records, edges=edges),
        )
        for arm, records in by_arm.items()
    }


def shift_rate_table(reports: Mapping[str, ArmShifts]) -> pd.DataFrame:
    """One row per arm per confidence band, in the order the reports were built.

    ``count`` is observations and ``points`` is the distinct decision points
    behind them; see :class:`ArmShifts` for why both are shown. The rate is
    ``count``-based, as the statistic is defined, but the two columns together
    say how far from independent those observations are.

    Both columns come off :class:`~council.evaluation.persuasion.ConfidenceShiftRate`
    rather than being recounted here, so this panel, ``results.json`` and the CLI
    table cannot disagree about the denominator behind one rate.
    """
    return pd.DataFrame(
        [
            {
                "arm": arm,
                "band": band.band.label,
                "confidence": _midpoint(band.band),
                "count": band.count,
                "points": band.point_count,
                "shifted_count": band.shifted_count,
                "reversed_count": band.reversed_count,
                "shift_rate": band.shift_rate,
                "reversal_rate": band.reversal_rate,
                "mean_distance": band.mean_distance,
            }
            for arm, entry in reports.items()
            for band in entry.report.bands
        ]
    )


# -- what each arm actually covers -------------------------------------------


def coverage_table(frame: pd.DataFrame) -> pd.DataFrame:
    """What each arm covers, so that two shift rates can be differenced knowingly.

    Under the shipped design the three treatment arms cover one identical point set
    -- a point with no placebo donor is withheld from **all three**
    (:func:`council.debate.sweep.servable_points`), precisely so no arm difference
    can be a coverage difference, and on the published run all three hold the same
    50 points. This table is the *check* on that invariant rather than a report of
    an expected gap: an interrupted sweep resumed against changed settings, or a
    future design change, is how the arms drift apart, and a coverage effect is
    invisible in a rate -- exactly the shape of an artefact that would be published
    as a finding.

    Columns:
        ``points`` distinct decision points the arm holds a row for;
        ``conversations`` distinct (committee, point) pairs; ``paired`` the
        observations that reached the shift rate; ``unpaired`` rows dropped for
        want of a partner round; ``failed_rows`` rows of complete pairs dropped
        because a generation produced nothing.
    """
    return pd.DataFrame(
        [
            _coverage_row(arm, select(frame, arm=arm))
            for arm in order_arms(row.arm for row in frame_to_rows(frame))
        ]
    )


def _coverage_row(arm: str, rows: pd.DataFrame) -> dict[str, object]:
    decisions = frame_to_rows(rows)
    return {
        "arm": arm,
        "points": len({row.point for row in decisions}),
        "conversations": len({(row.composition, row.point) for row in decisions}),
        "paired": len(shifts(rows)),
        "unpaired": len(unpaired_rows(rows)),
        "failed_rows": len(failed_rows(rows)),
    }


def coverage_note(coverage: pd.DataFrame) -> str | None:
    """A sentence when the debate arms did not answer the same decision points.

    ``None`` when they did, so the panel says nothing rather than reassuring the
    reader in a case it has not checked.
    """
    debated = coverage.loc[coverage["arm"] != str(Arm.INDEPENDENT)]
    if len(debated) < 2 or int(debated["points"].nunique()) == 1:
        return None
    return (
        "The debate arms do not cover the same decision points -- "
        + ", ".join(f"{row.arm} {row.points}" for row in debated.itertuples(index=False))
        + ". The design withholds a point any arm cannot serve from all three "
        "(council.debate.sweep.servable_points), so unequal coverage means this run "
        "violated that invariant -- an interrupted sweep resumed under different "
        "settings is the usual cause -- and part of any gap between the arms' rates "
        "is coverage rather than behaviour. Do not difference these rates."
    )


# -- calibration -------------------------------------------------------------


def calibration_table(report: CalibrationReport) -> pd.DataFrame:
    """Stated confidence against realised hit rate, one row per band.

    ``confidence`` is the band's midpoint, which is what the reference diagonal is
    drawn against. A perfectly calibrated agent puts every point on ``hit_rate ==
    confidence``; that the two axes are the same quantity is the whole reading of
    the panel.
    """
    return pd.DataFrame(
        [
            {
                "band": bucket.band.label,
                "confidence": _midpoint(bucket.band),
                "count": bucket.count,
                "hit_count": bucket.hit_count,
                "hit_rate": bucket.hit_rate,
            }
            for bucket in report.buckets
        ]
    )


# -- influence ---------------------------------------------------------------


def influence_table(matrix: InfluenceMatrix) -> pd.DataFrame:
    """The matrix in long form: one row per ordered pair of models.

    Carries the arm on every row. The placebo exists to be differenced against the
    real debate, so a heatmap lifted out of the page without its condition is a
    table that cannot be read at all.
    """
    return pd.DataFrame(
        [
            {
                "arm": matrix.arm,
                "conceder": conceder,
                "influencer": influencer,
                "conceded": int(matrix.conceded[row, column]),
                "opportunities": int(matrix.opportunities[row, column]),
                "amount": float(matrix.amount[row, column]),
                "rate": matrix.rate(conceder, influencer),
            }
            for row, conceder in enumerate(matrix.models)
            for column, influencer in enumerate(matrix.models)
        ]
    )


def net_influence_table(matrix: InfluenceMatrix) -> pd.DataFrame:
    """Concessions won minus concessions made, loudest first."""
    return pd.DataFrame(
        [{"model": model, "net_influence": net} for model, net in matrix.net_influence]
    )


def _midpoint(band: Band) -> float:
    return (band.lower + band.upper) / 2.0

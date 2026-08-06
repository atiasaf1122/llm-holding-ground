"""Turning a plan or a results object into something a person reads.

Two audiences, one module. ``plan`` is read before a night of GPU time is
committed, so its table has to make the cost obvious at a glance and has to say
which of its numbers were counted and which were assumed. ``evaluate`` is read
afterwards, and its job is to put the pre-registered comparison and the two
controls that qualify it on the same screen, so that a headline number cannot be
lifted out without them.

Presentation only. Nothing here computes a result; every number arrives already
decided by :mod:`council.planning` or :mod:`council.scoring`, which is what keeps
those two testable against arithmetic rather than against a string.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from typing import Any

from council.backtest.metrics import PerformanceMetrics
from council.domain.signal import Arm
from council.evaluation.calibration import CalibrationReport
from council.evaluation.influence import InfluenceMatrix
from council.evaluation.persuasion import ShiftRateReport
from council.evaluation.windows import WindowComparison
from council.planning import ExperimentPlan
from council.scoring import PRIMARY_RULE, ArmOutcome, ExperimentResults

RIGHT = ">"
LEFT = "<"


def render_table(
    headers: Sequence[str],
    rows: Sequence[Sequence[str]],
    *,
    aligns: Sequence[str],
    footer: Sequence[str] | None = None,
) -> str:
    """A fixed-width table with a rule under the head, and one over a footer row."""
    body = [list(row) for row in rows] + ([list(footer)] if footer is not None else [])
    widths = [
        max(len(str(headers[column])), *(len(row[column]) for row in body))
        if body
        else len(str(headers[column]))
        for column in range(len(headers))
    ]
    rule = "  ".join("-" * width for width in widths)

    def line(cells: Sequence[str]) -> str:
        return "  ".join(
            f"{cell:{align}{width}}"
            for cell, align, width in zip(cells, aligns, widths, strict=True)
        ).rstrip()

    lines = [line(headers), rule, *(line(row) for row in rows)]
    if footer is not None:
        lines += [rule, line(footer)]
    return "\n".join(lines)


def format_duration(seconds: float) -> str:
    """Wall clock, rounded to something a person plans an evening around."""
    total = round(seconds)
    hours, rest = divmod(total, 3600)
    minutes, remainder = divmod(rest, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {remainder:02d}s"
    return f"{remainder}s"


def format_count(value: int) -> str:
    """A count with thousands separators. Public, because
    :mod:`council.probe.render` renders its tables with the same primitives -- two
    renderers that formatted an empty rate differently would read as two different
    kinds of number."""
    return f"{value:,}"


def format_ratio(value: float | None, *, digits: int = 3) -> str:
    """A number, or a dash where there is nothing to report.

    A dash rather than a zero, following
    :attr:`council.evaluation.calibration.ConfidenceBucket.hit_rate`: an empty band
    has no rate, and printing 0.0 would draw a line no observation supports.
    """
    if value is None:
        return "-"
    if math.isinf(value):
        return "+inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def render_plan(plan: ExperimentPlan) -> str:
    """The cost of a configuration, per stage, with the total under a rule."""
    rows = [
        [
            f"{stage.stage}{'*' if stage.estimated else ''}",
            stage.arm,
            format_count(stage.inferences),
            format_count(stage.completed),
            format_count(stage.remaining),
            format_duration(stage.seconds(plan.seconds_per_inference)),
        ]
        for stage in plan.stages
    ]
    table = render_table(
        ("stage", "arm", "inferences", "stored", "remaining", "wall clock"),
        rows,
        aligns=(LEFT, LEFT, RIGHT, RIGHT, RIGHT, RIGHT),
        footer=(
            "total",
            "",
            format_count(plan.total),
            format_count(plan.completed),
            format_count(plan.remaining),
            format_duration(plan.seconds),
        ),
    )
    return "\n".join([table, "", *_plan_notes(plan)])


def _plan_notes(plan: ExperimentPlan) -> list[str]:
    """What the table cannot say in a column, said underneath it.

    The estimate marker is the important one. A debate arm counted from a measured
    contested share and one extrapolated from an assumption are different kinds of
    number, and somebody deciding whether to start tonight is entitled to know
    which they are looking at.
    """
    measured = "assumed" if plan.contested_estimated else "measured"
    notes = [
        f"{format_count(plan.decision_points)} decision points, "
        f"{format_count(plan.contested_points)} contested ({measured}).",
        f"Wall clock assumes {plan.seconds_per_inference:.2f}s per inference at each "
        "stage's own concurrency.",
    ]
    if plan.is_estimated:
        notes.append(
            "* estimated: the independent arm has not been generated, so the "
            "contested share is assumed rather than measured. Re-run plan after "
            "generate for a counted figure."
        )
    return notes


def render_results(results: ExperimentResults) -> str:
    """The arms, the shift rates, the windows and the loudest voice, in that order."""
    exploratory = results.aggregation_rule != PRIMARY_RULE
    sections = [
        f"Council -- results under {results.aggregation_rule} aggregation"
        + ("*" if exploratory else ""),
        f"{format_count(results.decision_count)} stored decisions; "
        f"{format_count(results.contested_points)} contested points "
        f"({results.contested_share:.1%} of the calendar).",
        "",
        _arms_table(results),
        "",
        _shift_table(results.shift_rates),
        "",
        _windows_block(results.windows),
        "",
        _influence_block(results.influence),
        "",
        *_results_notes(results),
    ]
    return "\n".join(sections)


def _results_notes(results: ExperimentResults) -> list[str]:
    """What the tables cannot say in a column, said underneath them.

    The marker follows :func:`_plan_notes`, and for the same reason: a number
    stated under the declared rule and one stated under a rule chosen after the
    data was visible are different kinds of number, and a reader handed a
    ``--rule median`` table has no other way to tell which they are holding.
    """
    notes = [
        f"Pre-registered comparison: {Arm.DEBATE} against {Arm.INDEPENDENT}, "
        f"under {PRIMARY_RULE} aggregation. Everything else here is exploratory."
    ]
    if results.aggregation_rule != PRIMARY_RULE:
        notes.append(
            f"* exploratory: this table is scored under {results.aggregation_rule} "
            f"aggregation, which is not the rule the comparison was declared in. "
            f"Re-run with --rule {PRIMARY_RULE} for the pre-registered figure."
        )
    return notes


def _arms_table(results: ExperimentResults) -> str:
    def row(name: str, metrics: PerformanceMetrics, baseline: str) -> list[str]:
        return [
            name,
            format_ratio(metrics.total_return),
            format_ratio(metrics.cagr),
            format_ratio(metrics.sharpe, digits=2),
            format_ratio(metrics.max_drawdown),
            format_ratio(metrics.turnover_per_period),
            baseline,
        ]

    rows = [
        row(
            outcome.arm,
            outcome.metrics,
            format_ratio(None if outcome.baseline is None else outcome.baseline.sharpe, digits=2),
        )
        for outcome in results.arms
    ]
    rows.append(row("buy_and_hold", results.buy_and_hold, "-"))
    return render_table(
        ("arm", "total", "CAGR", "Sharpe", "max DD", "turnover/p", "random Sharpe"),
        rows,
        aligns=(LEFT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT),
    )


def _shift_table(shift_rates: Mapping[str, ShiftRateReport]) -> str:
    """Shift rate by the confidence held before the debate: the primary statistic."""
    arms = sorted(shift_rates)
    if not arms:
        return "No debate rounds stored, so there is no shift rate to report."
    bands = shift_rates[arms[0]].bands
    rows = [
        [
            band.band.label,
            *(
                f"{format_ratio(shift_rates[arm].bands[index].shift_rate)}"
                f" ({shift_rates[arm].bands[index].count})"
                for arm in arms
            ),
        ]
        for index, band in enumerate(bands)
    ]
    return "\n".join(
        [
            "Shift rate by prior confidence (count in brackets)",
            render_table(
                ("prior confidence", *arms),
                rows,
                aligns=(LEFT, *(RIGHT for _ in arms)),
            ),
        ]
    )


def _windows_block(windows: Mapping[str, WindowComparison]) -> str:
    lines = ["Windows won against the independent arm"]
    lines += [
        f"  {arm}: {comparison.summary}"
        + (f", {comparison.ties} tied" if comparison.ties else "")
        for arm, comparison in sorted(windows.items())
    ]
    return "\n".join(lines)


def _influence_block(influence: Mapping[str, InfluenceMatrix]) -> str:
    lines = ["Net influence (concessions won minus made)"]
    for arm, matrix in sorted(influence.items()):
        scores = ", ".join(f"{model} {score:+d}" for model, score in matrix.net_influence)
        lines.append(f"  {arm}: {scores or 'no concessions'}")
    return "\n".join(lines)


def results_as_json(results: ExperimentResults) -> dict[str, Any]:
    """The whole results object, in plain types and in valid JSON.

    A non-finite ratio arrives as the string ``"Infinity"``, ``"-Infinity"`` or
    ``"NaN"`` rather than as a float or a null. A Sharpe of infinity is what
    :mod:`council.backtest.metrics` says when a return never varied, so nulling it
    would turn a deliberate, ugly signal into a missing cell -- but RFC 8259 has no
    token for it either, and Python's permissive writer emits one that jq,
    ``JSON.parse`` and every schema validator reject. The name survives; the file
    stays readable by something other than Python.
    """
    payload: dict[str, Any] = _in_valid_json(_results_payload(results))
    return payload


def _in_valid_json(value: Any) -> Any:
    """The payload with every non-finite float replaced by the name JSON lacks."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return _finite_or_name(value)
    if isinstance(value, Mapping):
        return {key: _in_valid_json(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_in_valid_json(item) for item in value]
    return value


def _finite_or_name(value: float) -> float | str:
    if math.isnan(value):
        return "NaN"
    if math.isinf(value):
        return "Infinity" if value > 0 else "-Infinity"
    return value


def _results_payload(results: ExperimentResults) -> dict[str, Any]:
    return {
        "aggregation_rule": results.aggregation_rule,
        "contested_share": results.contested_share,
        "contested_points": results.contested_points,
        "decision_count": results.decision_count,
        "buy_and_hold": asdict(results.buy_and_hold),
        "arms": [_arm_json(outcome) for outcome in results.arms],
        "shift_rates": {arm: _shift_json(report) for arm, report in results.shift_rates.items()},
        "calibration": {
            arm: _calibration_json(report) for arm, report in results.calibration.items()
        },
        "influence": {arm: _influence_json(matrix) for arm, matrix in results.influence.items()},
        "windows": {arm: _windows_json(window) for arm, window in results.windows.items()},
    }


def _arm_json(outcome: ArmOutcome) -> dict[str, Any]:
    return {
        "arm": outcome.arm,
        "point_count": outcome.point_count,
        "metrics": asdict(outcome.metrics),
        "baseline": None if outcome.baseline is None else asdict(outcome.baseline),
    }


def _shift_json(report: ShiftRateReport) -> dict[str, Any]:
    return {
        "skipped_count": report.skipped_count,
        "bands": [
            {
                "band": band.band.label,
                "count": band.count,
                "shifted_count": band.shifted_count,
                "reversed_count": band.reversed_count,
                "shift_rate": band.shift_rate,
                "reversal_rate": band.reversal_rate,
                "mean_distance": band.mean_distance,
            }
            for band in report.bands
        ],
    }


def _calibration_json(report: CalibrationReport) -> dict[str, Any]:
    return {
        "correlation": report.correlation,
        "hit_rate": report.hit_rate,
        "scored_count": report.scored_count,
        "skipped_count": report.skipped_count,
        "buckets": [
            {
                "band": bucket.band.label,
                "count": bucket.count,
                "hit_count": bucket.hit_count,
                "hit_rate": bucket.hit_rate,
            }
            for bucket in report.buckets
        ],
    }


def _influence_json(matrix: InfluenceMatrix) -> dict[str, Any]:
    return {
        "arm": matrix.arm,
        "models": list(matrix.models),
        "conceded": matrix.conceded.tolist(),
        "amount": matrix.amount.tolist(),
        "opportunities": matrix.opportunities.tolist(),
        "net_influence": [[model, score] for model, score in matrix.net_influence],
    }


def _windows_json(comparison: WindowComparison) -> dict[str, Any]:
    return {
        "window_count": comparison.window_count,
        "treatment_wins": comparison.treatment_wins,
        "ties": comparison.ties,
        "windows": [
            {
                "index": window.index,
                "treatment_return": window.treatment_return,
                "control_return": window.control_return,
                "margin": window.margin,
            }
            for window in comparison.windows
        ],
    }

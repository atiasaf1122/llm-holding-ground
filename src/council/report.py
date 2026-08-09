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
from council.planning import TREATMENT_ARMS, ExperimentPlan
from council.scoring import PRIMARY_RULE, ArmOutcome, ExperimentResults

RIGHT = ">"
LEFT = "<"


def _declared_order(mapping: Mapping[str, Any]) -> list[str]:
    """The treatment arms in the order the experiment declares them, then the rest.

    One report used to order the arms two ways: :func:`_arms_table` iterates
    ``results.arms`` and gets debate, rationale-only, placebo, while the shift,
    window and influence blocks sorted the keys and got debate, placebo,
    rationale-only. :data:`council.app.curves.SCORED_ARMS` asserts the opposite
    guarantee -- that the panel and the results table list the arms identically --
    so the declared tuple is what every block reads.

    Anything the tuple does not name is sorted after it rather than dropped: an
    unrecognised label is a fact about the frame and the reader has to see it.
    """
    declared = [str(arm) for arm in TREATMENT_ARMS]
    return [arm for arm in declared if arm in mapping] + sorted(set(mapping) - set(declared))


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
        f"({results.contested_share:.1%} of the decision points)"
        f"; scored over prices {results.calendar_start} to {results.calendar_end}, "
        f"{format_count(results.session_count)} sessions; "
        f"{_decision_span(results)}.",
        "",
        _arms_table(results),
        "",
        _shift_table(
            results.shift_rates,
            results.debated_points,
            dropped_pairs=results.dropped_pairs,
            unpaired=results.unpaired,
            short_committee_points=results.short_committee_points,
        ),
        "",
        _windows_block(results.windows),
        "",
        _influence_block(results.influence),
        "",
        *_results_notes(results),
    ]
    return "\n".join(sections)


def _decision_span(results: ExperimentResults) -> str:
    """The dates decisions were made on, beside the price range they were scored over.

    Named separately because the two come apart. An interrupted ``generate`` leaves
    prices covering the configured range and decisions covering a fraction of it,
    and every backtest period before the first decision is flat in every arm -- so
    a header naming the price range alone reads as a run that covered it.
    """
    if results.decision_start is None or results.decision_end is None:
        return "no decision dates"
    return (
        f"decisions {results.decision_start} to {results.decision_end}, "
        f"{format_count(results.decision_date_count)} dates"
    )


def _results_notes(results: ExperimentResults) -> list[str]:
    """What the tables cannot say in a column, said underneath them.

    The marker follows :func:`_plan_notes`, and for the same reason: a number
    stated under the declared rule and one stated under a rule chosen after the
    data was visible are different kinds of number, and a reader handed a
    ``--rule median`` table has no other way to tell which they are holding.

    The equity comparison is labelled **secondary**. Two quantities are declared in
    the README and only one can decide the result; the primary outcome is the shift
    rate, which involves no returns and which "net of costs" cannot qualify. This
    line used to attach "Pre-registered comparison" to the arms table, so both
    declared outcomes carried the primary label and whichever came out favourable
    could be reported as the pre-registered one.
    """
    notes = [
        f"Secondary declared comparison: {Arm.DEBATE} against {Arm.INDEPENDENT}, "
        f"under {PRIMARY_RULE} aggregation, net of costs. The primary outcome is "
        "the shift rate above. Everything else here is exploratory."
    ]
    unmatched = [outcome.arm for outcome in results.arms if outcome.baseline is None]
    if unmatched:
        # The dash in the `random Sharpe` column means two different things in one
        # table: an absent null here, and "not applicable" in the buy_and_hold row
        # directly underneath. The only statement of the reason was a log line that
        # never reached the report, and results.json records `"baseline": null`
        # with no reason either.
        notes.append(
            "No turnover-matched random baseline for "
            + ", ".join(unmatched)
            + ": the null draws from the arm's own exposures, and no subset of its "
            "revision dates reaches its turnover. That dash is an absent null, not "
            "a zero, and not the 'not applicable' the buy_and_hold row's dash means."
        )
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


def _shift_table(
    shift_rates: Mapping[str, ShiftRateReport],
    debated_points: Mapping[str, int],
    *,
    dropped_pairs: Mapping[str, int] | None = None,
    unpaired: Mapping[str, int] | None = None,
    short_committee_points: Mapping[str, int] | None = None,
) -> str:
    """Shift rate by the confidence held before the debate: the primary statistic.

    The bar is printed in the caption, which is what
    :attr:`council.evaluation.persuasion.Shift.threshold` exists for: a shift rate
    cannot be read without knowing what counted as a shift, and a rate published
    beside no bar is the drift that attribute was added to prevent.

    The decision points each arm debated are printed underneath, because the placebo
    arm abandons every contested point with no usable earlier donor, so part of any
    debate-minus-placebo gap is coverage rather than persuasion -- and
    :attr:`council.scoring.ArmOutcome.point_count` cannot show it.

    Each cell carries both denominators. ``obs`` is the unit the statistic is
    declared over; ``pts`` is the distinct decision points behind them, and every
    point is answered once per seat of every committee, so the observations repeat.
    ``debated_points`` underneath cannot supply this: it is per arm, not per band.

    The pairs the rate's denominator lost are printed underneath too, when there
    are any. The drop is not random across the arms -- round 1 only exists in the
    debate arms, so a crashed round 1 lands its phantom shift entirely on the
    treatment -- and both counts previously reached the dashboard's coverage table
    and neither the CLI nor ``results.json``.

    The points the *exposure series* lost sit on the same line, for the same reason
    and with a heavier consequence: a committee short of one post-debate row is
    dropped by :func:`council.scoring.committee_exposures` and the point falls back
    to that committee's independent view, so a conversation that happened is scored
    as the control it is being compared against.
    """
    arms = _declared_order(shift_rates)
    if not arms:
        return "No debate rounds stored, so there is no shift rate to report."
    bands = shift_rates[arms[0]].bands
    rows = [
        [
            band.band.label,
            *(
                f"{format_ratio(shift_rates[arm].bands[index].shift_rate)}"
                f" ({shift_rates[arm].bands[index].count} obs"
                f" / {shift_rates[arm].bands[index].point_count} pts)"
                for arm in arms
            ),
        ]
        for index, band in enumerate(bands)
    ]
    bar = next(
        (report.threshold for report in shift_rates.values() if report.threshold is not None),
        None,
    )
    caption = (
        "Shift rate by prior confidence (observations / decision points in brackets)"
        if bar is None
        else (
            f"Shift rate by prior confidence, bar {bar:.2f} "
            "(observations / decision points in brackets)"
        )
    )
    lines = [
        caption,
        render_table(
            ("prior confidence", *arms),
            rows,
            aligns=(LEFT, *(RIGHT for _ in arms)),
        ),
        "Decision points debated: "
        + ", ".join(f"{arm} {format_count(debated_points.get(arm, 0))}" for arm in arms),
    ]
    dropped = dropped_pairs or {}
    orphaned = unpaired or {}
    short = short_committee_points or {}
    if any(dropped.values()) or any(orphaned.values()) or any(short.values()):
        lines.append(
            "Pairs dropped for a failed generation: "
            + ", ".join(f"{arm} {format_count(dropped.get(arm, 0))}" for arm in arms)
            + "; rows with no partner round: "
            + ", ".join(f"{arm} {format_count(orphaned.get(arm, 0))}" for arm in arms)
            + "; points scored as the control for a committee short of a seat or "
            "absent from this arm: "
            + ", ".join(f"{arm} {format_count(short.get(arm, 0))}" for arm in arms)
        )
    return "\n".join(lines)


def _windows_block(windows: Mapping[str, WindowComparison]) -> str:
    lines = ["Windows won against the independent arm"]
    lines += [
        f"  {arm}: {windows[arm].summary}"
        + (f", {windows[arm].ties} tied" if windows[arm].ties else "")
        for arm in _declared_order(windows)
    ]
    return "\n".join(lines)


def _influence_block(influence: Mapping[str, InfluenceMatrix]) -> str:
    lines = ["Net influence (concessions won minus made)"]
    for arm in _declared_order(influence):
        matrix = influence[arm]
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
        # Which calendar this artefact scored. Provenance, not a measurement: a
        # published results file that records decision counts and per-arm periods
        # but names neither end of the period they were counted over cannot show
        # that a run covered six months of a two-year configured range.
        "calendar": {
            "start": results.calendar_start.isoformat(),
            "end": results.calendar_end.isoformat(),
            "sessions": results.session_count,
        },
        # And which dates decisions were made on, which is not the same span. The
        # calendar above is the price file's range, so on its own it catches a short
        # run only when the prices are short -- not when an interrupted generate
        # leaves a full price range holding a fraction of a run's decisions.
        "decisions": {
            "start": None if results.decision_start is None else results.decision_start.isoformat(),
            "end": None if results.decision_end is None else results.decision_end.isoformat(),
            "dates": results.decision_date_count,
        },
        "contested_share": results.contested_share,
        "contested_points": results.contested_points,
        "decision_count": results.decision_count,
        "debated_points": dict(results.debated_points),
        "dropped_pairs": dict(results.dropped_pairs),
        "short_committee_points": dict(results.short_committee_points),
        "unpaired_rows": dict(results.unpaired),
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
        "threshold": report.threshold,
        "bands": [
            {
                "band": band.band.label,
                "count": band.count,
                "points": band.point_count,
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
                # How the period was cut. `split_windows` gives earlier windows the
                # extra period on an uneven split, so the windows are not all the
                # same length -- and every arm is flat over the `lookback_days - 1`
                # warm-up, so an early window can be partly or wholly dead. Wins and
                # ties alone cannot show either.
                "start": window.start,
                "stop": window.stop,
                "length": window.length,
                "treatment_return": window.treatment_return,
                "control_return": window.control_return,
                "margin": window.margin,
            }
            for window in comparison.windows
        ],
    }

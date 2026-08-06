"""A probe report, as something a person reads.

Presentation only, the way :mod:`council.report` is: every number arrives already
decided by :mod:`council.probe.report`, which is what keeps that module testable
against arithmetic rather than against a string. The table primitives are imported
from :mod:`council.report` rather than rewritten, so an absent rate prints as the
same dash here as it does in the market results.

Three blocks, in the order the numbers qualify each other: the two movement rates
per condition, then capitulation cut by the confidence held *before* the challenge,
then what was left out. The exclusions go last and are never omitted -- a headline
computed over the trials that survived reads exactly like one computed over the run.
"""

from __future__ import annotations

from council.probe.report import ConditionReport, ProbeReport
from council.report import LEFT, RIGHT, format_count, format_ratio, render_table


def render_probe(report: ProbeReport, *, model: str) -> str:
    """The whole report under one heading naming the model it describes."""
    heading = f"Council -- capitulation probe on {model}"
    if not report.conditions:
        return f"{heading}\n\nNo trials were scored, so there is nothing to report."
    return "\n".join(
        [
            heading,
            "",
            _rates_table(report),
            "",
            _bands_table(report),
            "",
            *_notes(report),
        ]
    )


def _rates_table(report: ProbeReport) -> str:
    """Both directions side by side.

    The correction rate is printed beside the capitulation rate rather than under
    it: the same mechanism produces both, and a table showing only the unflattering
    half describes a model that never listens to anybody as ideal.
    """
    rows = [
        [
            str(condition.condition),
            format_count(condition.overall.right_before),
            format_count(condition.overall.capitulated),
            format_ratio(condition.overall.capitulation_rate),
            format_count(condition.overall.wrong_before),
            format_count(condition.overall.corrected),
            format_ratio(condition.overall.correction_rate),
            format_ratio(condition.overall.opening_accuracy),
            format_count(condition.ungraded_count),
        ]
        for condition in report.conditions
    ]
    return render_table(
        (
            "condition",
            "right",
            "gave in",
            "capitulation",
            "wrong",
            "corrected",
            "correction",
            "accuracy",
            "excluded",
        ),
        rows,
        aligns=(LEFT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT, RIGHT),
    )


def _bands_table(report: ProbeReport) -> str:
    """Capitulation by prior confidence, one column per condition.

    Every condition was cut at the same edges, so the rows line up by construction
    and the two columns can be read against each other. The count in brackets is the
    denominator: a rate over two items is not the same claim as a rate over twenty.
    """
    rows = [
        [
            band.band.label,
            *(
                f"{format_ratio(condition.bands[index].counts.capitulation_rate)} "
                f"({condition.bands[index].counts.right_before})"
                for condition in report.conditions
            ),
        ]
        for index, band in enumerate(report.conditions[0].bands)
    ]
    return render_table(
        ("prior confidence", *(str(condition.condition) for condition in report.conditions)),
        rows,
        aligns=(LEFT, *(RIGHT for _ in report.conditions)),
    )


def _notes(report: ProbeReport) -> list[str]:
    """What the tables cannot say in a column, said underneath them."""
    return [
        "Capitulation above placebo: "
        + format_ratio(report.capitulation_above_placebo)
        + ". An upper bound, not a measurement: the placebo peer argues visibly "
        "about another question, so a model that notices discounts it for a reason "
        "the real condition never offers.",
        "",
        *(_exclusions(condition) for condition in report.conditions),
    ]


def _exclusions(condition: ConditionReport) -> str:
    """One condition's dropped trials, split by the position that was dropped.

    Split because the two halves bias opposite rates: an excluded trial the model
    had opened correctly can only raise the capitulation rate, and one it had opened
    wrongly can only raise the correction rate.
    """
    counts = condition.ungraded
    return (
        f"{condition.condition}: {format_count(counts.total)} trial(s) excluded "
        f"({format_count(counts.after_correct)} after a correct opening, "
        f"{format_count(counts.after_incorrect)} after a wrong one, "
        f"{format_count(counts.opening_unusable)} with no readable opening); "
        f"{format_count(condition.skipped_count)} scored but outside every band."
    )

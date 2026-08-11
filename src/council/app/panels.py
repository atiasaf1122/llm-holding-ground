"""The five result panels, in the order a reader should meet them.

Layout only, like :mod:`council.app.dashboard`, which assembles these. Every
number drawn here is computed by a tested function in :mod:`council.app`: a chart
cannot be asserted against, so the chart is all these functions contain.

Each panel handles its own empty case by name. A run that stored only the
independent arm has no shift to measure and nobody to concede to anyone, and
saying that is a different statement from drawing an axis with nothing on it.

**Every panel names its population.** The committee selector applies to the whole
page, so each panel states whether it is showing the eight committees pooled --
the declared scope -- or the single one the reader chose. A page whose panels
silently described different populations would let a reader difference two
numbers that were never comparable.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from council.app.artefacts import ARM_ORDER, Results
from council.app.curves import POOLED_LABEL, CurveSet, curves_frame, metrics_frame
from council.app.tables import (
    calibration_table,
    coverage_note,
    coverage_table,
    forward_return_lookup,
    influence_table,
    net_influence_table,
    select,
    shift_rate_table,
    shift_reports,
)
from council.app.transcripts import (
    SeatUtterance,
    Transcript,
    read_transcripts,
    seat_label,
    transcript_table,
)
from council.config import Settings
from council.domain.signal import Arm
from council.evaluation.calibration import CalibrationReport, calibrate
from council.evaluation.influence import influence_matrix
from council.evaluation.persuasion import OPENING_ROUND, REBUTTAL_ROUND
from council.scoring import PRIMARY_RULE

TRANSCRIPT_CHOICES = 50
"""How many conversations the reader picks from, widest disagreement first.

The list is ordered by opening dispersion, so the cut keeps the points where the
agents were furthest apart -- which are the ones the panel exists to show.
"""

DECLARED: str = "Pre-registered"
"""The word marking a number that was fixed before any result was generated.

Only two numbers on the page carry it. Everything else is a cut somebody chose
after seeing data, and the difference is the difference between a measurement and
a search.
"""


def _population_caption(composition: str | None) -> None:
    """Name the rows the panel above is drawn from, on every panel."""
    named = POOLED_LABEL if composition is None else f"the {composition} committee only"
    st.caption(f"Population: {named}.")


def _rounds_in(arm: str | None) -> list[int]:
    """Which rounds a population can be asked about.

    The independent arm has one round by construction, and offering a second
    would return an empty report that reads as a run with nothing in it.
    """
    if arm == str(Arm.INDEPENDENT):
        return [OPENING_ROUND]
    return [OPENING_ROUND, REBUTTAL_ROUND]


def _round_label(round_index: int) -> str:
    # "first rebuttal", not "after the debate": conversations run to ~6 rounds and
    # the arm ordering reverses between round 1 and the endpoint (CLAIMS C28), so a
    # label calling round 1 the debate's end would misread the study on the chart.
    return "0 - opening view" if round_index == OPENING_ROUND else "1 - first rebuttal"


# -- 2. equity curves --------------------------------------------------------


def equity_panel(
    curve_set: CurveSet, settings: Settings, *, composition: str | None, rule_name: str
) -> None:
    st.header("Equity curves")
    _declaration_note(composition=composition, rule_name=rule_name)
    st.caption(
        f"Net of {settings.total_cost_bps:.0f} bps a side. The random baseline "
        "trades as often as the control and holds the same exposure sizes, "
        "shuffled in time; buy-and-hold trades once and pays nothing."
    )
    frame = curves_frame(curve_set.curves)
    chart = (
        alt.Chart(frame)
        .mark_line()
        .encode(
            x=alt.X("date:T", title=None),
            y=alt.Y("equity:Q", title="growth of 1.0", scale=alt.Scale(zero=False)),
            color=alt.Color("series:N", title="arm"),
            tooltip=["date:T", "series:N", alt.Tooltip("equity:Q", format=".3f")],
        )
        .properties(height=340)
    )
    st.altair_chart(chart, width="stretch")
    st.dataframe(metrics_frame(curve_set.curves), hide_index=True)
    if curve_set.baseline_note is not None:
        st.warning(curve_set.baseline_note)


def _declaration_note(*, composition: str | None, rule_name: str) -> None:
    """Whether these curves are the declared comparison or a cut of it.

    The declared comparison names one aggregation rule and reads the
    balanced design as one experiment. Either control moved off those values and
    the panel is exploratory -- which is worth stating on the panel rather than
    leaving to a reader to reconstruct from two sidebar widgets.

    **Secondary**, not primary. The equity comparison and the shift rate are two
    different quantities, and only one of them can decide the result; the primary
    outcome is the shift rate on the next panel. Labelling both as the primary
    declared one is the freedom a pre-registration exists to remove.
    """
    if composition is None and rule_name == PRIMARY_RULE:
        st.success(
            f"{DECLARED} secondary declared comparison: the committee after debate "
            f"against the same committee before, under `{PRIMARY_RULE}` aggregation, "
            "pooled over every committee in the balanced design, net of costs. The "
            "primary outcome is the shift rate below. This is the same computation "
            "`python -m council evaluate` reports."
        )
        return
    st.warning(
        "Exploratory cut, not the declared comparison: "
        + " and ".join(
            note
            for note in (
                None if composition is None else f"one committee ({composition})",
                None if rule_name == PRIMARY_RULE else f"`{rule_name}` aggregation",
            )
            if note is not None
        )
        + f". The declared comparison is `{PRIMARY_RULE}` over {POOLED_LABEL}."
    )


# -- 3. the primary statistic ------------------------------------------------

_RATE_TOOLTIP = alt.Tooltip("shift_rate:Q", format=".3f")
"""Three decimals, because two round a rate of 0.004 to the same figure as zero."""


def shift_panel(results: Results, settings: Settings, *, composition: str | None) -> None:
    st.header("Shift rate against prior confidence")
    st.success(
        f"{DECLARED} primary statistic: the share of agent-conversation observations "
        "-- one agent, one committee, one contested point, one arm -- in which the "
        f"agent's exposure shifted by at least {settings.shift_threshold:.2f} between "
        "its opening and post-debate view, partitioned by the confidence it reported "
        "before seeing its peers. Observations repeat across seats and committees and "
        "are not independent, and pairs with a failed generation are excluded."
    )
    st.caption(
        "Reading the debate arm against the placebo is exploratory, not declared: "
        "the declaration states a per-arm rate and registers no contrast and no "
        "direction. The placebo shows movement in response to peers whose prose is "
        "about another day -- and, half the time, the other instrument -- so it bounds "
        "what the argument's content contributes; it does not isolate contradiction "
        "(CLAIMS D8, D14). These are first-rebuttal rates: over whole conversations "
        "the arm ordering reverses (CLAIMS C28). Read it against the coverage "
        "table below, which says whether the two arms answered the same points."
    )
    _population_caption(composition)
    reports = shift_reports(results.decisions)
    table = shift_rate_table(reports)
    if table.empty or table["count"].sum() == 0:
        st.warning(
            "No paired rounds in this run, so no shift can be measured. The "
            "independent arm has one round by construction; a debate arm has two."
        )
        return

    chart = (
        alt.Chart(table)
        .mark_bar()
        .encode(
            x=alt.X("band:N", title="prior confidence"),
            xOffset=alt.XOffset("arm:N", sort=list(ARM_ORDER)),
            y=alt.Y("shift_rate:Q", title="share that shifted", scale=alt.Scale(domain=[0, 1])),
            color=alt.Color("arm:N", sort=list(ARM_ORDER), title="arm"),
            tooltip=["arm:N", "band:N", "count:Q", "points:Q", _RATE_TOOLTIP],
        )
        .properties(height=320)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "`count` is observations -- the unit the statistic is declared over -- and "
        "`points` is the distinct decision points behind them: each point is "
        "answered once per seat of every committee, so the observations are "
        "repeated rather than independent, and `points` is what says by how much. "
        + " ".join(
            f"{entry.arm}: {len(entry.records)} observations from {entry.points} decision points."
            for entry in reports.values()
            if entry.records
        )
        + " Bands with no observations are absent rather than drawn at zero. "
        + " ".join(
            f"{entry.arm}: {entry.report.skipped_count} outside every band."
            for entry in reports.values()
            if entry.report.skipped_count
        )
    )
    st.dataframe(table, hide_index=True)
    _coverage(results)


def _coverage(results: Results) -> None:
    """What each arm actually ran, under the rates computed from it."""
    st.subheader("Coverage")
    coverage = coverage_table(results.decisions)
    st.dataframe(coverage, hide_index=True)
    note = coverage_note(coverage)
    if note is not None:
        st.warning(note)
    else:
        st.caption(
            "Rows an arm produced but could not pair, and pairs dropped for a "
            "failed generation, are counted here rather than left out of sight of "
            "the rate they were removed from."
        )


# -- 4. calibration ----------------------------------------------------------


def calibration_panel(results: Results, *, composition: str | None) -> None:
    st.header("Calibration")
    st.caption(
        "Does a stated confidence predict being right? Confidence is never used "
        "to weight an aggregation, because using it before measuring it would "
        "answer the question with itself. Exploratory: the arm and round below "
        "are chosen after the fact."
    )
    _population_caption(composition)
    arm = st.selectbox("population", results.arms, key="calibration_arm")
    rounds = _rounds_in(arm)
    round_index = st.selectbox("round", rounds, format_func=_round_label, key="calibration_round")
    if arm is None or round_index is None:
        return

    report = calibrate(
        select(results.decisions, arm=arm, round_index=round_index),
        forward_return_lookup(results.opens),
    )
    table = calibration_table(report)
    diagonal = pd.DataFrame({"confidence": [0.0, 1.0], "hit_rate": [0.0, 1.0]})
    points = (
        alt.Chart(table)
        .mark_line(point=True)
        .encode(
            x=alt.X("confidence:Q", title="stated confidence", scale=alt.Scale(domain=[0, 1])),
            y=alt.Y("hit_rate:Q", title="realised hit rate", scale=alt.Scale(domain=[0, 1])),
            size=alt.Size("count:Q", title="decisions"),
            tooltip=["band:N", "count:Q", alt.Tooltip("hit_rate:Q", format=".3f")],
        )
    )
    reference = (
        alt.Chart(diagonal)
        .mark_line(strokeDash=[6, 4], color="grey")
        .encode(x="confidence:Q", y="hit_rate:Q")
    )
    st.altair_chart((reference + points).properties(height=340), width="stretch")
    _calibration_footnote(report)


def _calibration_footnote(report: CalibrationReport) -> None:
    reading = "undefined" if report.correlation is None else f"{report.correlation:+.3f}"
    st.caption(
        f"{report.scored_count} decisions scored, {report.skipped_count} skipped -- "
        "a flat exposure has no direction to be right about, and the last two "
        "sessions have no forward return. Correlation between confidence and "
        f"being right: {reading}."
    )


# -- 5. influence ------------------------------------------------------------


def influence_panel(results: Results, *, composition: str | None) -> None:
    st.header("Influence")
    st.caption(
        "Who gave more ground than they got. One arm at a time: summing the real "
        "debate with its placebo would mix the answer into the question. "
        "Exploratory: no direction here was declared in advance."
    )
    _population_caption(composition)
    debate_arms = [arm for arm in results.arms if arm != str(Arm.INDEPENDENT)]
    if not debate_arms:
        st.warning("This run holds no debate rows, so nobody has conceded anything.")
        return

    arm = st.selectbox("arm", debate_arms, key="influence_arm")
    if arm is None:
        return
    matrix = influence_matrix(results.decisions, arm=arm)
    table = influence_table(matrix)
    chart = (
        alt.Chart(table)
        .mark_rect()
        .encode(
            x=alt.X("influencer:N", title="moved toward"),
            y=alt.Y("conceder:N", title="gave ground"),
            color=alt.Color("rate:Q", title="share", scale=alt.Scale(scheme="blues")),
            tooltip=["conceder:N", "influencer:N", "conceded:Q", "opportunities:Q"],
        )
        .properties(height=280)
    )
    st.altair_chart(chart, width="stretch")
    st.caption(
        "Rows concede to columns. Every committee seats each base model exactly "
        "once, so the diagonal is always zero and is there only to keep the "
        "matrix square."
    )
    st.dataframe(net_influence_table(matrix), hide_index=True)


# -- 6. the transcript reader ------------------------------------------------


def transcript_panel(results: Results, *, composition: str | None) -> None:
    st.header("Read one debate")
    st.caption(
        "Every panel above is a rate. This is the evidence under them: the point "
        "where the agents opened furthest apart, and what each said at its opening "
        "view and its first rebuttal. A conversation runs up to six rounds; this "
        "shows the first exchange, which is what the primary statistic measures."
    )
    _population_caption(composition)
    transcripts = read_transcripts(results.decisions)
    if not transcripts:
        st.warning(
            "No conversation in this run has both an opening and a final round, "
            "so there is nothing to read."
        )
        return

    chosen = st.selectbox(
        "decision point",
        transcripts[:TRANSCRIPT_CHOICES],
        format_func=lambda item: item.label,
        key="transcript",
    )
    if chosen is None:
        return
    _transcript_header(chosen)
    for seat in chosen.seats:
        _seat_row(seat)
    st.dataframe(transcript_table(chosen), hide_index=True)


def _transcript_header(transcript: Transcript) -> None:
    """The four numbers over a conversation, each labelled with who it covers.

    The spread is taken over every opening view and the two means over the seats
    that spoke in both rounds, because a before-and-after delta computed across
    two different populations would not be a delta at all. That makes the spread
    and the means describe different committees whenever a seat is silent, so the
    labels say which -- rather than leaving four numbers that look like one set.
    """
    columns = st.columns(4)
    columns[0].metric("opening dispersion (this committee)", f"{transcript.opening_std:.2f}")
    columns[1].metric("committee before (speaking seats)", f"{transcript.opening_mean:+.2f}")
    columns[2].metric(
        "committee after (speaking seats)",
        f"{transcript.final_mean:+.2f}",
        delta=f"{transcript.final_mean - transcript.opening_mean:+.2f}",
    )
    columns[3].metric("largest single move", f"{transcript.largest_move:.2f}")
    if transcript.silent:
        st.warning(
            "Seats missing one of the two rounds, and therefore not shown: "
            + ", ".join(f"{model}/{persona}" for model, persona in transcript.silent)
            + ". Their opening views are still in the dispersion above and not in "
            "the two means, which is why those metrics name their populations."
        )


def _seat_row(seat: SeatUtterance) -> None:
    shift = seat.shift
    st.subheader(seat_label(seat))
    opening, final = st.columns(2)
    opening.markdown(
        f"**opening** exposure `{shift.prior_exposure:+.2f}` "
        f"confidence `{shift.prior_confidence:.2f}`"
    )
    opening.write(seat.opening_rationale or "_no rationale stored_")
    verdict = "moved at the bar" if shift.changed_mind else "held at the bar"
    final.markdown(
        f"**first rebuttal** exposure `{shift.posterior_exposure:+.2f}` "
        f"confidence `{shift.posterior_confidence:.2f}` -- {verdict}"
    )
    final.write(seat.final_rationale or "_no rationale stored_")
    if seat.failed:
        st.warning("A generation failed here; the stored exposure is a placeholder.")

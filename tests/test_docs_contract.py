"""The prose states what the code does.

Every assertion here is a sentence that was wrong once, in a document a reader is
asked to check results against. None of them would have failed a run: a README that
declares a strict bar beside an inclusive predicate, a design note that promises the
arms cover the same points, an arithmetic block off by a factor of three -- each is
a defect only in the sense that it misdescribes the artefact, which is the whole
value of a pre-registration.

The rule these follow: assert against the shipped value or the shipped predicate
wherever one exists, so that changing the code without changing the prose fails
here rather than being published.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from council.config import PROJECT_ROOT, get_settings
from council.debate.protocol import DEFAULT_REBUTTAL_ROUNDS
from council.evaluation.threshold import exceeds, meets
from council.planning import SECONDS_PER_INFERENCE
from council.scoring import DEFAULT_WINDOW_COUNT

README = PROJECT_ROOT / "README.md"
RESEARCH = PROJECT_ROOT / "docs" / "research.md"
FINDINGS = PROJECT_ROOT / "docs" / "findings.md"
CLAIMS = PROJECT_ROOT / "docs" / "CLAIMS.md"
COMPOSITIONS = PROJECT_ROOT / "src" / "council" / "debate" / "compositions.py"

DOCUMENTS: tuple[Path, ...] = (README, RESEARCH, FINDINGS, CLAIMS)

SURVIVING_RUN = (
    PROJECT_ROOT / "docs" / "results" / "superseded" / "run-2models" / "decisions.parquet"
)
"""The two-model run's decisions. Four files said these had been *deleted* while the
same sections cited this path as the source of their recomputed figures."""

MOVED_ARTEFACT = "docs/results/run-2models"
"""Where the two-model run used to live. Cited in five places after it moved under
``superseded/``, one of them the test that pins findings.md to the shipped code --
which then errored on import, so the guard was off while claiming to be on."""


def text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def flowed(body: str) -> str:
    """One line, blockquote markers removed, so a wrap cannot hide a phrase."""
    return " ".join(line.lstrip("> ") for line in body.splitlines()).replace("  ", " ")


# -- the pre-registered bar is the one the code applies ---------------------------


def test_the_shipped_predicate_is_inclusive_at_the_bar() -> None:
    # The reason the wording matters at all: exposures land on a 0.05 grid, so
    # exactly-at-the-bar moves are the largest single cluster, and `meets` and
    # `exceeds` disagree on every one of them.
    for high, low in ((0.5, 0.3), (0.3, 0.1), (0.4, 0.6)):
        distance = abs(high - low)
        assert meets(distance, 0.20) is True
        assert exceeds(distance, 0.20) is False


def test_the_readme_does_not_declare_a_bar_the_code_does_not_use() -> None:
    # "more than 0.20" excludes the whole at-the-bar cluster that `Shift.shifted`
    # counts. Declaration and computation differing on the primary statistic is
    # the defect; the README is where the declaration lives.
    primary = text(README).split("**Primary statistic.**")[1].split("**Direction.**")[0]

    assert "more than `0.20`" not in primary
    assert "at least" in primary
    assert "meets" in primary


def test_the_dashboard_states_the_same_bar_as_the_readme() -> None:
    # The panel reprints the primary statistic beside the inclusively computed
    # number, so a reader is otherwise told one statistic and shown another.
    panel = (PROJECT_ROOT / "src" / "council" / "app" / "panels.py").read_text(encoding="utf-8")

    assert "shifted by more than" not in panel
    assert "shifted by at least" in panel


# -- the bounds the reader is asked to check results against ----------------------


def test_every_bound_config_declares_before_the_run_is_named_in_the_declaration() -> None:
    # config.py declares six bounds with the same before-the-run justification, and
    # `placebo_min_gap_sessions` changes which points each arm covers. Naming two of
    # them leaves four invisible in the paragraph a reader checks results against.
    declaration = text(README).split("Six bounds are fixed in")[1].split("**Why this")[0]

    for bound in (
        "shift_threshold",
        "dispersion_threshold",
        "agreement_spread",
        "stillness_rounds",
        "max_debate_rounds",
        "placebo_min_gap_sessions",
    ):
        assert bound in declaration, bound


def test_the_declared_values_are_the_values_that_ship() -> None:
    declaration = text(README).split("Six bounds are fixed in")[1].split("**Why this")[0]
    settings = get_settings()

    assert f"shift_threshold = {settings.shift_threshold:.2f}" in declaration
    assert f"dispersion_threshold = {settings.dispersion_threshold:.2f}" in declaration
    assert f"agreement_spread = {settings.agreement_spread:.2f}" in declaration
    assert f"stillness_rounds = {settings.stillness_rounds}" in declaration
    assert f"max_debate_rounds = {settings.max_debate_rounds}" in declaration
    assert f"placebo_min_gap_sessions = {settings.placebo_min_gap_sessions}" in declaration


def test_the_shipped_cap_is_the_pin_the_documents_describe() -> None:
    # Three documents described three protocols once. They now describe the one
    # that runs, and this is what ties that description to the code.
    assert get_settings().max_debate_rounds == DEFAULT_REBUTTAL_ROUNDS == 1


# -- which bounds actually predate the results ------------------------------------

PRE_EXISTING_BOUNDS: tuple[str, ...] = ("shift_threshold", "dispersion_threshold")
LATER_BOUNDS: tuple[str, ...] = (
    "agreement_spread",
    "stillness_rounds",
    "max_debate_rounds",
    "placebo_min_gap_sessions",
)
"""`git log -S` on src/council/config.py: the first pair arrives in `afce0ae`, the
first commit; the second four arrive in `cbf6a55`, which lands after the results
commits `fa436fa` and `98a4020`. Two of the four were calibrated from that run."""


def test_the_declaration_does_not_claim_all_six_bounds_predate_the_data() -> None:
    # "Declared before any result was generated" is what separates a measurement
    # from a search, and it was true of two of the six bounds it was said about.
    declaration = text(README).split("Six bounds are fixed in")[1].split("**Why this")[0]
    header = text(README).split("## Pre-registered primary comparison")[1].split(">")[0]

    assert "Declared before any result was generated." not in header
    assert "afce0ae" in declaration
    assert "cbf6a55" in declaration
    for commit in ("fa436fa", "98a4020"):
        assert commit in declaration, commit
    assert "not** pre-registered" in declaration


def test_config_says_which_of_its_bounds_were_added_after_the_runs() -> None:
    config = (PROJECT_ROOT / "src" / "council" / "config.py").read_text(encoding="utf-8")

    assert "All three bounds are declared here, before the run" not in config
    assert "were *not* declared before the run" in config
    assert "cbf6a55" in config


def _agreement_spread_shares() -> dict[str, float | int]:
    """Committee spreads at each round of the surviving run's debate arm.

    Read the way ``debate.protocol._agreed`` reads them: the widest gap between any
    two seats of one (composition, date, ticker) committee, tested with
    ``threshold.within``.
    """
    import pandas as pd

    from council.domain.signal import Arm
    from council.evaluation.frames import (
        ARM,
        COMPOSITION,
        DECISION_DATE,
        EXPOSURE,
        ROUND_INDEX,
        TICKER,
    )
    from council.evaluation.threshold import within

    frame = pd.read_parquet(SURVIVING_RUN)
    debate = frame.loc[frame[ARM].astype(str) == str(Arm.DEBATE)]
    shares: dict[str, float | int] = {}
    for round_index in (0, 1):
        held: dict[tuple[str, object, str], list[float]] = {}
        rows = debate.loc[debate[ROUND_INDEX] == round_index]
        for composition, day, ticker, exposure in zip(
            rows[COMPOSITION], rows[DECISION_DATE], rows[TICKER], rows[EXPOSURE], strict=True
        ):
            held.setdefault((str(composition), day, str(ticker)), []).append(float(exposure))
        spreads = [max(seats) - min(seats) for seats in held.values() if len(seats) > 1]
        shares[f"meets_{round_index}"] = (
            sum(1 for spread in spreads if within(spread, 0.20)) / len(spreads)
        )
        if round_index == 0:
            shares["meets_half"] = sum(1 for s in spreads if within(s, 0.50)) / len(spreads)
            shares["bare"] = sum(1 for s in spreads if s <= 0.20) / len(spreads)
            shares["on_bar"] = sum(1 for s in spreads if abs(s - 0.20) < 1e-6)
            shares["on_bar_bare"] = sum(
                1 for s in spreads if abs(s - 0.20) < 1e-6 and s <= 0.20
            )
    return shares


def test_the_agreement_spread_comment_does_not_attribute_its_shares_to_this_run() -> None:
    # The README flags `agreement_spread` as calibrated from a run rather than
    # chosen blind, so the provenance is the whole justification -- and the comment
    # named a run whose decisions are on disk and reproduce none of its six figures.
    config = (PROJECT_ROOT / "src" / "council" / "config.py").read_text(encoding="utf-8")
    comment = config.split("# Agreement:")[1].split("agreement_spread: float")[0]
    measured = _agreement_spread_shares()

    assert "do **not** reproduce" in comment
    assert "four-model run" in comment
    for share, quoted in (
        ("meets_0", "39.5%"),
        ("meets_1", "37.3%"),
        ("meets_half", "64.5%"),
        ("bare", "34.1%"),
    ):
        assert f"{measured[share]:.1%}" == quoted, (share, measured[share])
        assert quoted in comment, quoted
    assert measured["on_bar"] == 95
    assert measured["on_bar_bare"] == 35
    assert "95 committees" in comment
    assert "admits 35" in comment or "only 35 of them" in comment
    # The shares it does still quote are the four-model run's, and are named as such
    # rather than as measurements of the file above.
    assert "13.6%" not in comment
    assert "39 committees" not in comment


def test_the_agreement_spread_comment_names_the_predicate_the_protocol_applies() -> None:
    # The comment is the checklist a next engineer uses when touching the stopping
    # rule: "the same comparison `debate.protocol._agreed` applies, so the number
    # quoted here and the predicate that ships cannot drift apart". `_agreed`
    # applies `within`, not `meets` -- the opposite-direction predicate, whose own
    # docstring warns that the other spelling reads as its own opposite. Acting on
    # the comment and putting `meets` into `_agreed` inverts the bar silently: on
    # the file the comment names, `within` reads 39.5% at round 0 and `meets` 69.0%.
    from council.evaluation.threshold import meets, within

    config = (PROJECT_ROOT / "src" / "council" / "config.py").read_text(encoding="utf-8")
    comment = config.split("# Agreement:")[1].split("agreement_spread: float")[0]
    protocol = (PROJECT_ROOT / "src" / "council" / "debate" / "protocol.py").read_text(
        encoding="utf-8"
    )

    assert "return within(max(exposures) - min(exposures), spread)" in protocol
    assert "evaluation.threshold.meets" not in comment
    assert "`meets`' 39.5%" not in comment
    assert "evaluation.threshold.within" in comment
    assert "`within`'s 39.5%" in comment
    # The two are not interchangeable at the values the comment quotes: a spread
    # wider than the bar is agreement under one and not under the other.
    assert within(0.10, 0.20) is True
    assert meets(0.10, 0.20) is False
    assert within(0.30, 0.20) is False
    assert meets(0.30, 0.20) is True


def test_the_claims_register_names_only_the_bounds_that_predate_the_data() -> None:
    c18 = text(CLAIMS).split("C18.")[1].split("C19.")[0]

    for bound in PRE_EXISTING_BOUNDS:
        assert bound in c18, bound
    for bound in LATER_BOUNDS:
        assert bound in c18, bound
    assert "afce0ae" in c18
    assert "not pre-registered" in c18


# -- the primary statistic is the quantity the code computes ----------------------


def test_the_primary_statistic_is_declared_over_the_unit_the_code_counts() -> None:
    # `shifts` emits one record per (composition, arm, date, ticker, model, persona)
    # and `shift_rate_by_confidence` divides by that record count. Declared over
    # decision points, the published 0.2737 would have to have been 0.9714.
    primary = text(README).split("**Primary statistic.**")[1].split("**Direction.**")[0]

    assert "decision points at which" not in primary
    assert "agent-conversation observations" in primary
    assert "not\n> independent" in primary or "not independent" in " ".join(primary.split())


def test_the_dashboard_does_not_name_a_finding_the_declaration_omits() -> None:
    # Directly under the box labelled as the declared primary statistic, the panel
    # printed "The gap between debate and placebo is the finding". The declaration
    # contains no such contrast: it states a per-arm share partitioned by
    # confidence, registers no direction ("No prediction is registered") and names
    # no comparison. Supplying a decisive contrast the declaration does not contain
    # is the freedom a pre-registration exists to close -- and CLAIMS D6 records
    # that every debate-minus-placebo number this project has produced came from a
    # placebo arm that was not inert.
    panel = " ".join(
        (PROJECT_ROOT / "src" / "council" / "app" / "panels.py")
        .read_text(encoding="utf-8")
        .split()
    )
    declared = flowed(text(README).split("**Primary statistic.**")[1].split("**Secondary")[0])

    assert "placebo" not in declared
    assert "No prediction is registered." in declared
    assert "gap between debate and placebo is the finding" not in panel
    assert "against the placebo is exploratory, not declared" in panel


def test_the_dashboard_declares_the_same_unit_as_the_readme() -> None:
    panel = (PROJECT_ROOT / "src" / "council" / "app" / "panels.py").read_text(encoding="utf-8")

    assert "share of contested decision points" not in panel
    assert "agent-conversation observations" in panel


# -- how many windows two years actually holds ------------------------------------


@pytest.mark.parametrize("document", [README, RESEARCH])
def test_the_window_count_named_in_the_documents_is_the_shipped_one(document: Path) -> None:
    # "roughly ten independent six-month windows" is 2.5x the real figure, in the
    # direction that makes the design look better powered. Two years is four
    # non-overlapping six-month windows, and the tool cuts it into five.
    body = text(document)

    assert "roughly ten independent six-month windows" not in body
    assert f"DEFAULT_WINDOW_COUNT = {DEFAULT_WINDOW_COUNT}" in body
    assert "four non-overlapping six-month windows" in body


def test_research_does_not_illustrate_a_bar_the_tool_cannot_print() -> None:
    # `WindowComparison.summary` renders "{wins} of {count} windows" at the shipped
    # count, so "8 of 10 windows" is a string no run can produce.
    body = text(RESEARCH)

    assert "8 of 10 windows" not in body
    assert f"4 of {DEFAULT_WINDOW_COUNT} windows" in body


# -- the prices are synthetic, and the front door says so -------------------------


def test_the_readme_says_the_reported_runs_used_synthetic_prices() -> None:
    # The README describes a real-market study end to end -- a ticker-selection rule
    # by market capitalisation, "one asset class, one market, one path" -- and never
    # said that every reported run was a geometric random walk.
    body = text(README)

    assert "synthetic_prices" in body
    assert "geometric random walks" in body
    assert "this repository fetches no" in body


def test_the_limitations_name_the_half_of_the_question_nothing_measures() -> None:
    # The headline question is "does it defend its position or abandon it -- and does
    # being right make any difference?". The second half has no instrument in the
    # market arms: the forward-return lookup reaches `calibration.calibrate` and
    # nothing else, `persuasion.Shift` carries prior/posterior exposure and
    # confidence and no return field, and `influence_matrix` consumes only shifts.
    # research.md discloses it and hands it to the probe; findings.md then records
    # that every probe trial landed in the top confidence bucket. README's
    # Limitations, introduced as "Named here rather than left for a reader to
    # find", omitted it.
    from dataclasses import fields

    from council.evaluation.persuasion import Shift

    limitations = flowed(text(README).split("## Limitations")[1])
    persuasion = (PROJECT_ROOT / "src" / "council" / "evaluation" / "persuasion.py").read_text(
        encoding="utf-8"
    )

    assert not [field.name for field in fields(Shift) if "return" in field.name]
    assert "forward_return" not in persuasion
    assert "Half the headline question has no measurement in the market arms." in limitations
    assert "no artefact joins them" in limitations
    assert "answered nowhere" in limitations


def test_the_hindsight_limitations_are_qualified_to_a_run_with_real_history() -> None:
    limitations = text(README).split("## Limitations")[1]
    hindsight = limitations.split("**Ticker selection carries hindsight.**")[1]

    assert "real history" in hindsight


# -- what the placebo actually varies ---------------------------------------------


def test_the_placebo_module_does_not_claim_only_the_day_differs() -> None:
    # The candidate filter keys on (date, ticker) and constrains only the date, so
    # in a two-ticker universe about half the donors are another instrument too.
    placebo = (PROJECT_ROOT / "src" / "council" / "debate" / "placebo.py").read_text(
        encoding="utf-8"
    )

    assert "One thing differs" not in placebo
    assert "often another instrument" in placebo


def test_the_placebo_pool_docstring_names_the_arm_the_sweep_actually_draws_from() -> None:
    # `sweep.placebo_pool_for` builds the pool from the independent arm, and its own
    # docstring argues that it must not be a debate arm -- a pool of debated points
    # holds contested days only, so the earliest contested day loses its donor in
    # the placebo arm alone. The module defining the placebo's contract said the
    # opposite of the decision the sweep makes and defends.
    placebo = (PROJECT_ROOT / "src" / "council" / "debate" / "placebo.py").read_text(
        encoding="utf-8"
    )
    sweep = (PROJECT_ROOT / "src" / "council" / "debate" / "sweep.py").read_text(
        encoding="utf-8"
    )

    assert "Assembled by the caller from a debate arm" not in placebo
    assert "from the **independent** arm" in placebo
    assert "placebo_pool_for" in placebo
    assert "str(Arm.INDEPENDENT)" in sweep.split("def placebo_pool_for")[1].split("\ndef ")[0]


def test_the_anonymity_claim_is_qualified_to_the_handle_the_code_enforces() -> None:
    # `peers_for` constructs the label; the rationale is copied unmodified, and the
    # only transform anywhere is `prompt._flatten`'s whitespace collapse and
    # truncation. No anonymisation audit of the completions archive exists, although
    # `store.py` and `config.py` both justify keeping the archive by one.
    source = (PROJECT_ROOT / "src" / "council" / "debate" / "peers.py").read_text(
        encoding="utf-8"
    )
    peers = " ".join(source.split())
    c5 = " ".join(text(CLAIMS).split("C5.")[1].split("C6.")[0].split())

    assert "**Anonymity of the handle.**" in peers
    assert "unvalidated model output" in peers
    assert "instruction rather than a check" in peers
    assert "no audit of the completions archive" in peers
    assert "rationale=view.rationale" in peers
    for phrase in ("handle", "instruction rather than a check", "no audit"):
        assert phrase in c5, phrase


@pytest.mark.parametrize("document", [README, RESEARCH])
def test_the_arms_table_says_the_placebo_may_change_the_instrument(document: Path) -> None:
    body = text(document)

    assert "(and possibly another instrument)" in body
    assert "instrument identity" in body


# -- the coverage figures are in one unit -----------------------------------------


@pytest.mark.parametrize("document", [README, RESEARCH])
def test_the_coverage_note_reports_dates_and_points_in_their_own_units(
    document: Path,
) -> None:
    # "60 of 461 decision dates ... 118 of 138" put two different units under one
    # label: `has_donor` refuses whole dates, and 118 of 138 counts (date, ticker)
    # points. A reader could reproduce neither figure from the other's definition.
    #
    # The six-month pair was then wrong as well as mismatched. "59 of 69" is not
    # the sentence's own quantity -- 69/138 is what the stored placebo arm happened
    # to cover, while the slice has 70 decision dates and 140 contested points and
    # every other arm keeps all of them. Recomputed on the surviving run:
    # `placebo_pool_for(independent, rotation-0)` holds 140 points over 70 dates,
    # `has_donor(..., required_seats=2, min_gap=60)` admits 20 points on 10 dates,
    # so 120 points on 60 dates lose their donor -- which is the quantity the
    # configured-range half of the sentence counts too.
    body = " ".join(text(document).split())

    assert "60 of 461 decision dates" not in body
    assert "60 of 461 dates (120 of 922 points)" in body
    assert "59 of 69 dates (118 of 138 contested points)" not in body
    assert "60 of 70 dates (120 of 140 contested points)" in body


def _six_month_donor_loss() -> tuple[int, int, int, int]:
    """What the shipped gap costs on the run the coverage sentence names.

    Returns ``(dates, points, lost_dates, lost_points)`` for the surviving
    two-model six-month run, measured through the shipped ``has_donor`` over the
    shipped ``placebo_pool_for``. The quoted pair was 59/69 and 118/138, which is
    the *stored* placebo arm's coverage rather than the slice's calendar.
    """
    import pandas as pd

    from council.debate.compositions import balanced_design
    from council.debate.sweep import has_donor, placebo_pool_for
    from council.domain.signal import Arm
    from council.evaluation.dispersion import contested_points
    from council.evaluation.frames import ARM

    frame = pd.read_parquet(SURVIVING_RUN)
    control = frame.loc[frame[ARM].astype(str) == str(Arm.INDEPENDENT)]
    contested = [point.point for point in contested_points(control, threshold=0.25)]
    committee = balanced_design(models=tuple(sorted(frame["model"].unique())))[0]
    pool = placebo_pool_for(frame, composition=committee)
    kept = {
        point
        for point in contested
        if has_donor(pool, point, required_seats=committee.size, min_gap=60)
    }
    lost = set(contested) - kept
    return (
        len({point[0] for point in contested}),
        len(contested),
        len({point[0] for point in lost}),
        len(lost),
    )


@pytest.mark.parametrize("document", [README, RESEARCH])
def test_the_six_month_coverage_pair_is_the_one_the_shipped_pre_flight_measures(
    document: Path,
) -> None:
    # The sentence says "the first 60 decision dates in either case" and then quoted
    # 59 of 69 for the six-month half -- its own sentence contradicted, because 69
    # and 138 are the stored placebo arm's coverage rather than the slice's
    # calendar. The slice holds 70 decision dates and 140 contested points, and the
    # arms it is differenced against keep every one of them.
    dates, points, lost_dates, lost_points = _six_month_donor_loss()
    body = " ".join(text(document).split())

    assert (dates, points) == (70, 140)
    assert (lost_dates, lost_points) == (60, 120)
    assert f"{lost_dates} of {dates} dates ({lost_points} of {points} contested points)" in body


def test_the_pre_flight_docstring_uses_the_same_units_as_the_documents() -> None:
    sweep = (PROJECT_ROOT / "src" / "council" / "debate" / "sweep.py").read_text(
        encoding="utf-8"
    )
    has_donor = sweep.split("def has_donor")[1].split("@dataclass")[0]

    assert "118 of 138 points" not in has_donor
    assert "120 of 140 contested points" in has_donor
    # One docstring carried two answers to one question: the `min_gap` paragraph
    # said 116 while the paragraph above it said 118, and the wrong one was attached
    # to the argument whose default is the hazard being described. Recomputed on
    # `docs/results/superseded/run-2models/decisions.parquet`, the pre-flight at gap
    # 0 holds for 138 of the 140 contested points and at gap 60 for 20, so the
    # disagreement is 118.
    assert "116 of 138" not in has_donor
    # 118 of 138 is a real figure and a *different* quantity from the coverage pair
    # above -- a gapless pre-flight admits 138 points, the shipped gap 20 -- so it
    # is stated against its own denominator rather than left to be read against the
    # 140 the sentence above counts.
    assert "118 of the 138 points a gapless pre-flight admits" in " ".join(has_donor.split())


def test_the_falling_range_in_findings_is_the_one_its_own_table_shows() -> None:
    # The paragraph opens "all four models" and then gives the falling range as
    # "0.03 to 0.30", dropping phi4's -0.20 and -0.15 -- the one model the section
    # exists to report, in the direction that makes the screen look tidier. CLAIMS
    # C13 has the correct range.
    section = text(FINDINGS).split("### What the corrected check actually found")[1]
    paragraph = " ".join(section.split("\n\n")[1].split())
    c13 = " ".join(text(CLAIMS).split("C13.")[1].split("C14.")[0].split())

    assert "-0.20 to +0.30" in c13
    assert "roughly 0.03 to 0.30" not in paragraph
    assert "between -0.20 and +0.30" in paragraph
    assert "runs backwards for phi4" in paragraph
    assert "forty times more in the extreme case" not in paragraph


def test_the_readme_does_not_claim_a_decision_on_every_session_the_prices_hold() -> None:
    # `agents.runner.decision_calendar` drops the first `lookback_days - 1` sessions
    # as warm-up, so at the shipped configuration 59 of the price table's 520
    # sessions carry no decision -- the 461 the README quotes elsewhere.
    body = " ".join(text(README).split())

    assert "on every session the price table holds" not in body
    assert "full `lookback_days` window behind it" in body
    assert "are warm-up" in body


# -- there is no frequency axis in this repository --------------------------------


def test_no_document_names_a_frequency_arm_the_code_does_not_have() -> None:
    # There is no frequency setting, no resampling and no non-daily path in src/.
    # "every frequency arm" and "the two-year daily arm" both imply a set of them.
    for document in (README, FINDINGS):
        body = " ".join(text(document).split())
        assert "frequency arm" not in body, document.name
        assert "two-year daily arm" not in body, document.name
        assert "two-year run at daily decision frequency" in body, document.name

    assert "only decision frequency this repository implements" in " ".join(
        text(README).split()
    )


# -- what the rationale-only arm actually withholds -------------------------------


@pytest.mark.parametrize("document", [README, RESEARCH])
def test_the_rationale_only_arm_is_not_described_as_removing_numbers(document: Path) -> None:
    # `_render_peer` drops the structured exposure field and nothing else; a figure
    # a peer wrote into its own prose reaches the reader unchanged. "no numbers"
    # claims the arm isolates anchoring when it only bounds it.
    body = text(document)

    assert "**no numbers**" not in body
    assert "stated exposure removed" in body


def test_the_only_numeric_redaction_in_the_prompt_module_is_the_exposure_field() -> None:
    # The claim above is about the code, so it is checked there too: if a numeric
    # redaction is ever added, this fails and the documents can be relaxed.
    prompt = (PROJECT_ROOT / "src" / "council" / "agents" / "prompt.py").read_text(
        encoding="utf-8"
    )
    peer_renderer = prompt.split("def _render_peer")[1].split("def _flatten")[0]

    assert "show_exposure" in peer_renderer
    assert "position" in peer_renderer
    # Nothing else in the renderer touches the rationale but the flattener.
    assert peer_renderer.count("_flatten(peer.rationale)") == 1


# -- the arms do not cover the same points ----------------------------------------


@pytest.mark.parametrize("document", [README, RESEARCH])
def test_the_placebo_coverage_gap_is_stated_where_the_arms_are_described(
    document: Path,
) -> None:
    # research.md asserted the peer rendering was the only thing that could differ
    # between arms. The placebo also needs a donor `placebo_min_gap_sessions` back,
    # so points without one are abandoned in that arm alone and backfilled with the
    # independent view -- the placebo curve is part control by construction.
    body = text(document)

    assert "coverage_note" in body
    assert "placebo_min_gap_sessions" in body


def test_research_no_longer_claims_the_arm_is_the_only_input_that_can_differ() -> None:
    assert "cannot come from anything else" not in text(RESEARCH)


# -- the balanced design answers fewer questions than the grid --------------------


@pytest.mark.parametrize("document", [README, RESEARCH])
def test_the_balanced_design_is_not_claimed_equivalent_to_the_full_grid(
    document: Path,
) -> None:
    # compositions.py's own docstring says what is given up: the interaction between
    # particular pairings. Both documents printed "the same questions" unqualified.
    body = text(document)

    assert "the same questions" not in body
    assert "main effects" in " ".join(body.split())


def test_claims_states_separability_rather_than_equivalence() -> None:
    c3 = text(CLAIMS).split("C3.")[1].split("C4.")[0]

    assert "separates model main effects" in c3
    assert "not\n    equivalent" in c3 or "not equivalent" in c3


# -- the contested gate gates almost nothing --------------------------------------


GATE_SAVING_FILES: tuple[Path, ...] = (
    README,
    PROJECT_ROOT / "src" / "council" / "config.py",
    PROJECT_ROOT / "src" / "council" / "evaluation" / "dispersion.py",
    PROJECT_ROOT / "src" / "council" / "debate" / "protocol.py",
)
"""Everywhere the withdrawn saving was asserted. The repair reached README only,
and the guard checked README only, so `config.py` -- the file the pre-registration
points readers at -- went on stating it as fact."""


@pytest.mark.parametrize("path", GATE_SAVING_FILES, ids=lambda path: path.name)
def test_no_file_claims_the_dispersion_gate_saves_the_budget(path: Path) -> None:
    # `is_contested` is `std > limit or is_split`, and the personas are crossed on
    # stance precisely so opposite signs appear on the same series, so `is_split`
    # fires on nearly every point. findings.md already recorded that the saving did
    # not materialise; README:198 now calls it "an open question rather than a
    # settled one" and CLAIMS C14 draws no conclusion, but three source files still
    # asserted it. The assertion is on the present tense: the README's surviving
    # "was expected to *be* most of the compute budget" is the withdrawal, and the
    # three source files each said the gate *is* it.
    body = " ".join(text(path).split())

    assert "is most of the compute budget" not in body, path.name
    assert "so it saved nothing" in body, path.name


def test_the_readme_does_not_claim_the_dispersion_gate_saves_the_budget() -> None:
    body = text(README)

    assert "also most of the compute budget" not in body
    assert "so it saved nothing" in body


# -- the compute arithmetic -------------------------------------------------------


def test_the_full_grid_estimate_is_the_arithmetic_the_planner_prices_it_with() -> None:
    # The earlier repair fixed the understatement by adopting a figure that assumed
    # one inference per second and no parallelism -- neither of which the repo's own
    # planner believes. `StagePlan.seconds` divides by parallelism, and
    # `_debate_stage` sets that to the number of distinct base models, which is four
    # for the full grid.
    days = 2_048_000 * SECONDS_PER_INFERENCE / 4 / 86_400

    assert pytest.approx(8.9, abs=0.1) == days
    assert "one to two weeks" not in text(RESEARCH)
    assert "three and a half weeks" not in text(RESEARCH)
    assert f"{days:.1f} days" in text(RESEARCH)
    assert "SECONDS_PER_INFERENCE = 1.5" in text(RESEARCH)
    assert "over three weeks of continuous compute" not in text(README)
    assert "over three weeks of continuous generation" not in text(COMPOSITIONS)


def test_the_full_grid_estimate_counts_every_arm_the_sweep_runs() -> None:
    # `compositions.py` was repaired on one side only: it multiplied the
    # eight-committee figure by the three treatment arms and left the 256-committee
    # side at one arm's cost, and the README carried the unfixed version with no
    # per-arm caveat at all. `run_debate_arms` loops over every entry of
    # `TREATMENT_ARMS`, so the grid this design would actually run costs three times
    # the quoted figure.
    from council.planning import TREATMENT_ARMS

    per_arm = 256 * 8 * 1_000
    full_grid = per_arm * len(TREATMENT_ARMS)
    days = full_grid * SECONDS_PER_INFERENCE / 4 / 86_400

    assert (per_arm, full_grid) == (2_048_000, 6_144_000)
    assert pytest.approx(26.7, abs=0.1) == days
    assert "about nine days of continuous compute" not in text(README)
    assert "**six million inferences**" in text(README)
    assert "about twenty-seven days of continuous compute" in text(README)
    assert "about nine\ndays of continuous generation" not in text(COMPOSITIONS)
    assert "6,144,000" in text(COMPOSITIONS)
    assert "about\ntwenty-seven\ndays of continuous generation" in text(COMPOSITIONS)
    # The ratio is unaffected -- both sides of it are per-arm figures -- so it stays.
    assert "one thirty-second of the compute" in text(README)


# -- superseded numbers are marked as superseded ----------------------------------


@pytest.mark.parametrize(
    "heading", ["## 3. Cost, planned and measured", "## 5. The main experiment"]
)
def test_a_section_built_on_deleted_artefacts_says_so_under_its_heading(heading: str) -> None:
    # Withdrawing Findings 2-4 for the threshold defect left the section's own
    # figures -- 14,496 decisions, 3,344 debates, the whole shift table -- standing
    # as current results of a run whose artefacts are gone and whose window was
    # never chosen.
    section = text(FINDINGS).split(heading)[1].split("\n## ")[0]
    preamble = section.split("\n### ")[0]

    assert "Superseded in full" in preamble
    # "deleted" was the word, and it is false of a file that is on disk and cited
    # three lines further down as the source of the recomputed table.
    assert "superseded" in preamble.lower()
    assert "six-month" in preamble


def test_the_claims_register_marks_the_measurement_claims_as_retained_for_the_record() -> None:
    measurements = text(CLAIMS).split("## About the measurements")[1].split("C8.")[0]

    assert "superseded" in measurements
    assert "C8-C15" in measurements


def test_no_document_calls_the_surviving_artefacts_deleted() -> None:
    # Four files said the evidence behind the superseded run "has been deleted"
    # while `docs/results/superseded/run-2models/decisions.parquet` is present,
    # readable and cited two lines away as the source of a recomputed table. A
    # reader told the evidence is gone does not check numbers they are asked to
    # take on trust -- and README:329 said the opposite.
    #
    # `debate/sweep.py` was left out of the first repair and out of the guard that
    # pinned it, so `has_donor`'s docstring went on calling the same file deleted
    # while quoting figures measured from it.
    assert SURVIVING_RUN.exists()

    for path in (
        FINDINGS,
        CLAIMS,
        PROJECT_ROOT / "src" / "council" / "config.py",
        PROJECT_ROOT / "src" / "council" / "debate" / "sweep.py",
    ):
        body = " ".join(text(path).split())
        assert "have been deleted" not in body, path.name
        assert "artefacts have been deleted" not in body, path.name
        assert "the deleted run" not in body, path.name
    assert "deleted six-month-window runs" not in " ".join(text(CLAIMS).split())


# -- every cited artefact exists --------------------------------------------------


@pytest.mark.parametrize("document", DOCUMENTS)
def test_no_document_cites_the_artefact_where_it_no_longer_is(document: Path) -> None:
    assert f"`{MOVED_ARTEFACT}" not in text(document)


def test_the_guard_that_pins_findings_to_the_code_reads_a_file_that_exists() -> None:
    guard = (PROJECT_ROOT / "tests" / "test_docs_findings.py").read_text(encoding="utf-8")

    assert MOVED_ARTEFACT not in guard.replace("superseded", "")
    assert "superseded" in guard


@pytest.mark.parametrize("document", [FINDINGS, CLAIMS])
def test_no_document_cites_an_artefact_path_that_does_not_exist(document: Path) -> None:
    body = text(document)
    cited = {
        line.split("`")[index]
        for line in body.splitlines()
        for index in range(1, len(line.split("`")), 2)
        if "docs/results" in line.split("`")[index]
    }
    assert cited, f"{document.name} cites no artefact, so this test has stopped checking"

    for path in cited:
        assert (PROJECT_ROOT / path.rstrip("/")).exists(), f"{document.name} cites {path}"


# -- the contested gate is measured on the pooled grid ----------------------------

POOLED_COMMITTEE_SHARE = "449 of 1,120"
"""The contested share recomputed per committee on
``docs/results/superseded/run-2models/decisions.parquet`` at the shipped
``dispersion_threshold``: rotations 56, 125, 70 and 123 of 140, uniforms 3, 11, 30 and
31 of 140. The published 100% is the share over the pooled independent arm."""


@pytest.mark.parametrize("document", [README, RESEARCH, FINDINGS, CLAIMS])
def test_the_contested_share_is_named_as_a_pooled_grid_figure(document: Path) -> None:
    # The gate is justified per committee -- "a conversation cannot change *the
    # committee's* decision" -- and the 100% quoted as proof it saves nothing is
    # measured once over the whole independent arm, pooled across every model and
    # persona, then applied unchanged to all eight committees. At the unit the
    # justification is stated in, the gate is not vacuous.
    body = flowed(text(document))

    assert "pooled" in body, document.name
    assert POOLED_COMMITTEE_SHARE in body, document.name


def test_the_claims_register_no_longer_concludes_the_gate_saves_nothing() -> None:
    c14 = text(CLAIMS).split("C14.")[1].split("C15.")[0]

    assert "optimisation saves nothing here" not in c14
    assert "pooled-grid shares" in " ".join(c14.split())


def test_the_planner_docstring_does_not_cite_the_pooled_share_as_a_committee_share() -> None:
    # `ASSUMED_CONTESTED_SHARE` justifies itself from the same 100%, and the plan it
    # feeds is spent per committee.
    planning = (PROJECT_ROOT / "src" / "council" / "planning.py").read_text(encoding="utf-8")
    docstring = planning.split("ASSUMED_CONTESTED_SHARE: Final = 1.0")[1].split('"""')[1]

    assert "pooled" in docstring
    assert POOLED_COMMITTEE_SHARE in docstring


# -- the declaration was amended, and says so -------------------------------------

ORIGINAL_STATISTIC = (
    "The share of contested decision points at which an agent shifted by more than "
    "`0.20` exposure, partitioned by the confidence it reported *before* seeing its peers."
)
"""``git show 001c8ff:README.md``, verbatim. The working-tree README declares a
different unit and a different bar, and both changes were made after the results
committed in ``fa436fa`` and ``98a4020``."""


def test_the_readme_quotes_the_statistic_it_originally_declared() -> None:
    # The section presented its current wording as what was declared. The unit
    # changed from decision points to agent-conversation observations -- a ~32x
    # change of denominator -- and the change was undisclosed.
    body = flowed(text(README))

    assert "Amendments." in body
    assert flowed(ORIGINAL_STATISTIC) in body
    assert "001c8ff" in body


def test_both_amendments_are_named_beside_the_results_they_postdate() -> None:
    amendments = flowed(text(README).split("**Amendments.**")[1].split("\n\n")[0])

    for commit in ("fa436fa", "98a4020"):
        assert commit in amendments, commit
    assert "The unit*" in amendments or "unit* changed" in amendments
    assert "bar* changed" in amendments
    assert "persuasion.shifts" in amendments
    assert "threshold.py" in amendments


def test_the_third_amendment_the_declaration_made_is_disclosed_with_the_other_two() -> None:
    # `git show 001c8ff:README.md` headed the equity comparison **Primary
    # comparison.** beside a different **Primary statistic.**; the working tree
    # demotes it to the secondary declared outcome that "does not decide the
    # result". That changes which declared quantity decides -- the thing a
    # pre-registration exists to fix -- and the Amendments block claimed
    # completeness at two.
    header = text(README).split("## Pre-registered primary comparison")[1].split(">")[0]
    amendments = flowed(text(README).split("**Amendments.**")[1].split("\n\n")[0])

    assert "amended twice" not in header
    assert "both changes are named there" not in header
    assert "three times" in header
    assert "all three changes are named there" in header
    assert "Two, both to the statistic" not in amendments
    assert "Three, all made after" in amendments
    assert "deciding outcome* changed" in amendments
    assert "001c8ff" in amendments


def test_the_opening_italic_no_longer_presents_the_amended_text_as_declared() -> None:
    header = text(README).split("## Pre-registered primary comparison")[1].split(">")[0]

    assert "The primary comparison and the two thresholds it is stated in were declared" \
        not in header
    assert "amended" in header


# -- one primary outcome, not two -------------------------------------------------


def test_the_readme_names_one_primary_outcome_and_one_secondary() -> None:
    # Two declared quantities both labelled the declared one leaves no rule for
    # which decides the result, so whichever comes out favourable can be reported
    # as the pre-registered one.
    body = flowed(text(README))

    assert "This is the single primary outcome." in body
    assert "Secondary declared outcome." in body
    assert "No multiplicity correction is applied across the two." in body
    # The equity comparison was declared as "Primary comparison" beside a "Primary
    # statistic" that is a different quantity, with no rule for which decides. The
    # heading survives only inside the Amendments block, quoted as what `001c8ff`
    # said -- which is the disclosure, not the claim.
    declaration, _, amendments = text(README).partition("**Amendments.**")

    assert "**Primary comparison.**" not in declaration
    assert "**Primary comparison.**" in amendments


def test_the_primary_outcome_is_not_qualified_by_a_cost_convention() -> None:
    # "Net of costs" cannot qualify a per-agent shift rate: it involves no returns
    # and never touches settings.total_cost_bps or any equity curve.
    primary = flowed(text(README).split("**Primary statistic.**")[1].split("**Direction.**")[0])

    assert "net of costs" not in primary


EQUITY_COMPARISON_MODULES: tuple[str, ...] = (
    "report.py",
    str(Path("app") / "panels.py"),
    # The first repair reached the two modules above -- the two this test
    # parametrised -- and stopped there. Four more called the debate-versus-
    # independent equity comparison "the pre-registered comparison" or "the primary
    # comparison", including the module that computes it, so the source told a
    # reader the opposite of the declaration the README makes.
    "scoring.py",
    "arms.py",
    str(Path("app") / "curves.py"),
    str(Path("app") / "dashboard.py"),
)


@pytest.mark.parametrize("module", EQUITY_COMPARISON_MODULES)
def test_the_cli_and_the_dashboard_call_the_equity_comparison_secondary(module: str) -> None:
    source = (PROJECT_ROOT / "src" / "council" / module).read_text(encoding="utf-8")

    assert "Pre-registered comparison:" not in source
    assert "primary comparison:" not in source
    # Not the whole phrase: two of these modules qualify it as the "secondary
    # declared (equity) comparison", to say which of the two declared quantities
    # the aggregation rule applies to.
    assert "econdary declared" in source


@pytest.mark.parametrize("module", EQUITY_COMPARISON_MODULES[2:])
def test_no_module_names_the_equity_comparison_the_pre_registered_one(module: str) -> None:
    # The README demotes the equity comparison to "Secondary declared outcome" and
    # says it "does not decide the result", naming the shift rate as "the single
    # primary outcome". These four modules each attached the deciding label to the
    # equity comparison instead -- `scoring.PRIMARY_RULE` going as far as to say the
    # declared comparison is *stated in* an aggregation rule, which the shift rate
    # does not depend on at all.
    source = (PROJECT_ROOT / "src" / "council" / module).read_text(encoding="utf-8")

    assert "pre-registered comparison" not in source
    assert "The primary comparison is" not in source


# -- which stage is actually the expensive one ------------------------------------


def test_the_readme_does_not_call_generation_the_only_expensive_stage() -> None:
    # The debate sweep is the larger bill by an order of magnitude and does not
    # checkpoint on (model, persona, ticker): `_Sweep.group` checkpoints per
    # (composition, arm, ticker), so a reader planning a resume was given the wrong
    # granularity for the stage that will actually be interrupted.
    body = flowed(text(README))

    assert "Generation is the only expensive stage" not in body
    assert "(model, persona, ticker)" in body
    assert "(committee, arm, ticker)" in body


# -- the front page describes this repository -------------------------------------


def test_the_status_list_does_not_leave_finished_work_unchecked() -> None:
    status = text(README).split("## Status")[1]

    assert "- [ ] End-to-end dry run on the mock provider" not in status
    assert "- [x] End-to-end dry run on the mock provider" in status
    assert "- [ ] Dashboard and the results write-up" not in status
    assert "- [x] Dashboard and the results write-up" in status


def test_the_status_list_says_both_completed_runs_are_superseded() -> None:
    status = flowed(text(README).split("## Status")[1])

    assert "docs/results/superseded/" in status
    assert "There is no current run." in status


def test_the_status_list_does_not_promise_artefacts_for_the_four_model_run() -> None:
    # `docs/results/superseded/` holds the two-model run and nothing else, and no
    # four-model artefact exists in the tree or in git history -- but CLAIMS C14 and
    # findings section 4 report on that run, so a reader was told they could inspect
    # evidence that is not there.
    status = flowed(text(README).split("## Status")[1])
    kept = PROJECT_ROOT / "docs" / "results" / "superseded"
    superseded = {path.name for path in kept.iterdir()}

    assert superseded == {"results-2models-6months.json", "run-2models"}, superseded
    assert "Their artefacts are under" not in status
    assert "two-model run's artefacts are under" in status
    assert "four-model run's" in status
    assert "cannot be rechecked" in status


def test_the_layout_lists_every_package_under_src() -> None:
    layout = text(README).split("## Layout")[1].split("```")[1]
    packages = {
        path.name
        for path in (PROJECT_ROOT / "src" / "council").iterdir()
        if path.is_dir() and (path / "__init__.py").is_file()
    }

    for package in packages:
        assert f"{package}/" in layout, package


def test_the_readme_says_how_to_start_the_dashboard_it_ships() -> None:
    assert "streamlit run src/council/app/dashboard.py" in text(README)

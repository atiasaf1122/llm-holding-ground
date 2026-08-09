"""The write-up prints the numbers the shipped code produces, and withdraws what
has been withdrawn.

``docs/findings.md`` is prose beside a stored artefact, which is the one place a
correction can be applied to the code and to ``docs/CLAIMS.md`` and still leave a
published table stating the superseded result. It happened once: ``cbf6a55``
replaced a bare ``distance >= threshold`` with
:func:`council.evaluation.threshold.meets`, rewrote C8-C11 and added D2-D5, and left
this document printing the pre-fix rates and reading conclusions off them.

So the table is recomputed here from the same stored decisions rather than trusted,
and the findings the claims register has withdrawn are checked to say so.
"""

from __future__ import annotations

import re

import pandas as pd
import pytest

from council.config import PROJECT_ROOT
from council.domain.signal import Arm
from council.evaluation.persuasion import shift_rate_by_confidence, shifts
from council.scoring import rows_in_arm

FINDINGS = PROJECT_ROOT / "docs" / "findings.md"
CLAIMS = PROJECT_ROOT / "docs" / "CLAIMS.md"
DECISIONS = (
    PROJECT_ROOT / "docs" / "results" / "superseded" / "run-2models" / "decisions.parquet"
)
"""Under ``superseded/``: the run these figures came from used a six-month window that
was never chosen, so its artefacts were retired there rather than deleted. This test
still recomputes the published table from them, because the table is published."""

PUBLISHED_ARMS: tuple[Arm, ...] = (Arm.DEBATE, Arm.DEBATE_PLACEBO, Arm.DEBATE_RATIONALE_ONLY)
"""The three columns of the shift-rate table, in the order it prints them."""

HEADING = "### Shift rate, by the confidence held before seeing any peer"

ROW = re.compile(r"^\|\s*(\d\.\d\d)\s")
"""A table row, keyed by the lower edge of the confidence band it reports. Stopping
at the whitespace after two decimals rather than at the label's dash, which is an en
dash in the document and has no business being pasted into a pattern."""

CELL = re.compile(r"(\d\.\d{3})\**\s*\((\d+)\)")
"""``rate (count)``, with the bold markers the placebo column wears ignored."""

Cells = list[tuple[str, int]]


@pytest.fixture(scope="module")
def published() -> pd.DataFrame:
    return pd.read_parquet(DECISIONS)


def published_table() -> dict[float, Cells]:
    """Every ``rate (count)`` in the shift-rate table, keyed by the band's lower edge."""
    document = FINDINGS.read_text(encoding="utf-8")
    assert HEADING in document, "the shift-rate table lost the heading this test finds it by"
    rows: dict[float, Cells] = {}
    for line in document.split(HEADING)[1].splitlines():
        match = ROW.match(line)
        if match is None:
            if rows:
                break
            continue
        rows[float(match.group(1))] = [(rate, int(count)) for rate, count in CELL.findall(line)]
    return rows


def recomputed(decisions: pd.DataFrame, arm: Arm) -> dict[float, tuple[str, int]]:
    """What the shipped code makes of the same artefact, keyed the same way."""
    report = shift_rate_by_confidence(shifts(rows_in_arm(decisions, arm), threshold=0.20))
    return {
        band.band.lower: (f"{band.shift_rate:.3f}", band.count)
        for band in report.bands
        if band.shift_rate is not None
    }


def test_the_published_shift_table_is_what_the_shipped_code_computes(
    published: pd.DataFrame,
) -> None:
    table = published_table()
    assert len(table) == 4, "the table lost or gained a confidence band"

    for column, arm in enumerate(PUBLISHED_ARMS):
        printed = {lower: cells[column] for lower, cells in table.items()}
        assert printed == recomputed(published, arm), arm


@pytest.mark.parametrize(("finding", "withdrawal"), [(2, "D2"), (3, "D3"), (4, "D4")])
def test_a_finding_the_claims_register_withdrew_does_not_still_state_its_result(
    finding: int, withdrawal: str
) -> None:
    # A withdrawal recorded only in CLAIMS.md leaves the conclusion standing in the
    # document a reader actually reads.
    document = FINDINGS.read_text(encoding="utf-8")
    heading = next(
        line for line in document.splitlines() if line.startswith(f"### Finding {finding} ")
    )
    body = document.split(heading)[1].split("\n### ")[0]

    assert "withdrawn" in heading.lower()
    assert withdrawal in body


def test_the_placebo_contamination_is_recorded_against_the_claim_that_rests_on_it() -> None:
    # The published placebo drew its donors from inside the lookback window, so the
    # arm C8 is read off was not inert. Recorded as a defect, and C8 pointed at it.
    claims = CLAIMS.read_text(encoding="utf-8")
    c8 = claims.split("C8.")[1].split("C9.")[0]

    assert "\nD6." in claims
    assert "D6" in c8


def test_the_write_up_no_longer_calls_the_placebo_day_unrelated() -> None:
    assert "about a different day entirely" not in FINDINGS.read_text(encoding="utf-8")


# -- the register does not re-assert what the write-up withdrew --------------------


@pytest.mark.parametrize(
    ("claim", "following", "finding"), [("C9.", "C10.", 2), ("C10.", "C11.", 3)]
)
def test_a_claim_over_a_withdrawn_column_states_no_conclusion(
    claim: str, following: str, finding: int
) -> None:
    # findings.md says of both columns: "The conclusion is withdrawn rather than
    # replaced. Nothing is asserted here about what the corrected column says." The
    # register quoted the same four figures and drew the conclusion anyway, so a
    # reader could not tell whether the project stood behind the reading.
    body = CLAIMS.read_text(encoding="utf-8").split(claim)[1].split(following)[0]

    assert "therefore" not in body
    assert "No conclusion" in body
    assert f"Finding {finding} is withdrawn" in " ".join(body.split())


def test_c12_does_not_order_two_models_on_a_difference_of_zero_events() -> None:
    # Both models capitulated exactly once on the probe (1/22 and 1/21), so the
    # rates differ only because the denominators do; and the market ordering
    # reverses in the placebo, which is the disagreement D4 cites.
    claims = CLAIMS.read_text(encoding="utf-8")
    c12 = claims.split("C12.")[1].split("C13.")[0]

    assert "held facts better" not in c12
    assert "UNSUPPORTED" in c12
    assert "D7" in c12
    assert "\nD7." in claims


def test_the_probe_reversal_subsection_is_withdrawn_in_the_write_up() -> None:
    document = FINDINGS.read_text(encoding="utf-8")
    heading = next(
        line for line in document.splitlines() if line.startswith("### The reversal against")
    )
    body = document.split(heading)[1].split("\n### ")[0].split("\n## ")[0]

    assert "withdrawn" in heading.lower()
    assert "D7" in body


def test_c13_carries_its_sample_rather_than_a_universal_quantifier() -> None:
    # Four models, two up-drifts and two down-drifts of one synthetic series, 16
    # calls each. Two falling windows cannot license "every", "none" or "whatever
    # persona" -- and the claim was then reused as a property of the study period.
    c13 = " ".join(CLAIMS.read_text(encoding="utf-8").split("C13.")[1].split("C14.")[0].split())

    assert "Every model separates" not in c13
    assert "whatever persona it wears" not in c13
    assert "16-call screen" in c13
    assert "not a demonstration" in c13


def test_the_write_up_softens_the_same_sentence_c13_carried() -> None:
    section = FINDINGS.read_text(encoding="utf-8").split("### What the corrected check")[1]
    body = " ".join(section.split("\n### ")[0].split())

    assert "none of these models will go long into a drawdown" not in body
    assert "hypothesis" in body


def test_c3_states_the_balance_property_the_design_actually_has() -> None:
    # rotations() puts each model at each persona once; uniform_references() puts it
    # there once more. Across all eight committees the count is two, not one.
    c3 = " ".join(CLAIMS.read_text(encoding="utf-8").split("C3.")[1].split("C4.")[0].split())

    assert "at every persona exactly once" not in c3
    assert "the same number of times" in c3
    assert "uniform references" in c3

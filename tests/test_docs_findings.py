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
DECISIONS = PROJECT_ROOT / "docs" / "results" / "run-2models" / "decisions.parquet"

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

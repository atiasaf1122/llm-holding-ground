"""The declaration on the page is the declaration in the repository.

The section extractor is tested against a hand-written document, and then the
real README is read once -- which is what pins the two headings. A renamed
heading is not a cosmetic change here: it is the pre-registration silently
vanishing from the top of the results page.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from council.app.preregistration import (
    PRIMARY_HEADING,
    QUESTION_HEADING,
    read_preregistration,
    section,
)
from council.config import PROJECT_ROOT

DOCUMENT = """# Title

Preamble.

## The question

> Does it hold?

More prose.

---

## Pre-registered primary comparison

The comparison.

### A detail

Still inside the comparison.

## Design

Something else.
"""


def test_a_section_is_the_body_under_its_heading() -> None:
    assert section(DOCUMENT, "## The question").startswith("> Does it hold?")


def test_a_section_stops_at_the_next_heading_of_the_same_level() -> None:
    assert "The comparison" not in section(DOCUMENT, "## The question")


def test_a_section_keeps_a_deeper_heading_inside_it() -> None:
    body = section(DOCUMENT, PRIMARY_HEADING)

    assert "### A detail" in body
    assert "Something else" not in body


def test_the_thematic_break_between_sections_is_not_part_of_either() -> None:
    # It belongs to the document's layout; rendered in a panel it draws a rule
    # across the bottom of a card.
    assert not section(DOCUMENT, QUESTION_HEADING).endswith("---")


def test_a_section_is_trimmed_of_the_blank_lines_around_it() -> None:
    body = section(DOCUMENT, QUESTION_HEADING)

    assert body == body.strip()


def test_a_renamed_heading_raises_rather_than_rendering_an_empty_panel() -> None:
    with pytest.raises(ValueError, match="no heading"):
        section(DOCUMENT, "## The Question")


def test_something_that_is_not_a_heading_is_refused() -> None:
    with pytest.raises(ValueError, match="not a markdown heading"):
        section(DOCUMENT, "The question")


def test_a_missing_readme_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="no README"):
        read_preregistration(tmp_path / "README.md")


def test_the_real_readme_still_carries_both_declared_passages() -> None:
    preregistration = read_preregistration(PROJECT_ROOT / "README.md")

    assert "confident agent is contradicted" in preregistration.question
    # The statistic is the primary outcome; the equity comparison is declared
    # beside it as the secondary one, and the section carries both.
    assert "Primary statistic" in preregistration.primary_comparison
    assert "Secondary declared outcome" in preregistration.primary_comparison
    assert "0.20" in preregistration.primary_comparison

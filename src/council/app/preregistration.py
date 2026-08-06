"""The declaration, lifted out of the README rather than retyped.

The dashboard shows what was declared before it shows a single number. That is
only worth anything if the text on the page is the text in the repository, so it
is extracted from ``README.md`` at render time instead of being copied into a
string here. A copy drifts, and a drifted pre-registration is worse than none: it
is a claim about what was declared that nobody can check.

A renamed heading therefore raises. The alternative -- rendering an empty panel --
would put the results on screen with the declaration silently missing, which is
exactly the failure this panel exists to prevent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

QUESTION_HEADING: Final = "## The question"
PRIMARY_HEADING: Final = "## Pre-registered primary comparison"

_HEADING = re.compile(r"^(#{1,6})\s")
_THEMATIC_BREAK: Final = "---"


def section(markdown: str, heading: str) -> str:
    """The body under one ATX heading, up to the next heading of the same level or higher.

    Args:
        heading: the heading line in full, hashes included, so the level is read
            from the document rather than guessed.

    Raises:
        ValueError: if the heading is absent, or is not an ATX heading.
    """
    match = _HEADING.match(heading)
    if match is None:
        raise ValueError(f"{heading!r} is not a markdown heading")
    level = len(match.group(1))

    lines = markdown.splitlines()
    try:
        start = lines.index(heading)
    except ValueError:
        raise ValueError(f"README has no heading {heading!r}") from None

    body: list[str] = []
    for line in lines[start + 1 :]:
        found = _HEADING.match(line)
        if found is not None and len(found.group(1)) <= level:
            break
        body.append(line)
    return _trim(body)


def _trim(lines: list[str]) -> str:
    """Drop the blank lines and the thematic break that separate one section from the next.

    The break belongs to the document's layout rather than to the section, and
    rendered inside a panel it draws a rule across the bottom of a card.
    """
    trimmed = list(lines)
    while trimmed and (not trimmed[-1].strip() or trimmed[-1].strip() == _THEMATIC_BREAK):
        trimmed.pop()
    while trimmed and not trimmed[0].strip():
        trimmed.pop(0)
    return "\n".join(trimmed)


@dataclass(frozen=True, slots=True)
class PreRegistration:
    """The two passages a reader must meet before any result."""

    question: str
    """The headline question, and the second one that comes free with it."""

    primary_comparison: str
    """The comparison and the statistic, as declared before any result existed."""


def read_preregistration(readme_path: Path) -> PreRegistration:
    """Extract both passages from a README on disk.

    Raises:
        FileNotFoundError: if the README is not there.
        ValueError: if either heading has been renamed.
    """
    if not readme_path.is_file():
        raise FileNotFoundError(f"no README at {readme_path}")
    markdown = readme_path.read_text(encoding="utf-8")
    return PreRegistration(
        question=section(markdown, QUESTION_HEADING),
        primary_comparison=section(markdown, PRIMARY_HEADING),
    )

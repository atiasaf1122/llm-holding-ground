"""Half-open bands over a [0, 1] score, shared by the two questions that need them.

Calibration asks whether confidence predicts being right. Persuasion asks whether
confidence protects a position. Those two curves are only readable side by side if
they are cut at the same places, so the cut lives here rather than twice.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Final

DEFAULT_EDGES: Final[tuple[float, ...]] = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
"""Five equal bands. Wide enough that a two-year run puts a usable count in each."""


@dataclass(frozen=True, slots=True)
class Band:
    """One half-open interval, except the last, which closes so that 1.0 lands."""

    lower: float
    upper: float
    closed_upper: bool

    def contains(self, value: float) -> bool:
        if value < self.lower:
            return False
        return value <= self.upper if self.closed_upper else value < self.upper

    @property
    def label(self) -> str:
        closing = "]" if self.closed_upper else ")"
        return f"[{self.lower:.2f}, {self.upper:.2f}{closing}"


def make_bands(edges: Sequence[float] = DEFAULT_EDGES) -> tuple[Band, ...]:
    """Build the bands between consecutive edges.

    Raises:
        ValueError: on fewer than two edges, or edges that do not strictly increase.
            Equal edges would create an empty band that silently swallows nothing
            while appearing in every report as a bucket with no observations.
    """
    if len(edges) < 2:
        raise ValueError("need at least two edges to form one band")
    for lower, upper in pairwise(edges):
        if upper <= lower:
            raise ValueError(f"edges must strictly increase; {lower} is followed by {upper}")

    last = len(edges) - 2
    return tuple(
        Band(lower=float(lower), upper=float(upper), closed_upper=index == last)
        for index, (lower, upper) in enumerate(pairwise(edges))
    )


def band_index(bands: Sequence[Band], value: float) -> int | None:
    """Which band a value falls in, or ``None`` if it falls outside all of them."""
    for index, band in enumerate(bands):
        if band.contains(value):
            return index
    return None

"""How far apart the agents are at a decision point.

This runs *before* generation, not after. It decides which days are worth debating
-- what skipping them saves has never been measured at the committee level; on the
pooled grid the contested share was 98.2%, so it saved almost nothing, and per
committee the same run gives 59.0%. It is also the
experiment's first checkpoint. If the agents barely disagree anywhere, there is
nothing for a debate to change and the headline question has no data behind it.
Better to learn that from the independent arm in an afternoon than from a finished
run.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from statistics import pstdev

import pandas as pd

from council.config import get_settings
from council.evaluation.frames import AgentKey, DecisionRow, PointKey, frame_to_rows


@dataclass(frozen=True, slots=True)
class Dispersion:
    """The spread of one decision point's exposures across its agents."""

    decision_date: date
    ticker: str
    agent_count: int
    exposure_std: float
    long_count: int
    short_count: int
    flat_count: int

    @property
    def point(self) -> PointKey:
        return (self.decision_date, self.ticker)

    @property
    def is_split(self) -> bool:
        """Whether agents disagree about direction, not merely about size."""
        return self.long_count > 0 and self.short_count > 0

    @property
    def minority_count(self) -> int:
        """How many agents are on the smaller side of a directional split."""
        return min(self.long_count, self.short_count)


def dispersion_by_point(frame: pd.DataFrame) -> tuple[Dispersion, ...]:
    """Measure the spread at every decision point in the frame, in date order.

    The frame is expected to hold one round of one arm -- ordinarily the independent
    arm, since that is what decides where a debate is worth running.

    Raises:
        ValueError: if an agent appears twice at the same point. That means the
            frame spans more than one arm or round, and the spread would then be
            measured across a mixture of conditions rather than across agents.
    """
    grouped: dict[PointKey, list[DecisionRow]] = defaultdict(list)
    for row in frame_to_rows(frame):
        grouped[row.point].append(row)

    return tuple(
        _measure(point, rows) for point, rows in sorted(grouped.items(), key=lambda item: item[0])
    )


def is_contested(dispersion: Dispersion, *, threshold: float | None = None) -> bool:
    """Whether this point is worth spending a debate on.

    Two independent sufficient conditions, and the second is not redundant. Agents
    that split on *direction* -- one long, one short -- can sit a hair either side of
    zero and produce a standard deviation below any sensible threshold, yet that is
    the sharpest disagreement the personas can express and precisely what the four
    of them were crossed to produce. Filtering on spread alone would throw those
    days away and leave the debate arm arguing about position sizing.

    Args:
        threshold: defaults to ``settings.dispersion_threshold``, declared in config
            before any debate ran. Passed explicitly only to test the boundary.
    """
    limit = get_settings().dispersion_threshold if threshold is None else threshold
    return dispersion.exposure_std > limit or dispersion.is_split


def contested_points(
    frame: pd.DataFrame, *, threshold: float | None = None
) -> tuple[Dispersion, ...]:
    """The subset of points a debate should be run on."""
    return tuple(
        dispersion
        for dispersion in dispersion_by_point(frame)
        if is_contested(dispersion, threshold=threshold)
    )


def contested_share(frame: pd.DataFrame, *, threshold: float | None = None) -> float:
    """Fraction of decision points that are contested; 0.0 for an empty frame.

    The number to look at before committing a GPU to the debate arm.
    """
    points = dispersion_by_point(frame)
    if not points:
        return 0.0
    contested = sum(1 for point in points if is_contested(point, threshold=threshold))
    return contested / len(points)


def _measure(point: PointKey, rows: list[DecisionRow]) -> Dispersion:
    seen: set[AgentKey] = set()
    for row in rows:
        if row.agent in seen:
            raise ValueError(
                f"{row.model}/{row.persona} appears twice at {point[0]} {point[1]}; "
                "the frame spans more than one arm or round"
            )
        seen.add(row.agent)

    exposures = [row.exposure for row in rows]
    return Dispersion(
        decision_date=point[0],
        ticker=point[1],
        agent_count=len(exposures),
        # Population, not sample. These agents are the whole committee rather than a
        # draw from a larger population of agents, and the sample form is undefined
        # for the committee of one that the control arm consists of.
        exposure_std=pstdev(exposures) if len(exposures) > 1 else 0.0,
        long_count=sum(1 for exposure in exposures if exposure > 0.0),
        short_count=sum(1 for exposure in exposures if exposure < 0.0),
        flat_count=sum(1 for exposure in exposures if exposure == 0.0),
    )

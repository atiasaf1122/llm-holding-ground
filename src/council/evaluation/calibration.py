"""Does an agent's self-reported confidence predict being right?

The headline question is whether a confident minority should be trusted over a
contrary majority. That is unanswerable without this module: "confident" has to
mean something before it can be worth deferring to. If the hit rate is flat across
confidence bands, then a debate result that says agents abandoned high-confidence
positions is a result about a number the models emit and not about conviction.

Confidence is never used to weight an aggregation -- see
:mod:`council.evaluation.aggregation`. It is measured here and only here.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from council.evaluation.buckets import DEFAULT_EDGES, Band, band_index, make_bands
from council.evaluation.frames import PointKey, frame_to_rows


@dataclass(frozen=True, slots=True)
class ConfidenceBucket:
    """One confidence band and how often decisions in it were right."""

    band: Band
    count: int
    hit_count: int

    @property
    def hit_rate(self) -> float | None:
        """``None`` rather than 0.0 when the band is empty.

        An empty band has no hit rate; reporting one as zero would draw a line
        through the calibration plot that no observation supports.
        """
        return self.hit_count / self.count if self.count else None


@dataclass(frozen=True, slots=True)
class CalibrationReport:
    """Hit rate by confidence band, plus the correlation over individual decisions."""

    buckets: tuple[ConfidenceBucket, ...]
    correlation: float | None
    """Pearson correlation between confidence and being right, across decisions.

    Computed on the individual decisions rather than on the band hit rates: five
    points fit a line far too well, and the bands exist to be looked at, not to be
    regressed through. ``None`` when it is undefined -- fewer than two scored
    decisions, or no variation in confidence or in outcome.
    """

    scored_count: int
    skipped_count: int
    """Decisions that could not be scored: a flat exposure makes "right" meaningless,
    and a point with no forward return has nothing to be right about. Reported so
    that a report built from almost nothing cannot look like a report built from
    everything."""

    @property
    def hit_rate(self) -> float | None:
        """Overall hit rate, ignoring confidence."""
        hits = sum(bucket.hit_count for bucket in self.buckets)
        return hits / self.scored_count if self.scored_count else None


def was_right(exposure: float, forward_return: float) -> bool | None:
    """Whether a directional call matched what the market then did.

    ``None`` when the question does not apply: a flat exposure expresses no
    direction to be right about, and a forward return of exactly zero gives nothing
    to be right against. Both are excluded rather than counted as misses, which
    would make cautious agents look wrong for declining to guess.
    """
    if exposure == 0.0 or forward_return == 0.0:
        return None
    return (exposure > 0.0) == (forward_return > 0.0)


def calibrate(
    frame: pd.DataFrame,
    forward_returns: Mapping[PointKey, float],
    *,
    edges: Sequence[float] = DEFAULT_EDGES,
) -> CalibrationReport:
    """Bucket decisions by confidence and measure the realised hit rate of each.

    Args:
        frame: stored decisions. Any arm and any round; the caller decides what
            population the question is being asked of, since calibration in the
            independent arm and calibration after a debate are different questions.
        forward_returns: the return each decision earned, keyed by decision point.
            See :func:`council.evaluation.frames.forward_returns_lookup` for the
            alignment this must already satisfy.
    """
    bands = make_bands(edges)
    counts = [0] * len(bands)
    hits = [0] * len(bands)
    confidences: list[float] = []
    outcomes: list[float] = []
    skipped = 0

    for row in frame_to_rows(frame):
        forward_return = forward_returns.get(row.point)
        hit = None if forward_return is None else was_right(row.exposure, forward_return)
        index = band_index(bands, row.confidence)
        if hit is None or index is None:
            skipped += 1
            continue
        counts[index] += 1
        hits[index] += int(hit)
        confidences.append(row.confidence)
        outcomes.append(float(hit))

    buckets = tuple(
        ConfidenceBucket(band=band, count=count, hit_count=hit)
        for band, count, hit in zip(bands, counts, hits, strict=True)
    )
    return CalibrationReport(
        buckets=buckets,
        correlation=pearson(confidences, outcomes),
        scored_count=len(confidences),
        skipped_count=skipped,
    )


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson correlation, or ``None`` where it is undefined.

    Undefined means what it says: with every confidence identical, or every outcome
    identical, the coefficient is 0/0. numpy returns NaN there and NaN propagates
    quietly through a report; ``None`` has to be handled.
    """
    if len(xs) != len(ys):
        raise ValueError(f"paired series must be the same length; got {len(xs)} and {len(ys)}")
    if len(xs) < 2:
        return None

    mean_x = math.fsum(xs) / len(xs)
    mean_y = math.fsum(ys) / len(ys)
    deviations = [(x - mean_x, y - mean_y) for x, y in zip(xs, ys, strict=True)]
    covariance = math.fsum(dx * dy for dx, dy in deviations)
    variance_x = math.fsum(dx * dx for dx, _ in deviations)
    variance_y = math.fsum(dy * dy for _, dy in deviations)
    if variance_x == 0.0 or variance_y == 0.0:
        return None
    return covariance / math.sqrt(variance_x * variance_y)

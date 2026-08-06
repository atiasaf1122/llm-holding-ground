"""Turning several agents' exposures into one committee exposure.

A single agent is a committee of one and runs the same code: ``mean([0.4])`` is
0.4. Special-casing the solo arm would put the control and the treatment on
different code paths, and a difference between the arms is the entire result -- it
must not be able to come from here.

Confidence is deliberately absent. Weighting by self-reported confidence before
measuring whether confidence is calibrated would answer that question with itself;
see :mod:`council.evaluation.calibration`, which measures it instead.
"""

from __future__ import annotations

import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Final

AggregationRule = Callable[[Sequence[float]], float]
"""Every rule has this shape, so an arm can be re-scored under a different one
without touching anything that produced the exposures."""


def mean(exposures: Sequence[float]) -> float:
    """The average view. One extreme agent moves the committee."""
    _require_members(exposures)
    return statistics.fmean(exposures)


def median(exposures: Sequence[float]) -> float:
    """The middle view, and the average of the middle two on an even committee.

    Robust where :func:`mean` is not: an agent that returned a flat exposure because
    its generation failed drags the mean and barely touches this.
    """
    _require_members(exposures)
    return float(statistics.median(exposures))


def direction_vote(exposures: Sequence[float]) -> float:
    """Majority direction, taken at full size.

    A vote settles direction and nothing else -- there is no coherent way to vote on
    a magnitude -- so the result is full exposure one way or the other. Sizing it at
    anything less would introduce a second parameter chosen by hand, and a knob set
    after seeing the equity curve is not a rule.

    A flat exposure abstains rather than voting for either side. A tie, including
    the all-flat committee, is flat: the committee reached no direction.
    """
    _require_members(exposures)
    longs = sum(1 for exposure in exposures if exposure > 0.0)
    shorts = sum(1 for exposure in exposures if exposure < 0.0)
    if longs > shorts:
        return 1.0
    if shorts > longs:
        return -1.0
    return 0.0


RULES: Final[Mapping[str, AggregationRule]] = {
    "direction_vote": direction_vote,
    "mean": mean,
    "median": median,
}
"""The rules by name, for reporting an arm under each of them."""

RULE_NAMES: Final[tuple[str, ...]] = tuple(sorted(RULES))


def _require_members(exposures: Sequence[float]) -> None:
    # An empty committee is never a legitimate state: an agent whose generation
    # failed still writes a row with a flat exposure, precisely so that no decision
    # point ever silently loses its members. Reaching here with nothing means rows
    # were filtered away upstream, and returning a confident 0.0 would bury that.
    if not exposures:
        raise ValueError("cannot aggregate an empty committee")

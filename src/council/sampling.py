"""Thinning the contested set so a debate finishes.

Priced honestly, the debate arms over the configured two years come to roughly
690,000 inferences -- about seventy-two hours of a single machine's evenings, and
that is the *upper* bound, assuming no conversation ever stops early. The obvious
economy is to shorten the study period. This module exists because that economy is
the wrong one.

Shortening the period changes what is measured. Two years of one ticker spans a
drawdown, a recovery and a flat stretch, and an agent's willingness to hold its
ground is not obviously the same in all three -- the one clear result the fitness
screen produced was regime-dependent (no model goes long into a 19% drawdown).
Measuring six months means measuring one regime and calling it the answer. Sampling
*within* the two years keeps every regime in the sample and costs only precision.

**Why this does not confound the return comparison.** It would, if a thinned debate
arm traded on fewer days than the control. It does not:
:func:`council.scoring.arm_exposures` starts every treatment arm from the
committee's own independent view and overwrites it only at the debated points, so
all four arms are backtested over the identical calendar at the identical
rebalancing rate. What thinning changes is how many days a treatment arm differs
from the control on -- which *dilutes* the market-side effect toward zero, and
leaves the behavioural measurements untouched, since shift rate, influence and stop
reason are each computed per debated point. That is the right trade for this study:
the market is a scoring function here, not the question.

**Why the donor pool is unaffected.** The placebo draws from the independent arm,
which covers every session whether or not it was debated
(:func:`council.debate.sweep.placebo_pool_for`). Thinning the points to debate does
not thin the days available to donate.

The one thing thinning must not do is bias *which* days are debated. Taking the
first N would score the treatment arms over an earlier market than they are
differenced against -- the same selection effect
:func:`council.debate.sweep.servable_points` refuses for the placebo. So the pick is
spread over each ticker's whole calendar, anchored at both ends.

**A larger budget is a superset of a smaller one.** The pick is a recursive
bisection -- both ends, then the midpoint, then the midpoints of the halves -- so
raising ``max_debate_points`` extends the sample rather than replacing it, and the
sweep skips every conversation already stored. That is what makes the run
interruptible in practice rather than only in principle: a night at 60 points can
be resumed to 120 without discarding the first night. Evenly *spaced* picks, which
this began as, do not have that property -- 60 evenly spaced points and 120 evenly
spaced points share only the ones that happen to coincide, and the rest of the first
night is spent again.
"""

from __future__ import annotations

import heapq
from collections.abc import Sequence

from council.evaluation.dispersion import Dispersion


def thin_contested(points: Sequence[Dispersion], *, keep: int | None) -> tuple[Dispersion, ...]:
    """At most ``keep`` contested points, evenly spread over the calendar.

    Deterministic, with no seed and nothing to draw: the pick is a function of the
    contested set alone, so a rerun on the same control arm debates the same days.
    A random sample of the same size would be unbiased too, and would additionally
    have to be reported, defended and reproduced from a seed.

    Each ticker is thinned separately and in proportion to how many contested points
    it has, so a ticker that disagrees more often keeps more of the budget rather
    than each ticker being cut to the same count. Within a ticker the pick spans the
    full date range, first and last included, and grows by refinement: raising
    ``keep`` keeps everything the smaller budget chose.

    Args:
        keep: the total across all tickers. ``None``, or a number at or above what
            is offered, keeps everything.

    Returns:
        The kept points, in the caller's own ordering. Thinning is a *filter*: it
        must not also reorder, because :func:`council.pipeline.select_contested`
        hands this straight to the sweep.
    """
    total = len(points)
    if keep is None or keep >= total or total == 0:
        return tuple(points)
    if keep < 1:
        raise ValueError(f"keep must be at least 1 or None, not {keep}")

    by_ticker: dict[str, list[Dispersion]] = {}
    for point in points:
        by_ticker.setdefault(point.ticker, []).append(point)

    # Insertion order, which is the order the tickers first appear in the contested
    # set -- deterministic without needing to be sorted, and the quotas depend only
    # on how many points each ticker brought.
    kept: set[tuple[object, ...]] = set()
    for own in by_ticker.values():
        quota = _quota(len(own), of=total, budget=keep)
        ordered = sorted(own, key=lambda point: point.decision_date)
        for index in _spread(len(ordered), take=quota):
            kept.add(_identity(ordered[index]))

    return tuple(point for point in points if _identity(point) in kept)


def _identity(point: Dispersion) -> tuple[object, ...]:
    """What makes two contested points the same one.

    The date and ticker rather than the whole record, because the returned tuple is
    rebuilt by filtering the *caller's* sequence: matching on identity would work
    equally, but matching on the point key is what the rest of the pipeline means by
    the same decision point.
    """
    return point.point


def _quota(share: int, *, of: int, budget: int) -> int:
    """This ticker's slice of the budget, never zero.

    Never zero because a ticker with any contested point at all must appear in the
    sample: dropping a whole ticker would turn a thinned two-ticker study into a
    one-ticker study, which is a change to the design rather than to its precision.
    """
    return max(1, round(budget * share / of))


def _spread(length: int, *, take: int) -> tuple[int, ...]:
    """``take`` indices spanning ``0..length-1``, both ends included.

    Both ends deliberately. The earliest contested days are the ones the placebo's
    donor gap withholds, and the latest are the only ones whose forward returns are
    short -- anchoring at both ends keeps the sample's span equal to the study's
    rather than quietly inset from it. A stride-based pick (``range(0, length,
    step)``) would anchor at the start and stop wherever the stride ran out, which
    on most lengths leaves the final weeks out of the sample entirely.

    The order is a recursive bisection: the two ends, then the midpoint of the
    widest remaining gap, repeatedly. Two properties follow, and the second is why
    this is not simply ``round(index * (length - 1) / (take - 1))``:

    * every prefix is spread over the whole range, not merely the full pick; and
    * ``take`` only decides where to stop, so a larger ``take`` returns a superset.

    Gaps are therefore near-uniform rather than exactly uniform -- between a bisection
    and the next, the widest is about twice the narrowest, and a little more than
    that where an odd gap splits unevenly. That is the price of the superset
    property, and it buys an experiment that can be extended a night at a time
    instead of restarted.
    """
    if take >= length:
        return tuple(range(length))
    if take <= 1:
        return (0,)

    chosen = [0, length - 1]
    # Widest gap first, and on a tie the leftmost, so the order is a function of
    # `length` alone -- which is what makes one budget's pick a prefix of another's.
    gaps = [(-(length - 1), 0, length - 1)]
    while len(chosen) < take and gaps:
        _, left, right = heapq.heappop(gaps)
        middle = (left + right) // 2
        if middle in (left, right):
            # Nothing between these two; the gap is spent, not skipped over.
            continue
        chosen.append(middle)
        heapq.heappush(gaps, (-(middle - left), left, middle))
        heapq.heappush(gaps, (-(right - middle), middle, right))
    return tuple(sorted(chosen))

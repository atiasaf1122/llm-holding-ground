"""Did the agent hold its ground?

Round 0 is the opening view; round 1 is the same agent, the same day, the same
ticker, after reading its peers. The difference is the whole experiment.

Rounds are only ever paired *inside* one conversation -- same composition, same
arm. A round 1 produced by a committee of four means nothing beside a round 0 from
a committee containing a fifth agent, and the placebo arm's round 1 must never be
subtracted from the real debate's round 0.

The headline output is the last function here: shift rate as a function of the
confidence the agent held *before* the debate. If a stated confidence of 0.9 sheds
its position as readily as one of 0.3, then confidence does not protect a view, and
that is the result whichever way the equity curve goes.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date

import pandas as pd

from council.config import get_settings
from council.evaluation.buckets import DEFAULT_EDGES, Band, band_index, make_bands
from council.evaluation.frames import (
    AgentKey,
    DebateKey,
    DecisionRow,
    debate_sort_key,
    frame_to_rows,
)

OPENING_ROUND = 0
REBUTTAL_ROUND = 1


@dataclass(frozen=True, slots=True)
class Shift:
    """One agent's movement between its opening view and its post-debate view."""

    decision_date: date
    ticker: str
    composition: str
    arm: str
    model: str
    persona: str
    prior_exposure: float
    posterior_exposure: float
    prior_confidence: float
    posterior_confidence: float
    threshold: float
    """The bar this record's :attr:`shifted` was judged against.

    Carried on the record rather than left in config, because a shift rate cannot be
    read without knowing what counted as a shift, and the two drifting apart in a
    written-up table is a quiet way to publish the wrong number.
    """

    @property
    def debate(self) -> DebateKey:
        return (self.composition, self.arm, self.decision_date, self.ticker)

    @property
    def delta(self) -> float:
        """Signed movement. Positive means the agent became more long."""
        return self.posterior_exposure - self.prior_exposure

    @property
    def distance(self) -> float:
        return abs(self.delta)

    @property
    def shifted(self) -> bool:
        """Whether the position moved far enough to count as changing its mind.

        Inclusive at the boundary, following config's wording that a move *of this
        size* counts.
        """
        return self.distance >= self.threshold

    @property
    def reversed_sign(self) -> bool:
        """Whether the agent came out the other side.

        Strict: both views must be directional and opposed. Retreating from long to
        flat is an abandonment but it is not a reversal, and folding the two together
        would let a wall of hedging read as a wall of about-turns.
        """
        if self.prior_exposure == 0.0 or self.posterior_exposure == 0.0:
            return False
        return (self.prior_exposure > 0.0) != (self.posterior_exposure > 0.0)

    @property
    def changed_mind(self) -> bool:
        """Either measure firing. A reversal counts however small the move was."""
        return self.shifted or self.reversed_sign


def shifts(frame: pd.DataFrame, *, threshold: float | None = None) -> tuple[Shift, ...]:
    """Pair each agent's round 0 with its round 1, in date order.

    Rows without a partner are dropped, which is ordinarily correct: the independent
    arm has no round 1, and an uncontested day never got a debate. Use
    :func:`unpaired_rows` to see exactly what was dropped rather than trusting that.

    A pair in which either round records a failed generation is dropped too, and for
    a sharper reason. A failure is stored with a flat exposure, so a round 1 that
    crashed reads as an agent walking away from its opening view -- and round 1 only
    exists in the debate arms, so those phantom shifts would land entirely on the
    treatment and leave the independent control untouched. :func:`failed_rows`
    reports them.

    Args:
        threshold: defaults to ``settings.shift_threshold``, declared in config
            before any debate ran.

    Raises:
        ValueError: on a round index past 1, or on the same agent appearing twice in
            one round of one conversation. Either means the frame is not what this
            function assumes, and the pairing would silently take whichever row
            happened to sort first.
    """
    limit = get_settings().shift_threshold if threshold is None else threshold
    return tuple(
        _to_shift(prior, posterior, limit) for prior, posterior in _pairs(frame_to_rows(frame))
    )


def unpaired_rows(frame: pd.DataFrame) -> tuple[DecisionRow, ...]:
    """Rows :func:`shifts` dropped for want of a partner, in the canonical row order."""
    return _rows_where(frame, lambda rounds: not _is_complete(rounds))


def failed_rows(frame: pd.DataFrame) -> tuple[DecisionRow, ...]:
    """Both rounds of every pair :func:`shifts` dropped for a failed generation.

    Reported for the same reason
    :attr:`council.evaluation.calibration.CalibrationReport.skipped_count` is: a shift
    rate computed over what is left of a run whose generations mostly crashed must
    not be readable as a shift rate over the run.

    Disjoint from :func:`unpaired_rows`, which covers the rows that had no partner at
    all -- a failure with no counterpart round is reported there.
    """
    return _rows_where(frame, lambda rounds: _is_complete(rounds) and _has_failure(rounds))


@dataclass(frozen=True, slots=True)
class ConfidenceShiftRate:
    """How readily agents in one prior-confidence band abandoned their position."""

    band: Band
    count: int
    shifted_count: int
    reversed_count: int
    total_distance: float

    @property
    def shift_rate(self) -> float | None:
        return self.shifted_count / self.count if self.count else None

    @property
    def reversal_rate(self) -> float | None:
        return self.reversed_count / self.count if self.count else None

    @property
    def mean_distance(self) -> float | None:
        return self.total_distance / self.count if self.count else None


@dataclass(frozen=True, slots=True)
class ShiftRateReport:
    """The bands, and what did not land in any of them."""

    bands: tuple[ConfidenceShiftRate, ...]
    skipped_count: int
    """Shifts whose prior confidence fell outside every supplied band.

    Reported rather than silently dropped, for the reason
    :attr:`council.evaluation.calibration.CalibrationReport.skipped_count` gives:
    a curve drawn from a handful of surviving records must not be readable as a
    curve drawn from the run.
    """


def shift_rate_by_confidence(
    shift_records: Sequence[Shift], *, edges: Sequence[float] = DEFAULT_EDGES
) -> ShiftRateReport:
    """Bucket shifts by the confidence held *before* the debate.

    Prior confidence, never posterior: the question is whether conviction protected
    a position, and a confidence restated after conceding is a description of the
    concession rather than a cause of it.

    Bands come from the same source calibration uses, so the two curves can be read
    against each other -- which is the point, since a shift rate only means something
    once the confidence axis is known to mean something.
    """
    bands = make_bands(edges)
    counts = [0] * len(bands)
    shifted = [0] * len(bands)
    reversed_ = [0] * len(bands)
    distance = [0.0] * len(bands)
    skipped = 0

    for shift in shift_records:
        index = band_index(bands, shift.prior_confidence)
        if index is None:
            skipped += 1
            continue
        counts[index] += 1
        shifted[index] += int(shift.shifted)
        reversed_[index] += int(shift.reversed_sign)
        distance[index] += shift.distance

    return ShiftRateReport(
        bands=tuple(
            ConfidenceShiftRate(
                band=band,
                count=count,
                shifted_count=shift_count,
                reversed_count=reversal_count,
                total_distance=total,
            )
            for band, count, shift_count, reversal_count, total in zip(
                bands, counts, shifted, reversed_, distance, strict=True
            )
        ),
        skipped_count=skipped,
    )


RoundsByAgent = dict[tuple[DebateKey, AgentKey], dict[int, DecisionRow]]

PairOrder = tuple[date, str, str, str, str, str]
"""``(decision_date, ticker, composition, arm, model, persona)`` -- the order
:func:`shifts` documents, which is :attr:`DecisionRow.sort_key` without the round."""


def _by_round(rows: Sequence[DecisionRow]) -> RoundsByAgent:
    by_round: RoundsByAgent = defaultdict(dict)
    for row in rows:
        if row.round_index > REBUTTAL_ROUND:
            raise ValueError(
                f"round {row.round_index} is past the protocol's two rounds "
                f"({row.model}/{row.persona} on {row.decision_date})"
            )
        rounds = by_round[(row.debate, row.agent)]
        if row.round_index in rounds:
            raise ValueError(
                f"{row.model}/{row.persona} appears twice in round {row.round_index} "
                f"of {row.composition or 'the independent arm'} on {row.decision_date}"
            )
        rounds[row.round_index] = row
    return by_round


def _is_complete(rounds: dict[int, DecisionRow]) -> bool:
    return OPENING_ROUND in rounds and REBUTTAL_ROUND in rounds


def _has_failure(rounds: dict[int, DecisionRow]) -> bool:
    return any(row.is_failure for row in rounds.values())


def _rows_where(
    frame: pd.DataFrame, predicate: Callable[[dict[int, DecisionRow]], bool]
) -> tuple[DecisionRow, ...]:
    return tuple(
        sorted(
            (
                row
                for rounds in _by_round(frame_to_rows(frame)).values()
                if predicate(rounds)
                for row in rounds.values()
            ),
            key=lambda row: row.sort_key,
        )
    )


def _pairs(rows: Sequence[DecisionRow]) -> tuple[tuple[DecisionRow, DecisionRow], ...]:
    return tuple(
        (rounds[OPENING_ROUND], rounds[REBUTTAL_ROUND])
        # Sorted chronologically rather than on the grouping key, which leads with
        # composition and arm: this function promises date order, and a frame
        # spanning three debate arms is the normal case here rather than the edge.
        for _, rounds in sorted(_by_round(rows).items(), key=_pair_order)
        if _is_complete(rounds) and not _has_failure(rounds)
    )


def _pair_order(item: tuple[tuple[DebateKey, AgentKey], dict[int, DecisionRow]]) -> PairOrder:
    (debate, agent), _ = item
    return (*debate_sort_key(debate), *agent)


def _to_shift(prior: DecisionRow, posterior: DecisionRow, threshold: float) -> Shift:
    return Shift(
        decision_date=prior.decision_date,
        ticker=prior.ticker,
        composition=prior.composition,
        arm=prior.arm,
        model=prior.model,
        persona=prior.persona,
        prior_exposure=prior.exposure,
        posterior_exposure=posterior.exposure,
        prior_confidence=prior.confidence,
        posterior_confidence=posterior.confidence,
        threshold=threshold,
    )

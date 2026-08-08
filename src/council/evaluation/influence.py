"""Is there a loudest voice that everyone converges on?

Take two agents who opened a debate apart and ask which one gave ground. The trap
is that meeting in the middle looks, from either agent's side, exactly like being
persuaded -- and counting it twice would manufacture a "loud" model out of two
polite ones.

The definition, for a pair *a* and *b* with opening exposures ``a0`` and ``b0`` and
closing exposures ``a1`` and ``b1``. Let ``d`` be the sign of ``b0 - a0``: the
direction *a* must move in to approach *b*. Then

    toward_a = (a1 - a0) * d        how far a moved toward b's opening view
    toward_b = (b1 - b0) * -d       how far b moved toward a's opening view

*a* is recorded as having conceded to *b* only when all three of these hold:

1. they opened apart at all, so there was ground to give;
2. ``toward_a`` is at least the bar -- *a* genuinely gave ground, rather than being
   dragged into the record by *b* moving away from it;
3. ``toward_a - toward_b`` is at least the bar -- *a* gave materially more than it
   got.

Condition 3 is what stops a symmetric meeting in the middle from being read as two
agents persuading each other: the quantity is antisymmetric, so at most one
direction of any pair can be credited, and two agents who each travel 0.4 toward
the other are credited to nobody however far they moved. Condition 2 is what stops
the mirror-image error, where *a* storms off and the motionless *b* is recorded as
having capitulated because the pair's midpoint drifted.

Both bars are deliberately set at the same pre-declared threshold and the test is
conservative on purpose. Under-attributing loses a real effect; over-attributing
invents a loudest voice out of noise, and this module exists to answer whether
there is one.

A concession is *not* a claim that the argument was good. It is a claim about who
moved, which is all the exposures can support.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from itertools import combinations

import numpy as np
import numpy.typing as npt
import pandas as pd

from council.config import get_settings
from council.evaluation.frames import ARM, DebateKey, debate_sort_key, frame_to_rows
from council.evaluation.persuasion import Shift, shifts
from council.evaluation.threshold import meets


@dataclass(frozen=True, slots=True)
class Concession:
    """One agent giving more ground than the agent it moved toward."""

    decision_date: date
    ticker: str
    composition: str
    arm: str
    conceder_model: str
    conceder_persona: str
    influencer_model: str
    influencer_persona: str
    amount: float
    """How far the conceder itself travelled toward the influencer's opening view."""

    asymmetry: float
    """How much more the conceder gave than it got. Always positive, always at most
    :attr:`amount` plus whatever ground the influencer moved away."""

    opening_gap: float
    """How far apart the two opened. A concession of 0.3 out of a gap of 0.35 is a
    capitulation; the same 0.3 out of a gap of 2.0 is a nudge."""


@dataclass(frozen=True, slots=True)
class InfluenceMatrix:
    """Who moved toward whom, aggregated over base models.

    Rows are the agent that conceded, columns the agent it conceded to. The diagonal
    is meaningful and is not self-influence: a model wears four personas here, so
    ``[i][i]`` counts one persona of a model giving ground to another persona of the
    same model.

    One arm. The placebo exists to be differenced against the real debate, so a table
    that had summed the two would answer the question the placebo was built to ask
    with a number in which both answers are already mixed.
    """

    arm: str
    """Which condition these counts come from. Carried on the record so that a table
    lifted into a write-up cannot lose it."""

    models: tuple[str, ...]
    conceded: npt.NDArray[np.int64]
    amount: npt.NDArray[np.float64]
    opportunities: npt.NDArray[np.int64]
    """Pairings where the two opened apart, and so had ground to give. Symmetric.
    The denominator; without it a model that simply debated more often looks
    persuasive."""

    def rate(self, conceder: str, influencer: str) -> float | None:
        """Share of their disagreements in which ``conceder`` gave ground."""
        row = self.models.index(conceder)
        column = self.models.index(influencer)
        chances = int(self.opportunities[row, column])
        return int(self.conceded[row, column]) / chances if chances else None

    @property
    def net_influence(self) -> tuple[tuple[str, int], ...]:
        """Concessions won minus concessions made, best first.

        The loudest voice, if there is one. Ties break on model name so that a run
        repeated on the same data reports the same order.
        """
        won = self.conceded.sum(axis=0)
        made = self.conceded.sum(axis=1)
        scores = [
            (model, int(won[index]) - int(made[index])) for index, model in enumerate(self.models)
        ]
        return tuple(sorted(scores, key=lambda item: (-item[1], item[0])))

    def to_frame(self) -> pd.DataFrame:
        """Concession counts, rows conceding to columns."""
        return pd.DataFrame(
            self.conceded, index=list(self.models), columns=list(self.models)
        ).rename_axis(index="conceder", columns="influencer")


def concessions(
    frame: pd.DataFrame, *, min_concession: float | None = None
) -> tuple[Concession, ...]:
    """Every asymmetric move in the frame, in date order.

    Args:
        min_concession: how much more one side must have given than the other before
            it counts. Defaults to ``settings.shift_threshold`` -- reusing the bar
            that was declared before any debate ran, rather than introducing a
            second one chosen once the results were visible.
    """
    limit = _resolve_limit(min_concession)
    return tuple(
        concession
        for group in _debates(frame, limit)
        for concession in _concessions_in(group, limit)
    )


def influence_matrix(
    frame: pd.DataFrame, *, arm: str, min_concession: float | None = None
) -> InfluenceMatrix:
    """Build the model-by-model matrix for one arm of a frame of debate rounds.

    The arm is required rather than defaulted. Summing the real debate and its
    placebo into one cell is not a mistake the reader of the finished table could
    detect, and the two conditions are meant to be subtracted from each other.

    The model universe comes from the arm's rows rather than from its concessions, so
    a model that never persuaded anyone still gets a row of zeros instead of
    vanishing from the report.

    Raises:
        ValueError: if the frame holds rows but none in this arm. That is a
            misspelled condition rather than an empty result, and a matrix of zeros
            would read as agents who never conceded.
    """
    limit = _resolve_limit(min_concession)
    rows = frame_to_rows(frame)
    present = sorted({row.arm for row in rows})
    if rows and arm not in present:
        raise ValueError(f"no rows in arm {arm!r}; this frame holds {', '.join(present)}")

    selected = frame.loc[frame[ARM].astype(str) == arm] if rows else frame
    models = tuple(sorted({row.model for row in rows if row.arm == arm}))
    index_of = {model: index for index, model in enumerate(models)}
    size = len(models)

    conceded = np.zeros((size, size), dtype=np.int64)
    amount = np.zeros((size, size), dtype=np.float64)
    opportunities = np.zeros((size, size), dtype=np.int64)

    for group in _debates(selected, limit):
        for left, right in combinations(group, 2):
            if left.prior_exposure == right.prior_exposure:
                continue
            first, second = index_of[left.model], index_of[right.model]
            opportunities[first, second] += 1
            # Two personas of one base model land on the diagonal, where the mirrored
            # write is the same cell: counting it twice would halve every same-model
            # rate, and with two models across four personas that is most of them.
            if second != first:
                opportunities[second, first] += 1
        for concession in _concessions_in(group, limit):
            row = index_of[concession.conceder_model]
            column = index_of[concession.influencer_model]
            conceded[row, column] += 1
            amount[row, column] += concession.amount

    return InfluenceMatrix(
        arm=arm,
        models=models,
        conceded=conceded,
        amount=amount,
        opportunities=opportunities,
    )


def _resolve_limit(min_concession: float | None) -> float:
    if min_concession is None:
        return get_settings().shift_threshold
    if min_concession < 0.0:
        raise ValueError("a concession bar below zero would credit agents for moving apart")
    return min_concession


def _debates(frame: pd.DataFrame, limit: float) -> tuple[tuple[Shift, ...], ...]:
    """Paired shifts grouped by conversation, conversations and agents both ordered.

    The bar is threaded through rather than left to config, so that every
    :class:`Shift` behind a concession carries the bar its concession was judged
    against -- which is the whole point of :attr:`Shift.threshold`.
    """
    by_debate: dict[DebateKey, list[Shift]] = defaultdict(list)
    for shift in shifts(frame, threshold=limit):
        by_debate[shift.debate].append(shift)
    return tuple(
        tuple(sorted(group, key=lambda shift: (shift.model, shift.persona)))
        for _, group in sorted(by_debate.items(), key=lambda item: debate_sort_key(item[0]))
    )


def _concessions_in(group: Sequence[Shift], limit: float) -> tuple[Concession, ...]:
    found: list[Concession] = []
    for left, right in combinations(group, 2):
        gap = right.prior_exposure - left.prior_exposure
        if gap == 0.0:
            continue
        direction = 1.0 if gap > 0.0 else -1.0
        toward_left = left.delta * direction
        toward_right = right.delta * -direction
        asymmetry = toward_left - toward_right

        # Exact symmetry is excluded on its own line rather than by the bar, so that
        # a caller passing a bar of 0.0 to inspect every asymmetric move still cannot
        # turn a perfect meeting-in-the-middle into two agents persuading each other.
        # Through `meets` rather than a bare comparison: exposures land on a coarse
        # grid, and `abs(0.3 - 0.1) < 0.20` is True in binary floating point. The
        # same defect halved the published influence figures.
        if asymmetry == 0.0 or not meets(abs(asymmetry), limit):
            continue
        if asymmetry > 0.0:
            conceder, influencer, travelled = left, right, toward_left
        else:
            conceder, influencer, travelled = right, left, toward_right
        # The side the asymmetry points at must have moved toward the other under its
        # own steam. Without this, an agent that stood still while its counterpart
        # stormed off would be recorded as having capitulated.
        # `meets`, not a bare comparison -- the same defect as the asymmetry gate
        # above, and missed when that one was fixed. This module has two bars and
        # routing only one of them through the shared comparison left the second
        # dropping every concession that travelled exactly the bar from the wrong
        # side of the grid.
        if travelled <= 0.0 or not meets(travelled, limit):
            continue

        found.append(
            Concession(
                decision_date=conceder.decision_date,
                ticker=conceder.ticker,
                composition=conceder.composition,
                arm=conceder.arm,
                conceder_model=conceder.model,
                conceder_persona=conceder.persona,
                influencer_model=influencer.model,
                influencer_persona=influencer.persona,
                amount=travelled,
                asymmetry=abs(asymmetry),
                opening_gap=abs(gap),
            )
        )
    return tuple(found)

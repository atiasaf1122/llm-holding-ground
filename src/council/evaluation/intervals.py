"""The paired interval the headline rests on, inside the audited perimeter.

The audit that prompted this module found that the one inferential statistic in the
write-up -- the placebo-minus-debate confidence interval -- was computed by no code in
the repository: the docs test asserted the interval *as a string* and its provenance
was a session transcript. Every descriptive figure here is recomputed from the
parquet by ``tests/test_docs_findings.py``; the interval now is too.

The design decisions, argued once here rather than restated in the write-up:

**Paired by decision point.** Observations inside one decision point share the day,
the prices and thirty-two correlated seats; the point is the unit that varies. Each
arm's observations are collapsed to a per-point mean first, and the two arms are
differenced within a point before anything is resampled.

**Bootstrap over points, seeded by index.** ``default_rng(i)`` for draw ``i`` --
deterministic without a global seed, so the interval is a pure function of the
artefact and reproduces to the digit anywhere. Percentile bounds, 5,000 draws. One
consequence worth knowing: every comparison over the same number of points draws the
same resample index patterns (common random numbers), so published intervals share
Monte Carlo noise across comparisons -- fine for each interval alone, and a reason
not to over-read joint statements like "both increments significant".

The failure conventions of the two statistics differ at the edge:
:func:`~council.evaluation.persuasion.shifts` drops the whole pair when either round
failed, while :func:`net_shift_gap` drops the failed row and reads first/last of the
survivors -- a conversation whose final round failed is scored to its penultimate
round. Zero failures exist in the published artefact, so the divergence is latent;
it is recorded here so a future run with failures does not inherit it silently.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from council.evaluation.frames import ARM
from council.evaluation.persuasion import shifts
from council.evaluation.threshold import meets

DRAWS = 5_000
"""Enough that the 2.5th and 97.5th percentiles are stable to the second decimal in
percentage points; doubling it moves the published bounds by less than 0.1pp."""


@dataclass(frozen=True, slots=True)
class PairedGap:
    """One arm-versus-arm difference in shift rate, paired by decision point."""

    mean_pp: float
    lower_pp: float
    upper_pp: float
    points: int
    """Decision points holding both arms -- the n that matters, not the observation
    count."""

    def excludes_zero(self) -> bool:
        return self.lower_pp > 0.0 or self.upper_pp < 0.0


def paired_shift_gap(
    decisions: pd.DataFrame,
    *,
    minuend_arm: str,
    subtrahend_arm: str,
    threshold: float | None = None,
    draws: int = DRAWS,
) -> PairedGap:
    """``minuend - subtrahend`` shift-rate gap with a percentile bootstrap interval.

    Positive means the minuend arm moved agents more. The frame is the stored
    decisions; any narrowing -- one committee kind, one model -- is done by the caller
    filtering the frame first, which is what keeps this function one comparison
    rather than a query language.
    """
    per_point: dict[str, pd.Series] = {}
    for arm in (minuend_arm, subtrahend_arm):
        arm_shifts = shifts(decisions.loc[decisions[ARM].astype(str) == arm], threshold=threshold)
        frame = pd.DataFrame(
            {
                "point": [(shift.decision_date, shift.ticker) for shift in arm_shifts],
                "shifted": [shift.shifted for shift in arm_shifts],
            }
        )
        if frame.empty:
            raise ValueError(f"no shift pairs in arm {arm!r}; nothing to compare")
        per_point[arm] = frame.groupby("point")["shifted"].mean()

    both = pd.concat(per_point, axis=1).dropna()
    if both.empty:
        raise ValueError(f"arms {minuend_arm!r} and {subtrahend_arm!r} share no decision point")
    difference = (both[minuend_arm] - both[subtrahend_arm]).to_numpy()

    return _bootstrap(difference, draws=draws)


def net_shift_gap(
    decisions: pd.DataFrame,
    *,
    minuend_arm: str,
    subtrahend_arm: str,
    threshold: float = 0.20,
    draws: int = DRAWS,
) -> PairedGap:
    """The same paired gap, measured over each conversation's whole span.

    ``paired_shift_gap`` above is a round-0-to-1 statement -- the declared primary
    comparison. This is its endpoint counterpart: a seat "moved" if its final round
    sits at least ``threshold`` from its opening view, whatever happened in between.
    The two orderings **disagree** on this study's data (C28): the placebo wins the
    first round and loses the conversation, so publishing either without the other
    is the drift this module exists to prevent.

    Endpoint rather than a sum of |moves|, because a seat that flinches and returns
    has not changed its mind, and the sum would count the flinch twice.

    Three conventions carried over from :func:`~council.evaluation.persuasion.shifts`
    so the two statistics cannot ask one question two ways: failed generations are
    excluded (a placeholder exposure is phantom movement, concentrated in whichever
    arm failed), the bar is :func:`~council.evaluation.threshold.meets`, and a seat
    holding only its opening round is dropped rather than counted as unmoved -- it
    has no endpoint to have moved to.
    """
    frame = decisions.loc[
        decisions[ARM].astype(str).isin((minuend_arm, subtrahend_arm))
        & (decisions["failure"].astype(str) == "none")
    ]
    seat = [ARM, "composition", "decision_date", "ticker", "model", "persona"]
    ordered = frame.sort_values("round_index", kind="mergesort")
    grouped = ordered.groupby(seat)
    conversations = grouped.agg(
        opening=("exposure", "first"), final=("exposure", "last"), rounds=("round_index", "nunique")
    ).reset_index()
    conversations = conversations.loc[conversations["rounds"] > 1]
    if conversations.empty:
        raise ValueError(f"no conversations in arms {minuend_arm!r} / {subtrahend_arm!r}")
    distance = (conversations["final"] - conversations["opening"]).abs()
    conversations["moved"] = [meets(value, threshold) for value in distance]
    per_point = conversations.groupby([ARM, "decision_date", "ticker"])["moved"].mean().unstack(0)
    both = per_point.dropna()
    difference = (both[minuend_arm] - both[subtrahend_arm]).to_numpy()
    return _bootstrap(difference, draws=draws)


def _bootstrap(difference: np.ndarray, *, draws: int) -> PairedGap:
    means = np.array(
        [
            np.random.default_rng(draw)
            .choice(difference, size=len(difference), replace=True)
            .mean()
            for draw in range(draws)
        ]
    )
    lower, upper = np.percentile(means, [2.5, 97.5])
    return PairedGap(
        mean_pp=float(difference.mean() * 100),
        lower_pp=float(lower * 100),
        upper_pp=float(upper * 100),
        points=len(difference),
    )

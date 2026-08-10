"""What the agents actually said to each other, at one decision point.

Every other panel is a rate. This one is the evidence behind the rates: two
agents, an opening view each, the peer block that sat between them, and what each
said afterwards. A reader who does not believe a shift rate can come here and read
one shift.

The prose lives in the stored ``rationale`` column, which
:mod:`council.evaluation.frames` deliberately does not carry -- no statistic reads
it. So the rows arrive through :func:`~council.evaluation.frames.frame_to_rows`
for their canonical order and validation, and the rationales are joined back on by
the same key the store uses to identify a decision.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from statistics import fmean, pstdev

import pandas as pd

from council.agents.store import DecisionKey
from council.app.artefacts import RATIONALE, require_columns
from council.config import get_settings
from council.evaluation.frames import (
    ARM,
    COMPOSITION,
    DECISION_DATE,
    MODEL,
    NO_COMPOSITION,
    PERSONA,
    ROUND_INDEX,
    TICKER,
    AgentKey,
    DecisionRow,
    frame_to_rows,
)
from council.evaluation.persuasion import (
    OPENING_ROUND,
    PAIRED_ROUNDS,
    REBUTTAL_ROUND,
    Shift,
)

RoundsByAgent = dict[AgentKey, dict[int, DecisionRow]]


def rationale_lookup(frame: pd.DataFrame) -> dict[DecisionKey, str]:
    """What each stored decision said, keyed the way the store identifies one.

    The key is :data:`~council.agents.store.KEY_COLUMNS` in order, so a column
    later added to a decision's identity reaches this join by construction rather
    than by someone remembering to.

    Raises:
        ValueError: if a column is absent, or if two rows share a decision key. A
            duplicate key means the parquet was concatenated twice; the transcript
            would then show whichever row happened to be read last, and nothing on
            the page would say which.
    """
    require_columns(frame)
    if frame.empty:
        return {}

    dates: list[date] = pd.to_datetime(frame[DECISION_DATE]).dt.date.tolist()
    lookup: dict[DecisionKey, str] = {}
    for decision_date, ticker, model, persona, arm, index, composition, rationale in zip(
        dates,
        frame[TICKER],
        frame[MODEL],
        frame[PERSONA],
        frame[ARM],
        frame[ROUND_INDEX],
        frame[COMPOSITION].fillna(NO_COMPOSITION),
        frame[RATIONALE].fillna(""),
        strict=True,
    ):
        key: DecisionKey = (
            decision_date,
            str(ticker),
            str(model),
            str(persona),
            str(arm),
            int(index),
            str(composition),
        )
        if key in lookup:
            raise ValueError(f"two stored rows share the decision key {key}")
        lookup[key] = str(rationale)
    return lookup


@dataclass(frozen=True, slots=True)
class TranscriptKey:
    """One conversation: a committee, an arm, and the point it was run on.

    The arm is part of the identity. The same committee debating the same day
    under the real debate and under the placebo held two different conversations,
    and showing one under the other's label would misrepresent both.
    """

    decision_date: date
    ticker: str
    arm: str
    composition: str


@dataclass(frozen=True, slots=True)
class SeatUtterance:
    """One agent's two turns, side by side."""

    shift: Shift
    opening_rationale: str
    final_rationale: str
    failed: bool
    """Whether either turn records a failed generation.

    Its exposure is a stored placeholder rather than a view, so the panel must not
    let it read as an agent that went flat.
    """

    @property
    def model(self) -> str:
        return self.shift.model

    @property
    def persona(self) -> str:
        return self.shift.persona


@dataclass(frozen=True, slots=True)
class Transcript:
    """One conversation, ready to read."""

    key: TranscriptKey
    seats: tuple[SeatUtterance, ...]
    """Seats that spoke in both rounds, ordered by model then persona."""

    silent: tuple[AgentKey, ...]
    """Seats missing one of the two rounds.

    Reported rather than dropped: a transcript showing two of four agents must
    not be readable as a debate between two.
    """

    opening_std: float
    """Spread of the opening exposures, over every seat including the silent ones.

    Population standard deviation over this committee's seats in this arm at round
    0. It is not the dispersion that gated the point --
    :func:`council.pipeline.select_contested` measures that over the whole
    independent arm, across every model and persona, so a uniform committee can show
    zero spread here on a point admitted by a directional split there.
    """

    is_split: bool
    """Whether the opening views disagreed about direction rather than about size."""

    @property
    def opening_mean(self) -> float:
        return fmean(seat.shift.prior_exposure for seat in self.seats)

    @property
    def final_mean(self) -> float:
        return fmean(seat.shift.posterior_exposure for seat in self.seats)

    @property
    def largest_move(self) -> float:
        return max(seat.shift.distance for seat in self.seats)

    @property
    def label(self) -> str:
        """One line, enough to choose between conversations in a list."""
        split = ", split" if self.is_split else ""
        return (
            f"{self.key.decision_date} {self.key.ticker} | {self.key.arm} | "
            f"{self.key.composition} | dispersion {self.opening_std:.2f}{split}"
        )


def read_transcripts(
    frame: pd.DataFrame, *, threshold: float | None = None
) -> tuple[Transcript, ...]:
    """Every conversation in the frame, widest opening disagreement first.

    A conversation with no seat that spoke in both rounds is not returned. The
    independent arm is the ordinary case of that -- it has no second round at all,
    so it drops out here without needing to be filtered by name.

    Rounds above the first rebuttal are set aside rather than shown. A conversation
    may run to six of them now, and the panel reads the same two rounds the primary
    statistic is computed over -- see
    :data:`council.evaluation.persuasion.PAIRED_ROUNDS`. What that costs is real and
    is worth stating: a reader cannot follow a long conversation to its end here, and
    the "final" column is the agent's first answer to its peers rather than its last
    word.

    Args:
        threshold: what counts as having shifted, for the :class:`Shift` carried by
            each seat. Defaults to ``settings.shift_threshold``.

    Raises:
        ValueError: on a duplicated decision key.
    """
    limit = get_settings().shift_threshold if threshold is None else threshold
    prose = rationale_lookup(frame)

    grouped: dict[TranscriptKey, RoundsByAgent] = defaultdict(lambda: defaultdict(dict))
    for row in frame_to_rows(frame):
        if row.round_index not in PAIRED_ROUNDS:
            # Set aside rather than refused. This raised on any index above the
            # first rebuttal, which was defensible while every conversation was
            # exactly two rounds long and takes the whole panel down now that one
            # can run to six -- on every artefact this project is about to produce.
            # The panel shows the same two rounds the primary statistic is computed
            # over, for `evaluation.persuasion.PAIRED_ROUNDS`' reason, so a reader
            # who does not believe a shift rate can still come here and read one
            # shift.
            continue
        key = TranscriptKey(
            decision_date=row.decision_date,
            ticker=row.ticker,
            arm=row.arm,
            composition=row.composition,
        )
        grouped[key][row.agent][row.round_index] = row

    found = (_transcript(key, agents, prose, limit) for key, agents in grouped.items())
    return tuple(sorted((item for item in found if item is not None), key=_reading_order))


def _transcript(
    key: TranscriptKey, agents: RoundsByAgent, prose: Mapping[DecisionKey, str], limit: float
) -> Transcript | None:
    seats = tuple(
        _utterance(rounds[OPENING_ROUND], rounds[REBUTTAL_ROUND], prose, limit)
        for _, rounds in sorted(agents.items())
        if OPENING_ROUND in rounds and REBUTTAL_ROUND in rounds
    )
    if not seats:
        return None

    openings = [
        rounds[OPENING_ROUND].exposure for rounds in agents.values() if OPENING_ROUND in rounds
    ]
    return Transcript(
        key=key,
        seats=seats,
        silent=tuple(
            agent
            for agent, rounds in sorted(agents.items())
            if not (OPENING_ROUND in rounds and REBUTTAL_ROUND in rounds)
        ),
        opening_std=pstdev(openings) if len(openings) > 1 else 0.0,
        is_split=any(value > 0.0 for value in openings) and any(value < 0.0 for value in openings),
    )


def _utterance(
    prior: DecisionRow,
    posterior: DecisionRow,
    prose: Mapping[DecisionKey, str],
    limit: float,
) -> SeatUtterance:
    return SeatUtterance(
        shift=Shift(
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
            threshold=limit,
        ),
        opening_rationale=prose[_key_of(prior)],
        final_rationale=prose[_key_of(posterior)],
        failed=prior.is_failure or posterior.is_failure,
    )


def _key_of(row: DecisionRow) -> DecisionKey:
    return (
        row.decision_date,
        row.ticker,
        row.model,
        row.persona,
        row.arm,
        row.round_index,
        row.composition,
    )


def _reading_order(transcript: Transcript) -> tuple[float, date, str, str, str]:
    """Widest disagreement first, then chronological.

    The panel exists to make the project legible in ten seconds, and the point
    where the agents were furthest apart is the one that does that.
    """
    key = transcript.key
    return (-transcript.opening_std, key.decision_date, key.ticker, key.arm, key.composition)


def transcript_table(transcript: Transcript) -> pd.DataFrame:
    """One row per seat: opening view beside final view."""
    return pd.DataFrame(
        [
            {
                "model": seat.model,
                "persona": seat.persona,
                "opening_exposure": seat.shift.prior_exposure,
                "opening_confidence": seat.shift.prior_confidence,
                "final_exposure": seat.shift.posterior_exposure,
                "final_confidence": seat.shift.posterior_confidence,
                "delta": seat.shift.delta,
                "shifted": seat.shift.shifted,
                "reversed": seat.shift.reversed_sign,
                "failed": seat.failed,
            }
            for seat in transcript.seats
        ]
    )


def seat_label(seat: SeatUtterance) -> str:
    """How one seat is named on the page: the base model wearing a persona.

    Both halves, always. Neither identifies an agent on its own -- the same model
    wears four personas and the same persona is worn by every model in turn.
    """
    return f"{seat.model} / {seat.persona}"

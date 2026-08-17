"""The coherent contradictor: peers who argue against the reader, on the right data.

The placebo bounds what the argument's *content* contributes, but a donor-day
argument is displaced in time and, half the time, in instrument -- it does not
merely disagree with the reader, it fails to engage the reader's data at all
(D8, D14). This arm closes the gap the placebo cannot: each of the reader's
three peers presents the strongest case it can make **against the reader's
opening view, on the same price context the reader is looking at**, with its
position rendered exactly as the debate arm renders one.

The adjudication it feeds is declared in ``docs/findings.md`` before the run:
at the round-0-to-1 shift rate, "moves like the placebo" means contradiction
itself is the mover; "moves like the debate arm" means the placebo's surplus
was about incoherence.

Three construction choices, each carrying a confound it exists to remove:

**The peers author their own counters.** In the debate arm the peer prose is
written by the peer models; a contradictor whose prose came from one voice --
the reader's own model asked to argue with itself, which is how the first
sketch in findings.md put it -- would fill three peer slots with one style, a
rendering difference riding along with the manipulation. The deviation from
the sketch is deliberate and this paragraph is its record.

**The opposite side is requested by the schema and enforced by validation.**
The first run trusted the grammar alone -- ``exposure`` bounds excluding the
reader's side -- and an audit then found the backend does not enforce numeric
``minimum``/``maximum`` at all: 6.3% of that run's counters agreed with the
reader, reaching 15.8% of readers (D15; the direction of the headline was
conservative -- clean readers shifted at 0.675 against the published 0.606 --
but the described treatment was not the delivered one, and the mock provider
never caught it because the mock obeys bounds the real backend ignores). So
the constraint is now verified after generation: a counter on the reader's own
side is retried once and then fails the conversation loudly. A flat reader has
no opposite side, so the counter is pushed to a decisive position on a side
chosen by digest parity -- deterministic, and balanced across the run rather
than within a committee.

**Every counter is archived.** The counter text a reader saw exists nowhere in
the decision rows -- only the reader's own output is stored -- and at
temperature zero the backend still drifts on 4.3% of identical prompts (D12),
so "regenerate it" is not provenance. One JSON line per counter, appended
beside the completions archive.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from pydantic import ValidationError

from council.agents.prompt import SIGNAL_SCHEMA
from council.agents.provider import Provider, ProviderError
from council.debate.compositions import Composition, Seat
from council.debate.peers import NoPeersError, SeatView
from council.domain.signal import MAX_RATIONALE_CHARS, Signal
from council.evaluation.frames import PointKey

CONTRA_ROUND_CAP: Final = 1
"""The contradictor arm's own cap on rebuttal rounds.

One, not the sweep's six, for two reasons that reinforce each other. The
pre-registered adjudicating metric is the round-0-to-1 shift rate, which one
rebuttal round fully serves. And no later-round peer schedule is defensible: a
counter regenerated each round against the reader's updated view copies the
placebo's novelty schedule, a counter frozen at round one argues against a
position the reader may have left -- either way a later-round comparison would
be measuring the schedule, which is the confound findings.md pins against.
"""

FLAT_SIDE_MAGNITUDE: Final = 0.25
"""Where a flat reader's contradictor is pushed to. A counter to "no position"
is a decisive position; anything smaller is agreement wearing a different
number."""

OPPOSITE_MARGIN: Final = 0.05
"""The contradictor's minimum commitment on the opposite side -- one grid step,
so "argue the other side" cannot be satisfied by 0.0, which contradicts nobody."""

_SYSTEM: Final = (
    "You are one of several analysts reviewing a price series. A colleague has "
    "just stated a position on it. Your task in this turn is adversarial review: "
    "present the strongest honest case for the OPPOSITE reading of the same "
    "data, and state the position that reading supports. Ground every claim in "
    "the series shown; do not invent facts beyond it. Keep the rationale to one "
    f"or two sentences, at most {MAX_RATIONALE_CHARS} characters."
)


@dataclass(frozen=True, slots=True)
class CounterRecord:
    """One archived counter-argument: who wrote it, against whom, and what it said."""

    decision_date: str
    ticker: str
    composition: str
    reader_model: str
    reader_persona: str
    author_model: str
    author_persona: str
    exposure: float
    confidence: float
    rationale: str
    prompt_hash: str
    generated_at: str

    def row(self) -> dict[str, Any]:
        return asdict(self)


def opposite_bounds(reader_exposure: float, *, token: str) -> tuple[float, float]:
    """The exposure range a counter to this view is constrained into.

    A long reader's contradictor is short, a short reader's is long, and a flat
    reader's is pushed to a decisive position on a side chosen by digest parity
    of ``token`` -- deterministic on a rerun, and balanced across the run's many
    flat readers rather than fixed to one side, which would make "contradict a
    flat view" a directional treatment.
    """
    if reader_exposure > 0.0:
        return (-1.0, -OPPOSITE_MARGIN)
    if reader_exposure < 0.0:
        return (OPPOSITE_MARGIN, 1.0)
    side = hashlib.blake2b(token.encode(), digest_size=8).digest()[0] % 2
    return (FLAT_SIDE_MAGNITUDE, 1.0) if side else (-1.0, -FLAT_SIDE_MAGNITUDE)


def counter_schema(reader_exposure: float, *, token: str) -> dict[str, Any]:
    """The signal schema with ``exposure`` bounded to the opposite side.

    The constraint lives in the grammar rather than the prompt because the
    grammar cannot be talked out of it: a model drifting back toward agreement
    has no tokens with which to do so.
    """
    schema = copy.deepcopy(SIGNAL_SCHEMA)
    low, high = opposite_bounds(reader_exposure, token=token)
    schema["properties"]["exposure"]["minimum"] = low
    schema["properties"]["exposure"]["maximum"] = high
    return schema


def counter_user_prompt(
    *, price_context: str, reader_exposure: float, reader_rationale: str
) -> str:
    """The user turn: the reader's own context, then the position to argue against."""
    stated = (
        f"A colleague states position {reader_exposure:+.2f} "
        f"(-1 fully short, +1 fully long) with this rationale: "
        f"{reader_rationale or '(no rationale given)'}"
    )
    return (
        f"{price_context}\n\n{stated}\n\n"
        "Present the strongest case for the opposite reading of this series, and "
        "state your position now as a JSON object with exposure, confidence and "
        "rationale."
    )


class CounterSideError(ValueError):
    """A generated counter sits on the reader's own side.

    Its own class rather than a bare ValueError so the caller can tell "the
    model agreed instead of opposing" from every other way a draw can fail --
    and because it exists as the audit trail of D15: the grammar was trusted to
    make this impossible, and it was not.
    """


def counter_opposes(counter_exposure: float, reader_exposure: float, *, token: str) -> bool:
    """Whether a counter actually landed in the range the schema requested."""
    low, high = opposite_bounds(reader_exposure, token=token)
    return low <= counter_exposure <= high


async def _opposed_signal(
    provider: Provider,
    *,
    user: str,
    schema: Mapping[str, Any],
    max_tokens: int | None,
    reader_exposure: float,
    token: str,
) -> Signal:
    """Generate a counter and verify its side, retrying once.

    One retry, not more: at temperature zero a second identical request usually
    reproduces the first answer, but the D12-measured regeneration noise gives
    a genuine second draw often enough to be worth one attempt before the
    conversation is abandoned -- and unbounded retries against a model that has
    decided to agree would spin forever.
    """
    for attempt in (0, 1):
        completion = await provider.generate(
            system=_SYSTEM, user=user, schema=schema, max_tokens=max_tokens
        )
        signal = Signal.model_validate(completion.data)
        if counter_opposes(signal.exposure, reader_exposure, token=token):
            return signal
        if attempt == 0:
            continue
    raise CounterSideError(
        f"exposure {signal.exposure:+.2f} sides with the reader's {reader_exposure:+.2f}"
    )


async def generate_counters(
    *,
    providers: Mapping[str, Provider],
    composition: Composition,
    point: PointKey,
    price_context: str,
    openings: Mapping[Seat, SeatView],
    max_tokens: int | None,
    archive: Path | None,
) -> dict[Seat, tuple[SeatView, ...]]:
    """Every reader's three counter-views, authored by its peers.

    Requires an opening view from **every** seat: a reader with no view cannot be
    contradicted, and a peer with no view of its own still authors counters (the
    counter is about the reader's position, not the author's). Refusing a partial
    committee keeps the peer count at three for every reader -- the dose parity
    the probe critique faulted -- at the price of abandoning the conversation,
    which the sweep counts and the resume retries.

    Raises:
        NoPeersError: if any seat's opening failed, or any counter generation
            failed. Caught by the sweep's ``hold`` and booked as an abandoned
            conversation rather than taking the run down.
    """
    if set(openings) != set(composition.seats):
        missing = [s for s in composition.seats if s not in openings]
        raise NoPeersError(
            f"counter-arguments need every opening view; missing {len(missing)} seat(s) "
            f"at {point[0]} {point[1]}"
        )

    day, ticker = point
    records: list[CounterRecord] = []
    views: dict[Seat, tuple[SeatView, ...]] = {}
    for reader in composition.seats:
        opening = openings[reader]
        token = (
            f"{composition.identifier}|{day.isoformat()}|{ticker}"
            f"|{reader.model}|{reader.persona.name}"
        )
        schema = counter_schema(opening.exposure, token=token)
        user = counter_user_prompt(
            price_context=price_context,
            reader_exposure=opening.exposure,
            reader_rationale=opening.rationale,
        )
        authored: list[SeatView] = []
        for author in composition.seats:
            if author == reader:
                continue
            provider = providers[author.model]
            try:
                signal = await _opposed_signal(
                    provider,
                    user=user,
                    schema=schema,
                    max_tokens=max_tokens,
                    reader_exposure=opening.exposure,
                    token=token,
                )
            except (ProviderError, ValidationError) as failure:
                raise NoPeersError(
                    f"counter-argument by {author.model} against {reader.model} at "
                    f"{day} {ticker} failed: {failure}"
                ) from failure
            except CounterSideError as agreed_instead:
                raise NoPeersError(
                    f"counter-argument by {author.model} against {reader.model} at "
                    f"{day} {ticker} sided with the reader twice: {agreed_instead}"
                ) from agreed_instead
            authored.append(
                SeatView(seat=author, exposure=signal.exposure, rationale=signal.rationale)
            )
            records.append(
                CounterRecord(
                    decision_date=day.isoformat(),
                    ticker=ticker,
                    composition=composition.identifier,
                    reader_model=reader.model,
                    reader_persona=reader.persona.name,
                    author_model=author.model,
                    author_persona=author.persona.name,
                    exposure=signal.exposure,
                    confidence=signal.confidence,
                    rationale=signal.rationale,
                    prompt_hash=hashlib.blake2b(
                        f"{_SYSTEM}\n{user}\n{json.dumps(schema, sort_keys=True)}".encode(),
                        digest_size=16,
                    ).hexdigest(),
                    generated_at=datetime.now(tz=UTC).isoformat(),
                )
            )
        views[reader] = tuple(authored)

    if archive is not None:
        _append_records(records, archive)
    return views


def _append_records(records: Sequence[CounterRecord], archive: Path) -> None:
    """Append, not replace: the sweep checkpoints per group and a resume holds only
    the conversations the last run lost, so replacing the file would discard every
    completed group's counters. A conversation retried after an abandonment appends
    a second copy of its counters; readers dedupe on the identity columns, and the
    duplicate is a cheaper defect than a hole."""
    archive.parent.mkdir(parents=True, exist_ok=True)
    with archive.open("a", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record.row(), sort_keys=True, ensure_ascii=False))
            handle.write("\n")

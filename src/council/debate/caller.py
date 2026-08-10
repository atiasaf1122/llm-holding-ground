"""Turning one seat's turn into one stored row.

:mod:`council.debate.protocol` decides who is shown to whom. This is the piece
that carries that decision to a model and writes down what came back, and it is
thin on purpose. What it does *not* do is the point: it renders no text, invents
no peer and handles no failure, because
:func:`~council.agents.inference.generate_decision` already does all three for the
independent arm. An arm that reimplemented any of them would differ from its own
control by more than the debate, and the difference would not be visible in a
stored row.

One instance per decision point. The date, the ticker and the committee are fixed
for a whole conversation, so passing them per call would let two turns of one
debate disagree about which point they belong to.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from council.agents.inference import DecisionPoint, generate_decision
from council.agents.prompt import PeerView, RenderedPrompt
from council.agents.provider import Provider
from council.agents.runner import utc_now
from council.agents.store import CompletionRecord
from council.debate.compositions import Composition, Seat
from council.debate.protocol import AgentReply
from council.domain.signal import Arm, Decision, Signal, StopReason


@dataclass(frozen=True, slots=True)
class SeatDecision:
    """One stored row and the archive line beside it."""

    decision: Decision
    record: CompletionRecord


class DecisionCaller:
    """An :class:`~council.debate.protocol.AgentCaller` that keeps what it generates.

    Args:
        providers: one backend per base model, supplied by the caller because
            which checkpoints may be resident at once is an operational decision
            this class has no way to make.
        composition: the committee. Carried whole rather than as its identifier so
            that a seat from another table is refused here, instead of being
            written down under a committee it never sat in.
    """

    def __init__(
        self,
        *,
        providers: Mapping[str, Provider],
        composition: Composition,
        ticker: str,
        decision_date: date,
        seed: int,
        clock: Callable[[], datetime] = utc_now,
    ) -> None:
        self._providers = dict(providers)
        self._composition = composition
        self._ticker = ticker
        self._decision_date = decision_date
        self._seed = seed
        self._clock = clock
        self.generated: list[SeatDecision] = []

    async def __call__(
        self,
        *,
        seat: Seat,
        price_context: str,
        peers: Sequence[PeerView],
        arm: Arm,
        round_index: int,
    ) -> AgentReply:
        point = DecisionPoint(
            model=seat.model,
            persona=seat.persona,
            ticker=self._ticker,
            decision_date=self._decision_date,
            arm=arm,
            round_index=round_index,
            composition=self._composition.identifier,
        )
        decision, record = await generate_decision(
            self._provider_for(seat),
            point=point,
            price_context=price_context,
            peers=peers,
            seed=self._seed,
            now=self._clock(),
        )
        self.generated.append(SeatDecision(decision=decision, record=record))
        return _as_reply(decision, record)

    def stamped(self, reason: StopReason | None) -> tuple[SeatDecision, ...]:
        """Everything generated here, carrying how the conversation ended.

        Stamped afterwards rather than at the call, and that is forced rather than
        chosen: a turn is taken before anyone knows whether the round it belongs to
        will turn out to be the last, so no argument to ``__call__`` could carry
        this. It goes on every row of the conversation, opening round included,
        because a stored row is the only unit this project persists -- a
        conversation has no record of its own to hang it from.

        ``None`` leaves the rows unstamped, which is the honest answer for a
        conversation that raised part way through: it reached no stopping condition,
        and :meth:`council.agents.store.DecisionStore.completed_conversations` has to
        keep reading it as unfinished so the next run holds it again.

        The archive line beside each decision is untouched: it records one request
        and one response, and the conversation's outcome is not a property of
        either.
        """
        if reason is None:
            return tuple(self.generated)
        return tuple(
            SeatDecision(
                decision=seat.decision.model_copy(update={"stop_reason": reason}),
                record=seat.record,
            )
            for seat in self.generated
        )

    def _provider_for(self, seat: Seat) -> Provider:
        if seat not in self._composition.seats:
            raise ValueError(f"{seat} is not a chair in {self._composition.identifier}")
        try:
            return self._providers[seat.model]
        except KeyError:
            raise ValueError(f"no provider was supplied for {seat.model}") from None


def _as_reply(decision: Decision, record: CompletionRecord) -> AgentReply:
    """Read the reply back off the row that was just written.

    Off the row rather than off the model's response, so that what the next round's
    peers are shown is exactly what the analysis will later read from disk. A
    failure is a flat placeholder in both places, and the protocol drops it from the
    peer block on the strength of ``failure`` rather than of the zero.
    """
    return AgentReply(
        signal=Signal(
            exposure=decision.exposure,
            confidence=decision.confidence,
            rationale=decision.rationale,
        ),
        prompt=RenderedPrompt(
            system=record.system, user=record.user, prompt_hash=record.prompt_hash
        ),
        failure=decision.failure,
    )

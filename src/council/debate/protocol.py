"""How one debate runs.

Three choices, each of which removes a confound rather than adds a feature.

**Simultaneous, not turn-taking.** Every agent gives its opening view without
seeing any other; then every agent sees all of them and answers. In a sequential
debate whoever speaks last has heard everyone and been heard by nobody, so with a
fixed order the experiment measures the order, and with a shuffled one it merely
spreads that effect around. Simultaneity removes the position entirely: the peer
block one agent reads is the same evidence every other agent reads, and no final
view is an input to any other final view. :func:`rebuttal_peers` is where that
holds -- a whole round's peer blocks are a pure function of the previous round,
built before any of them is sent.

**One rebuttal round in v1.** Opening view, peer block, final view: two calls per
seat, eight per committee. :data:`DEFAULT_REBUTTAL_ROUNDS` is a parameter rather
than an assumption threaded through the code, so a later run can raise it -- with
one caveat, that :mod:`council.evaluation.persuasion` rejects a round index above
1 today and would have to be taught what a middle round means first.

**Contested points only.** On a day the agents already agree, a conversation
cannot change the committee's decision, and skipping those days is most of the
compute budget. :func:`run_debate` refuses an uncontested point rather than
trusting its caller to have filtered.

Nothing here renders a prompt or writes a stored row. A seat's turn is taken
through :class:`AgentCaller`, which is handed exactly what
:func:`~council.agents.prompt.build_prompt` needs and is expected to do nothing
with it but call :func:`~council.agents.inference.generate_decision`. That is
deliberate: the peer fencing, the truncation of peer prose, the arm-dependent
exposure switch and the provenance hash are then the independent arm's, and the
arms cannot differ by anything this module chose. Which day the placebo arm's peer
views are taken from is :mod:`council.debate.placebo`'s. What is left here is the
protocol: who speaks when, who is shown to whom, and when a point is abandoned.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from enum import StrEnum
from typing import Protocol

from council.agents.prompt import PeerView, RenderedPrompt
from council.config import get_settings
from council.debate.compositions import Composition, Seat
from council.debate.peers import NoPeersError, SeatView, peers_for, seated_views
from council.debate.placebo import PlaceboPool, donor_views
from council.domain.signal import Arm, FailureMode, Signal
from council.evaluation.dispersion import Dispersion, is_contested
from council.evaluation.frames import PointKey
from council.evaluation.threshold import TOLERANCE, meets

DEFAULT_REBUTTAL_ROUNDS = 1
"""Rounds after the opening one."""

OPENING_ROUND = 0
"""The round every arm asks without peers: the independent question, put to a
committee. Matching :data:`council.evaluation.persuasion.OPENING_ROUND`, which is
the same round read back off disk."""


@dataclass(frozen=True, slots=True)
class AgentReply:
    """One completed turn, as the caller reports it.

    ``prompt`` is the text that was actually sent, digest included, and it is
    reported for a failed generation too: a prompt that was built and answered
    badly and a prompt that was never built are otherwise the same absence.

    A failure carries a placeholder signal rather than nothing, matching how a
    failed decision is stored: never dropped, because the rate of failure per model
    is itself a result. It is kept out of the peer block, though -- see
    :func:`live_views`.
    """

    signal: Signal
    prompt: RenderedPrompt
    failure: FailureMode = FailureMode.NONE

    @property
    def is_failure(self) -> bool:
        return self.failure is not FailureMode.NONE


class AgentCaller(Protocol):
    """Taking one seat's turn.

    The arguments are exactly :func:`~council.agents.prompt.build_prompt`'s, so an
    implementation has nothing left to decide: it pairs the seat's persona with
    them and calls :func:`~council.agents.inference.generate_decision`. Choosing a
    peer, rewording a block or handling a failure here would put the treatment arms
    on a different code path from the control.
    """

    async def __call__(
        self,
        *,
        seat: Seat,
        price_context: str,
        peers: Sequence[PeerView],
        arm: Arm,
        round_index: int,
    ) -> AgentReply: ...


@dataclass(frozen=True, slots=True)
class Turn:
    """One seat in one round: what it was shown, what was sent, and what came back.

    Both halves of the prompt are kept rather than the user turn alone.
    ``prompt_hash`` is taken over the pair -- length-prefixed, so a persona sentence
    sliding out of the system turn and into the user turn changes the digest -- and
    the anonymisation audit that reads a transcript afterwards has to be able to see
    the same two turns the digest saw.
    """

    seat: Seat
    round_index: int
    peers: tuple[PeerView, ...]
    reply: AgentReply

    @property
    def prompt(self) -> RenderedPrompt:
        return self.reply.prompt

    @property
    def user(self) -> str:
        return self.reply.prompt.user

    @property
    def view(self) -> SeatView | None:
        """This turn as a peer would see it; ``None`` if the generation failed."""
        if self.reply.is_failure:
            return None
        return SeatView(
            seat=self.seat,
            exposure=self.reply.signal.exposure,
            rationale=self.reply.signal.rationale,
        )


class StopReason(StrEnum):
    """Why a conversation ended.

    Not bookkeeping -- this is a measurement. The number of rounds is an outcome
    of the debate rather than a setting, and *which* condition ended it says
    something the round count alone does not.
    """

    AGREED = "agreed"
    """The seats came within :attr:`Settings.agreement_spread` of each other."""

    SETTLED = "settled"
    """Nobody moved for :attr:`Settings.stillness_rounds` consecutive rounds.

    The interesting one for this study. A committee that stops without agreeing
    has not converged: it has entrenched, every seat holding a position the others
    argued against and none of them shifting. Two rounds rather than one because
    an agent that ignored an argument on first reading may take it on the second,
    and calling it entrenched after a single quiet round would deny it that.
    """

    CAP = "cap"
    """The round limit was reached while the committee was still moving.

    Not a failure. A conversation still in motion at the cap has no equilibrium
    within the budget, and that is a result about the committee.
    """

    NO_SPEAKERS = "no_speakers"
    """A whole round failed to generate, leaving the next with nothing to answer."""


@dataclass(frozen=True, slots=True)
class DebateTranscript:
    """One conversation, start to finish.

    Rounds are indexed as they are stored: 0 is the opening, 1 is after seeing
    peers. A tuple of rounds rather than two named fields, so that raising
    :data:`DEFAULT_REBUTTAL_ROUNDS` needs no new shape here.
    """

    composition: Composition
    arm: Arm
    decision_date: date
    ticker: str
    rounds: tuple[tuple[Turn, ...], ...]
    stop_reason: StopReason = StopReason.CAP

    @property
    def rebuttal_rounds(self) -> int:
        """Rounds after the opening. The headline outcome of a variable-length debate."""
        return len(self.rounds) - 1

    @property
    def opening(self) -> tuple[Turn, ...]:
        return self.rounds[OPENING_ROUND]

    @property
    def final(self) -> tuple[Turn, ...]:
        return self.rounds[-1]


def live_views(turns: Sequence[Turn]) -> tuple[SeatView, ...]:
    """The views of a round that actually produced output, in seat order.

    A failed turn is stored with a flat exposure and an empty rationale. Rendered
    as a peer it would read as an analyst arguing for no position at all -- an
    argument nobody made, injected into the treatment arms alone.
    """
    return tuple(view for turn in turns if (view := turn.view) is not None)


def rebuttal_peers(
    *, composition: Composition, views: Sequence[SeatView]
) -> tuple[tuple[PeerView, ...], ...]:
    """Every seat's peer block for the next round, in seat order, as one pure
    function of one round.

    This signature is the simultaneity guarantee. There is no argument through
    which a later seat could receive an earlier seat's answer to this round, so no
    ordering effect can exist to be measured -- and a test can check the property
    without running a debate at all.

    An agent is never handed its own view. In the placebo arm that exclusion is
    what keeps the peer count equal to the real arm's, since the donor point was
    argued by the same committee.

    A round shows the round immediately before it and no earlier one. With the v1
    single rebuttal that is the whole conversation; a later version that wanted a
    running transcript would change what ``views`` is given, not this function.
    """
    ordered = seated_views(views, composition=composition)
    return tuple(peers_for(seat, views=ordered) for seat in composition.seats)


async def run_debate(
    *,
    composition: Composition,
    arm: Arm,
    dispersion: Dispersion,
    price_context: str,
    caller: AgentCaller,
    placebo_pool: PlaceboPool | None = None,
    seed: int | None = None,
    dispersion_threshold: float | None = None,
    max_rounds: int | None = None,
    agreement_spread: float | None = None,
    stillness_rounds: int | None = None,
    placebo_min_gap: int | None = None,
) -> DebateTranscript:
    """Run one committee through one decision point, in one arm.

    Args:
        dispersion: the point, and the disagreement measured on it in the
            independent arm. Both halves are load-bearing: it carries the date and
            ticker, and it is what decides the point is worth debating at all.
        price_context: the window every seat is shown, identical for all of them
            since the personas differ in the system turn alone.
        placebo_pool: required by, and only used by, the placebo arm.
        seed: defaults to ``settings.seed``. Only the placebo draw uses it.

    Raises:
        ValueError: for the independent arm, for an uncontested point, for fewer
            than one rebuttal round, or for a placebo arm with no usable donor.
        NoPeersError: if a whole round failed to generate, leaving the next round
            with nothing to react to.
    """
    settings = get_settings()
    cap = settings.max_debate_rounds if max_rounds is None else max_rounds
    spread = settings.agreement_spread if agreement_spread is None else agreement_spread
    stillness = settings.stillness_rounds if stillness_rounds is None else stillness_rounds

    _check_runnable(
        arm=arm,
        dispersion=dispersion,
        rebuttal_rounds=cap,
        threshold=dispersion_threshold,
    )
    point = (dispersion.decision_date, dispersion.ticker)
    if arm is Arm.DEBATE_PLACEBO:
        # Drawn once here purely to fail fast: an unusable pool would otherwise
        # cost a whole opening round before raising. The draw the debate uses is
        # made per round, inside the loop.
        donor_views(
            pool=placebo_pool,
            point=point,
            composition=composition,
            seed=seed,
            round_index=1,
            min_gap=placebo_min_gap,
        )

    seats = composition.seats
    opening = await _take_round(
        seats=seats,
        peer_blocks=((),) * len(seats),
        price_context=price_context,
        arm=arm,
        round_index=OPENING_ROUND,
        caller=caller,
    )
    _check_someone_spoke(opening, point=point)

    rounds = [opening]
    reason = StopReason.CAP
    still_streak = 0

    for round_index in range(1, cap + 1):
        if arm is Arm.DEBATE_PLACEBO:
            # A fresh donor each round. Repeating one frozen peer block would let
            # the control reach stillness for a reason the treatment never faces --
            # nothing new to answer -- and stillness is now a measurement.
            views = donor_views(
                pool=placebo_pool,
                point=point,
                composition=composition,
                seed=seed,
                round_index=round_index,
                min_gap=placebo_min_gap,
            )
        else:
            views = live_views(rounds[-1])

        if not views:
            reason = StopReason.NO_SPEAKERS
            break

        rounds.append(
            await _take_round(
                seats=seats,
                peer_blocks=rebuttal_peers(composition=composition, views=views),
                price_context=price_context,
                arm=arm,
                round_index=round_index,
                caller=caller,
            )
        )

        if _agreed(rounds[-1], spread=spread):
            reason = StopReason.AGREED
            break

        # Stillness is counted on consecutive quiet rounds, and the streak resets
        # the moment anyone moves: a committee that goes quiet, is stirred by one
        # seat, then goes quiet again has not been still for two rounds.
        still_streak = still_streak + 1 if _nobody_moved(rounds[-2], rounds[-1]) else 0
        if still_streak >= stillness:
            reason = StopReason.SETTLED
            break

    return DebateTranscript(
        composition=composition,
        arm=arm,
        decision_date=dispersion.decision_date,
        ticker=dispersion.ticker,
        rounds=tuple(rounds),
        stop_reason=reason,
    )


def _agreed(turns: Sequence[Turn], *, spread: float) -> bool:
    """Whether every seat that spoke is within ``spread`` of every other.

    A round in which fewer than two seats produced output cannot show agreement:
    one surviving voice agrees with nobody, and treating it as consensus would end
    conversations on a generation failure.
    """
    exposures = [view.exposure for turn in turns if (view := turn.view) is not None]
    if len(exposures) < 2:
        return False
    return meets(spread, max(exposures) - min(exposures))


def _nobody_moved(previous: Sequence[Turn], current: Sequence[Turn]) -> bool:
    """Whether no seat's position changed between two rounds.

    Any movement at all counts, not just movement past the shift threshold: the
    question here is whether the conversation is still alive, and a seat inching
    by 0.05 is still answering. A committee that oscillates by small amounts
    therefore runs to the cap -- which is the honest reading of a debate that
    never comes to rest, not a defect.

    A seat that failed to generate is skipped rather than read as having moved to
    flat; a phantom move would reset the streak and keep a dead conversation
    running.
    """
    before = {turn.seat: view.exposure for turn in previous if (view := turn.view) is not None}
    after = {turn.seat: view.exposure for turn in current if (view := turn.view) is not None}
    shared = before.keys() & after.keys()
    if not shared:
        return False
    return all(abs(before[seat] - after[seat]) <= TOLERANCE for seat in shared)


def _check_runnable(
    *, arm: Arm, dispersion: Dispersion, rebuttal_rounds: int, threshold: float | None
) -> None:
    if arm is Arm.INDEPENDENT:
        raise ValueError("the independent arm is the control; there is no debate to run")
    if rebuttal_rounds < 1:
        raise ValueError("a debate needs at least one round after the opening")
    if not is_contested(dispersion, threshold=threshold):
        raise ValueError(
            f"{dispersion.decision_date} {dispersion.ticker} is not contested; "
            "debating it cannot change the committee's decision"
        )


def _check_someone_spoke(opening: Sequence[Turn], *, point: PointKey) -> None:
    """Drop a point whose whole opening round failed -- in every arm, placebo included.

    The placebo shows a donor's views and so has something to render whatever
    happened here, which is exactly why it needs the check written out: whole-round
    failures land on long prompts, truncation-prone models and slow days rather
    than on random points, so an arm that kept them while the other two dropped
    them would be backtested over a different set of decision points, and the
    difference between the arms would absorb that selection.
    """
    if not live_views(opening):
        raise NoPeersError(
            f"every opening view at {point[0]} {point[1]} failed; there is nothing to debate"
        )


async def _take_round(
    *,
    seats: Sequence[Seat],
    peer_blocks: Sequence[Sequence[PeerView]],
    price_context: str,
    arm: Arm,
    round_index: int,
    caller: AgentCaller,
) -> tuple[Turn, ...]:
    """Put every seat's turn in flight at once and keep the answers in seat order.

    Concurrent because the protocol is simultaneous and there is nothing to wait
    for; ordered by seat because ``gather`` returns in argument order however the
    calls finish, and a round whose order depended on latency would render the next
    round's peer blocks differently on every run.
    """
    replies = await asyncio.gather(
        *(
            caller(
                seat=seat,
                price_context=price_context,
                peers=peers,
                arm=arm,
                round_index=round_index,
            )
            for seat, peers in zip(seats, peer_blocks, strict=True)
        )
    )
    return tuple(
        Turn(seat=seat, round_index=round_index, peers=tuple(peers), reply=reply)
        for seat, peers, reply in zip(seats, peer_blocks, replies, strict=True)
    )

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

**Ended by a condition rather than by a count.** An opening view, then rebuttal
rounds until one of three things happens: every seat is within ``agreement_spread``
of every other, nobody moves for ``stillness_rounds`` consecutive rounds, or the cap
is reached. The cap is whatever the caller passes as ``max_rounds``;
``settings.max_debate_rounds`` is what :func:`council.debate.sweep.run_debate_arms`,
:func:`council.planning.plan_experiment` and :mod:`council.scoring` all resolve it
from, so plan, run and score move together, and this module's fallback is only for a
caller that passes neither. At the shipped cap of six all four stop reasons are
reachable, :attr:`~council.domain.signal.StopReason.SETTLED` included -- a streak of
``stillness_rounds = 2`` quiet rounds needs two rebuttal rounds and now has room for
them. While the cap was pinned at one it could not occur, which is the whole reason
the pin was worth removing: entrenchment is the outcome this study is about, and the
protocol was configured so that it could never be recorded.

Both stop conditions require every chair to be filled. A committee that loses a seat
mid-conversation goes on debating with the survivors, and survivors agree sooner and
sit still more readily than the committee would have; letting either predicate fire
on them would publish three agents' agreement under four agents' label. See
:func:`_agreed`, :func:`_nobody_moved` and
:attr:`DebateTranscript.surviving_seats`, which is what a stop reason has to be read
against.

Which condition ended a conversation is returned on the transcript **and stored**:
:meth:`council.debate.sweep._Sweep.hold` stamps it onto every row of the
conversation through :meth:`council.debate.caller.DecisionCaller.stamped`, and
``stop_reason`` is a column of :data:`council.agents.store.STORED_COLUMNS`. It has to
be, because a variable-length conversation leaves no fixed set of rows: the reason is
what tells a resumed run that a conversation which stopped at round two is finished
rather than owing four more. See
:meth:`council.agents.store.DecisionStore.completed_conversations`.

**Contested points only.** On a day the agents already agree, a conversation
cannot change the committee's decision -- what skipping those days saves has never
been measured at the committee level; on the pooled grid the contested share was
98.2%, so it saved almost nothing. Per committee the same run gives 59.0%.
:func:`run_debate` refuses an uncontested point rather
than trusting its caller to have filtered.

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
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Protocol

from council.agents.prompt import PeerView, RenderedPrompt
from council.config import get_settings
from council.debate.compositions import Composition, Seat
from council.debate.contra import CONTRA_ROUND_CAP
from council.debate.peers import NoPeersError, SeatView, peers_for, seated_views
from council.debate.placebo import PlaceboPool, donor_views
from council.domain.signal import Arm, FailureMode, Signal
from council.domain.signal import StopReason as StopReason  # re-exported; see below
from council.evaluation.dispersion import Dispersion, is_contested
from council.evaluation.frames import PointKey
from council.evaluation.threshold import TOLERANCE, within

DEFAULT_REBUTTAL_ROUNDS = 6
"""The cap every shipped path runs at: rounds after the opening one.

A mirror of ``Settings.max_debate_rounds``, which is what
:func:`council.debate.sweep.run_debate_arms`, :func:`council.planning.plan_experiment`
and :mod:`council.scoring` actually resolve the cap from -- the setting cannot
import this constant without closing a cycle, so the two are kept equal by test
(``tests/test_docs_contract.py``) instead.

A cap rather than a length. Under a fixed round count this number was also how
many rounds every conversation held, and half the pipeline read it that way; it is
now an upper bound that most conversations do not reach, and nothing may read it as
the index of a conversation's last round.
"""

OPENING_ROUND = 0
"""The round every arm asks without peers: the independent question, put to a
committee. Matching :data:`council.evaluation.persuasion.OPENING_ROUND`, which is
the same round read back off disk."""

# `StopReason` is declared in `council.domain.signal`, beside `Arm` and
# `FailureMode`, because it is a stored column of `Decision` and those are declared
# where the stored row is. The predicates that decide it live here, so it is
# re-exported explicitly rather than by accident: `from council.debate.protocol
# import StopReason` is how the debate layer and its tests have always named it, and
# moving the declaration should not have moved the name.


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

    @property
    def surviving_seats(self) -> int:
        """How many seats were still generating when the conversation ended.

        Carried beside :attr:`stop_reason` because neither reason can be read
        without it. :attr:`StopReason.AGREED` and :attr:`StopReason.SETTLED` are
        now only reachable with every chair filled -- see :func:`_agreed` and
        :func:`_nobody_moved` -- so on those this equals
        ``composition.size`` and says so. On :attr:`StopReason.CAP` it is the
        number that decides whether "still moving at the budget" describes the
        committee or two survivors of it, and on
        :attr:`StopReason.NO_SPEAKERS` it is zero by definition.

        Derived rather than stored on the record, so it cannot disagree with the
        rounds it is counted from.
        """
        return len(live_views(self.final))


def live_views(turns: Sequence[Turn]) -> tuple[SeatView, ...]:
    """The views of a round that actually produced output, in seat order.

    A failed turn is stored with a flat exposure and an empty rationale. Rendered
    as a peer it would read as an analyst arguing for no position at all -- an
    argument nobody made, injected into the treatment arms alone.
    """
    return tuple(view for turn in turns if (view := turn.view) is not None)


def rebuttal_peers(
    *, composition: Composition, views: Sequence[SeatView], order_token: str
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

    Args:
        order_token: passed straight to :func:`~council.debate.peers.peers_for`,
            which permutes each block by a digest of it. Built by
            :func:`run_debate` from the seed, the committee, the point and the round
            index, so the numbering is a different one in each prompt and the same
            one on a rerun -- and deliberately **not** from the arm. Keying it by arm
            would vary the numbering *between* the arms as well as across prompts, so
            the rationale-only control would differ from the treatment in peer order
            on top of the withheld exposure. The comment in :func:`run_debate`, where
            the token is built, has the argument.
    """
    ordered = seated_views(views, composition=composition)
    return tuple(
        peers_for(seat, views=ordered, order_token=order_token) for seat in composition.seats
    )


PLACEBO_ARMS: frozenset[Arm] = frozenset({Arm.DEBATE_PLACEBO, Arm.DEBATE_PLACEBO_SAME})
"""Both placebo variants. One donor-draw code path serves the two; the only
difference is the ``same_instrument`` flag, which is the D14 manipulation."""

ContraViews = Callable[[Mapping[Seat, SeatView]], Awaitable[dict[Seat, tuple[SeatView, ...]]]]
"""Given every seat's opening view, every reader's counter-views, generated.

A callable rather than the providers themselves so that the protocol stays free
of provider plumbing -- the sweep, which owns the providers, injects this."""


def arm_round_cap(arm: Arm, cap: int) -> int:
    """The cap this arm actually runs to, given the sweep's cap.

    The contradictor is clamped to :data:`~council.debate.contra.CONTRA_ROUND_CAP`
    -- the adjudicating metric is round 0 to 1 and no later-round peer schedule
    for a targeted contradiction is defensible. One function, imported by the
    sweep that runs conversations and the planner that prices them, so the two
    cannot disagree about how long an arm's conversations are.
    """
    return min(cap, CONTRA_ROUND_CAP) if arm is Arm.DEBATE_CONTRADICTOR else cap


async def run_debate(
    *,
    composition: Composition,
    arm: Arm,
    dispersion: Dispersion,
    price_context: str,
    caller: AgentCaller,
    placebo_pool: PlaceboPool | None = None,
    contra_views: ContraViews | None = None,
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
        seed: defaults to ``settings.seed``. Used by the placebo draw and by the
            peer ``order_token``, so it permutes the analyst numbering in every
            arm; the comment on the token at the bottom of this function has the
            argument for keeping the arm out of it.

    Raises:
        ValueError: for the independent arm, for an uncontested point, for fewer
            than one rebuttal round, or for a placebo arm with no usable donor.
        NoPeersError: if the opening round left fewer than two speakers, or if a
            rebuttal round leaves a single seat with no peer to answer. A
            *rebuttal* round in
            which every seat failed does not raise: the conversation ends and is
            returned with :attr:`StopReason.NO_SPEAKERS`, which the caller has to
            read to tell it from a debate that ran to a stopping condition.
    """
    settings = get_settings()
    cap = arm_round_cap(arm, settings.max_debate_rounds if max_rounds is None else max_rounds)
    if arm is Arm.DEBATE_CONTRADICTOR and contra_views is None:
        raise ValueError(
            "the contradictor arm needs a counter-argument generator; the sweep "
            "injects one built over its own providers"
        )
    spread = settings.agreement_spread if agreement_spread is None else agreement_spread
    stillness = settings.stillness_rounds if stillness_rounds is None else stillness_rounds
    # Resolved the same way the placebo draw resolves it, and for the same reason:
    # the peer order has to reproduce exactly on a rerun of a configured seed, so
    # it may not depend on whether the caller happened to pass one.
    resolved_seed = settings.seed if seed is None else seed

    _check_runnable(
        arm=arm,
        dispersion=dispersion,
        rebuttal_rounds=cap,
        threshold=dispersion_threshold,
    )
    point = (dispersion.decision_date, dispersion.ticker)
    if arm in PLACEBO_ARMS:
        # Drawn once here purely to fail fast: an unusable pool would otherwise
        # cost a whole opening round before raising. The draw the debate uses is
        # made per round, inside the loop.
        #
        # At the *last* round the cap allows rather than at the first, because the
        # draw no longer wraps: a candidate set smaller than the cap can serve
        # round 1 and raise at round 3, and failing there would cost every round
        # generated before it. Refusing the point up front is what makes the
        # sweep's pre-flight -- `debate.sweep.has_donor`, which is asked the same
        # question with the same cap -- a promise rather than a hope.
        donor_views(
            pool=placebo_pool,
            point=point,
            composition=composition,
            seed=seed,
            round_index=cap,
            min_gap=placebo_min_gap,
            same_instrument=arm is Arm.DEBATE_PLACEBO_SAME,
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

    contra_blocks: tuple[tuple[PeerView, ...], ...] | None = None
    if arm is Arm.DEBATE_CONTRADICTOR:
        # Generated once, from the opening round, before the loop: the cap is one,
        # so there is no later round to regenerate for -- and the docstring on
        # `arm_round_cap` has the argument for why there never should be. Raises
        # NoPeersError on any incomplete opening or failed counter, which the
        # sweep books as an abandoned conversation.
        assert contra_views is not None  # checked above; narrows the type
        openings = {turn.seat: turn.view for turn in opening if turn.view is not None}
        counters = await contra_views(openings)
        contra_blocks = tuple(
            peers_for(
                seat,
                views=counters.get(seat, ()),
                order_token=(
                    f"{resolved_seed}|{composition.identifier}"
                    f"|{dispersion.decision_date.isoformat()}|{dispersion.ticker}|1"
                ),
            )
            for seat in seats
        )

    for round_index in range(1, cap + 1):
        if arm in PLACEBO_ARMS:
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
                same_instrument=arm is Arm.DEBATE_PLACEBO_SAME,
            )
            # Restricted to the seats that spoke in the round just taken. The donor
            # always holds every chair, so without this a failed seat costs each
            # survivor a peer in the two real arms -- which build their block from
            # `live_views(rounds[-1])` -- and costs the placebo nothing. The peer
            # count would then differ between the arms whenever a generation
            # failed: a second manipulation riding along with the intended one.
            spoke = {turn.seat for turn in rounds[-1] if turn.view is not None}
            views = tuple(view for view in views if view.seat in spoke)
        else:
            views = live_views(rounds[-1])

        if not views:
            reason = StopReason.NO_SPEAKERS
            break

        rounds.append(
            await _take_round(
                seats=seats,
                peer_blocks=contra_blocks
                if contra_blocks is not None
                else rebuttal_peers(
                    composition=composition,
                    views=views,
                    # Deliberately *not* keyed by arm. Every other field varies the
                    # numbering across prompts, which is what removes the confound;
                    # adding the arm would additionally vary it *between* arms, so
                    # the rationale-only control would differ from the treatment in
                    # peer order as well as in the withheld exposure -- a second
                    # manipulation riding along with the intended one, which is
                    # what `agents.prompt` and design note 3 forbid.
                    order_token=(
                        f"{resolved_seed}|{composition.identifier}"
                        f"|{dispersion.decision_date.isoformat()}|{dispersion.ticker}"
                        f"|{round_index}"
                    ),
                ),
                price_context=price_context,
                arm=arm,
                round_index=round_index,
                caller=caller,
            )
        )

        # Read here rather than only at the top of the next iteration. The round
        # just taken may be the last the cap allows, and in the placebo arm the
        # next iteration reads a donor block that is never empty -- so a round in
        # which every seat failed would be returned as CAP and counted as a
        # conversation held -- in the placebo arm at any cap, and in every arm at
        # a cap of one.
        if not live_views(rounds[-1]):
            reason = StopReason.NO_SPEAKERS
            break

        if _agreed(rounds[-1], spread=spread, seats=len(seats)):
            reason = StopReason.AGREED
            break

        # Stillness is counted on consecutive quiet rounds, and the streak resets
        # the moment anyone moves: a committee that goes quiet, is stirred by one
        # seat, then goes quiet again has not been still for two rounds.
        still_streak = (
            still_streak + 1 if _nobody_moved(rounds[-2], rounds[-1], seats=len(seats)) else 0
        )
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


def _agreed(turns: Sequence[Turn], *, spread: float, seats: int) -> bool:
    """Whether every seat of the committee spoke and all of them are within ``spread``.

    The whole committee, not merely the seats that survived. A conversation may
    now run for several rounds, so a seat that drops out part way through is no
    longer a curiosity of the last round: the survivors go on answering each other
    and, being fewer, come inside the bar sooner than the committee would have.
    Recording that as :attr:`StopReason.AGREED` publishes the agreement of two
    agents under the label of four, and the seat that left is the one that was
    still arguing.

    Requiring every chair rather than a floor of two is why ``seats`` is passed in
    rather than counted off ``turns``: every seat takes a turn in every round --
    :func:`_take_round` calls each of them -- and a failed turn is still a turn, so
    ``len(turns)`` is the committee's size whatever happened and could never have
    detected this. What varies is how many of those turns produced a view.
    """
    exposures = [view.exposure for turn in turns if (view := turn.view) is not None]
    if len(exposures) < seats or len(exposures) < 2:
        return False
    return within(max(exposures) - min(exposures), spread)


def _nobody_moved(previous: Sequence[Turn], current: Sequence[Turn], *, seats: int) -> bool:
    """Whether no seat's position changed between two rounds.

    Any movement at all counts, not just movement past the shift threshold: the
    question here is whether the conversation is still alive, and a seat inching
    by 0.05 is still answering. A committee that oscillates by small amounts
    therefore runs to the cap -- which is the honest reading of a debate that
    never comes to rest, not a defect.

    A seat that failed to generate is skipped rather than read as having moved to
    flat; a phantom move would reset the streak and keep a dead conversation
    running.

    A round that lost or gained a speaker is not a quiet round. Intersecting the two
    speaker sets and asking only whether the survivors held still scores a round that
    lost two seats and gained two back -- with the returners' exposures flipped in
    sign -- as nobody moving, which advances the streak and ends the debate as
    :attr:`StopReason.SETTLED`, the verdict the design reads as entrenchment.

    Every chair has to be filled in both rounds, for :func:`_agreed`'s reason and
    one of its own. ``_nobody_moved`` reads stillness off the seats present in
    *both* rounds, so a seat that drops out stops being able to reset the streak:
    a committee still arguing through the seat it lost reads as entrenched, which
    is the finding this study exists to report and the last one that should be
    inferred from an absence. Two rounds of three seats out of four holding still
    is a fact about three seats.
    """
    before = {turn.seat: view.exposure for turn in previous if (view := turn.view) is not None}
    after = {turn.seat: view.exposure for turn in current if (view := turn.view) is not None}
    if before.keys() != after.keys() or len(before) < seats or len(before) < 2:
        return False
    return all(abs(before[seat] - after[seat]) <= TOLERANCE for seat in before)


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
    """Drop a point whose opening round left fewer than two speakers -- in every arm,
    placebo included.

    The placebo shows a donor's views and so has something to render whatever
    happened here, which is exactly why it needs the check written out: whole-round
    failures land on long prompts, truncation-prone models and slow days rather
    than on random points, so an arm that kept them while the other two dropped
    them would be backtested over a different set of decision points, and the
    difference between the arms would absorb that selection.

    Two speakers rather than one, because one is the same selection with a smaller
    number on it. A lone survivor has no peer, so :func:`rebuttal_peers` raises out
    of the real arms -- while the placebo, whose peer block comes from the donor
    pool and never reads this round, holds the point and debates on.
    """
    if len(live_views(opening)) < 2:
        raise NoPeersError(
            f"fewer than two opening views at {point[0]} {point[1]} survived; "
            "there is nobody to debate with"
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

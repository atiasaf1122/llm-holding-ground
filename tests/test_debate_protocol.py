"""The protocol's promises, checked as promises about prompts.

Four of them are load-bearing and none is visible in a stored row: that no final
view could have depended on another final view, that the placebo arm only ever
draws a day the decision has already lived through, that a point which lost its
whole opening round is dropped in every arm alike, and that the three arms differ
in the peer block and in nothing else. Each is asserted here against the exact text
the mock provider was handed.
"""

from __future__ import annotations

import re
from datetime import date, timedelta

import pytest

from council.agents.prompt import build_prompt
from council.debate.compositions import Composition, rotations
from council.debate.peers import NoPeersError
from council.debate.placebo import PlaceboPool, select_placebo_point
from council.debate.protocol import (
    DEFAULT_REBUTTAL_ROUNDS,
    DebateTranscript,
    StopReason,
    live_views,
    rebuttal_peers,
    run_debate,
)
from council.domain.signal import Arm
from council.evaluation.dispersion import Dispersion
from helpers_debate import (
    DAY,
    MODELS,
    OTHER_DAYS,
    PRICE_CONTEXT,
    TICKER,
    MockCaller,
    committee,
    contested,
    independent_user_turn,
    placebo_pool,
    settled,
    views_of,
)

SEED = 20260101

COMMITTEE_SIZE = len(MODELS)
"""What a donor day must hold a view for. The draw requires a complete donor, so a
placebo test that passed anything less would be exercising a pool no sweep admits."""

DEBATE_ARMS = (Arm.DEBATE, Arm.DEBATE_RATIONALE_ONLY, Arm.DEBATE_PLACEBO)

ORDER_TOKEN = "test-token"
"""Whatever varies the peer numbering. :func:`~council.debate.peers.peers_for`
requires one, so a test that omitted it would be asserting on a permutation no
prompt uses."""

POSITION = re.compile(r" \(position [-+]\d\.\d\d\)")
"""How the full arm prints a peer's number, and the only thing the rationale-only
arm removes."""


async def _run(
    arm: Arm,
    caller: MockCaller,
    *,
    composition: Composition | None = None,
    dispersion: Dispersion | None = None,
    pool: PlaceboPool | None = None,
    rebuttal_rounds: int = DEFAULT_REBUTTAL_ROUNDS,
) -> DebateTranscript:
    """One debate, with everything the test is not asserting on left at a default."""
    seated = committee() if composition is None else composition
    if arm is Arm.DEBATE_PLACEBO and pool is None:
        pool = placebo_pool(seated)
    return await run_debate(
        composition=seated,
        arm=arm,
        dispersion=contested() if dispersion is None else dispersion,
        price_context=PRICE_CONTEXT,
        caller=caller,
        placebo_pool=pool,
        seed=SEED,
        max_rounds=rebuttal_rounds,
        stillness_rounds=rebuttal_rounds + 1,
        # The fixtures hold a handful of sessions; the production gap of 60 cannot
        # be met by a pool that size, and it is not what these tests are about.
        placebo_min_gap=0,
        # A negative bar can never be met, which is how a test that is about the
        # round mechanics switches the early stops off. Leaving them on would make
        # these assertions depend on whether the mock's seats happen to agree.
        agreement_spread=-1.0,
    )


def _analysts(text: str) -> int:
    return sum(1 for line in text.splitlines() if line.startswith("Analyst"))


# -- shape --------------------------------------------------------------------


async def test_a_debate_is_two_calls_per_seat() -> None:
    # Arrange
    caller = MockCaller()

    # Act
    transcript = await _run(Arm.DEBATE, caller)

    # Assert -- eight calls for a four-seat committee is the arithmetic the
    # balanced design was sized against.
    assert len(transcript.rounds) == 2
    assert [len(round_turns) for round_turns in transcript.rounds] == [4, 4]
    assert len(caller.prompts) == 8


async def test_the_opening_round_is_the_independent_question_unchanged() -> None:
    # Arrange -- byte-for-byte, against the prompt the control arm would have sent.
    caller = MockCaller()

    # Act
    transcript = await _run(Arm.DEBATE, caller)

    # Assert
    assert {turn.user for turn in transcript.opening} == {independent_user_turn()}


async def test_turns_come_back_in_seat_order_whatever_the_calls_did() -> None:
    # Arrange
    composition = committee()
    caller = MockCaller()

    # Act
    transcript = await _run(Arm.DEBATE, caller, composition=composition)

    # Assert
    for round_turns in transcript.rounds:
        assert [turn.seat for turn in round_turns] == list(composition.seats)


async def test_a_turn_keeps_both_halves_of_the_prompt_that_was_sent() -> None:
    # Arrange -- the digest is taken over the pair, so an audit run from a
    # transcript has to be able to see the pair.
    caller = MockCaller()

    # Act
    transcript = await _run(Arm.DEBATE, caller)

    # Assert
    for turn in transcript.final:
        assert turn.prompt.system
        assert turn.prompt.user == turn.user
        assert turn.prompt.prompt_hash


# -- simultaneity -------------------------------------------------------------


async def test_no_final_view_can_depend_on_another_final_view() -> None:
    # Arrange
    caller = MockCaller()

    # Act
    transcript = await _run(Arm.DEBATE, caller)

    # Assert -- the mock's rationale is unique to the prompt that produced it, so
    # a final rationale appearing in any final prompt would mean one agent had
    # heard another's answer to the round it was itself answering.
    final_rationales = [turn.reply.signal.rationale for turn in transcript.final]
    for prompt in caller.prompts_in_round(1):
        assert not any(rationale in prompt for rationale in final_rationales)


def test_a_rounds_peer_blocks_are_a_pure_function_of_the_round_before_it() -> None:
    # Arrange -- the structural half of the same guarantee: there is no argument
    # through which a later seat could receive an earlier seat's answer.
    composition = committee()
    views = views_of(composition, marker="opening")

    # Act
    blocks = rebuttal_peers(composition=composition, views=views, order_token=ORDER_TOKEN)

    # Assert
    assert len(blocks) == composition.size
    assert blocks == rebuttal_peers(composition=composition, views=views, order_token=ORDER_TOKEN)


async def test_every_agent_sees_every_peer_and_never_itself() -> None:
    # Arrange
    caller = MockCaller()

    # Act
    transcript = await _run(Arm.DEBATE, caller)

    # Assert
    openings = {turn.seat: turn.reply.signal.rationale for turn in transcript.opening}
    for turn in transcript.final:
        assert openings[turn.seat] not in turn.user
        for seat, rationale in openings.items():
            if seat != turn.seat:
                assert rationale in turn.user


# -- one code path for the arms ----------------------------------------------


async def test_the_arms_ask_the_same_opening_question() -> None:
    # Act
    prompts = {arm: (await _run(arm, MockCaller())).opening[0].user for arm in DEBATE_ARMS}

    # Assert
    assert set(prompts.values()) == {independent_user_turn()}


async def test_withholding_the_numbers_is_the_only_difference_it_makes() -> None:
    # Arrange -- the two arms share every opening, so their rebuttal prompts may be
    # compared line for line.
    full = await _run(Arm.DEBATE, MockCaller())
    rationale_only = await _run(Arm.DEBATE_RATIONALE_ONLY, MockCaller())

    # Act & Assert
    for shown, withheld in zip(full.final, rationale_only.final, strict=True):
        assert POSITION.sub("", shown.user) == withheld.user
        assert POSITION.search(shown.user) is not None
        assert POSITION.search(withheld.user) is None


def test_the_peer_block_claims_nothing_about_what_the_peers_read() -> None:
    # Arrange -- in the placebo arm the peers argued about a different session's
    # window, so a header saying they had reviewed this one would be false in that
    # arm alone and would invite the agent to check. The same goes for a claim of
    # independence, which stops being true from the second rebuttal round on.
    composition = committee()
    blocks = rebuttal_peers(
        composition=composition,
        views=views_of(composition, marker="x"),
        order_token=ORDER_TOKEN,
    )

    # Act
    rendered = build_prompt(
        persona=composition.seats[0].persona,
        price_context=PRICE_CONTEXT,
        arm=Arm.DEBATE_PLACEBO,
        peers=blocks[0],
        round_index=1,
    )

    # Assert
    for claim in ("same price history", "same series", "same window", "independent"):
        assert claim not in rendered.user.lower()


# -- the placebo --------------------------------------------------------------


def test_the_placebo_never_draws_a_later_day() -> None:
    # Arrange -- every day in the pool is also asked for, so a draw could land on
    # the day itself or on one after it if the ordering were not enforced. A donor
    # from the future is the failure that does not announce itself: its rationales
    # quote moves this decision has not seen, and the arm is backtested anyway.
    composition = committee()
    days = (*OTHER_DAYS, DAY)
    pool = placebo_pool(composition, days=days)

    # Act
    drawn = {
        day: select_placebo_point(
            min_gap=0,
            required_seats=COMMITTEE_SIZE,
            pool=pool, point=(day, TICKER), composition=composition.identifier, seed=SEED
        )
        for day in days[1:]
    }

    # Assert
    for day, donor in drawn.items():
        assert donor[0] < day


def test_a_pool_of_only_later_days_is_refused() -> None:
    # Arrange
    composition = committee()
    pool = placebo_pool(composition, days=(date(2022, 6, 1), date(2022, 9, 1)))

    # Act & Assert
    with pytest.raises(ValueError, match=r"no earlier date|no session at least"):
        select_placebo_point(
            min_gap=0,
            required_seats=COMMITTEE_SIZE,
            pool=pool, point=(DAY, TICKER), composition=composition.identifier, seed=SEED
        )


async def test_a_debate_whose_pool_is_all_later_days_never_runs() -> None:
    # Arrange -- the refusal has to happen before any generation, since the point
    # of it is that nothing downstream can tell a future donor from a past one.
    caller = MockCaller()
    pool = placebo_pool(committee(), days=(date(2022, 6, 1),))

    # Act & Assert
    with pytest.raises(ValueError, match=r"no earlier date|no session at least"):
        await _run(Arm.DEBATE_PLACEBO, caller, pool=pool)
    assert caller.prompts == []


def test_the_placebo_draw_is_reproducible_under_a_fixed_seed() -> None:
    # Arrange
    composition = committee()
    pool = placebo_pool(composition)
    point = (DAY, TICKER)

    # Act
    first = select_placebo_point(
            min_gap=0,
            required_seats=COMMITTEE_SIZE,
        pool=pool, point=point, composition=composition.identifier, seed=SEED
    )
    second = select_placebo_point(
            min_gap=0,
            required_seats=COMMITTEE_SIZE,
        pool=pool, point=point, composition=composition.identifier, seed=SEED
    )

    # Assert
    assert first == second


def test_the_donor_moves_when_the_pool_gains_earlier_candidates() -> None:
    # The docstring promised the draw was deterministic "given the seed, the
    # committee and the point being debated -- and given nothing else", and the
    # comment on the digest sort credited it with stopping "a rerun over a different
    # date range" from rewriting an arm already on disk. It stops dependence on draw
    # *order* only: the donor is `ordered[0]` over a digest ranking of the
    # candidates, so a candidate a wider range admits can rank first and redraw the
    # donor for a point whose seed, committee and date are unchanged. This is the
    # scenario now pending -- the run moves from a six-month window to the
    # configured two years -- and nothing detects it, because
    # `check_prompt_provenance` validates round-0 rows, which carry no peer block.
    composition = committee()
    later = tuple(date(2022, 4, day) for day in range(11, 31))
    earlier = tuple(date(2022, 4, day) for day in range(1, 11))
    narrow = placebo_pool(composition, days=later)
    wider = placebo_pool(composition, days=earlier + later)

    def draw(pool: PlaceboPool, day: date) -> tuple[date, str]:
        return select_placebo_point(
            min_gap=0,
            required_seats=COMMITTEE_SIZE,
            pool=pool,
            point=(day, TICKER),
            composition=composition.identifier,
            seed=SEED,
        )

    moved = [day for day in later[1:] if draw(narrow, day) != draw(wider, day)]

    assert moved, "widening the pool left every donor unchanged, so the hazard is gone"
    # And the docstring says so rather than promising the opposite.
    from council.debate import placebo as placebo_module

    doc = " ".join((placebo_module.select_placebo_point.__doc__ or "").split())
    assert "and given nothing else" not in doc
    assert "the candidate set" in doc


def test_a_different_seed_draws_a_different_donor() -> None:
    # Arrange -- checked across many points rather than one, since any single draw
    # may coincide. The first day is drawn for by nobody: it has no earlier day.
    composition = committee()
    days = tuple(date(2022, 4, day) for day in range(1, 21))
    pool = placebo_pool(composition, days=days)

    def draw_all(seed: int) -> tuple[tuple[date, str], ...]:
        return tuple(
            select_placebo_point(
            min_gap=0,
            required_seats=COMMITTEE_SIZE,
                pool=pool, point=(day, TICKER), composition=composition.identifier, seed=seed
            )
            for day in days[1:]
        )

    # Act & Assert
    assert draw_all(SEED) != draw_all(SEED + 1)


def test_two_committees_on_one_day_do_not_share_a_donor_by_construction() -> None:
    # Arrange -- the draw is keyed by committee as well as by point, so one day's
    # placebo evidence is not the same text for every configuration.
    composition = committee()
    days = tuple(date(2022, 4, day) for day in range(1, 21))
    pool = placebo_pool(composition, days=days)

    def draw_all(identifier: str) -> tuple[tuple[date, str], ...]:
        return tuple(
            select_placebo_point(
            min_gap=0,
            required_seats=COMMITTEE_SIZE,
                pool=pool, point=(day, TICKER), composition=identifier, seed=SEED
            )
            for day in days[1:]
        )

    # Act & Assert
    assert draw_all("rotation-0") != draw_all("uniform-momentum-bold")


def test_the_configured_gap_leaves_one_candidate_so_every_round_repeats_it() -> None:
    # The comment on the modulo said wrapping needs "a conversation [to outlast] the
    # pool, which the production gap and round cap make impossible". The gap makes
    # it routine: the filter is `key[0] <= cutoff` with `cutoff = earlier[-gap]`, so
    # a point with exactly `gap` earlier sessions has a one-element candidate set
    # and every round draws the same donor. Only the cap of one hides it -- raising
    # the cap would hand the placebo an identical peer block in consecutive rounds,
    # which is the failure the per-round redraw was added to prevent.
    from council.config import get_settings
    from council.debate.sweep import _check_cap

    gap = get_settings().placebo_min_gap_sessions
    composition = committee()
    days = tuple(date(2022, 1, 3) + timedelta(days=offset) for offset in range(gap + 1))
    pool = placebo_pool(composition, days=days)

    drawn = {
        select_placebo_point(
            min_gap=gap,
            required_seats=COMMITTEE_SIZE,
            pool=pool,
            point=(days[-1], TICKER),
            composition=composition.identifier,
            seed=SEED,
            round_index=round_index,
        )
        for round_index in (1, 2, 3)
    }

    assert drawn == {(days[0], TICKER)}
    # So raising the cap has to deal with it: the checklist names it, and so does
    # the refusal a caller actually sees.
    assert "select_placebo_point" in (_check_cap.__doc__ or "")
    with pytest.raises(ValueError, match="select_placebo_point"):
        _check_cap(DEFAULT_REBUTTAL_ROUNDS + 1)


def test_a_pool_holding_only_this_day_is_refused() -> None:
    # Arrange
    composition = committee()
    pool = placebo_pool(composition, days=(DAY,))

    # Act & Assert -- falling back to the day's own views would make the placebo
    # arm a second debate arm, and every stored row would still look correct.
    with pytest.raises(ValueError, match=r"no earlier date|no session at least"):
        select_placebo_point(
            min_gap=0,
            required_seats=COMMITTEE_SIZE,
            pool=pool, point=(DAY, TICKER), composition=composition.identifier, seed=SEED
        )


async def test_the_placebo_shows_another_days_rationales_and_not_todays() -> None:
    # Arrange
    caller = MockCaller()

    # Act
    transcript = await _run(Arm.DEBATE_PLACEBO, caller)

    # Assert
    openings = [turn.reply.signal.rationale for turn in transcript.opening]
    for turn in transcript.final:
        assert "donor 2022-" in turn.user
        assert not any(rationale in turn.user for rationale in openings)


async def test_the_placebo_gives_each_agent_as_many_peers_as_a_real_debate() -> None:
    # Arrange -- a peer count that differed between the arms would be a second
    # manipulation riding along with the intended one.
    real = await _run(Arm.DEBATE, MockCaller())
    placebo = await _run(Arm.DEBATE_PLACEBO, MockCaller())

    # Act & Assert
    assert [_analysts(turn.user) for turn in placebo.final] == [
        _analysts(turn.user) for turn in real.final
    ]


async def test_a_failed_seat_costs_every_arm_the_same_peer() -> None:
    # Arrange -- the real arms build the block from the seats that spoke, so a seat
    # that failed its opening costs each survivor a peer. The donor day always holds
    # every chair, so the placebo felt nothing: a generation failure changed the peer
    # count in two arms and not the third.
    counts = {}
    for arm in DEBATE_ARMS:
        caller = MockCaller(failing_models=frozenset({"beta"}), failing_round=0)
        transcript = await _run(arm, caller)
        assert len(live_views(transcript.opening)) == COMMITTEE_SIZE - 1
        counts[arm] = [(turn.seat.model, _analysts(turn.user)) for turn in transcript.final]

    # Assert
    assert counts[Arm.DEBATE_PLACEBO] == counts[Arm.DEBATE]
    assert counts[Arm.DEBATE_RATIONALE_ONLY] == counts[Arm.DEBATE]


async def test_a_pool_argued_by_another_committee_is_refused() -> None:
    # Arrange -- same models, different personas, so nothing about the pool looks
    # wrong; every seat would simply get one peer more than the real arm gives.
    seated = rotations(models=MODELS)[0]
    strangers = placebo_pool(rotations(models=MODELS)[1])
    caller = MockCaller()

    # Act & Assert
    with pytest.raises(ValueError, match="not a chair"):
        await _run(Arm.DEBATE_PLACEBO, caller, composition=seated, pool=strangers)
    assert caller.prompts == []


async def test_the_donor_views_are_numbered_by_the_committee_not_by_the_pool() -> None:
    # Arrange -- a pool assembled from a groupby holds its views in whatever order
    # that produced; if the analyst numbers followed it, a rerun would rewrite an
    # arm that had already been generated.
    seated = committee()
    forward = placebo_pool(seated)
    backward = {point: tuple(reversed(views)) for point, views in forward.items()}

    # Act
    in_order = await _run(Arm.DEBATE_PLACEBO, MockCaller(), pool=forward)
    shuffled = await _run(Arm.DEBATE_PLACEBO, MockCaller(), pool=backward)

    # Assert
    assert [turn.user for turn in in_order.final] == [turn.user for turn in shuffled.final]


async def test_the_placebo_arm_without_a_pool_is_refused() -> None:
    with pytest.raises(ValueError, match="pool"):
        await run_debate(
            composition=committee(),
            arm=Arm.DEBATE_PLACEBO,
            dispersion=contested(),
            price_context=PRICE_CONTEXT,
            caller=MockCaller(),
            seed=SEED,
        )


# -- what is refused ----------------------------------------------------------


async def test_an_uncontested_point_is_not_debated() -> None:
    # Act & Assert -- on a day the agents agree, a conversation cannot change the
    # committee's decision, and the compute is most of the budget.
    with pytest.raises(ValueError, match="not contested"):
        await _run(Arm.DEBATE, MockCaller(), dispersion=settled())


async def test_the_independent_arm_is_not_a_debate() -> None:
    with pytest.raises(ValueError, match="control"):
        await _run(Arm.INDEPENDENT, MockCaller())


async def test_a_debate_needs_at_least_one_round_after_the_opening() -> None:
    with pytest.raises(ValueError, match="at least one round"):
        await _run(Arm.DEBATE, MockCaller(), rebuttal_rounds=0)


# -- failures -----------------------------------------------------------------


async def test_a_failed_opening_is_not_shown_as_a_peer() -> None:
    # Arrange -- a failure is stored with a flat exposure, which as a peer would
    # read as an analyst arguing for no position at all.
    caller = MockCaller(failing_models=frozenset({"beta"}), failing_round=0)

    # Act
    transcript = await _run(Arm.DEBATE, caller)

    # Assert
    assert len(live_views(transcript.opening)) == 3
    for turn in transcript.final:
        assert _analysts(turn.user) == (3 if turn.seat.model == "beta" else 2)


@pytest.mark.parametrize("arm", DEBATE_ARMS)
async def test_a_round_in_which_everyone_failed_stops_the_debate(arm: Arm) -> None:
    # Arrange
    caller = MockCaller(failing_models=frozenset(MODELS))

    # Act & Assert -- the placebo has a donor to show and would otherwise sail
    # through a point the other two arms dropped, so the arms would end up
    # backtested over different sets of decision points.
    with pytest.raises(NoPeersError):
        await _run(arm, caller)


async def test_one_surviving_opening_view_is_dropped_by_every_arm() -> None:
    # Arrange -- three of four seats fail their opening, leaving a lone survivor.
    # It has no peer, so `rebuttal_peers` raises out of the two real arms; the
    # placebo's round-1 block comes from the donor pool and never reads this
    # round, so it would hold the point and debate on. Same selection effect as
    # the whole-round case above, with a smaller number on it.
    messages: dict[Arm, str] = {}

    # Act
    for arm in DEBATE_ARMS:
        caller = MockCaller(failing_models=frozenset(MODELS[1:]), failing_round=0)
        with pytest.raises(NoPeersError) as raised:
            await _run(arm, caller)
        messages[arm] = str(raised.value)

    # Assert -- identically, or the arms cover different point sets.
    assert len(set(messages.values())) == 1
    assert "fewer than two opening views" in messages[Arm.DEBATE_PLACEBO]


async def test_a_rebuttal_round_that_all_failed_ends_the_debate_without_raising() -> None:
    # The opening round raises; a rebuttal round does not. It sets NO_SPEAKERS and
    # returns, so a caller that only writes `except NoPeersError` books the point
    # as a conversation held. `debate.sweep._Sweep.hold` reads the stop reason for
    # exactly this reason, and run_debate's Raises block has to say so.
    caller = MockCaller(failing_models=frozenset(MODELS), failing_round=1)

    transcript = await _run(Arm.DEBATE, caller, rebuttal_rounds=3)

    assert transcript.stop_reason is StopReason.NO_SPEAKERS
    assert transcript.rebuttal_rounds == 1


@pytest.mark.parametrize("arm", DEBATE_ARMS)
async def test_the_last_round_the_cap_allows_is_read_for_speakers_too(arm: Arm) -> None:
    # Arrange -- the test above runs to a cap of three, so the empty round is
    # noticed at the top of the *next* iteration. `run_debate_arms` ships
    # DEFAULT_REBUTTAL_ROUNDS, where there is no next iteration: the reason stayed
    # CAP and `_Sweep.hold` booked a conversation with no usable rebuttal row as
    # held. In the placebo arm the top-of-loop view is the donor block, which is
    # never empty, so NO_SPEAKERS was unreachable there at any cap at all.
    caller = MockCaller(failing_models=frozenset(MODELS), failing_round=1)

    # Act
    transcript = await _run(arm, caller, rebuttal_rounds=DEFAULT_REBUTTAL_ROUNDS)

    # Assert
    assert transcript.stop_reason is StopReason.NO_SPEAKERS
    assert live_views(transcript.final) == ()


# -- room for more rounds -----------------------------------------------------


async def test_a_second_rebuttal_round_reacts_to_the_first() -> None:
    # Arrange
    caller = MockCaller()

    # Act
    transcript = await _run(Arm.DEBATE, caller, rebuttal_rounds=2)

    # Assert
    assert len(transcript.rounds) == 3
    first_rebuttal = {turn.seat: turn.reply.signal.rationale for turn in transcript.rounds[1]}
    for turn in transcript.rounds[2]:
        assert first_rebuttal[turn.seat] not in turn.user
        assert any(
            rationale in turn.user
            for seat, rationale in first_rebuttal.items()
            if seat != turn.seat
        )

"""When a conversation ends, and why.

The number of rounds is an outcome of the debate rather than a setting, so the
stopping rule is a measuring instrument and is tested as one. Three conditions can
end a debate and each means something different about the committee: it agreed, it
stopped moving without agreeing, or it was still moving when the budget ran out.

A bug that ended conversations early would read as "committees converge quickly",
and one that never ended them would read as "committees never settle". Neither
raises anything, so neither would be noticed without these.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import pytest

import council.debate.sweep as council_sweep
from council.agents.mock import MockProvider
from council.agents.prompt import PeerView, build_prompt
from council.agents.provider import Provider
from council.config import get_settings
from council.debate.compositions import Seat
from council.debate.placebo import select_placebo_point
from council.debate.protocol import (
    DEFAULT_REBUTTAL_ROUNDS,
    AgentReply,
    DebateTranscript,
    StopReason,
    Turn,
    _agreed,
    _nobody_moved,
    run_debate,
)
from council.debate.sweep import run_debate_arms
from council.domain.signal import Arm, FailureMode, Signal
from council.pipeline import open_store, select_contested, stored_decisions
from helpers_debate import FLAT, PRICE_CONTEXT, committee, contested, placebo_pool
from helpers_pipeline import make_prices, make_settings, run_independent

SEAT_ORDER = [seat.persona.name for seat in committee().seats]


class ScriptedCaller:
    """Answers from a per-round script, so a test can steer the committee.

    ``script[round][seat]`` is the exposure that seat returns in that round. A
    round past the end of the script repeats the last row, which is how a test
    asks for stillness without spelling it out.
    """

    def __init__(self, script: Sequence[Sequence[float]]) -> None:
        self.script = [list(row) for row in script]

    async def __call__(
        self,
        *,
        seat: Seat,
        price_context: str,
        peers: Sequence[PeerView],
        arm: Arm,
        round_index: int,
    ) -> AgentReply:
        row = self.script[min(round_index, len(self.script) - 1)]
        rendered = build_prompt(
            persona=seat.persona,
            price_context=price_context,
            arm=arm,
            peers=peers,
            round_index=round_index,
        )
        return AgentReply(
            prompt=rendered,
            signal=Signal(
                exposure=row[SEAT_ORDER.index(seat.persona.name)],
                confidence=0.6,
                rationale=f"round {round_index}",
            ),
            failure=FailureMode.NONE,
        )


async def debate(script: Sequence[Sequence[float]], **overrides: Any) -> DebateTranscript:
    settings: dict[str, Any] = {
        "agreement_spread": 0.20,
        "stillness_rounds": 2,
        "max_rounds": 6,
        "placebo_min_gap": 0,
    }
    settings.update(overrides)
    return await run_debate(
        composition=committee(),
        arm=Arm.DEBATE,
        dispersion=contested(),
        price_context=PRICE_CONTEXT,
        caller=ScriptedCaller(script),
        seed=1,
        **settings,
    )


class TestAgreement:
    async def test_a_committee_that_converges_stops_and_says_so(self) -> None:
        # Arrange: wide apart, then inside the bar.
        script = [[1.0, 0.5, -0.5, -1.0], [0.10, 0.15, 0.20, 0.25]]

        # Act
        transcript = await debate(script)

        # Assert
        assert transcript.stop_reason is StopReason.AGREED
        assert transcript.rebuttal_rounds == 1

    async def test_a_spread_exactly_on_the_bar_counts_as_agreement(self) -> None:
        # Inclusive, like every other threshold here -- and pinned on the side of
        # the representation error that a bare `max - min <= spread` gets wrong.
        # 0.9 - 0.7 is 0.20000000000000007, so the bare form calls this committee
        # still apart; `within` calls it agreed, which is what the bar says in
        # prose. Checking only the other side is what made this test vacuous
        # once: the rule could be reverted with the whole suite still green.
        transcript = await debate([[1.0, 0.5, -0.5, -1.0], [0.9, 0.9, 0.7, 0.7]])
        assert transcript.stop_reason is StopReason.AGREED

    async def test_a_spread_on_the_bar_that_subtracts_short_also_agrees(self) -> None:
        # The other side of the same error: 0.30 - 0.10 is 0.19999999999999998,
        # which a bare comparison accepts. Both are nominally 0.20, and the
        # stopping rule must not depend on where on the grid the spread landed.
        transcript = await debate([[1.0, 0.5, -0.5, -1.0], [0.30, 0.20, 0.10, 0.10]])
        assert transcript.stop_reason is StopReason.AGREED

    async def test_a_spread_just_wider_than_the_bar_does_not(self) -> None:
        transcript = await debate([[1.0, 0.5, -0.5, -1.0], [0.35, 0.20, 0.10, 0.10]])
        assert transcript.stop_reason is not StopReason.AGREED


class TestStillness:
    async def test_one_quiet_round_is_not_enough(self) -> None:
        # The reason for two: an agent that ignored an argument on first reading
        # may take it on the second, and stopping at the first quiet round would
        # record that committee as entrenched without giving it the chance.
        script = [[1.0, -1.0, 1.0, -1.0], [0.9, -0.9, 0.9, -0.9], [0.9, -0.9, 0.9, -0.9]]
        transcript = await debate(script, max_rounds=2)
        assert transcript.stop_reason is StopReason.CAP

    async def test_two_quiet_rounds_end_it(self) -> None:
        transcript = await debate([[1.0, -1.0, 1.0, -1.0], [0.9, -0.9, 0.9, -0.9]])

        assert transcript.stop_reason is StopReason.SETTLED
        # Round 1 moved; rounds 2 and 3 did not.
        assert transcript.rebuttal_rounds == 3

    async def test_the_streak_resets_when_anybody_moves(self) -> None:
        # Quiet, then one seat stirs, then quiet again. That is not two
        # consecutive still rounds and the conversation is not over.
        script = [
            [1.0, -1.0, 1.0, -1.0],
            [1.0, -1.0, 1.0, -1.0],
            [0.8, -1.0, 1.0, -1.0],
            [0.8, -1.0, 1.0, -1.0],
        ]
        transcript = await debate(script)

        assert transcript.stop_reason is StopReason.SETTLED
        assert transcript.rebuttal_rounds == 4

    async def test_entrenchment_is_recorded_as_settling_not_as_agreement(self) -> None:
        # The case this study cares about most: they stopped moving and are still
        # far apart. Recording that as agreement would erase the finding.
        transcript = await debate([[1.0, -1.0, 1.0, -1.0]])
        exposures = [turn.reply.signal.exposure for turn in transcript.final]

        assert transcript.stop_reason is StopReason.SETTLED
        assert max(exposures) - min(exposures) > 0.20

    async def test_settled_is_reachable_at_the_shipped_cap_and_the_shipped_stillness(
        self,
    ) -> None:
        # The point of raising the cap. At a cap of one rebuttal round, SETTLED could
        # not occur with the configured `stillness_rounds = 2` -- a streak of two
        # quiet rounds needs two rebuttal rounds -- so the outcome this study is
        # about, a committee that stops without agreeing, was unrecordable by
        # construction. Asserted at both shipped values together, because that pair
        # is what a run actually uses.
        settings = get_settings()
        assert settings.stillness_rounds == 2, "the arithmetic below assumes the shipped value"

        transcript = await debate(
            [[1.0, -1.0, 1.0, -1.0], [0.9, -0.9, 0.9, -0.9]],
            max_rounds=settings.max_debate_rounds,
            stillness_rounds=settings.stillness_rounds,
        )

        assert transcript.stop_reason is StopReason.SETTLED
        # Round 1 moved; rounds 2 and 3 did not. Well inside the cap of six, which
        # is what makes the reason reachable rather than the cap's own verdict.
        assert transcript.rebuttal_rounds == 3
        assert transcript.rebuttal_rounds < settings.max_debate_rounds

    async def test_settled_still_needs_only_one_quiet_round_at_a_stillness_of_one(
        self,
    ) -> None:
        # The other bound the reason rests on, and the one that used to be its only
        # route: `stillness_rounds` is `Field(default=2, ge=1, le=10)` and the sweep
        # threads the run's own value through, so `COUNCIL_STILLNESS_ROUNDS=1` is a
        # supported configuration and one quiet rebuttal round ends the conversation.
        transcript = await debate([[1.0, -1.0, 1.0, -1.0]], max_rounds=6, stillness_rounds=1)

        assert transcript.stop_reason is StopReason.SETTLED
        assert transcript.rebuttal_rounds == 1

    def test_the_documents_no_longer_call_the_entrenchment_verdict_unreachable(self) -> None:
        # Three docstrings said SETTLED "cannot occur in any run this repository
        # makes" and was "Unreachable at the shipped cap, and arithmetically so".
        # That was true, and it was a defect in the experiment rather than a fact
        # worth documenting. The claim has to go with the cap that made it true, or
        # the next reader is told the study cannot report its own subject.
        from council.config import PROJECT_ROOT

        config = (PROJECT_ROOT / "src" / "council" / "config.py").read_text(encoding="utf-8")
        protocol = (PROJECT_ROOT / "src" / "council" / "debate" / "protocol.py").read_text(
            encoding="utf-8"
        )
        signal = (PROJECT_ROOT / "src" / "council" / "domain" / "signal.py").read_text(
            encoding="utf-8"
        )
        # The enum's per-member docstring is source-only, so it is read as source.
        member = " ".join(signal.split('SETTLED = "settled"')[1].split('"""')[1].split())

        for body in (" ".join(config.split()), " ".join(protocol.split()), member):
            assert "cannot occur in any run this repository makes" not in body
            assert "Unreachable at the shipped cap" not in body
        assert "unreachable" in member.lower()
        assert "cap was pinned at one" in member


class PartialCaller:
    """Answers only the seats a round's script names; the rest fail to generate.

    ``script[round]`` maps a seat's index in the committee to its exposure. A round
    past the end repeats the last row.
    """

    def __init__(self, script: Sequence[Mapping[int, float]]) -> None:
        self.script = [dict(row) for row in script]

    async def __call__(
        self,
        *,
        seat: Seat,
        price_context: str,
        peers: Sequence[PeerView],
        arm: Arm,
        round_index: int,
    ) -> AgentReply:
        row = self.script[min(round_index, len(self.script) - 1)]
        index = SEAT_ORDER.index(seat.persona.name)
        rendered = build_prompt(
            persona=seat.persona,
            price_context=price_context,
            arm=arm,
            peers=peers,
            round_index=round_index,
        )
        if index not in row:
            return AgentReply(signal=FLAT, prompt=rendered, failure=FailureMode.MALFORMED)
        return AgentReply(
            prompt=rendered,
            signal=Signal(
                exposure=row[index], confidence=0.6, rationale=f"round {round_index}"
            ),
        )


class TestStillnessNeedsTheSameVoices:
    async def test_a_round_that_changed_speakers_is_not_a_quiet_round(self) -> None:
        # `_nobody_moved` intersected the two rounds' speaking seats and asked only
        # whether the *shared* ones held still. A round that lost two speakers and
        # gained two back -- with a returner's exposure flipped in sign -- was
        # scored "nobody moved", so the streak advanced and the debate ended as
        # SETTLED, the verdict this design reads as entrenchment.
        script: list[Mapping[int, float]] = [
            {0: 1.0, 1: -1.0, 2: 0.9, 3: -0.9},
            {0: 1.0, 1: -1.0},
            {1: -1.0, 2: 0.2, 3: 0.9},
            {0: 1.0, 1: -1.0, 2: 0.2, 3: 0.9},
        ]

        transcript = await run_debate(
            composition=committee(),
            arm=Arm.DEBATE,
            dispersion=contested(),
            price_context=PRICE_CONTEXT,
            caller=PartialCaller(script),
            seed=1,
            agreement_spread=-1.0,
            stillness_rounds=2,
            max_rounds=4,
            placebo_min_gap=0,
        )

        # Seat 3 went -0.9 to +0.9 across the rounds the old rule called quiet.
        assert transcript.stop_reason is StopReason.CAP
        assert transcript.rebuttal_rounds == 4

    def test_one_surviving_voice_cannot_show_the_committee_settled(self) -> None:
        # `_agreed` already refuses to read consensus off fewer than two voices;
        # the stillness predicate had no such floor, so a round in which a single
        # seat survived and held still counted towards entrenchment.
        seats = committee().seats
        rounds = tuple(
            (
                Turn(
                    seat=seats[0],
                    round_index=index,
                    peers=(),
                    reply=AgentReply(
                        prompt=build_prompt(persona=seats[0].persona, price_context=PRICE_CONTEXT),
                        signal=Signal(exposure=0.5, confidence=0.6, rationale="held"),
                    ),
                ),
            )
            for index in (1, 2)
        )

        assert _nobody_moved(*rounds, seats=len(seats)) is False

    def test_a_committee_short_a_seat_cannot_settle_or_agree(self) -> None:
        # The stronger requirement the longer cap forces. Both predicates read only
        # the seats that spoke, so three survivors of a four-seat committee holding
        # still for two rounds ended the debate as SETTLED -- entrenchment inferred
        # from an absence, with the seat that was still arguing being the one that
        # went missing. `_nobody_moved` already refused a round that *changed*
        # speakers; a seat that drops out and stays out changes nothing after the
        # round it left in, so it was invisible until conversations got long enough
        # for one to happen mid-way.
        seats = committee().seats

        def round_of(present: int, index: int) -> tuple[Turn, ...]:
            return tuple(
                Turn(
                    seat=seat,
                    round_index=index,
                    peers=(),
                    reply=AgentReply(
                        prompt=build_prompt(persona=seat.persona, price_context=PRICE_CONTEXT),
                        signal=Signal(exposure=0.5, confidence=0.6, rationale="held"),
                    ),
                )
                for seat in seats[:present]
            )

        three = (round_of(3, 1), round_of(3, 2))

        assert _nobody_moved(*three, seats=len(seats)) is False
        assert _agreed(three[1], spread=0.20, seats=len(seats)) is False
        # And the whole committee still settles and still agrees, so the fix is not
        # "refuse everything".
        assert _nobody_moved(round_of(4, 1), round_of(4, 2), seats=len(seats)) is True
        assert _agreed(round_of(4, 2), spread=0.20, seats=len(seats)) is True

    async def test_a_seat_lost_mid_conversation_does_not_read_as_entrenchment(self) -> None:
        # The same thing end to end, on the path a run takes. Seat 3 argues, drops
        # out at round 2 and never comes back; the survivors hold still from there.
        # The old predicates called that SETTLED at round 3.
        script: list[Mapping[int, float]] = [
            {0: 1.0, 1: -1.0, 2: 0.9, 3: -0.9},
            {0: 1.0, 1: -1.0, 2: 0.9, 3: 0.9},
            {0: 1.0, 1: -1.0, 2: 0.9},
        ]

        transcript = await run_debate(
            composition=committee(),
            arm=Arm.DEBATE,
            dispersion=contested(),
            price_context=PRICE_CONTEXT,
            caller=PartialCaller(script),
            seed=1,
            agreement_spread=-1.0,
            stillness_rounds=2,
            max_rounds=5,
            placebo_min_gap=0,
        )

        assert transcript.stop_reason is StopReason.CAP
        assert transcript.rebuttal_rounds == 5
        # And the reason can be read against how many were alive to produce it.
        assert transcript.surviving_seats == 3
        assert transcript.surviving_seats < committee().size


def restless(rounds: int = 9) -> list[list[float]]:
    """A committee that never agrees and never stops moving."""
    return [[n / 10 - 0.5, 0.9, -0.9, 0.4 - n / 20] for n in range(rounds)]


class TestTheCap:
    async def test_a_committee_still_moving_runs_out_of_budget(self) -> None:
        transcript = await debate(restless(), max_rounds=4)

        assert transcript.stop_reason is StopReason.CAP
        assert transcript.rebuttal_rounds == 4

    @pytest.mark.parametrize("cap", [1, 2, 5])
    async def test_the_cap_is_the_only_thing_bounding_it(self, cap: int) -> None:
        assert (await debate(restless(), max_rounds=cap)).rebuttal_rounds == cap


class TestThePlaceboKeepsUp:
    async def test_a_fresh_donor_is_drawn_for_every_round(self) -> None:
        # A frozen peer block would let the control settle for a reason the
        # treatment never faces -- nothing new to answer -- and stillness is one
        # of the things this design measures.
        days = (date(2021, 1, 4), date(2021, 2, 1), date(2021, 3, 1), date(2021, 4, 1))
        transcript = await run_debate(
            composition=committee(),
            arm=Arm.DEBATE_PLACEBO,
            dispersion=contested(),
            price_context=PRICE_CONTEXT,
            caller=ScriptedCaller([[1.0, -1.0, 1.0, -1.0], [0.9, -0.8, 0.7, -0.6]]),
            placebo_pool=placebo_pool(committee(), days=days),
            seed=1,
            agreement_spread=-1.0,
            stillness_rounds=99,
            max_rounds=3,
            placebo_min_gap=0,
        )

        shown = [
            frozenset(turn.user for turn in round_turns) for round_turns in transcript.rounds[1:]
        ]
        assert len(shown) == 3
        assert len(set(shown)) == 3, "a round reused another round's peer block"


class TestTheDonorGap:
    @staticmethod
    def eight_sessions() -> tuple[date, ...]:
        return tuple(date(2022, 1, day) for day in (3, 4, 5, 6, 7, 10, 11, 12))

    @pytest.mark.parametrize("gap", [1, 2, 3, 5])
    async def test_no_donor_is_nearer_than_the_configured_gap(self, gap: int) -> None:
        pool = placebo_pool(committee(), days=self.eight_sessions())
        sessions = sorted({key[0] for key in pool})
        reader = sessions[-1]

        chosen = select_placebo_point(
            pool=pool,
            point=(reader, "AAPL"),
            composition="rotation-0",
            required_seats=committee().size,
            seed=1,
            min_gap=gap,
        )

        assert sessions.index(reader) - sessions.index(chosen[0]) >= gap

    async def test_a_pool_too_shallow_for_the_gap_is_refused(self) -> None:
        # Refusing beats quietly drawing a nearer donor: the arm would still run,
        # and the control it provides would be weaker than the one reported.
        with pytest.raises(ValueError, match="fewer than the"):
            select_placebo_point(
                pool=placebo_pool(committee(), days=(date(2022, 1, 3), date(2022, 1, 4))),
                point=(date(2022, 1, 4), "AAPL"),
                composition="rotation-0",
                required_seats=committee().size,
                seed=1,
                min_gap=5,
            )

    async def test_a_gap_of_zero_still_refuses_the_day_itself(self) -> None:
        # The gap check computes a cutoff, and with a gap of zero that cutoff is
        # the decision date. A bound written as "<= cutoff" alone would admit the
        # day being decided as its own donor: lookahead, introduced by the very
        # check meant to strengthen the control. It was, once.
        chosen = select_placebo_point(
            pool=placebo_pool(committee(), days=(date(2022, 1, 3), date(2022, 1, 4))),
            point=(date(2022, 1, 4), "AAPL"),
            composition="rotation-0",
            required_seats=committee().size,
            seed=1,
            min_gap=0,
        )
        assert chosen[0] < date(2022, 1, 4)


class ConstantFactory:
    """One fixed exposure per base model, held for every round.

    A committee driven by this has a spread that is known on paper and never
    moves, so a debate run on it stops for exactly one reason: the bar it was
    given. The mock replays a single supplied response, cycling, which is what
    makes it constant across rounds as well as across seats.
    """

    def __init__(self, exposures: Mapping[str, float]) -> None:
        self._exposures = dict(exposures)

    def __call__(self, model: str) -> Provider:
        return MockProvider(
            responses=[
                {
                    "exposure": self._exposures[model],
                    "confidence": 0.5,
                    "rationale": f"{model} holds",
                }
            ],
            model=model,
        )


class TestTheSweepStopsOnItsOwnSettings:
    """`run_debate` resolves its bars from the process-wide `get_settings()` when
    its caller omits them, and the sweep must not omit them: a run configured with
    its own `Settings` -- which is every run `cli.settings_from` builds -- would
    otherwise have one read out from under it.

    Asserted on the arguments the sweep passes rather than on the round count. At
    the pinned cap of one rebuttal round the round count cannot distinguish the
    bars: `StopReason.SETTLED` needs a streak of two quiet rounds and so at least
    two rebuttal rounds, and AGREED and CAP both end the conversation after the one
    round the cap allows. That the count is now blind to them is the finding
    recorded beside `Settings.max_debate_rounds`, not a reason to stop checking the
    threading.
    """

    def _sweep(self, tmp_path: Path, settings_update: dict[str, Any]) -> dict[str, Any]:
        """Run one debate group and hand back the kwargs `run_debate` was called with."""
        base = make_settings(tmp_path)
        prices = make_prices()
        run_independent(base, prices)
        store = open_store(base)
        decisions = stored_decisions(store)
        settings = base.model_copy(update=settings_update)

        seen: list[dict[str, Any]] = []
        real = run_debate

        async def spy(**kwargs: Any) -> DebateTranscript:
            seen.append(kwargs)
            return await real(**kwargs)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(council_sweep, "run_debate", spy)
            asyncio.run(
                run_debate_arms(
                    settings=settings,
                    prices=prices,
                    decisions=decisions,
                    contested=select_contested(decisions, settings=base),
                    provider_factory=ConstantFactory({"alpha": 0.5, "beta": -0.5}),
                    store=store,
                    arms=(Arm.DEBATE,),
                )
            )
        assert seen, "the sweep held no conversation, so it passed no bars"
        return seen[0]

    @pytest.mark.parametrize(
        ("agreement_spread", "stillness_rounds"), [(1.0, 2), (0.20, 1), (0.20, 3)]
    )
    def test_the_bars_come_from_the_runs_settings(
        self, tmp_path: Path, agreement_spread: float, stillness_rounds: int
    ) -> None:
        # Act
        passed = self._sweep(
            tmp_path,
            {"agreement_spread": agreement_spread, "stillness_rounds": stillness_rounds},
        )

        # Assert -- explicitly, not left to `run_debate`'s fallback on the
        # process-wide settings.
        assert passed["agreement_spread"] == agreement_spread
        assert passed["stillness_rounds"] == stillness_rounds

    def test_the_cap_the_sweep_passes_is_the_configured_one(self, tmp_path: Path) -> None:
        # The sweep resolves `rebuttal_rounds` from `settings.max_debate_rounds`, so
        # plan, run and score move together -- and the shipped value is now six
        # rather than the pin of one.
        assert self._sweep(tmp_path, {})["max_rounds"] == DEFAULT_REBUTTAL_ROUNDS
        assert DEFAULT_REBUTTAL_ROUNDS == get_settings().max_debate_rounds > 1

    def test_a_cap_above_one_is_run_rather_than_refused(self, tmp_path: Path) -> None:
        # `run_debate_arms` used to refuse every cap but one, because eight consumers
        # read the cap as the index of each conversation's last round and corrupted a
        # run at anything longer. They read `stop_reason` and each conversation's own
        # final round now, so a longer cap is a longer conversation rather than a
        # broken artefact, and the refusal that stood in for the work is gone.
        assert self._sweep(tmp_path, {"max_debate_rounds": 3})["max_rounds"] == 3

    def test_a_cap_below_one_rebuttal_round_is_still_refused(self, tmp_path: Path) -> None:
        # What is left of the refusal, and it is arithmetic rather than a checklist:
        # at zero rebuttal rounds the treatment arms are the control.
        base = make_settings(tmp_path)
        prices = make_prices()
        run_independent(base, prices)
        store = open_store(base)
        decisions = stored_decisions(store)

        with pytest.raises(ValueError, match="at least one round"):
            asyncio.run(
                run_debate_arms(
                    settings=base,
                    prices=prices,
                    decisions=decisions,
                    contested=select_contested(decisions, settings=base),
                    provider_factory=ConstantFactory({"alpha": 0.5, "beta": -0.5}),
                    store=store,
                    arms=(Arm.DEBATE,),
                    rebuttal_rounds=0,
                )
            )

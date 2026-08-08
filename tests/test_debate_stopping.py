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

from council.agents.mock import MockProvider
from council.agents.prompt import PeerView, build_prompt
from council.agents.provider import Provider
from council.debate.compositions import Seat, balanced_design
from council.debate.placebo import select_placebo_point
from council.debate.protocol import AgentReply, DebateTranscript, StopReason, run_debate
from council.debate.sweep import run_debate_arms
from council.domain.signal import Arm, FailureMode, Signal
from council.pipeline import open_store, select_contested, stored_decisions
from helpers_debate import PRICE_CONTEXT, committee, contested, placebo_pool
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
        # still apart; `meets` calls it agreed, which is what the bar says in
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
            pool=pool, point=(reader, "AAPL"), composition="rotation-0", seed=1, min_gap=gap
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
    """`run_debate` resolves the two bars from the process-wide `get_settings()`
    when its caller omits them, and the sweep omitted them -- while threading
    `dispersion_threshold`, `max_rounds`, `seed` and `placebo_min_gap` from its own
    `Settings` precisely so that a run configured with its own would not have one
    read out from under it. `cli.settings_from` builds every run that way, and
    which condition ended a conversation is a declared measurement.
    """

    @pytest.mark.parametrize(
        ("agreement_spread", "stillness_rounds", "rounds"),
        [
            # Exposures of +0.5 and -0.5 sit exactly 1.0 apart, so this bar is met
            # in round 1 and the conversation ends AGREED after two rounds.
            (1.0, 2, 2),
            # Under the narrow bar nobody agrees, and nobody ever moves, so the
            # stillness streak is what ends it -- after one quiet round here.
            (0.20, 1, 2),
            # The process-wide values, which is what the sweep used to run
            # whatever its own Settings said: two quiet rounds, so three in all.
            (0.20, 2, 3),
        ],
    )
    def test_the_bars_come_from_the_runs_settings(
        self, tmp_path: Path, agreement_spread: float, stillness_rounds: int, rounds: int
    ) -> None:
        # Arrange -- the control arm is generated by the ordinary mock, since a
        # committee that all said the same thing would leave nothing contested.
        base = make_settings(tmp_path)
        prices = make_prices()
        run_independent(base, prices)
        store = open_store(base)
        decisions = stored_decisions(store)
        tables = balanced_design(models=base.agent_models)
        assert len({table.size for table in tables}) == 1
        seats = tables[0].size

        settings = base.model_copy(
            update={
                "agreement_spread": agreement_spread,
                "stillness_rounds": stillness_rounds,
            }
        )

        # Act
        report = asyncio.run(
            run_debate_arms(
                settings=settings,
                prices=prices,
                decisions=decisions,
                contested=select_contested(decisions, settings=base),
                provider_factory=ConstantFactory({"alpha": 0.5, "beta": -0.5}),
                store=store,
                arms=(Arm.DEBATE,),
                rebuttal_rounds=3,
            )
        )

        # Assert -- the round count is the only reading of the stop reason a
        # stored row currently offers.
        assert report.held > 0
        assert report.abandoned == 0
        assert report.generated == report.held * seats * rounds

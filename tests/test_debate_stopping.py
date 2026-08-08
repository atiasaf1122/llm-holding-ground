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

from collections.abc import Sequence
from datetime import date
from typing import Any

import pytest

from council.agents.prompt import PeerView, build_prompt
from council.debate.compositions import Seat
from council.debate.placebo import select_placebo_point
from council.debate.protocol import AgentReply, DebateTranscript, StopReason, run_debate
from council.domain.signal import Arm, FailureMode, Signal
from helpers_debate import PRICE_CONTEXT, committee, contested, placebo_pool

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
        # Inclusive, like every other threshold here, and checked on values whose
        # decimal difference does not survive binary subtraction.
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

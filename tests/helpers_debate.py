"""Fixtures shared by the debate tests.

Every agent in here answers through :class:`~council.agents.mock.MockProvider`, so
the whole debate layer is exercised with no daemon, no network and no GPU. The
mock derives its answer from the prompt, which is what makes two of these tests
possible at all: a round's rationales are unique to the prompts that produced
them, so a prompt can be searched for text that would only be there if the
protocol had leaked a later round into an earlier one.

:class:`MockCaller` builds its prompts with the same
:func:`~council.agents.prompt.build_prompt` the independent sweep uses. A fake
renderer here would let the tests pass on a peer block no run would ever send.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date
from typing import Any

from council.agents.mock import MockProvider
from council.agents.prompt import PeerView, build_prompt
from council.debate.compositions import Composition, Seat, rotations
from council.debate.peers import SeatView
from council.debate.placebo import PlaceboPool
from council.debate.protocol import AgentReply
from council.domain.persona import PERSONAS
from council.domain.signal import Arm, FailureMode, Signal
from council.evaluation.dispersion import Dispersion
from council.evaluation.frames import PointKey

MODELS: tuple[str, ...] = ("alpha", "beta", "gamma", "delta")

DAY = date(2022, 3, 1)
OTHER_DAYS: tuple[date, ...] = (
    date(2022, 2, 22),
    date(2022, 2, 23),
    date(2022, 2, 24),
    date(2022, 2, 25),
)
"""Sessions before :data:`DAY`, because that is the only kind of day a placebo
donor may be drawn from. A pool of later days would let every placebo test pass
while the arm read rationales about a window its decision had not reached."""

TICKER = "AAPL"

SIGNAL_SCHEMA: dict[str, Any] = Signal.model_json_schema()

FLAT = Signal(exposure=0.0, confidence=0.0, rationale="")
"""What a failed generation is stored with: a placeholder, not an opinion."""

PRICE_CONTEXT = "Daily returns %, oldest first.\n+1.20 -0.40 +0.15"


def committee() -> Composition:
    """A four-seat committee: one model per persona, the identity rotation."""
    return rotations(models=MODELS)[0]


def view(seat: Seat, *, exposure: float = 0.5, rationale: str = "trend intact") -> SeatView:
    return SeatView(seat=seat, exposure=exposure, rationale=rationale)


def views_of(composition: Composition, *, marker: str) -> tuple[SeatView, ...]:
    """One view per seat, each rationale carrying a marker a test can search for."""
    return tuple(
        SeatView(seat=seat, exposure=0.25 * index, rationale=f"{marker} from chair {index}")
        for index, seat in enumerate(composition.seats)
    )


def placebo_pool(composition: Composition, *, days: tuple[date, ...] = OTHER_DAYS) -> PlaceboPool:
    """Openings from several earlier days, all argued by the same committee."""
    pool: dict[PointKey, tuple[SeatView, ...]] = {}
    for day in days:
        pool[(day, TICKER)] = views_of(composition, marker=f"donor {day.isoformat()}")
    return pool


def contested(*, on: date = DAY, ticker: str = TICKER) -> Dispersion:
    """A point the agents split on by direction, so any threshold counts it."""
    return Dispersion(
        decision_date=on,
        ticker=ticker,
        agent_count=4,
        exposure_std=0.6,
        long_count=2,
        short_count=2,
        flat_count=0,
    )


def settled(*, on: date = DAY, ticker: str = TICKER) -> Dispersion:
    """A point with nothing to argue about, at any non-negative threshold."""
    return Dispersion(
        decision_date=on,
        ticker=ticker,
        agent_count=4,
        exposure_std=0.0,
        long_count=4,
        short_count=0,
        flat_count=0,
    )


def independent_user_turn(price_context: str = PRICE_CONTEXT) -> str:
    """The control's user turn, which every arm's opening round must match."""
    return build_prompt(persona=PERSONAS[0], price_context=price_context).user


class MockCaller:
    """An :class:`~council.debate.protocol.AgentCaller` with no GPU behind it.

    Records every prompt it is given, which is what the tests assert on: the
    protocol's promises are promises about prompts.
    """

    def __init__(self, *, failing_models: frozenset[str] = frozenset(), failing_round: int = 0):
        self.provider = MockProvider()
        self.prompts: list[tuple[Seat, int, str]] = []
        self._failing_models = failing_models
        self._failing_round = failing_round

    async def __call__(
        self,
        *,
        seat: Seat,
        price_context: str,
        peers: Sequence[PeerView],
        arm: Arm,
        round_index: int,
    ) -> AgentReply:
        rendered = build_prompt(
            persona=seat.persona,
            price_context=price_context,
            arm=arm,
            peers=peers,
            round_index=round_index,
        )
        self.prompts.append((seat, round_index, rendered.user))
        if seat.model in self._failing_models and round_index == self._failing_round:
            return AgentReply(signal=FLAT, prompt=rendered, failure=FailureMode.MALFORMED)
        # The mock derives its answer from the user turn alone. Every seat is handed
        # the same opening question -- the personas live in the system turn -- so
        # four seats would open with one identical rationale, which no real backend
        # would do and which would make a peer indistinguishable from the agent
        # reading it. The seat is folded in for the mock's benefit only; the prompt
        # the protocol built is what gets recorded, unchanged.
        completion = await self.provider.generate(
            system=rendered.system, user=f"{seat}\n{rendered.user}", schema=SIGNAL_SCHEMA
        )
        return AgentReply(signal=Signal.model_validate(completion.data), prompt=rendered)

    def prompts_in_round(self, round_index: int) -> tuple[str, ...]:
        return tuple(user for _, index, user in self.prompts if index == round_index)


def persona_names() -> tuple[str, ...]:
    return tuple(persona.name for persona in PERSONAS)

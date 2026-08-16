"""The two extension arms: the same-instrument placebo and the coherent contradictor.

Each test here guards a property the extension's *comparison* depends on rather
than a behaviour of the code for its own sake: donors that stay on the reader's
ticker, counters that cannot agree with the reader, a rendering the reader cannot
tell apart from a real debate's, and a cap the adjudication rule pinned before
the run.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from council.agents.mock import MockProvider
from council.debate.contra import (
    CONTRA_ROUND_CAP,
    counter_schema,
    generate_counters,
    opposite_bounds,
)
from council.debate.peers import NoPeersError, SeatView
from council.debate.placebo import select_placebo_point
from council.debate.protocol import PLACEBO_ARMS, StopReason, arm_round_cap, run_debate
from council.debate.sweep import has_donor
from council.domain.signal import Arm
from helpers_debate import (
    DAY,
    OTHER_DAYS,
    TICKER,
    MockCaller,
    committee,
    contested,
    placebo_pool,
    views_of,
)

OTHER_TICKER = "XOM"


def two_ticker_pool(days: tuple[date, ...] = OTHER_DAYS) -> dict:
    """Donor days on both tickers, so a draw is free to pick the wrong one."""
    table = committee()
    pool = dict(placebo_pool(table, days=days))
    for day in days:
        pool[(day, OTHER_TICKER)] = views_of(table, marker=f"foreign {day.isoformat()}")
    return pool


# -- the same-instrument placebo ---------------------------------------------------


def test_the_same_instrument_draw_never_leaves_the_readers_ticker() -> None:
    """The whole manipulation: D14 measured 49% cross-ticker donors in the
    unconstrained arm, and this arm exists to hold that factor at zero."""
    pool = two_ticker_pool()
    for round_index in (1, 2, 3):
        donor = select_placebo_point(
            pool=pool,
            point=(DAY, TICKER),
            composition=committee().identifier,
            required_seats=4,
            seed=7,
            round_index=round_index,
            min_gap=1,
            same_instrument=True,
        )
        assert donor[1] == TICKER, f"round {round_index} drew {donor}"


def test_the_unconstrained_draw_is_untouched_by_the_flag_existing() -> None:
    """The cross-instrument arm is already published; its draws must reproduce.

    The digest ordering runs over the candidate set, so if the flag's default
    leaked into the old arm's candidates, every stored placebo conversation
    would stop matching its own donor -- silently.
    """
    pool = two_ticker_pool()
    kwargs = dict(
        pool=pool,
        point=(DAY, TICKER),
        composition=committee().identifier,
        required_seats=4,
        seed=7,
        round_index=1,
        min_gap=1,
    )
    assert select_placebo_point(**kwargs) == select_placebo_point(**kwargs, same_instrument=False)


def test_per_round_redraw_stays_distinct_under_the_constraint() -> None:
    pool = two_ticker_pool()
    donors = {
        select_placebo_point(
            pool=pool,
            point=(DAY, TICKER),
            composition=committee().identifier,
            required_seats=4,
            seed=7,
            round_index=round_index,
            min_gap=1,
            same_instrument=True,
        )
        for round_index in (1, 2, 3, 4)
    }
    assert len(donors) == 4, "a repeated donor lets the arm settle on nothing new"


def test_has_donor_mirrors_the_constrained_draw() -> None:
    """The pre-flight and the draw must apply one test, or the sweep commits to a
    point the draw then refuses -- the drift class D-registered on the first run."""
    table = committee()
    pool = dict(placebo_pool(table))  # the reader's ticker only
    foreign_only = {(day, OTHER_TICKER): views_of(table, marker="foreign") for day in OTHER_DAYS}

    assert has_donor(
        pool, (DAY, TICKER), required_seats=4, min_gap=1, rounds=2, same_instrument=True
    )
    assert not has_donor(
        foreign_only, (DAY, TICKER), required_seats=4, min_gap=1, rounds=1, same_instrument=True
    )
    assert has_donor(
        foreign_only, (DAY, TICKER), required_seats=4, min_gap=1, rounds=1, same_instrument=False
    )


@pytest.mark.asyncio
async def test_the_same_instrument_arm_runs_the_full_protocol() -> None:
    """Same cap, same redraw, same rendering as the published placebo -- the two
    arms must differ in the donor's ticker and in nothing else."""
    caller = MockCaller()
    transcript = await run_debate(
        composition=committee(),
        arm=Arm.DEBATE_PLACEBO_SAME,
        dispersion=contested(),
        price_context="ctx",
        caller=caller,
        placebo_pool=two_ticker_pool(),
        seed=7,
        max_rounds=2,
        placebo_min_gap=1,
    )
    assert transcript.stop_reason in set(StopReason)
    assert Arm.DEBATE_PLACEBO_SAME in PLACEBO_ARMS
    prompts = caller.prompts_in_round(1)
    assert prompts, "the rebuttal round rendered no prompts"
    assert all("foreign" not in prompt for prompt in prompts), (
        "a cross-ticker donor's prose reached a same-instrument reader"
    )


# -- the coherent contradictor -----------------------------------------------------


def test_opposite_bounds_exclude_the_readers_side() -> None:
    assert opposite_bounds(0.6, token="t") == (-1.0, -0.05)
    assert opposite_bounds(-0.3, token="t") == (0.05, 1.0)
    low, high = opposite_bounds(0.0, token="t")
    assert (low, high) in {(0.25, 1.0), (-1.0, -0.25)}
    assert opposite_bounds(0.0, token="t") == opposite_bounds(0.0, token="t"), (
        "a flat reader's side must not depend on the draw"
    )


def test_counter_schema_carries_the_bounds_into_the_grammar() -> None:
    schema = counter_schema(0.8, token="t")
    assert schema["properties"]["exposure"]["minimum"] == -1.0
    assert schema["properties"]["exposure"]["maximum"] == -0.05
    from council.agents.prompt import SIGNAL_SCHEMA

    assert SIGNAL_SCHEMA["properties"]["exposure"].get("minimum") == -1.0
    assert SIGNAL_SCHEMA["properties"]["exposure"].get("maximum") == 1.0


@pytest.mark.asyncio
async def test_every_reader_gets_three_peer_authored_counters(tmp_path: Path) -> None:
    table = committee()
    providers = {model: MockProvider(model=model) for model in {s.model for s in table.seats}}
    openings = {seat: SeatView(seat=seat, exposure=0.5, rationale="up") for seat in table.seats}
    archive = tmp_path / "counters.jsonl"

    counters = await generate_counters(
        providers=providers,
        composition=table,
        point=(DAY, TICKER),
        price_context="ctx",
        openings=openings,
        max_tokens=64,
        archive=archive,
    )

    assert set(counters) == set(table.seats)
    for reader, authored in counters.items():
        assert len(authored) == 3, "dose parity: three peers, as the debate arm shows"
        assert reader not in {v.seat for v in authored}, "a reader cannot counter itself"
        # The mock invents numbers inside the schema's bounds, so the sign
        # constraint is exercised end to end rather than assumed.
        assert all(v.exposure < 0 for v in authored), "a counter to +0.5 must be short"

    rows = [json.loads(line) for line in archive.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 12, "one archived line per (reader, author) pair"
    assert {row["reader_model"] for row in rows} == {s.model for s in table.seats}


@pytest.mark.asyncio
async def test_a_missing_opening_abandons_the_conversation_rather_than_thinning_peers(
    tmp_path: Path,
) -> None:
    table = committee()
    providers = {model: MockProvider(model=model) for model in {s.model for s in table.seats}}
    openings = {seat: SeatView(seat=seat, exposure=0.5, rationale="up") for seat in table.seats[:3]}

    with pytest.raises(NoPeersError, match="every opening view"):
        await generate_counters(
            providers=providers,
            composition=table,
            point=(DAY, TICKER),
            price_context="ctx",
            openings=openings,
            max_tokens=64,
            archive=tmp_path / "counters.jsonl",
        )


def test_the_contradictor_cap_is_one_and_shared_by_plan_and_sweep() -> None:
    assert CONTRA_ROUND_CAP == 1
    assert arm_round_cap(Arm.DEBATE_CONTRADICTOR, 6) == 1
    assert arm_round_cap(Arm.DEBATE, 6) == 6
    assert arm_round_cap(Arm.DEBATE_PLACEBO_SAME, 6) == 6


@pytest.mark.asyncio
async def test_a_contradictor_conversation_stops_at_its_own_cap() -> None:
    table = committee()
    caller = MockCaller()
    providers = {model: MockProvider(model=model) for model in {s.model for s in table.seats}}

    async def contra(openings):
        return await generate_counters(
            providers=providers,
            composition=table,
            point=(DAY, TICKER),
            price_context="ctx",
            openings=openings,
            max_tokens=64,
            archive=None,
        )

    transcript = await run_debate(
        composition=table,
        arm=Arm.DEBATE_CONTRADICTOR,
        dispersion=contested(),
        price_context="ctx",
        caller=caller,
        contra_views=contra,
        seed=7,
        max_rounds=6,
    )

    assert transcript.rebuttal_rounds <= CONTRA_ROUND_CAP
    prompts = caller.prompts_in_round(1)
    assert len(prompts) == 4, "every seat answers the single rebuttal round"
    # Rendering parity: positions are shown, through the same peer block the
    # debate arm renders -- the "so must the rendering" pin.
    assert all("position" in prompt for prompt in prompts)


@pytest.mark.asyncio
async def test_the_contradictor_without_a_generator_is_refused() -> None:
    with pytest.raises(ValueError, match="counter-argument generator"):
        await run_debate(
            composition=committee(),
            arm=Arm.DEBATE_CONTRADICTOR,
            dispersion=contested(),
            price_context="ctx",
            caller=MockCaller(),
            seed=7,
        )


# -- the disposition prompts (D10's experiment) ------------------------------------


def test_the_disposition_briefs_differ_in_the_stance_voice_alone(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The D10 manipulation is one factor: identity phrasing against tendency
    phrasing. Everything else in the four briefs is byte-identical, so a result
    difference between the runs cannot be about anything but the voice -- and the
    loader's cache is keyed by directory, so one process can read both without
    serving either stale."""
    from council.agents.prompt import load_persona_brief
    from council.config import get_settings

    original = {
        name: load_persona_brief(name)
        for name in ("momentum-bold", "momentum-cautious", "reversion-bold", "reversion-cautious")
    }

    monkeypatch.setenv("COUNCIL_PROMPTS_DIR", "src/council/agents/prompts-disposition")
    get_settings.cache_clear()
    try:
        for name, before in original.items():
            after = load_persona_brief(name)
            assert after != before
            assert "tendency you have noticed in yourself" in after
            assert "not a rule you are bound to" in after
            # The sections outside the stance voice are untouched.
            assert (
                before.split("## How hard you commit")[1]
                == (after.split("## How hard you commit")[1])
            )
    finally:
        monkeypatch.delenv("COUNCIL_PROMPTS_DIR")
        get_settings.cache_clear()
    assert load_persona_brief("momentum-bold") == original["momentum-bold"]

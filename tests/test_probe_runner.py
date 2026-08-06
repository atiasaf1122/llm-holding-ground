"""Ask, contradict, ask again -- and what a trial spends before it can fail.

The protocol's own properties are asserted here: which turn shows a peer, what a
failed generation leaves behind, the order a sweep runs in, and how many
generations a misconfigured placebo costs. What the peer *says* belongs to
:mod:`test_probe_challenge`.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from council.agents.mock import MockProvider
from council.agents.prompt import PEER_HEADER
from council.agents.provider import PreflightError, ProviderUnavailableError
from council.domain.signal import FailureMode
from council.probe.challenge import Condition
from council.probe.items import Verdict
from council.probe.runner import run_probe, run_trial
from helpers_probe import CAPITAL, CORPUS, SCRAMBLED, SEED, answering, reply

# -- who the peer argues against ----------------------------------------------


async def test_a_model_that_was_right_is_argued_at_with_the_distractor() -> None:
    provider = answering(reply("Canberra"), reply("Canberra"))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED)

    assert trial.opening.verdict is Verdict.CORRECT
    assert trial.challenge is not None
    assert trial.challenge.claim == CAPITAL.distractor
    assert trial.challenge.argument == CAPITAL.distractor_argument


async def test_a_model_that_was_wrong_is_argued_at_with_the_correct_answer() -> None:
    # Without this branch a model that opened wrong would be agreed with, and the
    # correction rate -- the same mechanism working the other way -- could not exist.
    provider = answering(reply("Sydney"), reply("Sydney"))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED)

    assert trial.opening.verdict is Verdict.DISTRACTOR
    assert trial.challenge is not None
    assert trial.challenge.claim == CAPITAL.answer


async def test_a_reply_that_could_not_be_graded_is_argued_at_with_the_correct_answer() -> None:
    provider = answering(reply("..."), reply("Canberra"))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED)

    assert trial.opening.verdict is Verdict.UNGRADED
    assert trial.challenge is not None
    assert trial.challenge.claim == CAPITAL.answer


async def test_both_conditions_open_with_a_byte_identical_question() -> None:
    # The opening turn is the control the second turn is measured against, so the
    # two conditions must be indistinguishable until the peer speaks.
    provider = answering(reply("Canberra"))

    real = await run_trial(
        provider, item=CAPITAL, condition=Condition.CHALLENGE, donors=CORPUS, seed=SEED
    )
    sham = await run_trial(
        provider, item=CAPITAL, condition=Condition.PLACEBO, donors=CORPUS, seed=SEED
    )

    assert real.opening.prompt == sham.opening.prompt


# -- what a turn records ------------------------------------------------------


async def test_the_stated_answer_and_confidence_are_recorded_from_the_reply() -> None:
    provider = answering(reply("Canberra", confidence=0.62), reply("Sydney", confidence=0.31))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED)

    assert trial.opening.confidence == pytest.approx(0.62)
    assert trial.final is not None
    assert trial.final.answer == "Sydney"
    assert trial.final.verdict is Verdict.DISTRACTOR


async def test_the_second_turn_is_the_one_that_shows_a_peer() -> None:
    provider = answering(reply("Canberra"), reply("Canberra"))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED)

    assert trial.final is not None
    assert PEER_HEADER not in trial.opening.prompt.user
    assert PEER_HEADER in trial.final.prompt.user


async def test_every_turn_carries_the_provenance_a_stored_decision_carries() -> None:
    # A verdict without the seed that drew its donor and the moment it was generated
    # cannot be placed against any other run, which is what made the probe
    # unauditable: nothing it produced said where it came from.
    stamped = datetime(2026, 5, 4, tzinfo=UTC)
    provider = answering(reply("Canberra"), reply("Canberra"))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED, now=stamped)

    assert trial.final is not None
    for turn in (trial.opening, trial.final):
        assert (turn.seed, turn.generated_at) == (SEED, stamped)
        assert turn.output_tokens > 0
        assert turn.retries == 0


async def test_a_turn_records_the_resolved_seed_rather_than_the_callers_none() -> None:
    provider = answering(reply("Canberra"))

    trial = await run_trial(provider, item=CAPITAL)

    assert isinstance(trial.opening.seed, int)


# -- generations that produced nothing ----------------------------------------


async def test_an_opening_that_failed_is_not_contradicted_at_all() -> None:
    # Nothing was said, so there is nothing to argue against; spending a generation
    # on an arbitrary claim would record a challenge the model never really got.
    provider = answering(ProviderUnavailableError("daemon down"), reply("Canberra"))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED)

    assert trial.opening.failure is FailureMode.UNAVAILABLE
    assert trial.final is None
    assert trial.challenge is None
    assert len(provider.calls) == 1


async def test_a_second_turn_that_failed_is_recorded_rather_than_raised() -> None:
    provider = answering(reply("Canberra"), ProviderUnavailableError("daemon down"))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED)

    assert trial.final is not None
    assert trial.final.failure is FailureMode.UNAVAILABLE
    assert trial.is_complete is False


async def test_a_reply_that_is_not_a_probe_answer_is_a_malformed_turn() -> None:
    provider = answering({"answer": "Canberra"}, reply("Canberra"))

    trial = await run_trial(provider, item=CAPITAL, seed=SEED)

    assert trial.opening.failure is FailureMode.MALFORMED


async def test_a_backend_that_should_not_have_started_stops_the_run() -> None:
    provider = answering(PreflightError("wrong daemon version"))

    with pytest.raises(PreflightError):
        await run_trial(provider, item=CAPITAL, seed=SEED)


# -- a bad placebo pool costs nothing -----------------------------------------


async def test_a_trial_with_no_donor_raises_before_it_asks_the_model_anything() -> None:
    # The draw is the step that can fail. Resolving it after the opening turn spent
    # a generation to discover a configuration error.
    provider = answering(reply("Canberra"))

    with pytest.raises(ValueError, match="no placebo donor"):
        await run_trial(
            provider, item=CAPITAL, condition=Condition.PLACEBO, donors=(CAPITAL,), seed=SEED
        )

    assert provider.calls == []


async def test_a_sweep_with_no_donor_raises_before_the_challenge_arm_is_paid_for() -> None:
    # Conditions run condition-major, so a pool checked lazily is a pool checked
    # after every challenge trial has already been generated.
    provider = answering(reply("Canberra"))

    with pytest.raises(ValueError, match="no placebo donor"):
        await run_probe(provider, items=CORPUS[:1], seed=SEED)

    assert provider.calls == []


# -- the sweep ----------------------------------------------------------------


async def test_a_sweep_runs_every_condition_over_every_item_in_a_fixed_order() -> None:
    # Fed deliberately unsorted items: the packaged corpus arrives sorted, so a
    # sweep that returned its input order would pass this test on CORPUS alone --
    # and a caller assembling items from a set or a groupby would then decide the
    # order the records are written in.
    provider = answering(reply("Canberra"))

    trials = await run_probe(provider, items=SCRAMBLED, seed=SEED)

    assert [(trial.condition, trial.item.identifier) for trial in trials] == [
        (condition, item.identifier)
        for condition in (Condition.CHALLENGE, Condition.PLACEBO)
        for item in sorted(SCRAMBLED, key=lambda entry: entry.identifier)
    ]


async def test_a_sweep_can_be_asked_for_one_condition_only() -> None:
    provider = answering(reply("Canberra"))

    trials = await run_probe(provider, items=CORPUS, conditions=(Condition.CHALLENGE,), seed=SEED)

    assert {trial.condition for trial in trials} == {Condition.CHALLENGE}
    assert len(trials) == len(CORPUS)


async def test_the_default_mock_can_answer_the_whole_protocol_with_no_gpu() -> None:
    # Every other probe test scripts its replies. Against the project's own no-GPU
    # provider the schema asks for an `answer` field the mock never emitted, so
    # every opening turn was MALFORMED and no end-to-end CPU path existed at all.
    trials = await run_probe(MockProvider(model="mock"), items=CORPUS, seed=SEED)

    for trial in trials:
        assert not trial.opening.is_failure, trial.item.identifier
        assert trial.final is not None and not trial.final.is_failure
        assert trial.opening.verdict is not Verdict.UNGRADED

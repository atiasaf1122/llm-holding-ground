"""Whether a run that was interrupted can be told what it already has."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, date, datetime
from pathlib import Path

import pandas as pd
import pytest

from council.agents.prompt import prompt_hash
from council.agents.store import (
    KEY_COLUMNS,
    STORED_COLUMNS,
    CompletionRecord,
    DecisionStore,
    decision_key,
    part_filename,
    to_storage_frame,
)
from council.domain.signal import Arm, Decision, FailureMode, StopReason
from council.evaluation.frames import NO_COMPOSITION, frame_to_rows

GENERATED_AT = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)


def make_decision(
    *,
    on: date = date(2022, 3, 1),
    ticker: str = "AAPL",
    model: str = "qwen3:8b",
    persona: str = "momentum-bold",
    exposure: float = 0.5,
    failure: FailureMode = FailureMode.NONE,
    arm: Arm = Arm.INDEPENDENT,
    composition: str | None = None,
) -> Decision:
    return Decision(
        decision_date=on,
        ticker=ticker,
        model=model,
        persona=persona,
        arm=arm,
        composition=composition,
        exposure=exposure,
        confidence=0.6,
        rationale="trend intact",
        prompt_hash="abc123",
        seed=1,
        generated_at=GENERATED_AT,
        failure=failure,
    )


@pytest.fixture
def store(tmp_path: Path) -> DecisionStore:
    return DecisionStore(
        decisions_path=tmp_path / "decisions.parquet",
        completions_path=tmp_path / "completions.jsonl",
    )


def make_record(*, on: date = date(2022, 3, 1)) -> CompletionRecord:
    return CompletionRecord(
        decision_date=on,
        ticker="AAPL",
        model="qwen3:8b",
        persona="momentum-bold",
        arm=str(Arm.INDEPENDENT),
        round_index=0,
        composition=NO_COMPOSITION,
        prompt_hash="abc123",
        system="you are an analyst",
        user="returns: +1.00",
        response={"exposure": 0.5},
        error=None,
        latency_seconds=1.5,
    )


# -- the stored shape ----------------------------------------------------------


def test_every_decision_field_is_written_to_disk() -> None:
    # A field added to Decision and forgotten here is dropped from the whole run,
    # and is discovered on the night the column is first needed.
    assert tuple(Decision.model_fields) == STORED_COLUMNS


def test_the_stored_frame_is_readable_by_the_evaluation_layer() -> None:
    frame = to_storage_frame([make_decision()])

    rows = frame_to_rows(frame)

    assert len(rows) == 1
    assert rows[0].exposure == 0.5
    assert rows[0].composition == NO_COMPOSITION


def test_the_independent_arm_stores_an_empty_composition_not_a_null() -> None:
    # A null here silently drops the entire control arm out of any grouping keyed
    # on composition.
    frame = to_storage_frame([make_decision()])

    assert frame["composition"].to_list() == [NO_COMPOSITION]


def test_a_model_tag_with_a_colon_produces_a_legal_filename() -> None:
    name = part_filename(model="qwen3:8b", persona="momentum-bold", ticker="AAPL", keys=())

    assert ":" not in name
    assert name.endswith(".parquet")


def test_models_differing_only_in_punctuation_do_not_share_a_part_file() -> None:
    colon = part_filename(model="qwen3:8b", persona="momentum-bold", ticker="AAPL", keys=())
    dash = part_filename(model="qwen3-8b", persona="momentum-bold", ticker="AAPL", keys=())

    assert colon != dash


def test_the_same_agent_in_two_arms_does_not_share_a_part_file() -> None:
    # The triple is not what makes a decision unique: the arm, the round and the
    # composition are part of its identity too. One filename for two arms means the
    # second checkpoint deletes the first arm's rows before consolidation sees them.
    independent = make_decision(arm=Arm.INDEPENDENT)
    debated = make_decision(arm=Arm.DEBATE, composition="quad")

    assert part_filename(
        model="qwen3:8b", persona="momentum-bold", ticker="AAPL", keys=[decision_key(independent)]
    ) != part_filename(
        model="qwen3:8b", persona="momentum-bold", ticker="AAPL", keys=[decision_key(debated)]
    )


def test_the_archive_is_deduplicated_on_the_key_columns_and_not_on_the_prompt_hash() -> None:
    # `DecisionStore.checkpoint` tells a reader how to clean an archive a crash
    # duplicated, so the key it names has to be one. `prompt_hash` is not:
    # :func:`council.agents.prompt.prompt_hash` digests the two prompt turns
    # alone, carrying no model, arm, composition or round, so every model shown
    # the same text shares one. In `data/completions.jsonl` its 28,608 lines hold
    # 13,744 distinct values with 28 lines on the largest, and deduplicating on it
    # would delete 14,864 genuinely distinct decisions from the one artefact that
    # exists so a later question can be answered without another night of
    # inference.
    shared = prompt_hash("you are an analyst", "returns: +1.00")
    one = replace(make_record(), model="qwen3:8b", prompt_hash=shared)
    another = replace(make_record(), model="gemma4:12b", prompt_hash=shared)

    def archive_key(line: CompletionRecord) -> tuple[object, ...]:
        return tuple(line.as_json()[column] for column in KEY_COLUMNS)

    assert one.prompt_hash == another.prompt_hash
    assert archive_key(one) != archive_key(another)

    documented = DecisionStore.checkpoint.__doc__ or ""
    assert "KEY_COLUMNS" in documented
    assert "``prompt_hash`` and can be deduplicated" not in documented


def test_a_part_file_is_named_the_same_whatever_order_its_decisions_arrive_in() -> None:
    earlier = decision_key(make_decision(on=date(2022, 3, 1)))
    later = decision_key(make_decision(on=date(2022, 3, 2)))

    forward = part_filename(model="m", persona="p", ticker="AAPL", keys=[earlier, later])
    backward = part_filename(model="m", persona="p", ticker="AAPL", keys=[later, earlier])

    assert forward == backward


# -- checkpointing and resuming ------------------------------------------------


def test_a_checkpoint_is_visible_before_the_run_is_consolidated(store: DecisionStore) -> None:
    decision = make_decision()

    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[decision],
        completions=[make_record()],
    )

    assert store.completed_keys() == frozenset({decision_key(decision)})
    assert not store.decisions_path.exists()


def test_keys_survive_the_round_trip_through_parquet(store: DecisionStore) -> None:
    # The dates come back as a different type than they went in, and the two do
    # not compare equal; a resume that could not recognise its own rows would
    # regenerate the whole sweep.
    decision = make_decision()
    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[decision],
        completions=[],
    )
    store.consolidate()

    assert decision_key(decision) in store.completed_keys()


def test_consolidation_merges_the_parts_and_clears_the_directory(store: DecisionStore) -> None:
    first = make_decision(ticker="AAPL")
    second = make_decision(ticker="XOM")
    for decision in (first, second):
        store.checkpoint(
            model="qwen3:8b",
            persona="momentum-bold",
            ticker=decision.ticker,
            decisions=[decision],
            completions=[],
        )

    store.consolidate()

    stored = pd.read_parquet(store.decisions_path)
    assert len(stored) == 2
    assert not store.parts_dir.exists()


def test_consolidating_twice_changes_nothing(store: DecisionStore) -> None:
    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[make_decision()],
        completions=[],
    )
    store.consolidate()
    first = store.decisions_path.read_bytes()

    store.consolidate()

    assert store.decisions_path.read_bytes() == first


def test_consolidating_an_empty_store_writes_nothing(store: DecisionStore) -> None:
    store.consolidate()

    assert not store.decisions_path.exists()


def test_a_regenerated_decision_supersedes_the_one_already_stored(store: DecisionStore) -> None:
    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[make_decision(exposure=0.5)],
        completions=[],
    )
    store.consolidate()

    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[make_decision(exposure=-0.9)],
        completions=[],
    )
    store.consolidate()

    stored = pd.read_parquet(store.decisions_path)
    assert stored["exposure"].to_list() == [-0.9]


def test_two_arms_for_one_agent_both_survive_a_run(store: DecisionStore) -> None:
    # The reviewer's scenario: one triple checkpointed twice in one run, once per
    # arm. Keyed on the triple alone the second write lands on the first's
    # filename, and the independent row is gone before consolidation can see it --
    # no error, and nothing downstream able to notice a whole arm is short.
    store.checkpoint(
        model="alpha",
        persona="momentum-bold",
        ticker="AAA",
        decisions=[make_decision(arm=Arm.INDEPENDENT, exposure=0.4)],
        completions=[],
    )
    store.checkpoint(
        model="alpha",
        persona="momentum-bold",
        ticker="AAA",
        decisions=[make_decision(arm=Arm.DEBATE, composition="quad", exposure=-0.9)],
        completions=[],
    )
    store.consolidate()

    stored = pd.read_parquet(store.decisions_path)
    assert sorted(stored["arm"]) == [str(Arm.DEBATE), str(Arm.INDEPENDENT)]
    assert sorted(stored["exposure"]) == [-0.9, 0.4]


def test_a_checkpoint_with_nothing_in_it_does_not_change_the_stored_types(
    store: DecisionStore,
) -> None:
    # An all-object empty frame concatenated with real parts downgrades every
    # column of the published artefact to object, so the dtypes of the project's
    # primary output would depend on whether some triple happened to be empty.
    store.checkpoint(
        model="alpha", persona="momentum-bold", ticker="ZZZ", decisions=[], completions=[]
    )
    store.checkpoint(
        model="alpha",
        persona="momentum-bold",
        ticker="AAA",
        decisions=[make_decision()],
        completions=[],
    )
    store.consolidate()

    stored = pd.read_parquet(store.decisions_path)
    assert stored["exposure"].dtype == "float64"
    assert stored["round_index"].dtype == "int64"


# -- a failure is not an answer ------------------------------------------------


def test_a_point_the_backend_could_not_answer_is_regenerated(store: DecisionStore) -> None:
    # A transient outage otherwise bakes exposure=0.0 into the arm permanently: the
    # row exists, so a rerun on a healthy backend issues nothing at all.
    unreachable = make_decision(failure=FailureMode.UNAVAILABLE, exposure=0.0)
    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[unreachable],
        completions=[],
    )
    store.consolidate()

    assert store.completed_keys() == frozenset()


def test_a_malformed_answer_is_not_asked_for_again(store: DecisionStore) -> None:
    # Temperature is zero, so a second attempt reproduces the first exactly. This
    # one is a finding about the model, not an outage.
    refused = make_decision(failure=FailureMode.MALFORMED, exposure=0.0)
    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[refused],
        completions=[],
    )

    assert store.completed_keys() == frozenset({decision_key(refused)})


def test_a_regenerated_failure_is_replaced_by_the_answer(store: DecisionStore) -> None:
    for decision in (
        make_decision(failure=FailureMode.UNAVAILABLE, exposure=0.0),
        make_decision(exposure=0.5),
    ):
        store.checkpoint(
            model="qwen3:8b",
            persona="momentum-bold",
            ticker="AAPL",
            decisions=[decision],
            completions=[],
        )
        store.consolidate()

    stored = pd.read_parquet(store.decisions_path)
    assert stored["failure"].to_list() == [str(FailureMode.NONE)]
    assert stored["exposure"].to_list() == [0.5]


def test_a_store_cannot_be_told_to_regenerate_decisions_that_succeeded(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="never regenerated"):
        DecisionStore(
            decisions_path=tmp_path / "d.parquet",
            completions_path=tmp_path / "c.jsonl",
            retry_failures=frozenset({FailureMode.NONE}),
        )


def test_stored_rows_are_ordered_by_their_key(store: DecisionStore) -> None:
    later = make_decision(on=date(2022, 3, 2))
    earlier = make_decision(on=date(2022, 3, 1))
    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[later, earlier],
        completions=[],
    )

    store.consolidate()

    stored = pd.read_parquet(store.decisions_path)
    assert list(stored.columns[: len(KEY_COLUMNS)]) == list(KEY_COLUMNS)
    assert [str(value) for value in stored["decision_date"]] == ["2022-03-01", "2022-03-02"]


# -- the completions archive ---------------------------------------------------


def test_completions_are_appended_one_json_object_per_line(store: DecisionStore) -> None:
    for _ in range(2):
        store.checkpoint(
            model="qwen3:8b",
            persona="momentum-bold",
            ticker="AAPL",
            decisions=[make_decision()],
            completions=[make_record()],
        )

    lines = store.completions_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["system"] == "you are an analyst"


def test_the_archive_is_written_in_key_order_whatever_order_the_batch_arrived_in(
    tmp_path: Path,
) -> None:
    # The archive promises byte-identical output for two runs of one configuration.
    # `sort_keys` orders the fields inside a line and nothing else: a debate round
    # puts every seat in flight under `asyncio.gather` and `DecisionCaller` appends
    # each reply as it lands, so the list reaching this method is in completion
    # order -- whichever model was fastest that night.
    batch = [
        replace(make_record(), model=model, round_index=round_index)
        for round_index in (0, 1)
        for model in ("phi4:14b", "qwen3.5:9b")
    ]

    archives = []
    for index, order in enumerate((batch, list(reversed(batch)))):
        store = DecisionStore(
            decisions_path=tmp_path / f"decisions-{index}.parquet",
            completions_path=tmp_path / f"completions-{index}.jsonl",
        )
        store.checkpoint(
            model="rotation-0",
            persona=str(Arm.DEBATE),
            ticker="AAPL",
            decisions=[],
            completions=order,
        )
        archives.append(store.completions_path.read_text(encoding="utf-8"))

    assert archives[0] == archives[1]
    written = [json.loads(line) for line in archives[0].splitlines()]
    keys = [(line["model"], line["round_index"]) for line in written]
    assert keys == sorted(keys)


def test_the_archive_keeps_the_whole_prompt_and_the_raw_response(store: DecisionStore) -> None:
    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[],
        completions=[make_record()],
    )

    archived = json.loads(
        store.completions_path.read_text(encoding="utf-8").strip()
    )
    assert archived["user"] == "returns: +1.00"
    assert archived["response"] == {"exposure": 0.5}
    assert archived["error"] is None
    # An archive line is not progress: only a stored decision marks a point done.
    assert store.completed_keys() == frozenset()


# -- a conversation is finished when it says why it ended ----------------------


def conversation(
    *,
    rounds: int,
    stop_reason: StopReason | None,
    failure: FailureMode = FailureMode.NONE,
    failed_round: int = 0,
) -> list[Decision]:
    """One committee's stored conversation: two seats over ``rounds`` rounds."""
    return [
        make_decision(
            arm=Arm.DEBATE,
            composition="rotation-0",
            model=model,
            failure=failure if index == failed_round else FailureMode.NONE,
        ).model_copy(update={"round_index": index, "stop_reason": stop_reason})
        for index in range(rounds)
        for model in ("alpha", "beta")
    ]


def keep(store: DecisionStore, decisions: list[Decision]) -> None:
    store.checkpoint(
        model="rotation-0", persona="debate", ticker="AAPL", decisions=decisions, completions=[]
    )


DEBATE_CONVERSATION = (date(2022, 3, 1), "AAPL", str(Arm.DEBATE), "rotation-0")


def test_a_conversation_that_stopped_early_is_finished_rather_than_owing_rounds(
    store: DecisionStore,
) -> None:
    # The resume test used to demand a stored row for every round `0..cap`. A
    # conversation that agreed at round two never satisfies it, so the sweep
    # re-debates a point it already owns on every resume and the plan that prices the
    # run can never reach zero. Two rounds out of a cap of six, marked AGREED.
    keep(store, conversation(rounds=2, stop_reason=StopReason.AGREED))

    assert store.completed_conversations() == frozenset({DEBATE_CONVERSATION})


def test_an_ongoing_conversation_is_not_finished_at_rounds_zero_and_one(
    store: DecisionStore,
) -> None:
    # The other half, and the one that decides whether the marker is worth having:
    # rounds 0 and 1 of a conversation still running look exactly like a conversation
    # that agreed at round 1, except for the reason. Without the reason a resumed run
    # would abandon it half finished.
    keep(store, conversation(rounds=2, stop_reason=None))

    assert store.completed_conversations() == frozenset()


@pytest.mark.parametrize(
    "reason", [StopReason.AGREED, StopReason.SETTLED, StopReason.CAP, StopReason.NO_SPEAKERS]
)
def test_every_stop_reason_ends_a_conversation_including_the_abandoned_one(
    store: DecisionStore, reason: StopReason
) -> None:
    # NO_SPEAKERS is an abandoned conversation and a finished one at the same time: a
    # round every seat botched reproduces exactly at temperature zero, so re-holding
    # it would spend a night confirming it. The case a resume *is* for is the next
    # test, where the failure is the backend rather than the model.
    keep(store, conversation(rounds=2, stop_reason=reason))

    assert store.completed_conversations() == frozenset({DEBATE_CONVERSATION})


def test_one_unreachable_row_leaves_the_whole_conversation_unfinished(
    store: DecisionStore,
) -> None:
    # `completed_keys` already promised this per row -- an hour with the daemon down
    # must not bake a flat exposure into the arm permanently -- and the sweep re-holds
    # conversations rather than seats, so one retriable row unmakes the conversation.
    keep(
        store,
        conversation(
            rounds=3, stop_reason=StopReason.CAP, failure=FailureMode.UNAVAILABLE, failed_round=1
        ),
    )

    assert store.completed_conversations() == frozenset()


def test_a_malformed_row_does_not_reopen_a_finished_conversation(
    store: DecisionStore,
) -> None:
    # The other side of `RETRIED_FAILURES`: a completion the schema rejected is
    # reproduced exactly by a second attempt, and re-holding the conversation would
    # spend the whole night confirming it.
    keep(
        store,
        conversation(
            rounds=3, stop_reason=StopReason.CAP, failure=FailureMode.MALFORMED, failed_round=1
        ),
    )

    assert store.completed_conversations() == frozenset({DEBATE_CONVERSATION})


def test_the_independent_arm_holds_no_conversations(store: DecisionStore) -> None:
    # It has no stop reason to carry, and the empty string must not be read as one.
    store.checkpoint(
        model="qwen3:8b",
        persona="momentum-bold",
        ticker="AAPL",
        decisions=[make_decision()],
        completions=[],
    )

    assert store.completed_conversations() == frozenset()


def test_the_stop_reason_survives_the_round_trip_through_parquet(
    store: DecisionStore,
) -> None:
    keep(store, conversation(rounds=2, stop_reason=StopReason.SETTLED))
    store.consolidate()

    frame = pd.read_parquet(store.decisions_path)

    assert set(frame["stop_reason"]) == {str(StopReason.SETTLED)}


def test_a_store_written_before_the_column_existed_still_opens(
    store: DecisionStore,
) -> None:
    # `stop_reason` was added after decisions had already been written, and a parquet
    # file is read by name. Refusing to open those artefacts is a worse answer than
    # "no conversation in this file recorded a stopping condition" -- which is both
    # true and what makes the sweep hold them again.
    keep(store, conversation(rounds=2, stop_reason=StopReason.AGREED))
    store.consolidate()
    older = pd.read_parquet(store.decisions_path).drop(columns=["stop_reason"])
    older.to_parquet(store.decisions_path, index=False)

    assert store.completed_conversations() == frozenset()
    assert store.completed_keys()

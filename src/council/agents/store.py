"""Writing a run down as it goes, so an interrupted one can be resumed.

A full sweep is an overnight job. Anything that only exists in memory until the
end is a thing a reboot deletes, so decisions land on disk as they are produced,
in whole (model, persona, ticker) parts.

The parts are separate files rather than appends to one parquet. Parquet keeps its
index in a footer written at close, so a process killed while holding one open
leaves a file nothing can read -- and the run then cannot restart without someone
deleting artefacts by hand. A part file is written to a temporary name and moved
into place, so at every instant every file in the directory is complete.

:meth:`DecisionStore.consolidate` folds the parts back into the single frame the
evaluation package reads. It is safe to call at any point, including twice, which
is what lets a resumed run start from a clean directory.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Final

import pandas as pd

from council.domain.signal import COMPLETED_STOP_REASONS, Decision, FailureMode
from council.evaluation.frames import NO_COMPOSITION

STORED_COLUMNS: Final[tuple[str, ...]] = (
    "decision_date",
    "ticker",
    "model",
    "persona",
    "arm",
    "round_index",
    "composition",
    "stop_reason",
    "exposure",
    "confidence",
    "rationale",
    "prompt_hash",
    "seed",
    "generated_at",
    "failure",
    "retries",
    "latency_seconds",
    "output_tokens",
)
"""Every field of :class:`~council.domain.signal.Decision`, written out by name.

Spelled out rather than derived from the model so that a field added to
``Decision`` fails a test here instead of being silently dropped from eighty
thousand rows -- which is discovered, if at all, on the night the column is first
needed.
"""

KEY_COLUMNS: Final[tuple[str, ...]] = (
    "decision_date",
    "ticker",
    "model",
    "persona",
    "arm",
    "round_index",
    "composition",
)
"""What makes a decision the same decision. Two rows sharing these are one
regenerated point, not two observations, and the later one wins."""

DecisionKey = tuple[date, str, str, str, str, int, str]

ConversationKey = tuple[date, str, str, str]
"""``(decision_date, ticker, arm, composition)`` -- one conversation.

:data:`DecisionKey` without the seat and without the round, which is exactly the
unit :meth:`council.debate.sweep._Sweep.group` resumes on. It cannot resume on
:data:`DecisionKey` any more: the length of a conversation is now an outcome of the
debate, so there is no set of keys a finished conversation is guaranteed to hold.
"""

STOP_REASON: Final = "stop_reason"
"""The stored column carrying
:attr:`~council.domain.signal.Decision.stop_reason`."""

NO_STOP_REASON: Final = ""
"""How a row belonging to no finished conversation is written.

The empty string rather than null, for the reason
:data:`~council.evaluation.frames.NO_COMPOSITION` is: a null groups to nothing and
compares false against everything, so an arm's rows would drop out of any tally
keyed on this column with no error and no gap anybody would notice. Every
independent-arm row carries it, and so does every row of a conversation that raised
part way through.
"""

RETRIED_FAILURES: Final[frozenset[FailureMode]] = frozenset(
    {FailureMode.UNAVAILABLE, FailureMode.TRUNCATED}
)
"""Which recorded failures a later run treats as unfinished rather than as done.

Both of these are faults of the backend rather than answers from the model. An
hour with the daemon down, or a wrong ``ollama_base_url``, otherwise bakes a flat
exposure into the arm permanently: the rows exist, so a rerun of the identical
configuration issues no inference and reports nothing to do, and because the
failure rate per model is a published result the outage is then indistinguishable
from a finding about that model.

:attr:`~council.domain.signal.FailureMode.MALFORMED` is deliberately absent.
Temperature is zero, so a completion the schema rejected is reproduced exactly by
a second attempt; retrying it forever would spend a whole night confirming it.
"""

PARTS_SUFFIX: Final = ".parts"
PART_PREFIX: Final = "part-"
TEMP_SUFFIX: Final = ".tmp"

_UNSAFE_IN_FILENAME: Final = re.compile(r"[^A-Za-z0-9._-]+")
_PART_DIGEST_BYTES: Final = 6


@dataclass(frozen=True, slots=True)
class CompletionRecord:
    """One request and what came back from it, kept for later re-reading.

    Stored so that a question thought of after the run -- a different way of
    scoring rationales, an audit that the anonymisation held -- can be answered
    from the archive rather than from another night of inference.

    ``response`` is the parsed object rather than the literal completion text: a
    provider parses before this code sees anything. On a failure it is ``None``
    and ``error`` carries the message, which for a malformed completion includes
    the beginning of the text that could not be parsed.
    """

    decision_date: date
    ticker: str
    model: str
    persona: str
    arm: str
    round_index: int
    composition: str
    prompt_hash: str
    system: str
    user: str
    response: dict[str, Any] | None
    error: str | None
    latency_seconds: float

    def as_json(self) -> dict[str, Any]:
        return {
            "decision_date": self.decision_date.isoformat(),
            "ticker": self.ticker,
            "model": self.model,
            "persona": self.persona,
            "arm": self.arm,
            "round_index": self.round_index,
            "composition": self.composition,
            "prompt_hash": self.prompt_hash,
            "system": self.system,
            "user": self.user,
            "response": self.response,
            "error": self.error,
            "latency_seconds": self.latency_seconds,
        }


def decision_key(decision: Decision) -> DecisionKey:
    """The identity of one decision, in the form the resume check compares."""
    return (
        decision.decision_date,
        decision.ticker,
        decision.model,
        decision.persona,
        str(decision.arm),
        decision.round_index,
        decision.composition or NO_COMPOSITION,
    )


def _completion_sort_key(record: CompletionRecord) -> DecisionKey:
    """:data:`KEY_COLUMNS` read off an archive record, so a batch of them can be
    written in the order :meth:`DecisionStore.consolidate` sorts the parquet into."""
    return (
        record.decision_date,
        record.ticker,
        record.model,
        record.persona,
        record.arm,
        record.round_index,
        record.composition,
    )


def to_storage_frame(decisions: Sequence[Decision]) -> pd.DataFrame:
    """Flatten decisions into the frame that goes to parquet.

    ``composition`` is written as the empty string rather than as null, matching
    :data:`~council.evaluation.frames.NO_COMPOSITION`. A null there would drop the
    entire independent arm out of any grouping keyed on the column, without an
    error and without a gap anyone would notice.
    """
    records = [
        {
            "decision_date": decision.decision_date,
            "ticker": decision.ticker,
            "model": decision.model,
            "persona": decision.persona,
            "arm": str(decision.arm),
            "round_index": decision.round_index,
            "composition": decision.composition or NO_COMPOSITION,
            "stop_reason": str(decision.stop_reason or NO_STOP_REASON),
            "exposure": decision.exposure,
            "confidence": decision.confidence,
            "rationale": decision.rationale,
            "prompt_hash": decision.prompt_hash,
            "seed": decision.seed,
            "generated_at": decision.generated_at,
            "failure": str(decision.failure),
            "retries": decision.retries,
            "latency_seconds": decision.latency_seconds,
            "output_tokens": decision.output_tokens,
        }
        for decision in decisions
    ]
    return pd.DataFrame.from_records(records, columns=list(STORED_COLUMNS))


def part_filename(*, model: str, persona: str, ticker: str, keys: Sequence[DecisionKey]) -> str:
    """A stable, filesystem-legal name for one checkpoint's part file.

    The name is the readable triple over a digest of the decision keys the part
    holds. Both halves are load-bearing.

    The digest is taken over the keys rather than over the triple because the
    triple is not what makes a decision unique: :data:`KEY_COLUMNS` also carries
    the arm, the round and the composition. A name keyed on the triple alone gives
    one file to two checkpoints for the same agent in two different arms, and the
    second silently deletes the first before :meth:`DecisionStore.consolidate` ever
    sees it -- no error, no warning, and nothing downstream able to notice a whole
    arm is short. Deriving the name from :func:`decision_key` means any column
    later added to the identity is carried into the filename by construction
    rather than by someone remembering to.

    The readable half stays because a parts directory nobody can read during a
    long run is a directory nobody checks. It is sanitised because a model tag
    contains a colon, which Windows will not accept in a filename -- and because
    sanitising alone would map ``qwen3:8b`` and ``qwen3-8b`` onto one name.
    """
    # Keys sorted, so the same set of decisions names the same file whatever order
    # the caller assembled them in.
    fields = (model, persona, ticker, *("\0".join(map(str, key)) for key in sorted(keys)))
    digest = hashlib.blake2b(b"".join(map(_framed, fields)), digest_size=_PART_DIGEST_BYTES)
    slug = "-".join(_UNSAFE_IN_FILENAME.sub("-", part) for part in (model, persona, ticker))
    return f"{PART_PREFIX}{slug}-{digest.hexdigest()}.parquet"


def _framed(field: str) -> bytes:
    """Length-prefixed, so no two lists of fields ever produce the same bytes."""
    encoded = field.encode("utf-8")
    return len(encoded).to_bytes(8, "big") + encoded


class DecisionStore:
    """Incremental, resumable storage for one generation run."""

    def __init__(
        self,
        *,
        decisions_path: Path,
        completions_path: Path,
        retry_failures: frozenset[FailureMode] = RETRIED_FAILURES,
    ) -> None:
        if FailureMode.NONE in retry_failures:
            raise ValueError("a decision that did not fail is finished; it is never regenerated")
        self._decisions_path = decisions_path
        self._completions_path = completions_path
        self._parts_dir = decisions_path.with_name(decisions_path.stem + PARTS_SUFFIX)
        self._retry_failures = frozenset(str(mode) for mode in retry_failures)

    @property
    def decisions_path(self) -> Path:
        return self._decisions_path

    @property
    def completions_path(self) -> Path:
        return self._completions_path

    @property
    def parts_dir(self) -> Path:
        return self._parts_dir

    def completed_keys(self) -> frozenset[DecisionKey]:
        """Every decision point already answered, consolidated or not.

        Read at the granularity of a single decision rather than of a checkpoint,
        so that extending the date range of an existing run generates the new
        dates instead of skipping every triple that already has a file.

        A row recording a failure in :data:`RETRIED_FAILURES` is *not* answered.
        Those rows are still stored -- the failure rate is a result -- but a row
        saying the daemon was unreachable is a record of an outage, and treating it
        as a finished decision is what makes a rerun on a healthy backend issue
        nothing at all.
        """
        keys: set[DecisionKey] = set()
        for path in self._sources():
            frame = pd.read_parquet(path, columns=[*KEY_COLUMNS, "failure"])
            answered = frame[~frame["failure"].isin(self._retry_failures)]
            keys.update(_keys_of(answered))
        return frozenset(keys)

    def completed_conversations(self) -> frozenset[ConversationKey]:
        """Every conversation whose stored rows say it reached a stopping condition.

        The debate's counterpart to :meth:`completed_keys`, and it exists because
        that method cannot answer the question any more. A conversation now ends on
        agreement, on stillness or at the cap, so the rows it leaves behind are
        however many rounds it happened to need; a resume test that asks for a row
        at every round up to the cap is never satisfied by a conversation that
        agreed at round two, and the sweep re-debates a point it already owns while
        :func:`council.planning.plan_experiment` reports work that can never be
        finished.

        So the marker is :attr:`~council.domain.signal.Decision.stop_reason`, which
        is only written once a conversation is over. A conversation counts as
        finished when some stored row of it carries one of
        :data:`~council.domain.signal.COMPLETED_STOP_REASONS` and **no** stored row
        of it records a failure in :data:`RETRIED_FAILURES`. The second half is what
        :meth:`completed_keys` already promised at the row level: an hour with the
        daemon down must not bake a flat exposure into the arm permanently, and one
        unreachable seat leaves the whole conversation unfinished because the sweep
        re-holds conversations rather than seats.

        Rounds 0 and 1 of an ongoing conversation carry no reason, so a run
        interrupted mid-conversation is resumed rather than read as complete. The
        seats are not checked individually: a group's rows are written in one
        atomic part file, so a conversation cannot be half stored.
        """
        finished: set[ConversationKey] = set()
        unfinished: set[ConversationKey] = set()
        reasons = {str(reason) for reason in COMPLETED_STOP_REASONS}
        for path in self._sources():
            frame = self._with_stop_reason(path)
            if frame.empty:
                continue
            dates: list[date] = pd.to_datetime(frame["decision_date"]).dt.date.tolist()
            for decision_date, ticker, arm, composition, failure, reason in zip(
                dates,
                frame["ticker"],
                frame["arm"],
                frame["composition"].fillna(NO_COMPOSITION),
                frame["failure"],
                frame[STOP_REASON].fillna(NO_STOP_REASON),
                strict=True,
            ):
                key: ConversationKey = (
                    decision_date,
                    str(ticker),
                    str(arm),
                    str(composition),
                )
                if str(failure) in self._retry_failures:
                    unfinished.add(key)
                if str(reason) in reasons:
                    finished.add(key)
        return frozenset(finished - unfinished)

    def stored_prompts(self) -> dict[DecisionKey, str]:
        """Every stored decision's identity beside the digest of the text it answered.

        :meth:`completed_keys` cannot answer this. Its identity is
        :data:`KEY_COLUMNS`, and nothing in there *defines the prompt*: not
        ``lookback_days``, not the persona file's contents, not the price series. A
        run resumed onto a directory generated under different prompt-defining
        settings therefore reports every existing row as already stored and writes
        one arm holding two different treatments. ``prompt_hash`` is the column that
        can tell, so it is read back out and compared -- see
        :func:`council.agents.runner.check_prompt_provenance`.

        Failed rows are included. Their hash is the prompt that was sent, which is
        exactly what a provenance check asks about, whatever came back.
        """
        prompts: dict[DecisionKey, str] = {}
        for path in self._sources():
            frame = pd.read_parquet(path, columns=[*KEY_COLUMNS, "prompt_hash"])
            prompts.update(
                zip(_keys_of(frame), (str(value) for value in frame["prompt_hash"]), strict=True)
            )
        return prompts

    def checkpoint(
        self,
        *,
        model: str,
        persona: str,
        ticker: str,
        decisions: Sequence[Decision],
        completions: Sequence[CompletionRecord],
    ) -> Path | None:
        """Commit one (model, persona, ticker) triple.

        Returns:
            The part file written, or ``None`` if there was nothing to write. A
            checkpoint carrying no decisions is a legitimate call -- a triple whose
            every point was already stored -- and it deliberately leaves no file:
            see :meth:`_write_part`.

        The completions log is appended first and the part file written second,
        because the part file is what marks the triple done. A crash between the
        two therefore repeats the triple on the next run, adding duplicate lines
        to an append-only archive whose lines carry :data:`KEY_COLUMNS` and can be
        deduplicated on them. ``prompt_hash`` does not identify a line: it digests
        the two prompt turns alone, so every model shown the same text shares one.
        The other order would lose the archive for a triple whose decisions were
        kept, and nothing later could tell.
        """
        self._append_completions(completions)
        return self._write_part(model=model, persona=persona, ticker=ticker, decisions=decisions)

    def consolidate(self) -> Path:
        """Merge every part into the single frame the evaluation package reads.

        Idempotent, and safe to call before a run as well as after one: doing so
        empties the parts directory, which is what keeps each triple within a run
        writing its part exactly once.

        **Nothing to merge means nothing is written.** Not merely an optimisation:
        every read path in the project passes through here, so an unconditional
        rewrite meant that *inspecting* a published run re-serialised its parquet.
        The frame compared equal and the bytes did not, which on a repository
        whose evidence is byte-pinned is the difference between a checksum a
        reader can verify and one that changes because they looked at it -- and it
        put a rewritten artefact into `git status` for anyone running `git add -A`
        afterwards.
        """
        sources = self._sources()
        if not sources:
            return self._decisions_path
        if sources == (self._decisions_path,):
            return self._decisions_path

        combined = pd.concat([pd.read_parquet(path) for path in sources], ignore_index=True)
        # Sources are listed with the consolidated frame first, so a triple that
        # was regenerated in this run supersedes the row it replaces.
        merged = combined.drop_duplicates(subset=list(KEY_COLUMNS), keep="last")
        ordered = merged.sort_values(list(KEY_COLUMNS), kind="stable").reset_index(drop=True)

        _write_parquet(ordered, self._decisions_path)
        for path in sources:
            if path != self._decisions_path:
                path.unlink()
        if self._parts_dir.is_dir() and not any(self._parts_dir.iterdir()):
            self._parts_dir.rmdir()
        return self._decisions_path

    # -- internals ------------------------------------------------------------

    def _with_stop_reason(self, path: Path) -> pd.DataFrame:
        """One source, guaranteed to carry :data:`STOP_REASON`.

        Read whole rather than by column, which every other reader here does. A
        parquet file is read by name, so asking for a column it does not hold raises
        rather than returning nulls -- and ``stop_reason`` was added to
        :data:`STORED_COLUMNS` after decisions had already been written. A store
        pointed at a file from before it, which is what any half-finished run on disk
        now is, would fail to open at all; the truthful answer is "no conversation in
        this file recorded a stopping condition", and it is also the answer that
        makes the sweep hold those conversations again.

        Only that column is supplied. Any other absence is a frame this store did not
        write, and inventing a value for it -- an empty ``failure``, say, which
        compares unequal to :data:`~council.evaluation.frames.NO_FAILURE` and would
        read every row as a crash -- is how a missing column becomes a wrong result.
        """
        frame = pd.read_parquet(path)
        if STOP_REASON not in frame.columns:
            frame[STOP_REASON] = NO_STOP_REASON
        return frame

    def _sources(self) -> tuple[Path, ...]:
        """Everything holding decisions, consolidated frame first."""
        base = [self._decisions_path] if self._decisions_path.is_file() else []
        parts = (
            sorted(self._parts_dir.glob(f"{PART_PREFIX}*.parquet"))
            if self._parts_dir.is_dir()
            else []
        )
        return (*base, *parts)

    def _write_part(
        self, *, model: str, persona: str, ticker: str, decisions: Sequence[Decision]
    ) -> Path | None:
        """Write one part, or nothing at all when there is nothing to write.

        Nothing, because an empty frame has no column types to carry: pandas gives
        every column ``object``, and :meth:`consolidate` concatenating one of those
        with real parts downgrades the whole published artefact to ``object``. The
        dtypes of the project's primary output would then depend on whether some
        triple happened to be empty that night, which is the sort of difference
        between two runs that costs an afternoon to explain.
        """
        if not decisions:
            return None
        keys = [decision_key(decision) for decision in decisions]
        self._parts_dir.mkdir(parents=True, exist_ok=True)
        target = self._parts_dir / part_filename(
            model=model, persona=persona, ticker=ticker, keys=keys
        )
        _write_parquet(to_storage_frame(decisions), target)
        return target

    def _append_completions(self, completions: Sequence[CompletionRecord]) -> None:
        if not completions:
            return
        self._completions_path.parent.mkdir(parents=True, exist_ok=True)
        # Sorted keys and an explicit newline so two runs of the same
        # configuration produce byte-identical archives on any platform. Sorted
        # *records* as well, because `sort_keys` only orders the fields inside a
        # line: the debate arms hand this list over in completion order, since a
        # round's seats go out under `asyncio.gather` and `DecisionCaller` appends
        # as each one returns, so line order would otherwise be whichever model
        # was fastest that night. The key is `KEY_COLUMNS`, which is also the
        # order `consolidate` sorts the parquet into, and it leaves the
        # independent arm's existing date order untouched.
        with self._completions_path.open("a", encoding="utf-8", newline="\n") as handle:
            for record in sorted(completions, key=_completion_sort_key):
                handle.write(json.dumps(record.as_json(), sort_keys=True, ensure_ascii=False))
                handle.write("\n")


def _write_parquet(frame: pd.DataFrame, target: Path) -> None:
    """Write, then move into place, so no reader ever sees a half-written file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(TEMP_SUFFIX)
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, target)


def _keys_of(frame: pd.DataFrame) -> Iterable[DecisionKey]:
    if frame.empty:
        return ()
    # Parquet gives back a date column as `datetime.date` or as `datetime64`
    # depending on how it was written, and the two do not compare equal. Narrowing
    # here is what makes a resumed run recognise its own earlier rows.
    dates: list[date] = pd.to_datetime(frame["decision_date"]).dt.date.tolist()
    return [
        (
            decision_date,
            str(ticker),
            str(model),
            str(persona),
            str(arm),
            int(round_index),
            str(composition),
        )
        for decision_date, ticker, model, persona, arm, round_index, composition in zip(
            dates,
            frame["ticker"],
            frame["model"],
            frame["persona"],
            frame["arm"],
            frame["round_index"],
            frame["composition"].fillna(NO_COMPOSITION),
            strict=True,
        )
    ]

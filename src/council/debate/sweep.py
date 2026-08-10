"""The sweep that runs every committee through every debate arm.

:mod:`council.agents.runner` is this module's opposite number: it sweeps the
independent arm, one model resident at a time, over every session. This one sweeps
the treatment arms, every model resident at once, over the contested points only.
The asymmetry is the design -- a committee is four seats answering the same round,
so unloading between them would swap a checkpoint in and out per call, and a point
the agents already agreed on cannot be changed by a conversation.

What one conversation is remains :mod:`council.debate.protocol`'s. What this owns
is everything that only matters because there are thousands of them: the resume
check, the checkpoint granularity, the pool the placebo arm draws from, and what
happens to a conversation that cannot be held.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final

import pandas as pd

from council.agents.provider import Provider
from council.agents.runner import (
    ContextIndex,
    ProviderFactory,
    build_contexts,
    check_prompt_provenance,
)
from council.agents.store import ConversationKey, DecisionStore
from council.config import Settings, get_settings
from council.debate.caller import DecisionCaller, SeatDecision
from council.debate.compositions import Composition, balanced_design
from council.debate.peers import NoPeersError, SeatView
from council.debate.placebo import PlaceboPool
from council.debate.protocol import run_debate
from council.domain.signal import Arm, StopReason
from council.evaluation.dispersion import Dispersion
from council.evaluation.frames import (
    ARM,
    DECISION_DATE,
    EXPOSURE,
    FAILURE,
    MODEL,
    NO_FAILURE,
    PERSONA,
    TICKER,
    PointKey,
)
from council.planning import TREATMENT_ARMS, conversation_key

_LOG = logging.getLogger(__name__)

RATIONALE: Final = "rationale"
"""The stored column the placebo pool needs and
:class:`~council.evaluation.frames.DecisionRow` does not carry. The analysis reads
exposures and confidences; the placebo arm shows peers their *prose*, so it reads
the parquet directly rather than through the row shape."""


def placebo_pool_for(decisions: pd.DataFrame, *, composition: Composition) -> PlaceboPool:
    """Other days' views for one committee, taken from the independent arm.

    The independent arm rather than a debate arm that has already run, for one
    reason that matters: a pool of debated points holds contested days only, so the
    earliest contested day has no earlier donor and is dropped -- from the placebo
    arm alone. The arms would then be backtested over different sets of decision
    points, which is the selection effect
    :func:`council.debate.protocol._check_someone_spoke` exists to prevent one
    version of. Drawing from the control gives every session as a candidate.

    The text is of the same kind either way. A debate's opening round is the
    independent question put to a committee and renders byte-identically to the
    control's prompt, so a donor rationale from here is a rationale about another
    day written by this seat, which is exactly what the placebo is defined as.

    Failed generations are excluded. Their rationale is the empty string, and an
    analyst who said nothing is not a counter-argument.
    """
    if RATIONALE not in decisions.columns:
        raise ValueError(
            "the placebo arm shows peers' prose, so it needs the stored decisions "
            f"rather than the analysis frame; column {RATIONALE!r} is missing"
        )
    seats = {(seat.model, seat.persona.name): seat for seat in composition.seats}
    usable = decisions.loc[
        (decisions[ARM].astype(str) == str(Arm.INDEPENDENT))
        & (decisions[FAILURE].astype(str) == NO_FAILURE)
    ]
    pool: dict[PointKey, list[SeatView]] = defaultdict(list)
    for day, ticker, model, persona, exposure, rationale in zip(
        pd.to_datetime(usable[DECISION_DATE]).dt.date,
        usable[TICKER],
        usable[MODEL],
        usable[PERSONA],
        usable[EXPOSURE],
        usable[RATIONALE],
        strict=True,
    ):
        seat = seats.get((str(model), str(persona)))
        if seat is not None:
            pool[(day, str(ticker))].append(
                SeatView(seat=seat, exposure=float(exposure), rationale=str(rationale))
            )
    return {point: tuple(views) for point, views in sorted(pool.items())}


def has_donor(
    pool: PlaceboPool,
    point: PointKey,
    *,
    required_seats: int,
    min_gap: int | None = None,
    rounds: int = 1,
) -> bool:
    """Whether the pool holds enough usable earlier days for this point.

    Asked in advance so that a point with no donor is skipped before its opening
    round is generated rather than after: an opening round costs one call per seat
    and would be thrown away.

    It must apply **exactly** the test :func:`council.debate.placebo.select_placebo_point`
    applies over the whole conversation: minimum gap, seat completeness, and one
    distinct candidate per round it may run to. The two drifting apart is
    not a cosmetic mismatch: this returns True, the sweep commits to the point, and
    the real draw then raises a plain ``ValueError`` that ``except NoPeersError``
    does not catch, so the whole sweep exits and the current group's uncheckpointed
    rows are lost. The gap costs whole *dates*, because the test below counts the
    pool's distinct dates: at the configured gap the first 60 decision dates lose
    their donor either way -- 60 of 461 dates (120 of 922 points) at the configured
    range, and 60 of 70 dates (120 of 140 contested points) on the six-month slice.
    Those are figures from the superseded two-model six-month run whose artefacts
    are retained at ``docs/results/superseded/run-2models/``, kept here as
    provenance for the bound rather than as a current measurement.

    Args:
        required_seats: how many seats the donor must hold a view for, ordinarily
            ``composition.size``. Same keyword and same test as the draw, for the
            reason ``min_gap`` is: a donor day on which one seat failed to generate
            holds fewer views than the committee has chairs, and admitting it
            leaves the placebo's agents one peer short of the real arm's -- or, at
            a single surviving view, raises ``NoPeersError`` and books the point as
            abandoned in the placebo arm alone.
        min_gap: defaults to ``settings.placebo_min_gap_sessions``, resolved the
            same way the draw resolves it. A default of zero here would be that
            drift written into the signature: a caller omitting the keyword would
            pre-flight at no gap while the draw enforced sixty, which at the
            configured gap disagreed on 118 of the 138 points a gapless pre-flight
            admits -- again from that superseded run, and provenance rather than a
            current figure. That is a different quantity from the coverage figures
            above, which count the points the configured gap refuses outright, and
            the two do not share a denominator.
        rounds: how many rebuttal rounds the conversation may run to, ordinarily
            the sweep's cap. One donor is drawn per round and
            :func:`~council.debate.placebo.select_placebo_point` no longer wraps, so
            a point with four usable candidates cannot serve a six-round
            conversation: it would either repeat a donor or, now, raise at round
            five with four rounds already generated. Counting candidates here is
            what turns that into a point skipped before it costs anything. The
            default of one is the smallest conversation rather than a convenience --
            it is what the draw's own default ``round_index`` asks for, so an
            omitted keyword still mirrors the draw rather than out-admitting it.
    """
    if min_gap is None:
        min_gap = get_settings().placebo_min_gap_sessions
    decision_date = point[0]
    earlier = sorted({key[0] for key in pool if key[0] < decision_date})
    if len(earlier) < min_gap:
        return False
    cutoff = earlier[-min_gap] if min_gap > 0 else decision_date
    candidates = sum(
        1
        for key, views in pool.items()
        if key[0] < decision_date and key[0] <= cutoff and len(views) == required_seats
    )
    return candidates >= rounds


@dataclass(frozen=True, slots=True)
class DebateReport:
    """What a debate sweep did, per its own units.

    ``held`` counts conversations that ran to the end; ``abandoned`` counts the ones
    that could not, because a whole round failed to generate or because the placebo
    had no earlier donor.

    **Pooled over every composition and every arm.**
    :func:`run_debate_arms` merges one of these per (committee, arm, ticker) group
    into a single value and the CLI prints one line from it, and no field here
    carries a composition or an arm. So the counters say how many conversations
    were lost and not by which arm -- an abandonment could be a placebo donor skip
    or a whole-round failure in the debate arm, and this object cannot tell them
    apart. The arm is recorded in :meth:`_Sweep.group`'s and :meth:`_Sweep.hold`'s
    ``_LOG.warning`` lines, and per-arm coverage is read back off the artefact by
    :func:`council.app.tables.coverage_note`.
    """

    conversations: int = 0
    held: int = 0
    skipped: int = 0
    abandoned: int = 0
    generated: int = 0
    failures: int = 0

    offered_points: int = 0
    """Contested points handed to the sweep, before the placebo filter."""

    dropped_points: int = 0
    """Contested points withheld from **every** arm for want of a placebo donor.

    Points, not conversations: the number is decided once for the whole sweep by
    :func:`servable_points` and is the same for all three arms by construction,
    which is the entire purpose of it. It is reported rather than logged because an
    experiment that quietly answers fewer questions than it was asked is
    indistinguishable, in every other output, from one that answered all of them.
    ``offered_points - dropped_points`` is what each arm actually covers.
    """

    def merge(self, other: DebateReport) -> DebateReport:
        return DebateReport(
            conversations=self.conversations + other.conversations,
            held=self.held + other.held,
            skipped=self.skipped + other.skipped,
            abandoned=self.abandoned + other.abandoned,
            generated=self.generated + other.generated,
            failures=self.failures + other.failures,
            offered_points=self.offered_points + other.offered_points,
            dropped_points=self.dropped_points + other.dropped_points,
        )


@dataclass(frozen=True, slots=True)
class _Sweep:
    """Everything one debate sweep holds constant.

    Grouped so the per-conversation helpers take the two or three arguments that
    actually vary. Frozen, like everything else here; the counters are returned
    rather than accumulated in place.
    """

    settings: Settings
    store: DecisionStore
    providers: Mapping[str, Provider]
    contexts: ContextIndex
    done: frozenset[ConversationKey]
    """Conversations the store says reached a stopping condition.

    Whole conversations rather than :data:`~council.agents.store.DecisionKey` rows,
    because a conversation's length is an outcome now: there is no set of rows a
    finished one is guaranteed to hold, and asking for one row per round up to the
    cap means a conversation that agreed at round two is re-held on every resume
    while the plan that prices the run can never reach zero.
    """

    rebuttal_rounds: int
    threshold: float | None

    async def group(
        self,
        *,
        composition: Composition,
        arm: Arm,
        ticker: str,
        points: Sequence[Dispersion],
        pool: PlaceboPool,
    ) -> DebateReport:
        """Every contested point of one ticker, for one committee in one arm.

        Checkpointed as a unit, matching the independent sweep's (model, persona,
        ticker) triple: one part file per group rather than per conversation, which
        for a full run is the difference between fifty files and five thousand.

        The checkpoint's ``model`` and ``persona`` arguments carry the committee and
        the arm, because for a debate those are what a group *is*. Only the readable
        half of the part filename is affected: the identity a resume compares is
        :func:`council.agents.store.decision_key`, taken over the rows themselves.
        """
        report = DebateReport()
        rows: list[SeatDecision] = []
        for dispersion in points:
            report = report.merge(DebateReport(conversations=1))
            if (
                conversation_key(
                    composition=composition,
                    arm=arm,
                    decision_date=dispersion.decision_date,
                    ticker=dispersion.ticker,
                )
                in self.done
            ):
                report = report.merge(DebateReport(skipped=1))
                continue
            if arm is Arm.DEBATE_PLACEBO and not has_donor(
                pool,
                dispersion.point,
                required_seats=composition.size,
                min_gap=self.settings.placebo_min_gap_sessions,
                rounds=self.rebuttal_rounds,
            ):
                # Kept as a backstop rather than as the filter it used to be.
                # `run_debate_arms` now drops a point no committee's pool can serve
                # from *every* arm, so reaching this means one committee's pool is
                # thinner than another's on a point the others kept -- which would
                # cost the placebo arm a point the debate arm holds, the coverage
                # difference the filter exists to remove. Loud, and counted.
                _LOG.warning(
                    "no placebo donor precedes %s for %s over %d round(s); skipped",
                    dispersion.point,
                    composition.identifier,
                    self.rebuttal_rounds,
                )
                report = report.merge(DebateReport(abandoned=1))
                continue
            held, complete = await self.hold(
                composition=composition, arm=arm, dispersion=dispersion, pool=pool
            )
            rows.extend(held)
            report = report.merge(DebateReport(held=int(complete), abandoned=int(not complete)))

        self.store.checkpoint(
            model=composition.identifier,
            persona=str(arm),
            ticker=ticker,
            decisions=[row.decision for row in rows],
            completions=[row.record for row in rows],
        )
        return report.merge(
            DebateReport(
                generated=len(rows),
                failures=sum(1 for row in rows if row.decision.is_failure),
            )
        )

    async def hold(
        self,
        *,
        composition: Composition,
        arm: Arm,
        dispersion: Dispersion,
        pool: PlaceboPool,
    ) -> tuple[tuple[SeatDecision, ...], bool]:
        """One conversation: its rows, and whether it ran to the end.

        An abandoned conversation still hands back the rows it produced. They are
        stored, because a round 0 with no round 1 is what
        :func:`council.evaluation.persuasion.unpaired_rows` exists to report, and
        because the per-model failure rate is a published result.

        Every row goes back carrying how the conversation ended, which is what a
        resume reads. The three exits stamp three different things, and the
        differences are the point:

        * A conversation that reached a stopping condition is stamped with it, so a
          resume knows a debate that agreed at round two owes nothing more.
        * :attr:`~council.domain.signal.StopReason.NO_SPEAKERS` is stamped too, and
          is *both* an abandoned conversation and a finished one. A round every seat
          botched reproduces exactly at temperature zero, so re-holding it would
          spend a night confirming it; a round lost to an unreachable daemon is
          stored as a retriable failure and
          :meth:`~council.agents.store.DecisionStore.completed_conversations`
          refuses to call that finished, which is the case a resume is for.
        * A conversation that raised out of the protocol is stamped with nothing.
          It reached no stopping condition, and the next run holds it again.
        """
        caller = DecisionCaller(
            providers=self.providers,
            composition=composition,
            ticker=dispersion.ticker,
            decision_date=dispersion.decision_date,
            seed=self.settings.seed,
        )
        try:
            transcript = await run_debate(
                composition=composition,
                arm=arm,
                dispersion=dispersion,
                price_context=self.contexts[(dispersion.ticker, dispersion.decision_date)],
                caller=caller,
                placebo_pool=pool if arm is Arm.DEBATE_PLACEBO else None,
                seed=self.settings.seed,
                dispersion_threshold=self.threshold,
                max_rounds=self.rebuttal_rounds,
                # From this sweep's settings rather than left to the module
                # default, which reads the process-wide instance. A run
                # configured with its own Settings must not have one bound
                # silently read out from under it.
                placebo_min_gap=self.settings.placebo_min_gap_sessions,
                # The same, for the two bars that decide when a conversation
                # ends. `StopReason` is a declared measurement, so a sweep whose
                # Settings differ from the cached process-wide ones must stop its
                # debates on its own bars rather than on the environment's.
                agreement_spread=self.settings.agreement_spread,
                stillness_rounds=self.settings.stillness_rounds,
            )
        except NoPeersError as exhausted:
            _LOG.warning("abandoned %s in %s: %s", dispersion.point, arm, exhausted)
            return caller.stamped(None), False
        # A whole round failing to generate stops the conversation from inside
        # rather than by raising, so the counting has to read the stop reason.
        # Without this, zero survivors is booked as a conversation held while one
        # survivor -- the strictly better outcome -- is booked as abandoned.
        if transcript.stop_reason is StopReason.NO_SPEAKERS:
            _LOG.warning(
                "abandoned %s in %s: a whole round failed to generate", dispersion.point, arm
            )
            return caller.stamped(transcript.stop_reason), False
        _LOG.debug(
            "%s in %s ended %s after %d rebuttal round(s) with %d of %d seat(s) alive",
            dispersion.point,
            arm,
            transcript.stop_reason,
            transcript.rebuttal_rounds,
            transcript.surviving_seats,
            composition.size,
        )
        return caller.stamped(transcript.stop_reason), True


def _check_cap(rebuttal_rounds: int) -> None:
    """Refuse a cap no conversation could run to.

    What this used to refuse was *any* cap but one, because eight consumers read a
    fixed round index and seven of them corrupted a run rather than failing at a
    longer one. Each is wired for variable length now, and the list stays here as the
    record of what that took:

    * :meth:`_Sweep.group`'s resume check, and
      :func:`council.planning.plan_experiment`'s ``completed`` count, ask
      :attr:`~council.domain.signal.Decision.stop_reason` instead of demanding a row
      for every round up to the cap.
    * :func:`council.scoring.arm_exposures` and
      :func:`council.scoring._arm_reports` read each conversation's own last round
      rather than one fixed index that a conversation which agreed early never wrote.
    * :func:`council.evaluation.persuasion._by_round` and
      :func:`council.app.transcripts.read_transcripts` set the rounds above the first
      rebuttal aside instead of raising on them, and go on pairing rounds 0 and 1.
    * :func:`council.debate.placebo.select_placebo_point` refuses a point whose
      candidate set is shorter than the conversation rather than wrapping and
      repeating a donor.
    * :func:`council.app.curves.arms_in` needed no change once ``arm_exposures``
      stopped reading a fixed index: round 1 is the round every held conversation
      has, whatever the cap.
    * :func:`council.app.panels._rounds_in` is the one that was **not** changed. It
      still offers rounds 0 and 1, so the middle rounds of a long conversation cannot
      be asked about on the dashboard's calibration panel. That is a gap in an
      exploratory surface rather than a wrong number -- those two rounds are the ones
      the declared comparison is stated over -- and it is recorded here rather than
      quietly left.

    What is left is arithmetic. A cap below one is a conversation with no rebuttal
    round at all -- the control with extra steps -- and
    :func:`council.debate.protocol.run_debate` refuses it per conversation. Refusing
    it here as well costs one comparison and saves the sweep from raising once per
    point, after ``store.consolidate`` and the provider preflight have already run.
    """
    if rebuttal_rounds < 1:
        raise ValueError(
            f"a debate needs at least one round after the opening, not "
            f"{rebuttal_rounds}; at zero the treatment arms are the control"
        )


def servable_points(
    points: Sequence[PointKey],
    *,
    decisions: pd.DataFrame,
    committees: Sequence[Composition],
    min_gap: int,
    rounds: int,
) -> frozenset[PointKey]:
    """Which of these points every arm can be run on.

    The placebo cannot run on a point with no usable earlier donor, and at the
    configured gap that is every point in the first ``placebo_min_gap_sessions``
    sessions of the calendar -- plus, now that a donor is drawn per round and never
    repeated, every point whose candidate set is shorter than the cap. Those points
    are at the *start* of the calendar, so dropping them from one arm and not the
    others is not a coverage difference that averages out: the placebo would be
    scored over a later, and possibly calmer, market than the arms it is
    differenced against, and "debate minus placebo" would be part manipulation and
    part calendar.

    So the filter is applied to all three arms alike. What it withholds is counted
    rather than logged away: :func:`run_debate_arms` puts the number on its report
    and the command line prints it. An experiment that quietly shrinks is the
    failure this function exists to prevent, and a silent version of it would look
    exactly like a clean run.

    A point is kept only if **every** committee can serve it. The pool is per
    committee -- a donor day must hold a view from each of that committee's seats,
    so a day one model failed on serves some committees and not others -- and
    keeping a point that only six of eight committees can debate would put the
    coverage difference back one level down, between committees within the placebo
    arm rather than between the arms.

    :func:`council.planning.plan_experiment` prices the run through this same
    function, so the plan and the sweep cannot disagree about which points are in
    the experiment.
    """
    pools = {
        table.identifier: placebo_pool_for(decisions, composition=table) for table in committees
    }
    return frozenset(
        point
        for point in points
        if all(
            has_donor(
                pools[table.identifier],
                point,
                required_seats=table.size,
                min_gap=min_gap,
                rounds=rounds,
            )
            for table in committees
        )
    )


async def run_debate_arms(
    *,
    settings: Settings,
    prices: pd.DataFrame,
    decisions: pd.DataFrame,
    contested: Sequence[Dispersion],
    provider_factory: ProviderFactory,
    store: DecisionStore | None = None,
    compositions: Sequence[Composition] | None = None,
    arms: Sequence[Arm] = TREATMENT_ARMS,
    rebuttal_rounds: int | None = None,
    threshold: float | None = None,
) -> DebateReport:
    """Run every committee through every debate arm, on the contested points only.

    Unlike the independent sweep, every base model is resident at once: a committee
    is four seats answering the same round, and unloading between them would swap a
    checkpoint in and out per call.

    **All three arms are run on one point set.** The contested points handed in are
    filtered by :func:`servable_points` down to those a placebo donor can actually
    serve, and the survivors go to every arm. Filtering the placebo alone -- which is
    what skipping the point inside :meth:`_Sweep.group` amounted to -- left the three
    arms covering different calendars, and the points the placebo lost are the
    earliest ones rather than a random sample, so part of any debate-minus-placebo
    difference was a difference in which days each arm answered. The count withheld
    is on :attr:`DebateReport.dropped_points` and the command line prints it.

    Args:
        rebuttal_rounds: defaults to ``settings.max_debate_rounds``, so that raising
            the cap changes plan, run and score together rather than being read by
            one of them and ignored by the others. It is a cap and not a length:
            every conversation stops on whichever of agreement, stillness or the cap
            comes first.

    Raises:
        ValueError: for a cap below one rebuttal round. See :func:`_check_cap`, which
            is what is left of a refusal that used to cover every cap but one.
    """
    rebuttal_rounds = (
        settings.max_debate_rounds if rebuttal_rounds is None else rebuttal_rounds
    )
    _check_cap(rebuttal_rounds)
    store = store or DecisionStore(
        decisions_path=settings.decisions_path, completions_path=settings.completions_path
    )
    store.consolidate()
    committees = tuple(
        balanced_design(models=settings.agent_models) if compositions is None else compositions
    )
    offered = tuple(contested)
    servable = servable_points(
        [point.point for point in offered],
        decisions=decisions,
        committees=committees,
        min_gap=settings.placebo_min_gap_sessions,
        rounds=rebuttal_rounds,
    )
    contested = tuple(point for point in offered if point.point in servable)
    dropped = tuple(point for point in offered if point.point not in servable)
    if dropped:
        _LOG.warning(
            "%d of %d contested point(s) withheld from every arm: no placebo donor "
            "at least %d session(s) back holding every seat, for each of %d round(s) "
            "-- earliest %s, latest %s. All three arms answer the remaining %d so "
            "that no difference between them is a difference in coverage",
            len(dropped),
            len(offered),
            settings.placebo_min_gap_sessions,
            rebuttal_rounds,
            min(point.point for point in dropped),
            max(point.point for point in dropped),
            len(contested),
        )
    contexts = build_contexts(
        prices,
        tickers=settings.tickers,
        dates=sorted({point.decision_date for point in contested}),
        lookback_days=settings.lookback_days,
    )
    # The same guard the independent sweep applies, for the same reason: the resume
    # identity carries nothing that defines the prompt, so a sweep resumed onto rows
    # generated under a different lookback, persona file or price series would add
    # its own rows beside them and leave one arm holding two treatments.
    check_prompt_provenance(stored=store.stored_prompts(), contexts=contexts)
    sweep = _Sweep(
        settings=settings,
        store=store,
        providers={
            model: provider_factory(model)
            for model in sorted({seat.model for table in committees for seat in table.seats})
        },
        contexts=contexts,
        done=store.completed_conversations(),
        rebuttal_rounds=rebuttal_rounds,
        # From this sweep's settings rather than left as None, for the same
        # reason `placebo_min_gap` is threaded above: None reaches
        # `run_debate._check_runnable`, which falls back to the process-wide
        # settings, while `pipeline.select_contested` picked the points with the
        # caller's `dispersion_threshold`. A run configured with its own Settings
        # would select on one bar and validate on another, and a point the looser
        # caller bar kept raises a plain ValueError that `except NoPeersError`
        # does not catch -- taking the sweep down with the group's uncheckpointed
        # rows.
        threshold=settings.dispersion_threshold if threshold is None else threshold,
    )

    report = DebateReport(offered_points=len(offered), dropped_points=len(dropped))
    try:
        for provider in sweep.providers.values():
            await provider.preflight()
        for composition in committees:
            pool = placebo_pool_for(decisions, composition=composition)
            for arm in arms:
                for ticker in settings.tickers:
                    report = report.merge(
                        await sweep.group(
                            composition=composition,
                            arm=arm,
                            ticker=ticker,
                            points=[point for point in contested if point.ticker == ticker],
                            pool=pool,
                        )
                    )
    finally:
        for provider in sweep.providers.values():
            await provider.aclose()
    store.consolidate()
    return report


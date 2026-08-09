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
from council.agents.store import DecisionKey, DecisionStore
from council.config import Settings, get_settings
from council.debate.caller import DecisionCaller, SeatDecision
from council.debate.compositions import Composition, balanced_design
from council.debate.peers import NoPeersError, SeatView
from council.debate.placebo import PlaceboPool
from council.debate.protocol import DEFAULT_REBUTTAL_ROUNDS, StopReason, run_debate
from council.domain.signal import Arm
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
from council.planning import TREATMENT_ARMS, conversation_keys

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
    pool: PlaceboPool, point: PointKey, *, required_seats: int, min_gap: int | None = None
) -> bool:
    """Whether the pool holds a usable earlier day for this point.

    Asked in advance so that a point with no donor is skipped before its opening
    round is generated rather than after: an opening round costs one call per seat
    and would be thrown away.

    It must apply **exactly** the test :func:`council.debate.placebo.select_placebo_point`
    applies, minimum gap and seat completeness included. The two drifting apart is
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
    """
    if min_gap is None:
        min_gap = get_settings().placebo_min_gap_sessions
    decision_date = point[0]
    earlier = sorted({key[0] for key in pool if key[0] < decision_date})
    if len(earlier) < min_gap:
        return False
    cutoff = earlier[-min_gap] if min_gap > 0 else decision_date
    return any(
        key[0] < decision_date and key[0] <= cutoff and len(views) == required_seats
        for key, views in pool.items()
    )


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

    def merge(self, other: DebateReport) -> DebateReport:
        return DebateReport(
            conversations=self.conversations + other.conversations,
            held=self.held + other.held,
            skipped=self.skipped + other.skipped,
            abandoned=self.abandoned + other.abandoned,
            generated=self.generated + other.generated,
            failures=self.failures + other.failures,
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
    done: frozenset[DecisionKey]
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
            if self.done.issuperset(
                conversation_keys(
                    composition=composition,
                    arm=arm,
                    decision_date=dispersion.decision_date,
                    ticker=dispersion.ticker,
                    rebuttal_rounds=self.rebuttal_rounds,
                )
            ):
                report = report.merge(DebateReport(skipped=1))
                continue
            if arm is Arm.DEBATE_PLACEBO and not has_donor(
                pool,
                dispersion.point,
                required_seats=composition.size,
                min_gap=self.settings.placebo_min_gap_sessions,
            ):
                _LOG.warning(
                    "no placebo donor precedes %s for %s; skipped",
                    dispersion.point,
                    composition.identifier,
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
            return tuple(caller.generated), False
        # A whole round failing to generate stops the conversation from inside
        # rather than by raising, so the counting has to read the stop reason.
        # Without this, zero survivors is booked as a conversation held while one
        # survivor -- the strictly better outcome -- is booked as abandoned.
        if transcript.stop_reason is StopReason.NO_SPEAKERS:
            _LOG.warning(
                "abandoned %s in %s: a whole round failed to generate", dispersion.point, arm
            )
            return tuple(caller.generated), False
        return tuple(caller.generated), True


def _check_cap(rebuttal_rounds: int) -> None:
    """Refuse a cap the rest of the pipeline cannot read.

    :mod:`council.debate.protocol` ends a conversation on a condition, so a
    conversation may stop before the cap. Several consumers assume a fixed round
    count instead, and only one of them raises when the assumption breaks. This list
    is the checklist for raising the cap, so it names all of them rather than the
    three that were noticed first:

    * :meth:`_Sweep.group`'s resume check demands a stored row for every round
      ``0..cap``, so a conversation that stopped early never satisfies it and is
      re-generated on every resume -- and :func:`council.planning.plan_experiment`
      counts ``completed`` the same way, so it can never reach ``total``.
    * :func:`council.scoring.arm_exposures` reads the treatment arm at the single
      index ``rebuttal_rounds``. A conversation that agreed earlier has no row
      there, so the treatment silently reverts to the independent exposure on
      exactly the converged points.
    * :func:`council.evaluation.persuasion.shifts` raises on any round index above
      1, which is the one that fails loudly.
    * :func:`council.app.transcripts.read_transcripts` raises on any round index
      above 1 too, so a cap-3 artefact takes the dashboard's transcript panel down
      rather than rendering the extra rounds.
    * :func:`council.app.curves.arms_in` qualifies a treatment arm on the literal
      round 1 rather than on the run's cap. At a higher cap an arm holding only
      rounds 0-1 passes as present while ``arm_exposures`` reads the cap's index and
      finds nothing -- the flat curve that reads as a clean null, which is the exact
      failure that function says it prevents.
    * :func:`council.app.panels._rounds_in` offers rounds 0 and 1 only, so the rounds
      above the first rebuttal cannot be asked about on the dashboard at all.
    * :func:`council.scoring._arm_reports` calibrates the treatment arm at
      ``ROUND_INDEX == rebuttal_rounds``, a second fixed-index read beside
      ``arm_exposures`` and with the same consequence on a conversation that stopped
      early: an empty calibration report rather than the arm's final round.
    * :func:`council.debate.placebo.select_placebo_point` wraps its per-round donor
      draw modulo the candidate set, and at the configured gap the earliest
      admissible point has exactly one candidate. Above one round the placebo would
      be shown an identical peer block in consecutive rounds -- the failure the
      per-round redraw exists to prevent -- with nothing raising or logging.

    Pinning here rather than in the protocol keeps the variable-length machinery
    intact and testable, and makes the pin a stated refusal rather than an
    accidental default nobody overrode.
    """
    if rebuttal_rounds != DEFAULT_REBUTTAL_ROUNDS:
        raise ValueError(
            f"the sweep runs at {DEFAULT_REBUTTAL_ROUNDS} rebuttal round(s), not "
            f"{rebuttal_rounds}. Teach every consumer that assumes a fixed round "
            "count variable length first: _Sweep.group's resume check demands a row "
            "for every round up to the cap, scoring.arm_exposures reads the treatment "
            "arm at one fixed round index, scoring._arm_reports calibrates at that "
            "same fixed index, evaluation.persuasion._by_round rejects a round index "
            "above 1, app.transcripts.read_transcripts rejects one too, "
            "app.curves.arms_in qualifies a treatment arm on the literal round 1 "
            "rather than on the run's cap, app.panels._rounds_in offers rounds 0 "
            "and 1 only, and debate.placebo.select_placebo_point wraps its donor "
            "draw modulo the candidate set -- which at the configured gap holds one "
            "candidate on the earliest admissible point, so a second round would "
            "repeat that donor silently"
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

    Args:
        rebuttal_rounds: defaults to ``settings.max_debate_rounds``, so that raising
            the cap changes plan, run and score together rather than being read by
            one of them and ignored by the others.

    Raises:
        ValueError: for any cap other than :data:`DEFAULT_REBUTTAL_ROUNDS`. The
            protocol is variable-length; seven consumers are not, and at a higher
            cap they corrupt a run rather than failing. See :func:`_check_cap`.
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
        done=store.completed_keys(),
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

    report = DebateReport()
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


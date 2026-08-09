# Claims made about this project

Every substantive assertion, in one place, so an auditor has a target rather than a
prose search. Each is either supported by an artefact in this repository or it is not.

## About the design

C1. The four arms (independent / debate / rationale-only / placebo) each bound a
    distinct alternative explanation, and collapsing any of them destroys the inference.
    Rationale-only withholds the peer's *structured* exposure field only, so it bounds
    anchoring on a peer's stated position rather than isolating anchoring: a figure a
    peer wrote into its own prose reaches the reader unchanged (see D1). The placebo
    additionally requires a donor `placebo_min_gap_sessions` back, so it covers fewer
    decision points than the arms it is differenced against (see the coverage note).
C2. The stance axis (momentum vs reversion) is necessary because an aggression-only
    axis produces agreement on direction and therefore nothing to debate.
C3. A balanced design of 8 committees separates model main effects from persona main
    effects, because every model appears at every persona the same number of times --
    once across the four Latin-square rotations, and once more across the four
    uniform references. It is not
    equivalent to the full 4^4 = 256 grid: what it gives up is the interaction between
    particular pairings -- whether this model argues differently against that one --
    which this study does not ask about.
C4. Simultaneous debate removes an order confound that a turn-taking debate would have.
C5. Anonymous peer *handles* remove a lab-reputation confound. Only the handle is
    enforced, where the value is constructed. The peer's prose is unvalidated model
    output, passed through unchanged apart from whitespace collapsing and truncation,
    so a rationale naming its family or restating its persona reaches the reader; the
    persona briefs ask models not to describe their method, which is an instruction
    rather than a check, and no audit of the completions archive has been written.
C6. Agents see normalised returns only, and no date, ticker or price level reaches them.
C7. The backtest fills at the next session's open and cannot see the future.

## About the measurements

Every figure in C8-C15, C17 and C19 comes from superseded six-month-window runs and is
retained for the record only. The two-model run's decisions survive at
`docs/results/superseded/run-2models/decisions.parquet`, which is what the recomputations
below are made from; the four-model run's artefacts do not survive, so nothing quoted
from it can be rechecked.

C8.  Debate and placebo shift rates are close (debate 0.204 / 0.373 /
     0.248 / 0.407 against placebo 0.201 / 0.393 / 0.243 / 0.374; within 0.01 in
     2 of 4 buckets, placebo higher in 1 of 4). **Not supported as a claim about
     contradiction versus argument content: see D6.** The placebo arm these
     figures come from drew its donors from inside the decision window, which
     biases toward exactly that reading.
C9.  Shift rate across prior confidence on the superseded run is 0.204 / 0.373 /
     0.248 / 0.407. No conclusion about whether confidence predicts holding is
     drawn from it: findings.md Finding 2 is withdrawn and not replaced, and the
     run's window was never chosen.
C10. Removing peers' exposure numbers gives 0.293 / 0.274 / 0.221 / 0.311 on the
     same superseded run. No conclusion about what the exposure number does to the
     confidence signal is drawn from it: findings.md Finding 3 is withdrawn and not
     replaced, and the run's window was never chosen.
C11. gemma4's net concessions are +3 in debate and -3 in placebo.
C12. Both models capitulated exactly once on the probe (1/22 and 1/21), so the probe
     does not order them. gemma4's net concessions are +3 in debate and +56 in
     rationale-only and -3 in the placebo, so the market arms do not order them
     consistently either. **UNSUPPORTED as a fact-versus-judgement conclusion: see
     D7.** No fact-versus-judgement conclusion is drawn.
C13. On a 16-call screen over the two strongest up-drifts and two strongest
     down-drifts of one synthetic series, all four models separated stance far more
     on the rising windows (1.06 to 1.20) than the falling ones (-0.20 to +0.30).
     With two falling windows per model this is a hypothesis about drawdown
     behaviour, not a demonstration.
C14. The contested share was 100% on the two-model run and 138/140 on the four-model
     run. **Both figures are pooled-grid shares**: `pipeline.select_contested` measures
     dispersion once over the whole independent arm -- every model crossed with every
     persona -- and `debate.sweep.run_debate_arms` applies that one list unchanged to all
     eight committees. Neither is the share for the committee that debates, which is the
     unit the gate is justified in. Recomputed per committee on
     `docs/results/superseded/run-2models/decisions.parquet` at
     `dispersion_threshold = 0.25`: rotations 56, 125, 70 and 123 of 140; uniforms 3, 11,
     30 and 31 of 140 -- **449 of 1,120, 40%**. No conclusion about what the optimisation
     saves is drawn: at the committee level it has never been measured.
C15. Returns in this study are noise by construction (synthetic geometric random walks)
     and the behavioural columns do not depend on prices being real.

## About the process

C16. All three quality gates pass: pytest, ruff, mypy strict.
C17. Zero generation failures across every run reported.
C18. The primary comparison, `shift_threshold` and `dispersion_threshold` were fixed
     in the first commit (`afce0ae`), before any result existed. The four other
     bounds `config.py` declares -- `agreement_spread`, `stillness_rounds`,
     `max_debate_rounds` and `placebo_min_gap_sessions` -- were added in `cbf6a55`,
     after the results committed in `fa436fa` and `98a4020`, and `agreement_spread`
     and `placebo_min_gap_sessions` were calibrated from that run's measured
     spreads and donor distances. Those four are not pre-registered.
C19. Transcripts may be quoted from a partially complete run, but rates may not,
     because a partial run is ordered by committee and ticker rather than random.

## Known already-found defect, do not spend time rediscovering

D1. The document describes +1.34% in a placebo peer block as "a number that exists
    nowhere in the agent's context", implying fabrication. It is not fabricated: it is
    a real rationale a model produced on a neighbouring decision point, which is
    exactly how the placebo is built. The wording is wrong and is being corrected.

D2. C9 read 0.204 / 0.241 / 0.218 / 0.228 and was described as flat. Those figures
    were produced by the bare `>=` that cbf6a55 replaced with
    `evaluation.threshold.meets`; the loss they came from is concentrated by
    confidence band rather than spread evenly, so the flatness was the artefact
    and not the finding. Recomputed above from
    `docs/results/superseded/run-2models/decisions.parquet` with the shipped comparison.

D3. C10 read 0.287 / 0.171 / 0.183 / 0.148, from the same bare comparison and
    superseded by cbf6a55 for the same reason. The "the number, not the argument,
    suppresses the confidence signal" reading rested on a flat debate line that no
    longer exists to be tilted.

D4. C11 read +20 in debate and +21 in placebo, and concluded that the influence
    matrix measures yieldingness rather than persuasiveness. Both bars in
    `evaluation/influence.py` now run through `meets` (cbf6a55, 027184d) and the
    placebo net sign flips to -3, so the two conditions no longer agree and that
    conclusion is withdrawn rather than restated.

D5. C8 read "within ~0.01 in 3 of 4 buckets, placebo higher in 2 of 4". Those
    counts came from the same bare `>=` cbf6a55 replaced; recomputed with `meets`
    from `docs/results/superseded/run-2models/decisions.parquet` they are 2 of 4 and 1 of 4.

D6. Every published placebo number was generated before `cbf6a55` enforced a
    minimum gap between the donor day and the day being decided, so no gap was
    enforced when they were produced. Reconstructing that draw over the 1,104
    placebo conversations in `docs/results/superseded/run-2models/decisions.parquet`
    (candidates = every earlier pool day with views; chosen =
    `candidates[int.from_bytes(blake2b(f"{seed}|{composition}|{date}|{ticker}",
    8), "big") % len(candidates)]`) gives a median donor distance of **14
    sessions**, minimum 1 and maximum 69, with **98.82%** of conversations
    drawing a donor fewer than 60 sessions back and 51.2% drawing one 14 sessions
    back or nearer. Agents look back 60 sessions, so in almost every placebo
    conversation the donor's window overlapped the window under decision and the
    "unrelated day" shared most of its bars with it. C8 therefore rests on a
    placebo arm that was not inert, and the contamination pushes in the direction
    C8 concludes: a peer arguing about nearly the same data behaves like a real
    debate partner. The conclusion is not restated as supported.

D7. C12 read "qwen3 held facts better than gemma4 in the probe but yields more in
    the market arms, therefore defending a fact and defending a judgement are
    different behaviours". Neither half survives its artefact. In the probe both
    models capitulated exactly **once** -- qwen3:8b 1 of 22 right answers (0.045)
    and gemma4 1 of 21 (0.048) -- so the rates differ only because the denominators
    do, and the ordering rests on a difference of zero events. findings.md says of a
    one-trial gap in the same table that "it is not evidence of anything". In the
    market arms the ordering reverses in the placebo: recomputed from
    `docs/results/superseded/run-2models/decisions.parquet` with
    `influence_matrix(min_concession=0.20)`, net concessions are gemma4 **+3** in
    debate and **+56** in rationale-only against **-3** in the placebo -- which is
    the same disagreement D4 cites when withdrawing Finding 4. The
    fact-versus-judgement conclusion is withdrawn rather than restated.

Note. The published artefacts under `docs/results/superseded/` were written before those two
commits and still hold the pre-fix numbers; the figures above are recomputed from
the stored decisions with the current code. `docs/findings.md` was written from the
same pre-fix numbers and its Findings 2, 3 and 4 are marked withdrawn there, against
D2, D3 and D4; its Finding 1 is marked against D6.

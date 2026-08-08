# Claims made about this project

Every substantive assertion, in one place, so an auditor has a target rather than a
prose search. Each is either supported by an artefact in this repository or it is not.

## About the design

C1. The four arms (independent / debate / rationale-only / placebo) each rule out a
    distinct alternative explanation, and collapsing any of them destroys the inference.
C2. The stance axis (momentum vs reversion) is necessary because an aggression-only
    axis produces agreement on direction and therefore nothing to debate.
C3. A balanced design of 8 committees answers the same questions as the full 4^4 = 256
    grid, because every model appears at every persona exactly once.
C4. Simultaneous debate removes an order confound that a turn-taking debate would have.
C5. Anonymous peers remove a lab-reputation confound.
C6. Agents see normalised returns only, and no date, ticker or price level reaches them.
C7. The backtest fills at the next session's open and cannot see the future.

## About the measurements

C8.  Debate and placebo shift rates are close (debate 0.204 / 0.373 /
     0.248 / 0.407 against placebo 0.201 / 0.393 / 0.243 / 0.374; within 0.01 in
     2 of 4 buckets, placebo higher in 1 of 4). **Not supported as a claim about
     contradiction versus argument content: see D6.** The placebo arm these
     figures come from drew its donors from inside the decision window, which
     biases toward exactly that reading.
C9.  Shift rate across prior confidence is 0.204 / 0.373 / 0.248 / 0.407,
     therefore self-reported confidence does not predict who holds a position in
     any monotone way.
C10. Removing peers' exposure numbers gives 0.293 / 0.274 / 0.221 / 0.311,
     therefore the exposure number changes which confidence bands move but does
     not produce a confidence signal either.
C11. gemma4's net concessions are +3 in debate and -3 in placebo.
C12. qwen3 held facts better than gemma4 in the probe but yields more in the market
     arms, therefore defending a fact and defending a judgement are different
     behaviours.
C13. Every model separates stance strongly on rising series and barely on falling ones,
     therefore none will go long into a drawdown whatever persona it wears.
C14. The contested share was 100% on the two-model run and 138/140 on the four-model
     run, therefore the "debate only contested points" optimisation saves nothing here.
C15. Returns in this study are noise by construction (synthetic geometric random walks)
     and the behavioural columns do not depend on prices being real.

## About the process

C16. All three quality gates pass: pytest, ruff, mypy strict.
C17. Zero generation failures across every run reported.
C18. The primary comparison and both thresholds were fixed before any result existed.
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
    `docs/results/run-2models/decisions.parquet` with the shipped comparison.

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
    from `docs/results/run-2models/decisions.parquet` they are 2 of 4 and 1 of 4.

D6. Every published placebo number was generated before `cbf6a55` enforced a
    minimum gap between the donor day and the day being decided, so no gap was
    enforced when they were produced. Reconstructing that draw over the 1,104
    placebo conversations in `docs/results/run-2models/decisions.parquet`
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

Note. The published artefacts under `docs/results/` were written before those two
commits and still hold the pre-fix numbers; the figures above are recomputed from
the stored decisions with the current code. `docs/findings.md` was written from the
same pre-fix numbers and its Findings 2, 3 and 4 are marked withdrawn there, against
D2, D3 and D4; its Finding 1 is marked against D6.

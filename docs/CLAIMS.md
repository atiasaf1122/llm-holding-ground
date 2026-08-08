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

C8.  Debate and placebo shift rates are indistinguishable (within ~0.01 in 3 of 4
     buckets, placebo higher in 2 of 4), therefore what moves agents is contradiction
     rather than argument content.
C9.  Shift rate is flat across prior confidence (0.204 / 0.241 / 0.218 / 0.228),
     therefore self-reported confidence does not predict who holds a position.
C10. Removing peers' exposure numbers tilts that flat line (0.287 / 0.171 / 0.183 /
     0.148), therefore the number, not the argument, is what suppresses the
     confidence signal.
C11. gemma4's net concessions are +20 in debate and +21 in placebo, therefore the
     influence matrix measures yieldingness rather than persuasiveness.
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

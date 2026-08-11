# Claims made about this project

Every substantive assertion, in one place, so an auditor has a target rather than a
prose search. Each is either supported by an artefact in this repository or it is not.

**The run these figures come from.** Real daily bars for AAPL and XOM, split and
dividend adjusted, 2021-09-13 to 2023-12-29 (579 sessions), decisions on the 501
sessions from 2022-01-03 to 2023-12-29. Four models, four personas, eight committees.
40,968 stored decisions, **zero generation failures**. The debate arms ran on 50
decision points sampled evenly across the two years (`max_debate_points = 60`, of
which 10 had no placebo donor and were withheld from all three arms), giving 1,200
conversations and 24,936 debate rows. Artefacts: `docs/results/run-4models-2y/`
(decisions.parquet, results.json, prices.parquet, probe/ with one file per model) -- the pinned copies
the doc-contract tests recompute from; `data/` is the mutable working directory.

Every earlier figure in this file came from six-month synthetic-price runs and has been
replaced. Those runs survive at `docs/results/superseded/` and are provenance only.

## About the design

C1. The four arms (independent / debate / rationale-only / placebo) each bound a
    distinct alternative explanation, and collapsing any of them destroys the inference.
    Rationale-only withholds the peer's *structured* exposure field only, so it bounds
    anchoring on a peer's stated position rather than isolating anchoring: a figure a
    peer wrote into its own prose reaches the reader unchanged (see D1). The placebo
    additionally requires a donor `placebo_min_gap_sessions` back, and one distinct
    donor per round the cap allows; the points it cannot serve are withheld from all
    three treatment arms, so the three cover one identical set and the shortfall is
    against the independent control rather than between the arms. On this run that
    shortfall is 10 of 60 sampled points, all in the first sixty sessions.
C2. The stance axis (momentum vs reversion) is necessary because an aggression-only
    axis produces agreement on direction and therefore nothing to debate. **Supported
    on this run:** stance sets the sign for every model without exception -- momentum
    seats average +0.07 to +0.58 and reversion seats -0.16 to -0.77 across all four
    models -- while aggression only scales it.
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
C20. The debate arms are run on a sample of the contested points, not all of them
    (`council.sampling`). The sample is evenly spread over each ticker's calendar and
    nested, so a larger budget contains a smaller one. **What this costs is stated
    rather than hidden:** a treatment arm differs from the control on 50 of 1,002
    decision points, so the market-side comparison in C24 is diluted toward zero by
    construction and is not a test of anything. The behavioural measurements are
    unaffected, being computed per debated point.

## About the measurements

The previous C8-C15 were measurements from six-month synthetic-price runs, every one
of them qualified, withdrawn or marked unsupported. They are **superseded** by the run
described above, not amended by it; the figures they carried are not repeated here and
their artefacts remain at `docs/results/superseded/`. The defects found in them are
kept below, because the classes of error recur.

C8. **The placebo moves agents more than a real debate does.** Shift rates over 1,600
    observations per arm at 50 decision points: debate **0.324**, rationale-only
    **0.323**, placebo **0.383**. Paired by decision point and bootstrapped over the
    50 points, placebo minus debate is **+5.94pp, 95% CI [+3.31, +8.75]**, and placebo
    minus rationale-only is **+6.06pp, 95% CI [+3.00, +9.12]**. Neither interval
    contains zero. The placebo is the highest of the three for **all four models
    separately in the raw means** -- placebo against debate, gemma4:12b 29.0 vs 20.2,
    granite4.1:8b 52.2 vs 48.8, phi4:14b 45.8 vs 40.0, qwen3.5:9b 26.2 vs 20.5 -- and
    individually significant under the paired test in **three of the four**:
    gemma4 +8.75pp [+4.00, +13.51], phi4 +5.75pp [+1.00, +10.50], qwen3.5 +5.75pp
    [+1.50, +10.50], granite4.1 +3.50pp [-2.50, +9.50], whose interval spans zero.
    The spread between models is itself large: granite4.1:8b
    moves nearly half the time and gemma4:12b a fifth, so the pooled rate is a summary
    of four quite different behaviours rather than a property of "a language model".
    Stratified by committee type the gap is +8.75pp [+4.88, +12.62] in the uniform
    committees against **+3.12pp [+0.25, +6.12]** in the rotations -- and the uniform
    stratum's contrast is confounded (D11), so the rotation figure is the defensible
    one. Two bounds on that: the rotation gap is bar-sensitive (at a 0.15 bar its
    interval spans zero, [-0.25, +5.62], while the pooled gap excludes zero at every
    bar from 0.10 to 0.30), and the whole claim is a round-0-to-1 statement -- over
    the full conversation the ordering **reverses** (C28).
C28. **At the conversation's end the round-1 ordering inverts -- in the mixed
    committees -- and the inversion is a description of the arms as built, not a
    clean measurement of content.** Net |final - opening| >= 0.20 over whole
    conversations, rotation stratum (every arm runs ~5.9 rounds, so stopping-rule
    truncation is negligible): debate **0.484**, rationale-only **0.421**, placebo
    **0.372**. Increments, rotation: debate over rationale-only **+6.25pp
    [+3.12, +9.25]** (pooled +3.19 [+0.75, +5.44]); rationale-only over placebo
    **+4.88pp [+1.75, +8.00]**, which does **not** survive pooling
    (+1.88 [-0.81, +4.50]). Net placebo minus debate: **-11.12pp [-14.12, -8.00]**
    rotation, -5.06pp [-7.31, -2.75] pooled. Uniform endpoint rates are flat
    (0.34/0.34/0.35); per model the reversal is carried by phi4 (-31.5pp) and
    granite4.1 (-13.5pp) with gemma4 and qwen3.5 on zero. **The placebo renders its
    donors' position numbers exactly as the debate arm renders its peers'** (only
    rationale-only hides them), so rationale-vs-placebo varies rendering, content
    and schedule at once, and the endpoint comparison against the placebo also
    bundles its per-round fresh-donor schedule -- oscillation around fresh
    quasi-random anchors plausibly cancels by the endpoint, the same mechanism C11's
    entrenchment note concedes. Churn (consecutive movements reversing direction,
    rotation stratum, descriptive, no interval): debate **70.5%**, rationale-only
    **79.4%**, placebo **80.1%** -- the fixed-rendering pair (debate vs placebo)
    moves churn 9.6pp on content, the fixed-content pair (debate vs rationale-only)
    8.9pp on rendering; the three cells cannot rank the two factors.
C9. **The declared-bar null on withholding peers' numbers is the zero-crossing of a
    sign-changing dose-response, not the absence of an effect.** Debate minus
    rationale-only, round 0 to 1, by bar: **+4.19 [+2.19, +6.25]** at 0.10,
    **+2.12 [+0.31, +3.94]** at 0.15, +0.12 [-1.81, +2.06] at the declared 0.20,
    -1.88 [-4.00, +0.25] at 0.25, **-3.25 [-5.19, -1.31]** at 0.30. Showing the
    numbers produces significantly more *small* moves and significantly fewer
    *large* ones -- which is C10's anchoring read stated quantitatively -- and at
    the conversation's endpoint the numbers add **+6.25pp [+3.12, +9.25]** of
    lasting movement in the rotations (C28). The flat rate at 0.20 is the point
    where the two tails happen to cancel.
C10. **Seeing peers' numbers converges positions without changing minds.** Opening
    spreads are near-equal across arms (in debate / placebo / rationale-only order:
    rotations 1.264 / 1.281 / 1.267; uniforms
    0.541 / 0.561 / 0.552) -- *near*, not exactly, because the backend does not
    reproduce byte-identical prompts exactly even at temperature 0 with a fixed
    seed; see D12.
    Final spreads are not: rotations close to **0.703** in debate against 0.910 in
    rationale-only and 1.028 in placebo; uniforms to **0.153** against 0.364 and 0.232.
    So the exposure field does work -- it pulls seats toward one number -- but the work
    it does is too small to cross the 0.20 bar C9 measures. Anchoring and persuasion
    are different behaviours and this run separates them.
C11. **Mixed committees never agree.** Across all three arms, **zero** of 600 rotation
    conversations reached `agreement_spread`; 89.0 to 95.0% ran to the six-round cap
    still moving and the remaining 5.0 to 11.0% entrenched. Mean rounds held 5.87 to
    5.90 of 6. Uniform committees agreed in 92.0% (debate), 87.5% (placebo) and 68.0%
    (rationale-only) of conversations, in 1.94 to 3.06 rounds.
C12. **On questions with a verifiable answer, none of the four models capitulates.**
    Probe capitulation against its own placebo: qwen3.5:9b 1/20 vs 1/21 (**+0.002**),
    granite4.1:8b 1/22 vs 1/22 (**0.000**), phi4:14b 0/24 vs 0/24 (**0.000**),
    gemma4:12b 0/23 vs 0/23 (**0.000**). The same four models shift 32 to 38% of the
    time on the market question. The probe does not order the models -- three of four
    net rates are exactly zero -- and it is an upper bound rather than a measurement,
    for the reason its own output states. Artefact-backed since the D13 re-run:
    per-model files at `docs/results/run-4models-2y/probe/`, every row stamped with
    its model tag, recomputed by the doc tests. The re-run confirmed the previously
    printed-only table except phi4's denominator (24 for 23 -- one more trial passed
    the opening filter; D12-class regeneration noise).
C13. **The influence matrix measures concession-proneness, not persuasiveness.**
    granite4.1:8b nets **-228** in the debate arm and **-230** in the placebo, where
    the arguments are about another day -- and, half the time, another instrument
    (D14). A model that concedes equally to a
    relevant and an irrelevant argument is not being persuaded by either. Full net
    concessions, debate / rationale-only / placebo: gemma4 +188 / +185 / +131,
    qwen3.5 +184 / +155 / +149, phi4 -144 / -88 / -50, granite4.1 -228 / -252 / -230.
C14. **The dispersion gate is measured on the wrong unit, and this is the first run to
    say by how much.** `pipeline.select_contested` measures dispersion once over the
    pooled grid of all sixteen agents and hands one list to all eight committees. That
    pooled share is **984 of 1,002 (98.2%)**, and every contested share this project
    has published -- including that one -- is one of those pooled-grid shares.
    Recomputed per committee at
    `dispersion_threshold = 0.25` it is **4,728 of 8,016 (59.0%)**, ranging from
    **19.0%** (uniform-reversion-bold) to **98.5%** (rotation-3). So the gate as
    applied skips 1.8% of points and a committee-level gate would skip 41%. Nothing in
    this run is invalidated by that -- all four arms are run on one identical point set,
    so the comparisons hold -- but "contested" describes the grid and not the committee
    that debates, and a uniform committee is frequently debating a point it never
    disagreed about.
C15. **Prices are real.** Split and dividend adjusted daily bars from Yahoo Finance via
    `yfinance` with `auto_adjust=True`, fetched once and pinned at
    `data/prices.parquet`, with the vendor's response kept verbatim beside it. Over the
    decision window AAPL returned **+7.0%** and XOM **+69.2%** close-to-close (the
    backtest itself is open-to-open, where the same window reads +10.3% / +76.2%; the
    convention is stated because the two do not reconcile without it) -- AAPL fell hard
    through 2022 and recovered through 2023, which is what makes the window worth two
    years rather than one. Because the series are
    adjusted, an open here is not a price anyone could have traded at; it is the price
    that makes the return series continuous through a corporate action, which is the
    right input for a total-return backtest and the wrong one for claiming a fill was
    achievable.
C21. **The committee loses to buying and holding, and to chance.** Independent arm
    total return **-14.5%** (CAGR -7.6%, Sharpe **-1.38**) against buy-and-hold
    **+39.6%** (CAGR +18.3%, Sharpe +0.85). A turnover-matched
    random null scores Sharpe **-0.44** on the same calendar, so the arm is not merely
    unprofitable, it is worse than trading at random with the same activity. Both
    figures are over the 501 sessions decisions exist for; scoring the benchmark over
    the price file's full range instead credits it with a 78-session warm-up no arm
    could trade in, which read **+67.2%** and is the defect `_scoring_window` fixes.
C22. **The direction of that loss has an identified source.** Reversion seats commit
    roughly twice as hard as momentum seats -- averaging **-0.69** against **+0.35** for
    the bold pair (parquet: -0.688 / +0.348) -- so a committee mean leans net short
    (56.2% of decisions short, 39.0% long, 4.8% flat) through a window in which both
    instruments rose.
C23. **All four models agree about what each persona should do.** Mean exposure by
    persona holds its sign across every model: momentum-bold +0.12 to +0.58,
    momentum-cautious +0.07 to +0.26, reversion-cautious -0.16 to -0.36,
    reversion-bold -0.60 to -0.77. Architecture moves the magnitude, not the sign.
C24. **Debate changes nothing about performance.** Total returns: independent -0.145,
    debate -0.150, rationale-only -0.145, placebo -0.146. This is not evidence that
    debate cannot affect returns -- see C20; the arms differ on 50 of 1,002 points, so
    the comparison has almost no power and would look like this whether or not an
    effect existed.

C26. **Stated confidence carries no information about accuracy.** Correlation between
    an agent's self-reported confidence and whether its direction was right:
    **-0.015 over 15,206 independent-arm observations**, with hit rates of 0.482,
    0.507, 0.483 and 0.510 across the four populated bands. The overall hit rate is
    **48.8%**, and it is **not** claimed to be below chance: clustered by decision
    point (998 clusters, SE 0.0072) the z against 0.5 is -1.7, and the rate is pinned
    near 50% by design anyway, since momentum and reversion seats mirror each other on
    most points. Nothing in this study weights anything by confidence, which is why
    the question could be asked at all.
C27. **That confidence predicts *holding* is NOT claimed, and the raw correlation that
    suggests it is confounded.** Pooled over the treatment arms, shift rate falls
    across the four lowest confidence bands (1.000 / 0.836 / 0.443 / 0.304) and
    turns back up in the top band (0.391, n=115); the overall correlation is
    -0.142. But confidence and the extremity of the position correlate at **+0.675**,
    and an extreme position has less room to move before it reaches ±1. Controlling
    for extremity the relationship is non-monotone -- at extremity 0.75-1.00, under the
    study's own five confidence bands, the rates run 0.524 / 0.269 / 0.319
    (n = 42 / 1,174 / 72); splitting the wide middle band sharpens the reversal to
    0.524 / 0.186 / 0.297 / 0.319 but rests on n = 42 and 72 cells, so the five-band
    figures are the ones claimed. The claim is withheld. It is
    recorded rather than dropped because the uncontrolled figure looks like a result
    and is not one, and the same confound sits under any later analysis reading
    confidence against behaviour rather than against outcome.

## About the process

C16. All three quality gates pass: pytest, ruff, mypy strict.
C17. Zero generation failures across the whole run -- 40,968 decisions, no malformed
     output, no truncation, no unreachable backend recorded on any stored row.
C18. The primary comparison, `shift_threshold` and `dispersion_threshold` were fixed
     in the first commit (`afce0ae`), before any result existed. The four other
     bounds `config.py` declares -- `agreement_spread`, `stillness_rounds`,
     `max_debate_rounds` and `placebo_min_gap_sessions` -- were added in `cbf6a55`,
     after the results committed in `fa436fa` and `98a4020`, and `agreement_spread`
     and `placebo_min_gap_sessions` were calibrated from that run's measured
     spreads and donor distances. Those four are not pre-registered. The hashes
     name commits in the development repository; the shipped tree carries no git
     object store, so a reader of the tree alone cannot verify them -- they are
     testimony, not artefacts. Neither is
     `max_debate_points`, which was chosen after the control arm had run, from a
     measured pilot rather than from any result.
C19. Transcripts may be quoted from a partially complete run, but rates may not,
     because a partial run is ordered by committee and ticker rather than random.
C25. **The run was interrupted once and the interruption is visible in the artefacts.**
     Ollama updated itself from 0.32.5 to 0.32.6 mid-sweep; the restarted daemon
     returned a non-JSON response, which `PreflightError` raises rather than records,
     and the process stopped with 19 of 48 committee-arm-ticker groups complete. That
     is the designed behaviour: a systemic failure should halt a run rather than write
     tens of thousands of `UNAVAILABLE` rows. The resume skipped the 19 finished groups
     and no row in `decisions.parquet` carries a failure.

## Known already-found defects, do not spend time rediscovering

D1. The document describes +1.34% in a placebo peer block as "a number that exists
    nowhere in the agent's context", implying fabrication. It is not fabricated: it is
    a real rationale a model produced on a neighbouring decision point, which is
    exactly how the placebo is built. The wording is wrong and is being corrected.

D8. The placebo controls for being contradicted, but it does not control for being
    contradicted *incoherently*. A peer arguing about a different day describes
    conditions that do not match the data in front of the reader. C8 therefore
    supports "the movement is not a response to argument content" and does **not**
    distinguish "reacting to contradiction" from "reacting to an argument that cannot
    be reconciled". No design in this repository separates the two, and the hypothesis
    that an agent can evaluate and reject a relevant argument but hedges against an
    incoherent one is a hypothesis, not a finding.

D9. `buy_and_hold` is an equal-weight basket rebalanced on the same calendar as the
    arms, not a literal buy-and-hold. Measured difference against a true hold: 0.79pp
    on one seed and 8.8pp on another. The name overstates how passive the benchmark is
    and has not been changed.

D10. **The study cannot tell defending a position from staying in character, and this
    is the deepest of the open confounds.** Every agent is *instructed* into its view
    -- "you read price moves as momentum", "you read them as overshoot". So when a seat
    does not move, two readings fit equally: it holds a conclusion it reached, or it
    obeys a standing instruction. C23 makes the problem sharper rather than softer:
    the persona sets the sign for all four models without exception, so the prompt is
    doing more work than the architecture, and "does a model hold its ground" may be
    "does a model stay in role".

    Everything measured survives this. The arms are differenced against each other and
    all four carry the same instruction, so it cannot produce the placebo gap (C8), the
    convergence split (C10) or the agreement counts (C11). What it changes is what the
    results are *about*. Nothing here licenses a sentence beginning "language models
    defend their beliefs".

    Addressable without a new design: state the persona as a disposition the agent may
    revise ("you have tended to read moves as momentum") rather than as an identity,
    and run the two phrasings as a factor. Not run here.

D11. **The uniform stratum's placebo contrast confounds relevance with agreement.**
    In a uniform committee the "real debate" peers hold the reader's own persona and
    mostly agree with it (opening spread 0.541 against the rotations' 1.264), while
    the placebo's donor-day content does not -- so in that stratum "placebo minus
    debate" compares non-agreement against agreement, not irrelevance against
    relevance. The gap is 2.8× larger there (+8.75pp against +3.12pp), which means
    the pooled headline is dominated by the confounded stratum. C8 therefore quotes
    the rotation figure as the defensible one. Found in audit round 1, verified
    independently, and distinct from D8, which it compounds.

D12. **The run is not exactly deterministic, and one sentence claimed it was.**
    Temperature 0, fixed seed, byte-identical prompts (equal `prompt_hash` on every
    round-0 row across arms) -- and still 4.31% of round-0 seat-point exposures differ
    between the debate and placebo arms, 2.19% by at least the 0.20 shift bar, with a
    maximum difference of 1.65, a sign reversal. This puts a ~2pp noise floor under
    every shift rate. It is symmetric across arms and so cannot create the gaps
    between them, but findings §2 said "identical across arms by construction" and
    the artefact says otherwise; the sentence is corrected and the floor is now
    reported. The README's "single deterministic sample" limitation is corrected the
    same way.

D13. **RESOLVED. The probe table's first publication rested on an overwritten
    artefact.** `council probe` wrote to one fixed default path, so four sequential
    runs each overwrote the last: only the final model's 48 trials survived, the
    rows carried no model field, and three of the four published rows rested on
    printed output alone. Fixed twice over -- the default target is now a per-model
    file (`council.probe.session.probe_target`) and every trial row is stamped with
    its model tag -- and then re-run: the artefacts at
    `docs/results/run-4models-2y/probe/` confirm the printed table exactly, except
    phi4's denominator (24 for 23; one more trial passed the opening filter,
    D12-class regeneration noise). Kept in the register because the *class* of error
    -- a shared default output path across runs that differ in a field the row does
    not record -- is general, and because for a day the repository published a table
    it could not back.

D14. **The placebo is displaced in instrument as well as in time, and only a module
    docstring said so.** `select_placebo_point` constrains the donor's date and not
    its ticker: re-running the shipped selector over every round-1 placebo point,
    **49.0% of donors are the other instrument**, and the median donor sits 214
    calendar days from the day being decided (max 671) -- "at least 60 sessions"
    understates typical displacement by roughly a factor of three. Every
    reader-facing description said "an unrelated day"; the arm is better described
    as "an unrelated day, and on a coin flip an unrelated instrument". This
    strengthens D8's incoherence reading -- a peer describing another instrument's
    regime is talking past the reader, not merely disagreeing -- and C6's
    anonymisation means the reader has no way to notice: no ticker, date or price
    level reaches it.

### Superseded defects, retained for provenance

D2 to D7 recorded errors in figures from the two-model six-month synthetic runs -- a
bare `>=` comparison that dropped grid-exact movements (D2, D3, D5), an influence
sign that flipped once both bars ran through `evaluation.threshold.meets` (D4), a
placebo whose donors overlapped the decision window because no minimum gap was
enforced (D6), and a fact-versus-judgement conclusion resting on a difference of zero
events (D7). Every claim they attacked has been replaced by a measurement from the run
described at the top of this file. They are kept because the *classes* of error recur:
a floating-point comparison on a grid, a control that is not inert, and an ordering
built on one event. The artefacts are at `docs/results/superseded/`.

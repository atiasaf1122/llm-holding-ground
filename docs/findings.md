# Findings

What the run measured, and what it does not license anyone to say.

**The run.** One two-year run at daily decision frequency -- the only frequency this
repository has, and the reason no result below is stated as holding at another. Real
daily bars for AAPL and XOM, split and dividend adjusted, decisions
on the 501 sessions from 2022-01-03 to 2023-12-29. Four models (`qwen3.5:9b`,
`granite4.1:8b`, `phi4:14b`, `gemma4:12b`), four personas, eight committees --- four
Latin-square rotations and four uniform references. **40,968 stored decisions in the
four original arms, 11,296 more in the two extension arms --- 52,264 in all, zero
generation failures.** The debate arms ran on 50 decision points sampled evenly across
the two years, giving 1,200 conversations in the three original arms and 2,000 across
all five.

The core figures below --- the shift table, the pooled rates, the headline and
endpoint intervals, and the calibration numbers --- are recomputed by
`tests/test_docs_findings.py` from
`docs/results/run-4models-2y/decisions.parquet`, the artefact this run produced; a
recomputed number that drifts from the parquet fails the suite. The remaining
intervals (per-model, strata, bar sweeps, endpoint increments) and the descriptive
tables reproduce from the same parquet but are verified by audit rather than
locked by test. Section 4's probe table is recomputed
from the per-model artefacts at `docs/results/run-4models-2y/probe/`, each row
carrying its model tag --- the re-run that D13 required, which confirmed the
previously unbackable table to the digit bar one denominator.

---

## 1. The placebo moves agents more than a real debate does --- at the first exchange

This is the headline as first published, and it is the opposite of what the
experiment was built expecting to find. **Read it with the extension verdict at
the end of this document**: the surplus decomposed into its foreign-instrument
component and then inverted, and the force that actually dominates --- undiluted
coherent opposition --- turned out to be twice as large as anything in this
section.

### Shift rate, by the confidence held before seeing any peer

Bar 0.20. Cells are `rate (observations)`; a decision point contributes up to 32
observations, so the observation count is not a sample size.

| prior confidence | debate | debate_placebo | debate_rationale_only |
| --- | --- | --- | --- |
| 0.00 – 0.20 | 1.000 (14) | **1.000 (14)** | 1.000 (13) |
| 0.20 – 0.40 | 0.826 (23) | **0.864 (22)** | 0.818 (22) |
| 0.40 – 0.60 | 0.409 (276) | **0.519 (270)** | 0.401 (272) |
| 0.60 – 0.80 | 0.285 (1249) | **0.338 (1256)** | 0.289 (1254) |
| 0.80 – 1.00 | 0.421 (38) | 0.395 (38) | 0.359 (39) |

Pooled across bands: debate **0.324**, rationale-only **0.323**, placebo **0.383**,
over 1,600 observations each at 50 decision points.

### It survives being tested properly

Observations inside one decision point are not independent --- the same day, the same
prices, thirty-two seats. So the comparison is paired by decision point and
bootstrapped over the 50 points, which is the unit that varies:

| comparison | difference | 95% CI | points where higher |
| --- | --- | --- | --- |
| placebo − debate | **+5.94pp** | [+3.31, +8.75] | 32 of 46 |
| placebo − rationale-only | **+6.06pp** | [+3.00, +9.12] | 32 of 42 |
| debate − rationale-only | +0.12pp | [−1.81, +2.06] | --- |

And it is not one model dragging the average. The placebo is the highest of the three
arms for **every model separately** in the raw means — with per-model paired intervals
that temper how far that replication goes:

| model | debate | rationale-only | placebo | placebo − debate, paired 95% CI |
| --- | --- | --- | --- | --- |
| gemma4:12b | 20.2% | 22.8% | **29.0%** | +8.75pp [+4.00, +13.51] |
| granite4.1:8b | 48.8% | 48.8% | **52.2%** | +3.50pp [−2.50, +9.50] |
| phi4:14b | 40.0% | 35.5% | **45.8%** | +5.75pp [+1.00, +10.50] |
| qwen3.5:9b | 20.5% | 22.0% | **26.2%** | +5.75pp [+1.50, +10.50] |

Three of the four intervals exclude zero. granite4.1's does not — the model with the
highest base shift rate is also the one whose placebo gap could be noise — so the
honest statement is "consistent in direction in all four, individually significant in
three".

### Where the effect lives, and the confound that lives with it

Split by committee type, the gap is 2.8× larger in the uniform committees:

| stratum | placebo − debate | 95% CI |
| --- | --- | --- |
| uniform (one persona ×4) | **+8.75pp** | [+4.88, +12.62] |
| rotation (four personas) | **+3.12pp** | [+0.25, +6.12] |

That difference matters, because in a *uniform* committee the "real debate" peers
mostly **agree** with the reader — same persona, same reading, opening spread 0.541
against the rotations' 1.264 — while the placebo's donor-day content does not. In that
stratum the comparison is not "relevant argument vs irrelevant argument"; it is
"agreement vs non-agreement", and being shown agreement rarely moves anyone. The
stratum where the real debate actually contains opposition — the rotations — still
shows the effect, at +3.12pp with an interval whose lower edge sits just above zero.

So the pooled +5.94pp overstates what the placebo *isolates*. The defensible claim is
the rotation stratum's: peers arguing about the wrong day move an agent somewhat more
than peers genuinely arguing against it about the right one. Registered as D11 in the
claims register, beside D8, which it compounds.

Two qualifications on that "defensible" figure, both found in audit:

**The placebo is displaced in instrument, not only in time.** The donor draw
constrains the date and not the ticker: **49.0% of round-1 donors are the other
instrument** (verified by re-running the shipped selector over every placebo point),
and the median donor sits **214 calendar days** away --- "at least 60 sessions"
understates typical displacement by a factor of three. A peer describing another
instrument in another regime is not so much contradicting the reader as talking past
it, which strengthens the incoherence reading of D8. Registered as D14.

**The rotation figure is bar-sensitive.** Recomputed at bars 0.10 / 0.15 / 0.20 /
0.25 / 0.30, the rotation gap is +3.62 / +2.62 / +3.12 / +4.25 / +3.75pp --- and the
interval at **0.15 spans zero** ([−0.25, +5.62]) while 0.20's lower edge sits at
+0.25pp. The pooled gap excludes zero at every bar; the stratum quoted as defensible
is the fragile one, and a reader should know the declared bar is doing work.

**And the extension run then unmade it.** With the donor constrained to the
reader's own instrument, this arm's surplus over debate *reverses* --- same-
instrument placebo minus debate is −5.00pp [−8.12, −1.50] --- so the +3.12pp was
carried by the cross-instrument donors D14 counted. The decomposition is at the
end of this document; this paragraph stands as the record of what the headline
looked like before it.

### The movement does not last --- in the mixed committees, the ordering reverses

Everything above is a round-0-to-1 statement, and it must be read as one, because
measured over the **whole conversation** --- net |final − opening| ≥ 0.20, rotation
stratum, where every arm runs ~5.9 rounds so stopping-rule truncation is negligible ---
the ordering inverts:

| net movement, opening to final | debate | rationale-only | placebo |
| --- | --- | --- | --- |
| rotation stratum | **0.484** | 0.421 | 0.372 |

Paired by decision point: net placebo − debate **−11.12pp [−14.12, −8.00]** in the
rotations, −5.06pp [−7.31, −2.75] pooled. The two increments, **rotation stratum**:
debate over rationale-only **+6.25pp [+3.12, +9.25]** (pooled +3.19 [+0.75, +5.44]),
rationale-only over placebo **+4.88pp [+1.75, +8.00]** --- and that second one does
**not** survive pooling (+1.88 [−0.81, +4.50]), so it is a mixed-committee fact only.
The reversal is a rotation-stratum fact throughout: in the uniform committees the
endpoint rates are flat (0.34 / 0.34 / 0.35) and nothing inverts. Per model it is
carried by phi4 (−31.5pp [−39.0, −23.5]) and granite4.1 (−13.5pp [−22.0, −5.0]) while
gemma4 and qwen3.5 sit on zero --- so neither the round-1 gap nor its reversal is a
uniform law.

Churn --- the share of a seat's consecutive movements that reverse direction, rotation
stratum, no interval attached: debate **70.5%**, rationale-only **79.4%**, placebo
**80.1%**. Read against what each arm actually renders (the placebo **shows** its
donors' position numbers, exactly as the debate arm shows its peers' --- only the
rationale-only arm hides them): the debate-placebo pair holds the rendering fixed and
varies the content, and churn moves 9.6pp; the debate-rationale pair holds content
fixed and varies the rendering, and churn moves 8.9pp. Both matter, and the three
cells cannot rank them. What a seat oscillates least around is its *actual* peers'
visible, persistent positions; whether a stable anchor, a relevant argument, or both
is doing the steadying, this design cannot say.

One mechanical caveat, and it cuts against the endpoint comparison too: the placebo
redraws a **fresh** donor every round, so its seats face a novel stimulus --- with a
novel, quasi-randomly-directed anchor number --- each round, while debate peers
converge and repeat. Oscillation around fresh anchors plausibly cancels by the
endpoint, which would depress the placebo's net movement mechanically; §3 concedes
exactly this mechanism for its entrenchment shares, and consistency demands it here.
The endpoint ordering is a finding about *the arms as built*, schedule included ---
not a clean measurement of content.

### What it means --- two clocks, and what each is allowed to say

At the first exchange, the arm whose argument is about the day being decided moves
agents least; the arm whose argument is about another day --- and, half the time,
another instrument (D14) --- moves them most, at identical rendering, before any
schedule difference exists (round 1 is each arm's first stimulus). So the immediate
reaction is not content-*blind* --- it is content-sensitive in the wrong direction:
arguments that cannot be reconciled with the data in view move agents *more* than
arguments that engage it. Whether that is a response to contradiction or to
incoherence is D8's boundary.

At the conversation's end the ordering inverts: the arms whose peers engage the
actual day retain the most movement, the placebo the least. But the endpoint
comparison bundles content with rendering and with the placebo's fresh-donor
schedule, and the one increment that looked like "relevant prose alone outlasts
irrelevant prose" is null when pooled --- so the endpoint licenses a description, not
a mechanism.

So neither headline survives alone, and neither should be quoted without the other:
"the placebo moves agents more" is true of the first exchange and false of the
outcome; "debate produces the most lasting movement" is true of the outcome and
confounded as an explanation. What both clocks agree on is the negative result the
study was built to test: at no point, on either clock, does any arm produce the
round-1 reasoned mind-change that "persuasion" usually names.

It does **not** establish that agents are reacting to contradiction as such. The
placebo peer argues about a different day, so its argument is not merely irrelevant,
it is **incoherent** against the data the reader is holding. Reacting to contradiction
and failing to reconcile an argument with the evidence are two different behaviours,
and no arm in this design separates them. The hypothesis --- an agent can evaluate and
reject a relevant argument, but hedges when it cannot place the argument at all --- is a
hypothesis. It is recorded as D8 in the claims register and is not a finding.

---

## 2. Seeing peers' numbers converges positions without changing minds

The rationale-only arm withholds the peer's structured exposure field. On the shift
rate at the declared bar it does nothing: +0.12pp against the full debate arm, on an
interval centred on zero. That null is the **zero-crossing of a dose-response**, not
an absence: by bar, debate minus rationale-only runs **+4.19 [+2.19, +6.25]** at
0.10, **+2.12 [+0.31, +3.94]** at 0.15, +0.12 at the declared 0.20, −1.88
[−4.00, +0.25] at 0.25, **−3.25 [−5.19, −1.31]** at 0.30. Showing the numbers
produces significantly more *small* moves and significantly fewer *large* ones ---
this section's story told quantitatively: the visible number pulls seats into small
convergent steps that stop short of big jumps --- and the declared bar happens to sit
where the two tails cancel.

On the *spread between seats* it does a great deal. Opening spreads are *near*-equal
across arms --- round 0 renders the identical prompt everywhere, and the stored
`prompt_hash` matches across all four arms on every row --- but they are not identical,
and the reason is worth a paragraph. At temperature 0 with a fixed seed, the backend
still does not reproduce exactly: **4.3% of round-0 seat-point exposures differ between
the debate and placebo arms on byte-identical prompts**, 2.19% differ by at least the
0.20 shift bar, and the largest single difference is 1.65 --- a full sign reversal. That
puts an unreported-until-now noise floor of roughly **2pp** under every shift rate in
this document. It is symmetric across arms, so it cannot manufacture the gaps between
them, but "identical by construction" was an assertion about the backend, not about the
artefact, and the artefact disagrees. Registered as D12; the opening spreads below are
the measured ones per arm's own rows:

| committee | opening | debate | rationale-only | placebo |
| --- | --- | --- | --- | --- |
| rotation (four personas) | 1.264 | **0.703** | 0.910 | 1.028 |
| uniform (one persona ×4) | 0.541 | **0.153** | 0.364 | 0.232 |

So the number on the page does work. It pulls seats toward one another --- a 44%
narrowing in mixed committees against 20% in the placebo --- and **within a single
round** that pull is too small to carry a seat across the 0.20 bar, which is why the
round-0-to-1 shift rate cannot tell the debate and rationale-only arms apart. Over a
whole conversation the small pulls do accumulate into bar-crossing movement (§1's
endpoint table, where the visible numbers add +6.25pp of lasting movement over prose
alone) --- and they accumulate in a *straighter line*: with the numbers visible only
70.5% of consecutive movements reverse direction, against 79.4% without them. Same
mechanism, two timescales: per round it nudges, per conversation it steers.

The anchoring magnitudes are read off the **rotation** row deliberately. The uniform
row's final spreads are truncated on exactly the quantity being compared: a
conversation stops when the spread reaches 0.20, the three arms agree at different
rates (92% / 68% / 87.5%) and stop at different rounds, so a uniform arm's "final
spread" partly measures when its stopping rule fired. The rotations never agree, run
~5.9 rounds in every arm, and are free of that truncation.

That is anchoring in its plain form, and this run separates it from persuasion. A
study measuring only convergence would have called this agreement. A study measuring
only shift rate would have missed it.

The placebo, at the first exchange, is the exact inverse: the most individual
movement, the least convergence. Motion without direction --- and, by the endpoint,
mostly motion without residue (§1's reversal).

---

## 3. Mixed committees never agree

Not "rarely". Across all three arms, **zero of 600** rotation conversations reached
`agreement_spread`.

| committee | arm | mean rounds | agreed | hit the cap | entrenched |
| --- | --- | --- | --- | --- | --- |
| rotation | debate | 5.89 | 0% | 89.0% | 11.0% |
| rotation | rationale-only | 5.87 | 0% | 90.0% | 10.0% |
| rotation | placebo | 5.90 | 0% | 95.0% | 5.0% |
| uniform | debate | 1.94 | 92.0% | 7.0% | 1.0% |
| uniform | rationale-only | 3.06 | 68.0% | 27.5% | 4.5% |
| uniform | placebo | 2.50 | 87.5% | 11.5% | 1.0% |

Two things worth separating here.

**The rotations are indifferent to what they are shown.** 5.87, 5.89 and 5.90 rounds
across the three arms; 89–95% hit the cap. A mixed committee runs to the end of its
budget whether its peers are arguing about today or about a day eighteen months ago.

**The uniform committees' agreement is the anchoring of section 2, not consensus.**
They agree 92% of the time when they can see each other's numbers and 68% when they
cannot --- and take 60% longer to get there. They were never in conflict: measured per
committee, a uniform committee's decision points are contested only 19–43% of the time
(section 7). Four agents who already agreed, agreeing faster when shown the number.

`SETTLED` --- the stopping condition for a committee that stops without agreeing --- was
unreachable in every earlier version of this study, because the round cap was pinned
at one and a stillness streak needs two. It occurs 5–11% of the time in the rotations.
The outcome the study was built to observe was, until this run, unobservable by
construction.

The *cross-arm* comparison of those entrenchment shares (placebo 5% vs debate 11%)
should not be read behaviourally: the placebo redraws a fresh donor every round, so
its seats face a novel stimulus each round while debate peers converge and repeat ---
stillness is being measured under systematically different stimulus schedules, and
"the placebo entrenches less" is partly mechanical.

---

## 4. On questions with a right answer, none of them capitulates

The same four models, the same machine, the same challenge mechanism --- put to 24 items
with verifiable answers instead of a market judgement.

| model | capitulation under challenge | under placebo | net |
| --- | --- | --- | --- |
| qwen3.5:9b | 0.050 (1/20) | 0.048 (1/21) | **+0.002** |
| granite4.1:8b | 0.045 (1/22) | 0.045 (1/22) | **0.000** |
| phi4:14b | 0.000 (0/24) | 0.000 (0/24) | **0.000** |
| gemma4:12b | 0.000 (0/23) | 0.000 (0/23) | **0.000** |

Two capitulations in 89 challenged trials. The same models shift 32–38% of the time on
the market question.

**Provenance, resolved (D13):** the table's first publication rested on printed output
--- four runs had overwritten one default file. The probe was re-run with per-model
output paths and a `model` field stamped on every row; the artefacts are at
`docs/results/run-4models-2y/probe/`, this table is recomputed from them by
`tests/test_docs_findings.py`, and the re-run **confirmed the printed table** with one
exception worth naming: phi4's denominator is 24 where the printed run said 23 --- one
more trial passed the opening-correctness filter, the same regeneration noise D12
measures. The old gemma4 artefact from the superseded two-model era
(`docs/results/superseded/probe-gemma4.jsonl`, 1/21 and 2/21) remains what it was: a
different model tag from a different era, retired.

The probe does **not** order the models against each other: three of four net rates are
exactly zero and the fourth rests on a single event. It is also an upper bound rather
than a measurement, because its placebo peer argues visibly about another question, so
a model that notices discounts it for a reason the market placebo never offers.

The cross-domain contrast is also not apples-to-apples, in two ways a reader should
weigh. The metrics differ: probe capitulation is a discrete answer *flip*, while the
market 32–38% counts any |Δexposure| ≥ 0.20 including hedging that never changes
side --- the nearest market analogue of a flip is a **sign reversal**, which runs
15.8% (debate) / 18.9% (rationale-only) / 21.4% (placebo), roughly half the quoted
rate. And the stimulus differs: one anonymous challenger with a short fixed argument
against three peers with longer model-written rationales and stated positions. So the
contrast is qualitative --- near-zero against clearly-nonzero --- not a measured ratio.

With that said, what it establishes still frames everything above: the movement in
sections 1 to 3 is not a general willingness to be talked out of a position. Even on
the flip-for-flip comparison it is 0-to-2 events in 89 against 16–21% --- specific to
a question with no verifiable answer.

---

## 5. Stated confidence carries no information about accuracy

Every agent reports a confidence with its exposure, and nothing in this study weights
anything by it --- deliberately, because whether it means anything is one of the
questions. It does not.

| stated confidence | hit rate | n |
| --- | --- | --- |
| 0.20 – 0.40 | 0.482 | 56 |
| 0.40 – 0.60 | 0.507 | 2,479 |
| 0.60 – 0.80 | 0.483 | 12,189 |
| 0.80 – 1.00 | 0.510 | 482 |

Correlation between stated confidence and being right: **−0.015 over 15,206
observations**. A flat line on a coin. The overall hit rate is **48.8%** --- and that is
**not distinguishable from chance** once the observations are clustered by decision
point, which this document's own section 1 argues they must be: 998 clusters, SE
0.0072, z = −1.7. The naive per-observation z of −3.0 is exactly the mistake section 1
refuses elsewhere. Nor would "worse than chance" have meant much if it held: the
pooled hit rate is pinned near 50% by design, because momentum and reversion seats
take opposite directions on three quarters of the points, so one is right almost
whenever its mirror is wrong. The money in section 6 was lost through the *sizes* ---
reversion committing twice as hard as momentum --- which a directional hit rate cannot
see.

This is the largest sample in the study by two orders of magnitude, and it is the one
result here that transfers directly to anyone building on these models: a committee
weighted by self-reported confidence is weighted by noise.

### The half that does not survive its own check

Confidence *looks* like it predicts holding. Pooled over the three treatment arms the
shift rate falls across the four lowest bands --- 1.000, 0.836, 0.443, 0.304 --- and
then turns back **up** in the top band (0.391, n=115), at an overall correlation of
−0.142.

It is confounded. Confidence and the extremity of the position taken correlate at
**+0.675**: a confident agent takes a large position, and a large position has less
room to move before it hits ±1. Controlling for extremity, the relationship is not
monotone: at extremity 0.75–1.00 under this document's own five confidence bands the
shift rate runs **0.524, 0.269, 0.319** (n = 42, 1,174, 72) --- falling and then rising
again. (Splitting the wide middle band in half sharpens the reversal --- 0.524, 0.186,
0.297, 0.319 --- but rests on cells of n = 42 and 72, so the five-band figures are the
ones stated.)

So the claim is **not made**. It is recorded here rather than dropped because the raw
correlation is real, publishable-looking, and wrong, and because the same confound
sits under any future analysis that reads confidence against behaviour rather than
against outcome.

---

## 6. Nobody made any money, and debate did not change that

Scored over the 501 sessions decisions exist for.

| arm | total | CAGR | Sharpe | max DD | turnover/p | turnover-matched random |
| --- | --- | --- | --- | --- | --- | --- |
| independent | −0.145 | −0.076 | **−1.38** | 0.153 | 0.099 | −0.44 |
| debate | −0.150 | −0.079 | −1.39 | 0.157 | 0.115 | −0.76 |
| rationale-only | −0.145 | −0.076 | −1.38 | 0.153 | 0.107 | −0.64 |
| placebo | −0.146 | −0.077 | −1.37 | 0.154 | 0.110 | −0.88 |
| buy_and_hold | **+0.396** | +0.183 | +0.85 | 0.139 | 0.002 | --- |

The committee lost 14.5% over a window in which holding the basket returned 39.6%, and
scored worse than a random null matched to its own turnover. It is not merely
unprofitable; it is systematically wrong-way.

**The direction has a source.** Reversion seats commit about twice as hard as momentum
seats --- the bold pair averages −0.69 against +0.35 --- so the committee mean leans short
(56.2% of decisions short, 39.0% long, 4.8% flat) through a window in which both
instruments rose. In a rising market, "this has overshot and will revert" produces more
conviction than "this is trending and I should join".

All four models agree about what each persona should do. The sign holds for every one
of them:

| persona | gemma4 | granite4.1 | phi4 | qwen3.5 |
| --- | --- | --- | --- | --- |
| momentum-bold | +0.245 | +0.452 | +0.577 | +0.120 |
| momentum-cautious | +0.094 | +0.264 | +0.125 | +0.067 |
| reversion-cautious | −0.157 | −0.219 | −0.179 | −0.363 |
| reversion-bold | −0.662 | −0.718 | −0.601 | −0.771 |

Architecture moves the magnitude. The persona decides the sign. That is the
precondition the design was built on (C2) and it holds without exception.

**The arms' returns must not be read as a test of anything.** The debate arms
differ from the control on 50 of 1,002 decision points, because the debate was sampled
to fit the machine. A comparison that thin would look like this whether or not an
effect existed. The behavioural measurements are unaffected --- each is computed per
debated point --- but this table is description, not evidence.

---

## 7. The dispersion gate is measured on the wrong unit

`pipeline.select_contested` measures dispersion once over the pooled grid of all
sixteen agents, then hands one list of points to all eight committees. Every contested
share this project has ever published is one of those pooled-grid shares. Recomputed
per committee, the picture is completely different:

| committee | contested | share |
| --- | --- | --- |
| rotation-3 | 987 / 1,002 | 98.5% |
| rotation-0 | 927 / 1,002 | 92.5% |
| rotation-2 | 829 / 1,002 | 82.7% |
| rotation-1 | 725 / 1,002 | 72.4% |
| uniform-reversion-cautious | 434 / 1,002 | 43.3% |
| uniform-momentum-bold | 417 / 1,002 | 41.6% |
| uniform-momentum-cautious | 219 / 1,002 | 21.9% |
| uniform-reversion-bold | 190 / 1,002 | 19.0% |
| **across all committees** | **4,728 of 8,016** | **59.0%** |
| pooled over the grid, as applied | 984 / 1,002 | **98.2%** |

The gate as it ships skips 1.8% of points. A committee-level gate would skip 41%.

Nothing here is invalidated by that --- all five debate arms run on one identical point set, so
every comparison holds --- but "contested" describes the grid, not the committee that
debates. A uniform committee spends most of its budget arguing about points it never
disagreed on, which is exactly what section 3 shows it doing.

---

## Three caveats that matter more than the numbers

**The placebo is not a clean isolation of contradiction.** See section 1. It controls
for being disagreed with; it does not control for being disagreed with *incoherently*.
This is the weakest joint in the design and it sits directly under the headline result.

**Recognition is a live risk, not a hypothetical.** Every model was trained on data
covering 2022–2023, and AAPL and XOM are two of the most written-about instruments in
it. Agents see normalised returns with no dates, no tickers and no price levels, which
reduces recognition and cannot prove its absence. A distinctive drawdown shape is still
a shape.

**Two instruments, two years, four models, one machine.** The stance effect (section
6) is uniform enough across models to be worth stating. The placebo effect (section 1)
is highest in all four models' raw means and individually significant in three ---
granite4.1's paired interval spans zero --- which is the strongest internal replication
this design offers, and it is still one run of one design on one pair of instruments.

## What would settle the open question

D8 --- contradiction versus incoherence --- is answerable with one more arm, not a new
study: peers whose rationales are *relevant to the day* but argue for the *wrong
conclusion*, generated by asking a seat to argue against its own view on the correct
data. That arm contradicts coherently.

Three specifications the proposal needs now, which the first draft of this section
predated: **the adjudicating metric is the round-0-to-1 shift rate** --- the declared
primary comparison, and the one where "moves like the placebo" and "moves like the
debate arm" are distinct orderings, since at the endpoint they reverse (C28). **The
peer schedule must match the arm it is compared against** --- the placebo redraws a
fresh donor per round and the debate arm's peers persist, so a later-round comparison
against either is confounded by novelty unless the new arm copies that arm's
schedule. **So must the rendering**: the debate and placebo arms both show peers'
position numbers and rationale-only hides them, so the coherent contradictor must
show them too, or any gap it produces is partly the rendering. And the cheaper
decomposition should run first: **a same-instrument-only
placebo**, which D14 makes almost free --- constrain the donor's ticker and the
existing arm splits "wrong day" from "wrong instrument" with no new machinery.

If the coherent contradictor moves agents like the placebo at round 1, the finding is
about contradiction. If it moves them like the debate arm, the finding is about
incoherence.

### The extension arms, as built, and the rule that will read them

*This subsection is written and committed before either arm has generated a row.
Both arms run on the same 50 decision points as the published run, alongside it in
the same store, and nothing already published re-runs.*

**`debate_placebo_same_instrument`** is the published placebo with the donor's
ticker constrained to the reader's. Full protocol otherwise: six-round cap, fresh
donor per round, numbers rendered. A preflight over the published pool found a
minimum of 17 same-ticker candidates across all 400 (point, committee) pairs, so
the constraint costs no coverage.

**`debate_contradictor`** deviates from the sketch above in one declared way: the
counters are **peer-authored** rather than self-authored --- each of the reader's
three peers writes the case against the reader's opening view, on the reader's own
price context, because three self-authored counters would fill three peer slots
with one model's voice, a rendering difference riding along with the manipulation.
The opposite side was *requested* through the output grammar (a flat reader's
contradictor is pushed to a decisive position on a digest-chosen side) --- and the
post-run audit found the backend does not enforce numeric schema bounds at all;
see D15 in the verdict below. Every counter is archived to `counters.jsonl`; the
arm is capped at **one rebuttal round**, because the adjudicating metric below is
round 0-to-1 and no later-round schedule for a targeted contradiction is
defensible.

**The adjudication rule, fixed now.** Rotation stratum, round-0-to-1 shift rate at
the declared 0.20 bar, paired by decision point through the shipped
`council.evaluation.intervals.paired_shift_gap`, 5,000 seeded draws:

1. If CI(contradictor − placebo) excludes zero from below while CI(contradictor −
   debate) contains zero → the placebo's surplus was **incoherence**: coherent
   opposition moves agents no more than genuine debate.
2. If CI(contradictor − debate) excludes zero from above while CI(contradictor −
   placebo) contains zero → **contradiction itself** moves agents, and coherence
   does not protect against it.
3. Both exclude zero (the contradictor lands between the arms) → both factors
   contribute; the two gaps are the decomposition and are reported as such.
4. Neither excludes zero → inconclusive at this n; reported plainly, no narrative.

For the same-instrument placebo the primary contrast is (same-instrument −
cross-instrument placebo), same stratum, same machinery: a negative interval
excluding zero attributes part of the placebo surplus to instrument displacement;
an interval containing zero says displacement contributed nothing measurable.
Uniform-stratum and pooled figures will be reported beside these but do not
adjudicate, for D11's reason.

### The verdict --- and the number the rule did not imagine

*Everything above this line was committed before the arms ran. This subsection
reports what they produced: 11,296 decisions, zero generation failures, one
conversation abandoned to a validation error and completed on the resume, every
counter archived. Artefacts in `docs/results/run-4models-2y/`
(`counters.jsonl` is an append log; a resume may append a conversation's
counters twice, and readers keep the last line per (date, ticker, committee,
reader, author) --- 4,800 unique pairs).*

The round-0-to-1 shift rates, all five arms side by side:

| arm | shift rate |
| --- | --- |
| debate | 0.324 |
| rationale-only | 0.323 |
| same-instrument placebo | **0.336** |
| cross-instrument placebo | 0.383 |
| **coherent contradictor** | **0.606** |

*(Rates in the table are pooled; every interval quoted below is the
pre-registered rotation stratum.)*

**The rule's branch 3 fires --- both intervals exclude zero --- but not in the
way its parenthetical imagined.** Contradictor − placebo is **+23.75pp
[+20.12, +27.62]** and contradictor − debate is **+26.88pp [+23.12, +30.63]**.
The branch says "the contradictor lands between the arms"; it landed **above
both**, by twenty points --- an outcome the rule as written had no branch for,
reported under the branch whose condition fired. Unanimous targeted opposition
does not move agents *like* the placebo or *like* the debate arm --- it moves
them nearly **twice as much as either**. 97.6% of the movements go *toward* the
opposition; 77.6% of all movements --- 87.0% of those that began with a side ---
end on the opposite sign, and over all contradicted decisions (movers and
holders alike) an outright sign reversal occurs on **47.1%**: just under half
of every decision the arm touched, and just over half (51.4%) of the decisions
whose readers had a side to reverse.

**D15, found by the post-run audit: the side constraint did not hold.** The
counters' opposite side was written into the output grammar's numeric bounds,
and the backend turns out not to enforce numeric bounds at all: **6.3% of sided
counters agreed with the reader**, reaching 15.8% of readers --- who, having
been shown agreement instead of opposition, shifted at only 0.238. The
**per-protocol rate --- readers who received the described treatment --- is
0.675**, above the intent-to-treat 0.606 in the table, so the defect is
conservative for the headline; but the arm as described was not everywhere the
arm as delivered, the per-model figures below are per-protocol for that reason
(intent-to-treat in parentheses), and the test suite missed it because the mock
provider obeys the bounds the real backend ignores. The generation code now
verifies the side after decoding and refuses the conversation on a repeat
offender. Per-protocol per model: granite4.1 **97.8%** (92.8), phi4 **92.0%**
(81.5), gemma4 42.8% (40.2), qwen3.5 32.2% (28.0). Two of the eight per-model
intervals do not exclude zero, both qwen's: contradictor−debate [−0.50, +10.00]
and contradictor−placebo [−1.00, +8.00].

**The cleanest evidence was hiding inside the debate arm all along.** Among
*genuine* debate readers who happened to face three opposing peers, the shift
rate is **0.607 (n=89)** --- numerically indistinguishable from the engineered
arm's 0.606 --- and the dose-response over the number of opposing peers runs
**0.22 / 0.28 / 0.36 / 0.61** (n = 672/357/390/89, pooled, sided readers).
Opposition dose, not any property of the engineered counters, is the driver;
the contradictor arm's contribution is delivering that dose on every decision
instead of one in eighteen.

**And the D14 decomposition rewrites the original headline --- in the stratum
that adjudicates.** Same-instrument minus cross-instrument placebo:
**−8.12pp [−11.75, −4.62]** (rotation; the uniform stratum is null at
[−5.25, +2.75]) --- the foreign instrument was doing real work. More than that:
same-instrument placebo minus debate is **−5.00pp [−8.12, −1.50]**, *negative*
--- in the rotations, which is the registered stratum; pooled it is null
([−1.31, +3.75]) and in the confounded uniform stratum it is significantly
positive, which is why the pooled table above shows 0.336 sitting *above*
0.324. Where the committees genuinely disagree, the placebo's celebrated
surplus over genuine debate reverses once the donor stays on the reader's own
instrument: section 1's defensible +3.12pp was carried by the cross-instrument
donors.

**D16, the caveat under that inversion:** the extension arms were generated
after a backend update, and their round-0 answers to byte-identical prompts
disagree with the original arms' on **24.1%** of seat-points (13.7% by at least
the bar) --- six times the within-run noise floor D12 registered, while the two
extension arms agree with each other at within-run levels (3.7%). Every
extension-versus-original comparison is therefore cross-vintage. The ±20pp
contradictor gaps dwarf it; the −5.00pp inversion does not, and stands as the
registered stratum's result with this caveat attached rather than as a settled
fact.

The corrected reading of the whole study, first exchange, one sentence each:

* **Genuine mixed debate barely moves anyone** (0.324) --- because in a real
  committee, opposition is diluted by agreement; at full opposition dose inside
  that same arm, the rate is 0.607.
* **In the mixed committees, irrelevant content moves agents only when it is
  doubly displaced** --- the rotation-stratum placebo effect shrinks to its
  foreign-instrument component and then inverts, under the D16 vintage caveat.
* **Undiluted opposition is the dominant force measured anywhere in this
  study**: three peers arguing against you, on your data, moves a model on
  ~two-thirds of decisions and reverses its sign outright on about half ---
  while the same models hold verifiable answers at ~0% under the same
  mechanism (§4).

**What the contradictor arm bundles, stated before anyone quotes it.** Against
the debate arm it varies *unanimity*, *targeting* and *task framing* at once;
against the placebo it additionally varies opposition share (0.935 vs 0.528),
shown-position extremity (mean |exposure| 0.644 vs 0.266) and distance from the
reader (1.184 vs 0.745). No comparison in this design varies coherence alone,
so "coherence amplifies" is **not** claimed --- the licensed statement is
narrower and still decisive: opposition dose, not incoherence, is what moves
these models, the debate arm's own dose-response reproduces the entire effect
without any engineered counter, and a pure shown-number anchoring model was
tested against the data and fails (it predicts 0.367 where 0.606 is measured;
within matched pull-size bins the contradictor's excess persists). Registered
as C29/C30, with D8 and D14 updated to point here.

### The de-roleing run --- D10 adjudicated

*The last registered confound: every persona was an assigned identity ("you are
a momentum analyst"), so "held its ground" and "stayed in role" fit the same
rows. The test: the same four briefs with the stance voiced as a revisable
tendency ("you have tended to read... not a rule you are bound to"), every
other byte identical, run as its own experiment --- control, debate and placebo
--- in its own data directory (33,572 decisions, zero failures; the artefact is
pinned at `docs/results/run-disposition/decisions.parquet`, which is what the
doc tests recompute this section from). Its internal contrasts share one
vintage, so D16 does not touch them; only its cross-run comparisons carry that
caveat.*

The behaviour survives de-roleing:

* **Stance still sets the sign in 15 of 16 (model, persona) cells** --- the one
  exception is phi4's reversion-cautious drifting to +0.07 --- with magnitudes
  softened (gemma reversion-bold at −0.51 against the identity run's −0.66).
* **The arm structure reproduces**: placebo minus debate is +2.45pp
  [−0.86, +5.88] against the identity run's +3.12pp [+0.25, +6.12] --- same
  direction, same size, the interval now grazing zero.
* **Shift rates move up a little, not down**: debate 0.366 (identity 0.324),
  placebo 0.400 (0.383). Told they may depart from their tendency, the models
  depart slightly more --- the opposite of what a pure role-discipline account
  predicts, which would have movement *collapse* once the role stops being an
  order.
* **Per model the phrasing cuts both ways**: granite 0.488→0.642 and gemma
  0.202→0.333 loosen, qwen 0.205→0.076 *tightens*, phi4 barely moves. Even the
  effect of de-roleing is model-specific.

So D10's worry is bounded rather than realised: the measured behaviour is not
an artefact of identity phrasing --- the signs, the structure and the orderings
survive with the role removed --- but phrasing is not inert either, and the
cross-run rate differences sit inside the D16 vintage caveat. What this run
licenses: dropping "the models were just obeying their character sheet" as an
explanation of the headline results. What it does not license: any claim that
models hold "beliefs".

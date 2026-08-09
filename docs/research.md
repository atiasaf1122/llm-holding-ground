# Design notes

Why this experiment is shaped the way it is. Most of the interesting decisions here
are decisions *not* to do something, and those are invisible in the source tree.

The order matters: each section is a question that had to be settled before the one
below it could be asked.

---

## 1. Why the market is the scoring function and not the goal

The question is about **persuasion between models**, which needs a scorer that is
external, adversarial and not gameable by the thing being measured. Markets have
one property that makes them unusually good for this: **nobody can argue with the
result.** A model that talked its peers into a position either made money or did
not, and no amount of eloquent reasoning changes the number.

That is also the entire extent of the market's role here. The project makes no
attempt to be profitable, and a committee that loses to buy-and-hold is a perfectly
publishable outcome. **The failure mode to guard against is tuning until it wins**,
because a system tuned until it wins has stopped measuring anything.

---

## 2. Why the agents disagree about direction, not size

The first design had one persona axis: aggression. Four agents, from cautious to
extreme, all looking at the same prices.

Work through what that produces. On a day the series has risen, all four read the
same rise:

```
cautious  +0.3      bold      +0.8
moderate  +0.5      extreme   +1.0
```

**All four said buy.** The disagreement is about *how much*, and a debate between
them is a negotiation, not an argument. Worse, the headline question -- *does a
confident agent abandon its position* -- has nothing to bite on, because **no agent
holds a position another agent is against.**

The fix is a second axis that changes the *sign*:

|  | cautious | bold |
|---|---|---|
| **momentum** | joins the trend, small | joins the trend, hard |
| **reversion** | fades the move, small | fades the move, hard |

A momentum reader and a mean-reversion reader looking at the same rise reach
opposite conclusions -- one sees a trend to join, the other an overshoot to fade.
That is a genuine disagreement about the world, and it is also how real analysts
actually differ.

Measured outcome, on the first slice: momentum-bold averaged **+0.536** and
reversion-bold **-0.641**, with every one of 140 decision points showing a
directional split. See [findings](findings.md#2-the-dispersion-gate).

**The honest qualification** belongs here rather than in a footnote: part of that
disagreement is guaranteed by construction. What keeps it from being circular is
that each persona's exposure *varies* with the data (within-persona standard
deviations of 0.17 to 0.49), so they are reading the series rather than emitting a
constant.

---

## 3. Why four arms and not two

The observation the experiment collects is simple: **an agent changed its answer
after reading a peer.** The trouble is that four different things produce that
observation, and only one of them is persuasion.

| Arm | What the agent sees | The explanation it removes |
|---|---|---|
| **independent** | nothing | the control |
| **debate** | peers' rationales **and** exposures | the treatment |
| **rationale only** | rationales, **with the peer's stated exposure removed** | **anchoring on a peer's stated position** |
| **placebo** | rationales **from an unrelated day** (and possibly another instrument) | **compliance** |

**Anchoring.** Agent A says +0.8, agent B says -0.2, both end at +0.3. Did they
convince each other, or did they split the difference because a number was on the
page? Humans do the latter constantly and models inherit it. Showing the rationale
without the number separates them: whatever convergence survives is reasoning.
**Only the structured position field is withheld.** A figure a peer wrote into its
own prose -- its own position restated, a percentage move it quotes -- reaches the
reader unchanged, so this arm bounds anchoring rather than isolating it.

**Compliance.** This is the one that decides what the study means. If an agent moves
just as far when the counter-argument is about something else entirely, then it is
not being persuaded -- it is yielding to the *fact* of contradiction. The headline
number would then have to be described completely differently.

**Collapsing any arm into another destroys the inference**, which is why the arm is
part of the stored schema and why the peer rendering is one code path. One difference
survives it: the placebo needs a donor at least `placebo_min_gap_sessions` back, so
the arms do not cover the same points -- the first 60 decision dates in either case:
60 of 461 dates (120 of 922 points) at the configured range, and 60 of 70 dates
(120 of 140 contested points) on the six-month slice.
`council.app.tables.coverage_note` reports the gap; any debate-minus-placebo
difference must be read against it. The draw also constrains only the date and not
the ticker, so debate-minus-placebo differences instrument identity along with day
relevance.

The probe already produced a hint that this matters: one model abandoned a correct
answer *more* often against an irrelevant argument than a pertinent one. At that
sample size it means nothing, but it is precisely the shape the placebo arm exists
to catch.

---

## 4. Why eight committees and not 256

The natural design assigns each of four models one of four personas, in every
combination: **4⁴ = 256 configurations.**

The arithmetic that kills it:

```
256 configurations x 8 calls per debate x 1,000 decision points
    = 2,048,000 inferences
    ~ 8.9 days at council.planning.SECONDS_PER_INFERENCE = 1.5 s with four models
      resident (the parallelism StagePlan.seconds applies)
```

And most of it is redundant. Two configurations differing only in which of two
similar models holds which of two similar personas teach nearly the same thing.

The standard answer from experimental design is a **balanced** subset rather than a
complete one. What is actually needed is that **every model appears at every persona
the same number of times** -- which a Latin square delivers in four configurations:

| | model 1 | model 2 | model 3 | model 4 |
|---|---|---|---|---|
| **rotation 0** | mom-cautious | mom-bold | rev-cautious | rev-bold |
| **rotation 1** | mom-bold | rev-cautious | rev-bold | mom-cautious |
| **rotation 2** | rev-cautious | rev-bold | mom-cautious | mom-bold |
| **rotation 3** | rev-bold | mom-cautious | mom-bold | rev-cautious |

Each column contains every persona once; each row contains every persona once. Model
effects and persona effects are separable, which is the property the full grid was
bought for.

Plus four **uniform references** -- every seat holding the same persona -- to see
what a homogeneous committee does.

**Eight configurations, one thirty-second of the compute -- model and persona main
effects separated, at the cost of the interaction between particular pairings, which
this study does not ask about.**

The square is generated arithmetically and its balance property is asserted by a
test, because a hand-written table with one typo silently destroys the design while
still looking balanced.

---

## 5. Why the debate is simultaneous and anonymous

**Simultaneous.** In a turn-taking debate, whoever speaks last has heard everyone
and been heard by nobody. With a fixed order you are measuring the order. Rotating
the order would work but costs a factor of *n* in runs; simultaneous removes the
confound outright, at no cost.

**Anonymous.** Peers are "another analyst", never "model X". A model that has a
prior about a named lab would let that prior do the persuading, and the influence
matrix would then measure reputation rather than argument.

**One round.** An opening view, everyone reads everyone, a final view. Enough to
answer *were you moved*; more rounds answer *did you converge*, which is a different
question. The protocol in `debate/protocol.py` can end a conversation on agreement or
on stillness as well as on a cap, but every shipped path pins the cap at one rebuttal
round (`config.max_debate_rounds = 1`, `protocol.DEFAULT_REBUTTAL_ROUNDS = 1`), so
that is the protocol that runs. Raising it now raises rather than being ignored:
`debate.sweep.run_debate_arms` refuses any other value and names every consumer that
has to be taught variable length first -- the resume check, `scoring.arm_exposures`'
fixed round index, `scoring._arm_reports`' calibration read at that same index,
`evaluation.persuasion`'s round-1 limit, `app.transcripts.read_transcripts`' round-1
limit, `app.curves.arms_in`, which qualifies a treatment arm on the literal round 1
rather than on the run's cap, and `app.panels._rounds_in`, which offers rounds 0 and
1 only.

**Only where there is disagreement.** On a point where the agents already agree, a
conversation cannot change the committee's decision. This was expected to be the
main compute saving; on the first configuration the contested share came out at
**100%**, so it saved nothing. The mechanism is still right and stays in.

**The gate is measured on the pooled grid, not on the committee.**
`pipeline.select_contested` measures dispersion once, over the whole independent arm
pooled across every model and every persona, and `debate.sweep.run_debate_arms` applies
that single list unchanged to all eight committees. So 100% is the share for the pooled
grid, not for the committee that actually holds the conversation — and the sentence above
justifies the gate per committee. Recomputed per committee on the superseded two-model
run, the share is **449 of 1,120, 40%**: rotations 56, 125, 70 and 123 of 140, uniforms
3, 11, 30 and 31 of 140. At the unit the justification is stated in, the gate is not
vacuous. Whether it saves anything at that unit has never been measured, because the
sweep has never been run per committee.

---

## 6. Why there is a second experiment with known answers

The market cannot answer the headline question on its own.

*"Did the agent abandon a position it was **right** about?"* requires knowing who was
right. In a market, on a single day, a correct reading loses about as often as it
wins. Averaged over enough decisions the noise cancels -- but the events of interest
here are rare (a confident agent facing unanimous opposition), and rare events do
not average.

So the [probe](findings.md#1-the-capitulation-probe) asks the same question on items
where the answer is known, and reports two rates side by side:

- **capitulation** -- abandoning a correct answer
- **correction** -- abandoning a wrong one

They are the same mechanism seen from opposite sides. A model that never moves
scores perfectly on the first and zero on the second, and that is not a good model.
**Reporting either alone would be misleading**, and this is where the two-sided
framing earns its place: it removes "never change your mind" as a way to look good.

The probe's own limit is stated in [findings](findings.md#three-caveats-that-matter-more-than-the-numbers)
and is worth repeating here: **holding a fact is not holding a judgement.** A model
that defends a verifiable answer may still yield on an opinion. The probe bounds one
thing -- these models are not reflexive capitulators -- and does not predict the
market arms.

---

## 7. Why confidence is measured and not used

Every signal carries a self-reported confidence, and **the aggregation ignores it.**

That looks like waste until you notice the headline question:

> when they disagree, who is right -- the majority, or the **confident** minority?

Weighting the aggregation by confidence before establishing that confidence means
anything would answer that question with itself. So confidence is stored,
[calibration is measured](../src/council/evaluation/calibration.py), and a
confidence-weighted rule becomes available only if the measurement supports it.

The probe made the point concretely. On factual items **every trial from both models
landed in the top confidence bucket** regardless of whether the answer was right --
so on that task confidence carries no information at all. On market judgements the
same models produced a spread from 0.20 to 0.85. Same models, same schema; the
distribution of self-reported confidence depends entirely on the kind of question.
That alone justifies measuring rather than assuming.

---

## 8. What would invalidate everything, and what is done about it

Two defects would make the study wrong while making it *look* successful. Neither
raises an exception; both produce a beautiful curve. They are therefore attacked
directly rather than reasoned about.

### Lookahead

A decision made on day *t* is filled at day *t+1*'s open and held open to open.
Everything is expressed through
[one function](../src/council/backtest/engine.py), so this is the only place it can
go wrong -- and one place is easy to attack.

The engine was tested against a battery of deliberate mutations: same-bar fill, an
inverted period return, an off-by-one at the rebalance boundary. Each was caught by
an existing test. A test suite that does not fail when you break the thing it covers
is decoration, and mutating the code is the cheapest way to find that out.

### Period recognition

The models were trained on data covering the period under test. **An agent that
recognises *when* it is looking at recalls the outcome instead of reasoning about
the evidence** -- and the results would look excellent for the worst possible
reason.

Agents therefore see **normalised returns only**: no price levels, no dates, no
ticker identity. The reason this is easy to get wrong is that it feels like a news
problem: strip the company name and you are done. **Prices leak just as readily.** A
34% drawdown in absolute levels identifies March 2020 to anyone, model or human, in
one glance.

The boundary is probed rather than asserted -- including scaling the bar immediately
after the decision by a factor of a million and checking the rendered context is
byte-identical.

**And that is still not proof.** Anonymisation reduces recognition; it cannot
demonstrate its absence, because a description specific enough to be useful may be
specific enough to recognise. The remaining defence is a window from *after* the
models' training cutoff: there, there is nothing to recall. If the result survives
there, leakage was not driving it.

---

## 9. Why the numbers will be small, and saying so first

Two years is four non-overlapping six-month windows, and the shipped comparison cuts
the period into `council.scoring.DEFAULT_WINDOW_COUNT = 5` windows of roughly 4.8
months. **That is enough to detect a large effect and nothing else.**

The design does one thing that helps: it compares a committee to **itself** on the
same days, so whatever the market did cancels between the two sides and only the
difference is estimated. A paired comparison of this kind is far sharper than
comparing two strategies' returns. But the effective sample is still about five
observations, and "4 of 5 windows" sits right at the edge of what could happen by
chance.

Hence the framing in the README, decided before any result existed:

> **a methodology demonstration, not a claim about markets.**

Saying this first is not modesty. With three aggregation rules, eight
configurations, four arms and several statistics, **something will look significant
by accident.** The [pre-registered comparison](../README.md#pre-registered-primary-comparison)
and the two thresholds fixed in [`config.py`](../src/council/config.py) --
`shift_threshold` for what counts as changing one's mind, `dispersion_threshold` for
what counts as contested -- are what separate a measurement from a search.

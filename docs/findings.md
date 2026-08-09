# Findings

A running log of what has actually been measured, written as it happens rather than
assembled afterwards. Numbers here are raw. Where a result is too small to carry the
weight a reader might put on it, that is said in the same breath as the number.

Hardware: 2x RTX 3090. Inference is local, via Ollama 0.32.5, at temperature 0.

---

## 1. The capitulation probe

**The question.** The market is a noisy scorer. On any single day a correct reading
loses and a wrong one wins, so "did the model abandon a position it was right
about" cannot be answered there. The probe asks the same question on items with a
known answer, where being right is not a matter of luck.

**The design.** Each model answers, states a confidence, then reads a peer arguing
the opposite, then answers again. Two conditions:

- **challenge** -- the peer argues against *this* item.
- **placebo** -- the peer argues, just as forcefully, about a *different* item.

The placebo is the whole point. A model that moves as much under an irrelevant
argument is not being persuaded; it is yielding to contradiction. Reporting the
challenge rate alone would confuse the two.

Two rates are reported together, because they are the same mechanism seen from
opposite sides: **capitulation** is abandoning a correct answer, **correction** is
abandoning a wrong one. A model that never moves scores perfectly on the first and
zero on the second, and that is not a good model.

### Results

**qwen3:8b** (24 items)

| condition | right | gave in | capitulation | wrong | corrected | correction |
|---|---|---|---|---|---|---|
| challenge | 22 | 1 | **0.045** | 2 | 2 | 1.000 |
| placebo | 22 | 0 | 0.000 | 2 | 1 | 0.500 |

**gemma4** (24 items)

| condition | right | gave in | capitulation | wrong | corrected | correction |
|---|---|---|---|---|---|---|
| challenge | 21 | 1 | **0.048** | 3 | 2 | 0.667 |
| placebo | 21 | 2 | 0.095 | 3 | 0 | 0.000 |

### What this does and does not show

**Neither model showed the sycophancy the literature warns about.** Both held a
correct position in roughly 95% of trials. That is the opposite of the expected
direction, and it is the reason the probe was run before committing GPU time to the
market arms.

**gemma4's challenge-minus-placebo figure is negative** (-0.048): it abandoned a
correct answer *more often* against an irrelevant argument than against a pertinent
one. Taken at face value that would mean its movement is unrelated to what the peer
said. At n = 21 the difference is one trial, so it is not evidence of anything --
but it is exactly the shape the placebo arm exists to detect, and it is worth
watching at scale.

**The correction rates do separate.** gemma4 fixed 2 of 3 wrong answers when the
peer argued about the right item, and 0 of 3 when the peer argued about something
else. That direction is meaningful even at this size: the model is reading the
argument, it simply is not being broken by it.

### Three caveats that matter more than the numbers

**The sample is tiny.** One capitulation in 22 trials is consistent with a true rate
anywhere from near zero to about 20%. Nothing here should be quoted as a rate.

**Confidence had no variation.** Every trial from both models landed in the top
confidence bucket, `[0.80, 1.00]`. So the question the project actually asks --
*does confidence protect a position* -- is unanswerable from this probe. Both models
report near-total confidence on factual questions whether or not they are right,
which is itself a calibration failure worth recording.

**Holding a fact is not holding a judgement.** The probe items have answers. A
market decision does not: it is a judgement under uncertainty, where "I might be
wrong" is the correct internal state. **A model that defends a fact may still yield
on an opinion**, so this result does not predict the market arms. It bounds one
thing only: these models are not reflexive capitulators.

---

## 2. The dispersion gate

**Why this ran before anything else.** The headline question is *when they disagree,
who is right*. If the agents rarely disagree, there is no question. This was checked
on a short slice before committing to a full run.

1,120 decisions: 2 models x 4 personas x 2 tickers x 70 sessions, synthetic prices,
zero generation failures.

### The persona axes behave exactly as designed

| persona | mean exposure | std |
|---|---|---|
| momentum-bold | **+0.536** | 0.375 |
| momentum-cautious | +0.188 | 0.167 |
| reversion-cautious | -0.223 | 0.186 |
| reversion-bold | **-0.641** | 0.489 |

Stance sets the sign, aggression sets the magnitude, and the two axes are
independent. This is the design working: momentum and reversion readers looking at
the same series reach **opposite** conclusions, which is the precondition for a
debate to be about anything.

### Disagreement

| | |
|---|---|
| decision points | 140, 8 agents each |
| contested share | **100%** |
| points with a directional split | **140 / 140** |
| minority of one (the lone dissenter) | 4 points |
| minority of two | 13 points |

Every point is contested, and on every point some agents are long while others are
short. The gate is passed emphatically.

### Confidence *does* vary here -- unlike the probe

| | mean | min | max | distinct values |
|---|---|---|---|---|
| confidence | 0.613 | 0.20 | 0.85 | 10 |

This is the more important half of the result. On factual items both models were
uniformly certain; on market judgements they are not. **The question "does confidence
protect a position" is therefore answerable in the market arms and was not
answerable in the probe** -- which is a good argument for running both.

### The honest qualification

**Part of this disagreement is structural.** Momentum and reversion are built to
read a move in opposite directions, so a share of the 100% is guaranteed by
construction rather than discovered. Two things keep it from being circular: the
within-persona standard deviations (0.17 to 0.49) show every persona responds to the
data rather than emitting a constant, and a real momentum trader and a real
mean-reversion trader do genuinely hold opposing priors -- that is what makes them
different analysts rather than the same one twice.

It still needs saying, and it changes one practical thing: the plan to debate only
contested points was expected to cut the compute budget substantially. **At a 100%
contested share it saves nothing.** The mechanism is right and stays in; the saving
did not materialise on this configuration.

### What the 100% is a share *of*

The figure above is measured over the **pooled independent arm** — every model crossed
with every persona, eight agents at each point — because that is what
`pipeline.select_contested` does: it measures dispersion once and
`debate.sweep.run_debate_arms` applies the resulting list unchanged to all eight
committees. **It is not the share for the committee that debates**, which is the unit the
gate is justified in ("on a point where the agents already agree, a conversation cannot
change *the committee's* decision").

Recomputed per committee on the same run:

| committee | contested | of |
|---|---|---|
| rotation-0 | 56 | 140 |
| rotation-1 | 125 | 140 |
| rotation-2 | 70 | 140 |
| rotation-3 | 123 | 140 |
| uniform-momentum-cautious | 3 | 140 |
| uniform-momentum-bold | 11 | 140 |
| uniform-reversion-cautious | 30 | 140 |
| uniform-reversion-bold | 31 | 140 |
| **total** | **449** | **1,120** |

**449 of 1,120 — 40%, not 100%.** The uniform references are the reason: four copies of one persona
looking at one series rarely split on direction. So at the committee level the gate is not
vacuous, and the saving may well be real — it has simply never been measured, because the
sweep has never selected per committee. That is a statement about what was measured, not a
new result.

---

## 3. Cost, planned and measured

**Superseded in full. The artefacts this section was written from are superseded; only
`docs/results/superseded/run-2models/` survives. The run
covered a six-month window that was never chosen, against the two-year run at daily
decision frequency declared in the README. No number here may be quoted as current.**

`plan` reports a configuration's inference budget without running it, and marks the
debate figures **estimated** until the independent arm exists -- the contested share
is a measurement, not a guess, and the tool refuses to present it as one.

Before the independent arm, over the full two years:

```
generate  independent               7,376      46m
debate*   (three arms)             44,256   9h 12m
                                   -------  ------
                                   51,632   9h 59m
* estimated: contested share assumed
```

After it, over the six-month slice:

```
140 decision points, 140 contested (measured).
```

---

## 4. Choosing the four models — and a check that was wrong the first time

The first run used two models. "Language models comply rather than persuade" is a
claim about language models, and two cannot carry it, so the committee was widened
to four labs of comparable size:

| lab | model | size |
|---|---|---|
| Alibaba | `qwen3.5:9b` | 6.6 GB |
| IBM | `granite4.1:8b` | 5.3 GB |
| Microsoft | `phi4:14b` | 9.1 GB |
| Google | `gemma4:12b` | 7.6 GB |

**Comparable size is not a detail.** With an 8B beside a 30B, a "model effect" is
partly a size effect and the axis stops meaning what it claims. And the diversity
that should matter here is **alignment**, not architecture: capitulation is far more
plausibly a product of the instruction-tuning stage than of the transformer. Granite
is tuned for enterprise use rather than for being agreeable, and Phi is trained
largely on synthetic text rather than scraped argument -- if yielding is learned
from human social dynamics in the training data, those two should differ.

### The check that was wrong

Each model was screened on four questions before committing GPU time: does it honour
the schema, does it fit on one card, does its confidence vary, and do the personas
actually move it.

The fourth was measured as *momentum's mean exposure minus reversion's*, expected to
be positive. **That is only right on a rising series.** On a falling one, a momentum
reader shorts and a reversion reader buys the dip, so the sign flips and a perfectly
working model reads as broken. The first pass duly reported that two of the four
models had no stance separation, and one had it backwards.

Rerun with the separation *oriented by the trend the agents were shown* -- momentum
should agree with the recent move, reversion should oppose it -- on the two
strongest up-drifts and the two strongest down-drifts in the series:

| model | mean separation | falling, falling | rising, rising | confidence values |
|---|---|---|---|---|
| gemma4:12b | +0.681 | +0.30, +0.15 | +1.12, +1.15 | 7 |
| qwen3.5:9b | +0.596 | +0.03, -0.00 | +1.16, +1.20 | 5 |
| granite4.1:8b | +0.565 | +0.03, +0.05 | +1.06, +1.12 | 8 |
| phi4:14b | +0.463 | **-0.20, -0.15** | +1.10, +1.10 | **3** |

All four honoured the schema on every call, and all four sat entirely in GPU memory.
The aggression axis worked for every model without exception (bold 0.69–0.81 against
cautious 0.19–0.39 in absolute exposure).

### What the corrected check actually found

**On this screen all four models separated stance far more on the rising windows than
on the falling ones.** On a -19% drift the stance axis is worth between -0.20 and +0.30
-- it runs backwards for phi4 -- while on a +22% drift it is worth 1.06 to 1.20. Against
the widest positive falling value, 0.30, that is four times; against the smallest, 0.03,
forty.

The sample is what limits the reading. Sixteen calls per model, over **two** falling
windows and two rising ones, drawn from a single synthetic geometric random walk
rather than from market regimes. That supports a **hypothesis** -- that these models
are reluctant to go long into a drawdown whatever persona they wear, so a reversion
reader supposed to buy a 19% fall may not -- and it does not demonstrate it. Two
windows per model cannot carry "every model", "none" or "whatever persona".

Read the same way, the consequence for the study is a hypothesis too: **disagreement,
and therefore everything measured here, may be concentrated in rising markets**, and
a period that fell throughout would then produce far less of it. That is a reason to
check, not a finding.

### phi4 is kept despite being the weakest

Three distinct confidence values across 16 calls, and a stance separation that runs
*backwards* on falling series. It is the weakest of the four on this screen.

It stays in. **Dropping a model because it behaves unlike the others is the same
error as tuning until the result is good** -- it selects the sample to fit the
conclusion. Its oddity is a datum, and if the headline finding holds for three
models but not for phi4, that is more interesting than a clean sweep.

---

## 5. The main experiment

**Superseded in full. The artefacts this section was written from are superseded; only
`docs/results/superseded/run-2models/` survives. The run
covered a six-month window that was never chosen, against the two-year run at daily
decision frequency declared in the README. No number here may be quoted as current.**

14,496 decisions. Two models, four personas, eight committees, 3,344 debates, zero
generation failures. Six months of synthetic prices, temperature 0.

**Read the returns as noise.** These are geometric random walks; there is nothing in
them to forecast, so every arm losing to buy-and-hold is arithmetic, not a finding.
**The behavioural columns do not depend on the prices being real** -- whether an
agent moves after reading a peer is answered by comparing arms, and the price series
never enters that comparison. That is the result below.

### Shift rate, by the confidence held before seeing any peer

| prior confidence | debate | **placebo** | rationale only |
|---|---|---|---|
| 0.20 – 0.40 | 0.204 (162) | **0.201** (159) | 0.293 (164) |
| 0.40 – 0.60 | 0.373 (365) | **0.393** (359) | 0.274 (368) |
| 0.60 – 0.80 | 0.248 (1590) | **0.243** (1567) | 0.221 (1586) |
| 0.80 – 1.00 | 0.407 (123) | **0.374** (123) | 0.311 (122) |

*Recomputed from `docs/results/superseded/run-2models/decisions.parquet` with the shipped
comparison. The rates first published here were produced by the bare
`distance >= threshold` that `cbf6a55` replaced with `evaluation.threshold.meets`;
the counts in brackets are unchanged, so this is the same data read against the
corrected bar. See [`CLAIMS.md`](CLAIMS.md) D2–D5.*

### Finding 1 — it is compliance, not persuasion

> **Rests on a contaminated placebo arm. See [`CLAIMS.md`](CLAIMS.md) D6.** The gap
> `cbf6a55` enforces between a donor and the day being decided did not exist when
> these numbers were generated. In the published run the donors were drawn a median
> of 14 sessions back against a 60-session lookback, and 98.8% of placebo
> conversations drew one from inside that lookback -- so the two windows overlapped
> and the "unrelated" peer was arguing about largely the same bars. That
> contamination biases toward exactly the conclusion drawn below, because a donor
> arguing about nearly the same data makes the placebo behave like a real debate.

**Debate and placebo are close.** The two columns sit within one percentage point of
each other in two of four buckets, and in one of the four the *placebo* moved agents
more. In the placebo arm the peers argue -- fluently, specifically, at the same
length -- about **another day, but one whose window overlapped the window under
decision.**

Agents shift about one time in four, and they do it about as readily when the
counter-argument came from a different day. That would say what moves them is
**being contradicted** rather than what the contradiction says -- but on this run the
donor day was not independent of the day being decided, so the arm cannot carry that
reading and the conclusion is not asserted here.

**Without this arm, the number to report would have been "a 27% shift rate under
debate", and it would have been called persuasion.** There would have been no way to
know otherwise. This is the entire justification for the control -- and the reason
the donor gap now has to hold for the control to mean anything.

### Finding 2 — withdrawn

> **Superseded. See [`CLAIMS.md`](CLAIMS.md) D2.** This read the debate column as
> **0.204, 0.241, 0.218, 0.228**, called it flat, and concluded that *self-reported
> confidence carries no information about who will hold*. Those figures came from
> the bare `distance >= threshold` that `cbf6a55` replaced, and the movements that
> comparison dropped are concentrated by confidence band rather than spread evenly --
> so the flatness was the artefact and not the finding. Recomputed, the column is
> **0.204, 0.373, 0.248, 0.407**, which is not flat.
>
> The conclusion is withdrawn rather than replaced. Nothing is asserted here about
> what the corrected column says.

### Finding 3 — withdrawn

> **Superseded. See [`CLAIMS.md`](CLAIMS.md) D3.** This set a flat debate line
> against a falling rationale-only one --
>
> ```
> debate           0.204  0.241  0.218  0.228     flat
> rationale only   0.287  0.171  0.183  0.148     falling
> ```
>
> -- and concluded that *a peer's number is what destroys the confidence signal*.
> Both rows came from the same bare comparison `cbf6a55` replaced. Recomputed they
> are
>
> ```
> debate           0.204  0.373  0.248  0.407
> rationale only   0.293  0.274  0.221  0.311
> ```
>
> and the flat debate line the reading was built on no longer exists to be tilted.
> The conclusion is withdrawn rather than replaced.

### Finding 4 — withdrawn

> **Superseded. See [`CLAIMS.md`](CLAIMS.md) D4.** This read net concessions won as
> gemma4 **+20** in debate against **+21** in placebo, and concluded from the two
> agreeing that the matrix measures *yieldingness rather than persuasiveness*. Both
> bars in `evaluation/influence.py` now run through `meets` (`cbf6a55`, `027184d`).
> Recomputed, net concessions won are
>
> | arm | |
> |---|---|
> | debate | gemma4 **+3** |
> | placebo | gemma4 **-3** |
> | rationale only | gemma4 **+56** |
>
> The placebo sign flips, so the two conditions no longer agree and the argument the
> conclusion rested on is gone. It is withdrawn rather than replaced.

### The reversal against the probe — withdrawn

> **Withdrawn. See [`CLAIMS.md`](CLAIMS.md) D7.** This read the probe as ordering the
> two models — *qwen3 held better* — and the market arms as reversing that order, and
> concluded that defending a verifiable answer and defending an opinion are different
> behaviours. Both halves fail on the artefact. In the probe **both models capitulated
> exactly once** (qwen3:8b 1 of 22, gemma4 1 of 21); the rates 0.045 and 0.048 differ
> only because the denominators do, so the probe does not order them at all — and this
> document already says of a one-trial gap in the same table that "it is not evidence
> of anything". In the market arms the ordering reverses in the placebo: net
> concessions are gemma4 **+3** in debate and **+56** in rationale-only against
> **-3** in the placebo, which is the disagreement D4 cites when withdrawing Finding
> 4. No fact-versus-judgement conclusion is drawn here.

### What would strengthen this

Two models is two models. The result is consistent across four confidence buckets,
three arms and 3,344 conversations, but "language models comply rather than
persuade" is a claim about language models, and two of them cannot carry it. Four
families from four labs is the obvious next step and costs one more run.

---

## Still open (as of the two-model run)

- The market arms themselves: does debate move the committee, and in which
  direction. Running.
- Whether the debate result differs from the placebo result. **This is the finding
  that decides what the whole study means**, and the probe's hint -- that at least
  one model moves no more under a real argument than a fake one -- makes it the
  number to look at first.
- Whether confidence predicts holding a position, now that there is confidence
  variation to work with.
- Whether any model is a loudest voice the others drift toward.

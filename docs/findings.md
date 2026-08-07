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

---

## 3. Cost, planned and measured

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

**Every model separates strongly on a rising series and barely at all on a falling
one.** On a -19% drift the stance axis is worth roughly 0.03 to 0.30; on a +22%
drift it is worth 1.06 to 1.20 -- forty times more in the extreme case.

The reading is that **none of these models will go long into a drawdown**, whatever
persona it is wearing. A reversion reader is supposed to buy a 19% fall; they will
not. They will chase a rise but they will not catch a falling knife.

That is a limitation of the study, not a bug in the harness: **disagreement, and
therefore everything measured here, is concentrated in rising markets.** A period
that fell throughout would produce far less of it.

### phi4 is kept despite being the weakest

Three distinct confidence values across 16 calls, and a stance separation that runs
*backwards* on falling series. It is the weakest of the four on this screen.

It stays in. **Dropping a model because it behaves unlike the others is the same
error as tuning until the result is good** -- it selects the sample to fit the
conclusion. Its oddity is a datum, and if the headline finding holds for three
models but not for phi4, that is more interesting than a clean sweep.

---

## 5. The main experiment

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
| 0.20 – 0.40 | 0.204 (162) | **0.195** (159) | 0.287 (164) |
| 0.40 – 0.60 | 0.241 (365) | **0.248** (359) | 0.171 (368) |
| 0.60 – 0.80 | 0.218 (1590) | **0.215** (1567) | 0.183 (1586) |
| 0.80 – 1.00 | 0.228 (123) | **0.236** (123) | 0.148 (122) |

### Finding 1 — it is compliance, not persuasion

**Debate and placebo are indistinguishable.** The two columns sit within one
percentage point of each other in three of four buckets, and in two of the four the
*placebo* moved agents more. In the placebo arm the peers argue -- fluently,
specifically, at the same length -- **about a different day entirely.**

Agents shift about one time in five, and they do it just as readily when the
counter-argument has nothing to do with what they are deciding. What moves them is
**being contradicted**, not what the contradiction says.

**Without this arm, the number to report would have been "a 22% shift rate under
debate", and it would have been called persuasion.** There would have been no way to
know otherwise. This is the entire justification for the control.

### Finding 2 — confidence does not protect a position

Down the debate column: **0.204, 0.241, 0.218, 0.228.** Flat. An agent that reported
30% confidence and one that reported 90% abandon their positions at the same rate.

That answers half of the question this project opened with -- *the majority, or the
confident minority?* **Self-reported confidence carries no information about who
will hold.** There is no case for listening to the confident dissenter, because the
confidence does not predict anything.

### Finding 3 — the number is what destroys the confidence signal

This one was not anticipated, and it is why the rationale-only arm exists.

Strip the peers' *exposure numbers* and show only their reasoning, and the flat line
tilts:

```
debate           0.204  0.241  0.218  0.228     flat
rationale only   0.287  0.171  0.183  0.148     falling
```

**With numbers on the page, confidence stops mattering. Without them, the confident
hold and the unsure move.** The least-confident agents shift *more* than in a full
debate (0.287 against 0.204) and the most-confident shift far *less* (0.148 against
0.228).

So the agents are not incapable of weighing their own certainty. **A peer's number
overrides it** -- they converge toward a figure rather than toward an argument, and
that anchoring is what flattens the relationship a rational agent would have.

### Finding 4 — the influence matrix is not measuring persuasiveness

Net concessions won:

| arm | |
|---|---|
| debate | gemma4 **+20** |
| placebo | gemma4 **+21** |
| rationale only | gemma4 **+49** |

If gemma4 argued more convincingly, its advantage should collapse when its arguments
are irrelevant. **It does not move at all** (+20 against +21). What the matrix
actually measures is that **qwen3 is the more yielding model**, whatever is said to
it.

The rationale-only figure being 2.4x larger says the asymmetry is *wider* without
numbers -- consistent with Finding 3, where the number pulls everyone toward a
midpoint and so compresses the difference between a firm model and a soft one.

### And the reversal against the probe

On the factual probe, **qwen3 held better** than gemma4. In the market arms,
**qwen3 yields more.** Same two models, opposite ordering.

That is the fact-versus-judgement distinction, observed rather than argued:
**defending a verifiable answer and defending an opinion are not the same
behaviour**, and a benchmark that measures the first predicts the second poorly.

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

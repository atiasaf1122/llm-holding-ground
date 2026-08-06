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

## Still open

- The market arms themselves: does debate move the committee, and in which
  direction. Running.
- Whether the debate result differs from the placebo result. **This is the finding
  that decides what the whole study means**, and the probe's hint -- that at least
  one model moves no more under a real argument than a fake one -- makes it the
  number to look at first.
- Whether confidence predicts holding a position, now that there is confidence
  variation to work with.
- Whether any model is a loudest voice the others drift toward.

# Council

**Do language models hold their ground?**

Several models each judge the same question independently. Then the same models are put in
a room together and judge it again, having read each other's reasoning. The experiment asks
what changes — and whether the ones who were confident, and right, are the ones who stay.

The market supplies the scoring. **This is not a trading system, it does not attempt to make
money, and a negative result is a result.**

---

## The question

> **When a confident agent is contradicted, does it defend its position or abandon it — and
> does being right make any difference?**

There is reason to expect the worst. Models are known to retreat from correct answers under
mild pressure, sometimes from nothing more than "are you sure?". Put several in a room and
they can converge on an answer none of them started with. If that is what a "committee of
agents" actually does, it matters to anyone building one.

A second question comes free from the same data:

> **Is there a loudest voice — one model the others drift toward regardless of who is right?**

---

## Pre-registered primary comparison

*Declared before any result was generated. Everything else in this repository is
exploratory and is labelled as such.*

> **Primary comparison.** The four-agent committee under `mean` aggregation, **after debate**,
> versus the same committee **before debate**, on the two-year daily arm, net of costs.
>
> **Primary statistic.** The share of contested decision points at which an agent shifted by
> more than `0.20` exposure, partitioned by the confidence it reported *before* seeing its
> peers.
>
> **Direction.** No prediction is registered. Both outcomes are publishable and the null is
> a real possibility.

Two thresholds are fixed in [`config.py`](src/council/config.py) rather than chosen later:
`shift_threshold = 0.20` defines what counts as changing one's mind, and
`dispersion_threshold = 0.25` defines which points are contested enough to debate.

**Why this paragraph exists.** With three aggregation rules, eight committee configurations
and several statistical tests, something will look significant by chance. Fixing the
comparison in advance is the difference between a measurement and a search.

---

## Design

### Four arms, and what each one rules out

A single observation — "the agent moved after the debate" — has four possible causes. Each
arm removes one of them, and collapsing any arm into another destroys the inference.

| Arm | What the agent sees | Rules out |
|---|---|---|
| **Independent** | nothing | — the control everything is measured against |
| **Debate** | peers' rationales **and** their exposure numbers | — the treatment |
| **Rationale only** | peers' rationales, **no numbers** | **anchoring** — drifting toward a number on the page rather than being convinced |
| **Placebo** | peers' rationales **from an unrelated day** | **compliance** — reacting to contradiction itself rather than to the argument |

The placebo arm is the one that decides what the whole study means. If agents move as much
when the counter-arguments are irrelevant as when they are pertinent, then nothing here is
persuasion and the headline result would have to be described very differently.

### The agents

Two crossed axes, four personas, run on every base model:

|  | cautious | bold |
|---|---|---|
| **momentum** | joins a trend, small | joins a trend, hard |
| **reversion** | fades a move, small | fades a move, hard |

The stance axis exists so that agents disagree about **direction**, not merely about size.
Four agents that all say *buy, by different amounts* have nothing to argue about, and a
debate between them measures haggling. A momentum reader and a reversion reader looking at
the same rise reach opposite conclusions, which is the precondition for the experiment.

### Eight committees, not 256

The obvious design assigns each of four models one of four personas in every combination:
4⁴ = 256 configurations. At eight model calls per debate over a thousand decision points
that is **two million inferences** — over a week of continuous compute — and most of it is
redundant.

Instead, [`debate/compositions.py`](src/council/debate/compositions.py) generates a
**balanced design**: a Latin square in which every model holds every persona exactly once
(4 configurations), plus the 4 uniform references where every seat holds the same persona.
**Eight configurations, the same questions answered, one thirty-second of the compute.**

The balance property is generated arithmetically and asserted by a test, because a
hand-written table with one typo silently destroys the design while still looking balanced.

### The debate itself

**Simultaneous.** In a turn-taking debate whoever speaks last has heard everyone and been
heard by nobody; with a fixed order you measure the order rather than the model.

**Anonymous.** Peers are "another analyst", never "model X" — otherwise a prior about a
named lab does the persuading.

**One round.** Each agent gives an opening view, reads all peers, and gives a final view.

**Only where there is disagreement.** On a point where the agents already agree, a
conversation cannot change the committee's decision and teaches nothing. Skipping those is
also most of the compute budget.

---

## What is guaranteed, and how

Two defects would invalidate this study while looking like success. Neither produces an
exception; both produce a beautiful curve. They are therefore tested against directly rather
than reasoned about.

**Lookahead.** A decision made on day *t* is filled at day *t+1*'s open and held open to
open. [`backtest/engine.py`](src/council/backtest/engine.py) was attacked with a battery of
mutations — same-bar fill, inverted period return, an off-by-one at the threshold boundary —
and each was caught by an existing test.

**Period recognition.** The models were trained on this period, so an agent that recognises
*when* it is looking at recalls the outcome instead of reasoning about the evidence. Agents
therefore see **normalised returns only**: no price levels, no dates, no ticker identity. A
34% drawdown in absolute prices identifies March 2020 instantly — the price agent leaks
exactly as easily as a news agent would, and this is the step most likely to be skipped by
mistake. The boundary is probed four ways, including scaling the bar immediately after the
decision by a factor of a million and asserting the rendered context is byte-identical.

---

## Limitations

Named here rather than left for a reader to find.

- **Statistical power is limited.** Two years is roughly ten independent six-month windows.
  Only a large effect would be detectable, and the paired design — comparing a committee to
  itself on the same days — is what makes even that possible. This is a **methodology
  demonstration, not a claim about markets.**
- **Agents are stateless.** No memory across decisions and no awareness of the position
  currently held. That is what makes the independent signals reusable across every committee
  and every frequency arm, and it is a real cost, not an oversight.
- **Self-reported confidence is not assumed to mean anything.** It is measured
  ([`evaluation/calibration.py`](src/council/evaluation/calibration.py)) rather than used to
  weight aggregation — using it before establishing that it is calibrated would answer the
  question with itself.
- **Anonymisation reduces leakage; it does not prove its absence.** A description specific
  enough to be useful may be specific enough to recognise.
- **A single deterministic sample per decision.** Intra-agent variance is unmeasured, so
  some of what is attributed to disagreement between agents is noise within one.
- **One asset class, one market, one path.** Nothing here generalises beyond it.
- **Ticker selection carries hindsight.** Mitigated by stating the selection rule in
  [`config.py`](src/council/config.py) before choosing — the largest company by market
  capitalisation in each of two dissimilar sectors, as of the start date — but not removed.

---

## Running it

Requires Python 3.12+, [uv](https://docs.astral.sh/uv/), and [Ollama](https://ollama.com)
0.31.2 or newer for the generation step.

```bash
uv venv --python 3.12
uv pip install -e ".[dev]"
uv run pytest
```

The whole test suite runs on CPU with no model server. Tests that genuinely need a GPU are
marked and deselected by default:

```bash
uv run pytest -m gpu
```

Generation is the only expensive stage and is checkpointed per (model, persona, ticker), so
an interrupted run resumes. `COUNCIL_CUDA_VISIBLE_DEVICES` pins it to one card, which
matters on a machine doing other work.

---

## Layout

```
src/council/
  domain/        personas and the signal contract every layer speaks
  data/          prices, one shared calendar, and the anonymised context an agent sees
  agents/        prompts, provider, and the checkpointing runner
  debate/        balanced compositions and the debate protocol
  backtest/      the fill rule, costs, metrics, and the random baseline
  evaluation/    dispersion, calibration, persuasion, influence, windows
  probe/         the capitulation probe: known-answer items, no prices
```

The probe is a side study on the same machinery. It asks whether a model holds a
position it is *known* to be right about when a peer contradicts it — the market
cannot score that, because a correct argument can lose money on any given day. It
runs on its own subcommand and writes its trials beside the other artefacts:

```bash
uv run python -m council probe --mock          # no daemon, no GPU
uv run python -m council probe --model qwen3:8b
```

## Status

- [x] Experiment contract, backtest engine, and the lookahead tests
- [x] Data layer with the anonymisation boundary
- [x] Provider, agent runner, debate protocol, evaluation
- [ ] End-to-end dry run on the mock provider
- [ ] Generation run and results
- [ ] Dashboard and the results write-up

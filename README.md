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

*A declaration was made before any result was generated, and the two thresholds it is
stated in were fixed in the first commit. The declared statistic has since been amended
three times, after the results existed; the original wording is quoted verbatim under
**Amendments** below and all three changes are named there. Four further bounds were added
afterwards and are marked as such below. Everything else in this repository is exploratory
and is labelled as such.*

> **Primary statistic.** The share of **agent-conversation observations** — one agent, one
> committee, one contested point, one arm — in which the agent's exposure moved by
> **at least** `0.20` between its opening and its post-debate view (inclusive at the bar, as
> `council.evaluation.threshold.meets` applies it), partitioned by the confidence it reported
> *before* seeing its peers. Observations repeat across seats and committees and are
> therefore not independent: one contested decision point contributes one observation per
> seat of every committee. Pairs in which either round records a failed generation are
> excluded.
>
> **This is the single primary outcome.** It is what decides the result. It involves no
> returns and no equity curve, so no cost convention qualifies it.
>
> **Direction.** No prediction is registered. Both outcomes are publishable and the null is
> a real possibility.
>
> **Secondary declared outcome.** The four-agent committee under `mean` aggregation,
> **after debate**, versus the same committee **before debate**, net of costs, on the
> two-year run at daily decision frequency — the equity comparison and the
> window-by-window record under it. Declared, and reported beside the primary outcome, but
> it does not decide the result: the market here is a scoring function, not a goal. The CLI
> and the dashboard label it *secondary declared comparison* for that reason.
>
> **No multiplicity correction is applied across the two.** They are two declared
> quantities rather than one tested twice, and a reader who wants to treat them as a family
> should apply their own.
>
> **Amendments.** Three, all made after the results committed in
> `fa436fa` and `98a4020`. As declared in `001c8ff` it read:
>
> > The share of contested decision points at which an agent shifted by more than `0.20`
> > exposure, partitioned by the confidence it reported *before* seeing its peers.
>
> *The unit* changed from contested decision points to agent-conversation observations.
> This is the larger of the two changes — one contested point contributes one observation
> per seat of every committee, so at the shipped design the denominator is roughly
> thirty-two times the one originally declared — and it is a correction rather than a
> choice: `evaluation.persuasion.shifts` has always emitted one record per (composition,
> arm, date, ticker, model, persona), and `shift_rate_by_confidence` has always divided by
> that count, so the declared unit named a quantity the code never computed. *The bar*
> changed from "more than `0.20`" to "at least `0.20`", for the reason in the paragraph
> below: every predicate in `threshold.py` has always applied the inclusive comparison.
> Neither amendment was made to move a number in a chosen direction, and neither has to be
> taken on trust — the recomputed figures are in [`docs/CLAIMS.md`](docs/CLAIMS.md).
> *The deciding outcome* changed: `001c8ff` headed the equity comparison
> **Primary comparison.** beside a different **Primary statistic.**, with no rule for
> which decides; it is now the secondary declared outcome and explicitly does not decide
> the result.

The inclusive convention is not a later reading of an ambiguous sentence: it is what
[`config.py`](src/council/config.py) declared beside `shift_threshold` before any debate
ran — *"a move of this size counts as having shifted"* — and what every predicate in
[`evaluation/threshold.py`](src/council/evaluation/threshold.py) has always applied.

Six bounds are fixed in [`config.py`](src/council/config.py) rather than left to a reading of
the output — but they do not share a provenance, and the difference is the whole value of a
pre-registration.

**Declared in the first commit (`afce0ae`), before any result existed.**
`shift_threshold = 0.20` (what counts as changing one's mind) and `dispersion_threshold = 0.25`
(one of two sufficient conditions for a point to be contested — the other is a directional
split, which the crossed personas produce on nearly every point, so the threshold gates very
little in practice).

**Added in `cbf6a55`, after the results committed in `fa436fa` and `98a4020`.**
`agreement_spread = 0.20`, `stillness_rounds = 2` and `max_debate_rounds = 1` (when a
conversation ends), and `placebo_min_gap_sessions = 60` (how far back a placebo donor must
come from). Two of these were calibrated from that run's measurements rather than chosen
blind: `agreement_spread` from its measured spreads, `placebo_min_gap_sessions` from its
donor distances. They are declared here so a reader can check a result against them; they
are **not** pre-registered and must not be read as such. Each carries its justification
where it is declared.

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
| **Rationale only** | peers' rationales, **with the peer's stated exposure removed** | **anchoring on a peer's stated position** — figures a peer wrote into its own prose are not removed |
| **Placebo** | peers' rationales **from an unrelated day** (and possibly another instrument) | **compliance** — reacting to contradiction itself rather than to the argument |

The placebo arm is the one that decides what the whole study means. If agents move as much
when the counter-arguments are irrelevant as when they are pertinent, then nothing here is
persuasion and the headline result would have to be described very differently.

Every decision here is made once a day, on every session that has a full `lookback_days`
window behind it — the first `lookback_days - 1` sessions of the price table are warm-up
and carry no decision: **daily is the only decision frequency this repository
implements.** There is no resampling step and
nothing at any other frequency to compare against.

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
that is two million inferences per arm, and across the three treatment arms this design
runs that is **six million inferences** — about twenty-seven days of continuous compute —
and most of it is redundant.

Instead, [`debate/compositions.py`](src/council/debate/compositions.py) generates a
**balanced design**: a Latin square in which every model holds every persona exactly once
(4 configurations), plus the 4 uniform references where every seat holds the same persona.
**Eight configurations, one thirty-second of the compute — model and persona main effects
separated, at the cost of the interaction between particular pairings, which this study does
not ask about.**

The balance property is generated arithmetically and asserted by a test, because a
hand-written table with one typo silently destroys the design while still looking balanced.

### The debate itself

**Simultaneous.** In a turn-taking debate whoever speaks last has heard everyone and been
heard by nobody; with a fixed order you measure the order rather than the model.

**Anonymous.** Peers are "another analyst", never "model X" — otherwise a prior about a
named lab does the persuading.

**One round.** Each agent gives an opening view, reads all peers, and gives a final view. The
protocol can also stop on agreement or on stillness, but every shipped path pins the cap at
one rebuttal round, so that is what runs; raising it fails loudly rather than being ignored.

**Only where there is disagreement.** On a point where the agents already agree, a
conversation cannot change the committee's decision and teaches nothing. This was expected to
be most of the compute budget; the measured contested share was 100%, so it saved nothing.

**That 100% is not the share for the committee that debates.**
[`pipeline.select_contested`](src/council/pipeline.py) measures dispersion **once**, over
the whole independent arm pooled across every model and every persona, and
[`debate.sweep.run_debate_arms`](src/council/debate/sweep.py) applies that one list
unchanged to all eight committees. The justification above is stated per committee; the
figure is a pooled-grid figure, and the two are not the same number. Recomputed per
committee on the superseded two-model run, the contested share is **449 of 1,120, 40%**
— by committee: rotations 56, 125, 70 and 123 of 140, uniforms 3, 11, 30 and 31 of 140.
At the unit the justification is stated in the gate is not vacuous, and the saving it was
expected to produce was never measured. The mechanism is kept because it is correct, and
what it costs or saves is an open question rather than a settled one.

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

- **Statistical power is limited.** Two years is four non-overlapping six-month windows, and
  the shipped comparison cuts the period into `council.scoring.DEFAULT_WINDOW_COUNT = 5`
  windows of roughly 4.8 months. Only a large effect would be detectable, and the paired
  design — comparing a committee to itself on the same days — is what makes even that
  possible. This is a **methodology demonstration, not a claim about markets.**
- **Agents are stateless.** No memory across decisions and no awareness of the position
  currently held. That is what makes the independent signals reusable across every committee,
  and it is a real cost, not an oversight.
- **Self-reported confidence is not assumed to mean anything.** It is measured
  ([`evaluation/calibration.py`](src/council/evaluation/calibration.py)) rather than used to
  weight aggregation — using it before establishing that it is calibrated would answer the
  question with itself.
- **Half the headline question has no measurement in the market arms.** Nothing shipped
  crosses *was the agent right* with *did the agent hold*: calibration relates confidence
  to being right, persuasion relates confidence to shifting, and no artefact joins them.
  The probe is the only instrument that pairs the two, and its confidences had no
  variation, so "does being right make any difference" is posed here and answered nowhere.
- **Anonymisation reduces leakage; it does not prove its absence.** A description specific
  enough to be useful may be specific enough to recognise.
- **A single deterministic sample per decision.** Intra-agent variance is unmeasured, so
  some of what is attributed to disagreement between agents is noise within one.
- **The arms do not cover the same decision points.** The placebo needs a donor at least
  `placebo_min_gap_sessions` back, so its earliest points are abandoned while the other arms
  keep them — the first 60 decision dates in either case: 60 of 461 dates (120 of 922 points)
  at the configured range, and 60 of 70 dates (120 of 140 contested points) on the six-month
  slice. `council.app.tables.coverage_note` reports the gap; any debate-minus-placebo
  difference must be read against it. The donor draw also constrains only the date, not the
  ticker, so a debate-minus-placebo difference differences instrument identity along with day
  relevance.
- **Every run reported here used synthetic prices.** `council.data.prices.synthetic_prices`
  generates geometric random walks over the configured calendar; this repository fetches no
  market data, and `load_prices` requires a `data/prices.parquet` the user supplies. There is
  nothing in these series to forecast, so the return columns are arithmetic and only the
  behavioural columns are readable.
- **One asset class, one market, one path.** Nothing here generalises beyond it. This applies
  to a run supplied with real history; under synthetic prices there is no market to
  generalise from at all.
- **Ticker selection carries hindsight.** Mitigated by stating the selection rule in
  [`config.py`](src/council/config.py) before choosing — the largest company by market
  capitalisation in each of two dissimilar sectors, as of the start date — but not removed.
  This too applies only to a run supplied with real history: under synthetic prices the
  ticker is an RNG seed and carries no hindsight.

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

Two stages are expensive and both resume from what is already on disk. Generation
checkpoints per (model, persona, ticker). The debate sweep is the larger bill by an order
of magnitude — at the shipped four-model settings it is 177,024 inferences against
generation's 14,752, about twelve times the independent arm — and it checkpoints per
(committee, arm, ticker), which is what a group *is* for a debate. An interrupted run of
either regenerates only what is missing. `COUNCIL_CUDA_VISIBLE_DEVICES` pins the process
to one card, which matters on a machine doing other work.

The dashboard reads whatever artefacts are in `data/`:

```bash
uv run streamlit run src/council/app/dashboard.py
```

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
  app/           the dashboard: artefact loading, panels, curves, pre-registration panel
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
- [x] End-to-end dry run on the mock provider
- [x] Dashboard and the results write-up
- [ ] A current generation run and results. Two runs were completed — two models, then
      four — and **both are superseded**: each covered a six-month window that was never
      chosen, against the two-year range `config.py` declares, and each was scored with a
      floating-point comparison since replaced. The two-model run's artefacts are under
      [`docs/results/superseded/`](docs/results/superseded/) and are marked as such; the
      four-model run's were kept only in `data/` and are gone, so every figure quoted
      from it — `CLAIMS.md` C14's 138/140 and `findings.md` section 4's screen — cannot
      be rechecked. **There is no current run.**

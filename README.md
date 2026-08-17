# Council

**Do language models hold their ground when they are contradicted?**

[![ci](https://github.com/atiasaf1122/llm-holding-ground/actions/workflows/ci.yml/badge.svg)](https://github.com/atiasaf1122/llm-holding-ground/actions/workflows/ci.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](pyproject.toml)

Four open-weight models judge the same market question independently. Then the same models
are put in a room together and judge it again, having read each other's reasoning. A
**placebo arm** — peers' real arguments about an unrelated day — measures how much of any
movement is a response to the *argument* rather than to being disagreed with at all.

**85,836 decisions on real prices, zero generation failures, on two consumer GPUs** — 52,264
in the published run and 33,572 in the de-roleing run that tests whether the whole result is
an artefact of the personas. The adjudication rule for the deciding experiment was committed
to git before that experiment generated a single row.

---

## The finding

**Movement is a response to opposition, not to argument.** The dose of disagreement
predicts the shift; the content of it barely registers.

| arm | what the agent reads | shift rate |
|---|---|---|
| debate | its peers, genuinely | 0.324 |
| rationale-only | peers' prose, positions withheld | 0.323 |
| same-instrument placebo | a real argument about an unrelated **day** | 0.336 |
| cross-instrument placebo | a real argument about an unrelated day **and instrument** | 0.383 |
| **coherent contradictor** | **three peers arguing against it, on its own data** | **0.606** |

*Share of agent-conversation observations moving ≥ 0.20 exposure between the opening view
and the first post-debate view. Rates pooled over strata; every interval below is the
pre-registered rotation stratum.*

Three results carry the paper:

**1. Unanimous targeted opposition nearly doubles the shift rate.** The contradictor arm
lands **+23.75pp [+20.12, +27.62]** above the placebo and **+26.88pp [+23.12, +30.63]**
above genuine debate. 97.6% of movements go *toward* the opposition, and an outright sign
reversal occurs on 47.1% of all the decisions the arm touched.

**2. The same effect was hiding inside the genuine debate all along.** Counting how many
opposing peers a debate reader actually faced — 0, 1, 2 or 3 — the shift rate climbs
**0.22 / 0.28 / 0.36 / 0.61**. At full opposition it is **0.607 (n=89)**, statistically
indistinguishable from the engineered arm's 0.606. Two independent routes, one number: the
engineered arm's only contribution is delivering that dose on every decision instead of on
one in eighteen.

**3. It is not a general weakness — it is specific to judgement.** Put to 24 questions with
verifiable answers and challenged by the same mechanism, the same four models capitulated
**twice in 89 challenged trials**. They do not fold under pressure. They fold under
pressure *about opinions*.

And it is not uniform across models. Under unanimous opposition, per-protocol capitulation
runs from **granite4.1 at 97.8%** and phi4 at 92.0% down to gemma4 at 42.8% and
**qwen3.5 at 32.2%**. There is no such thing as "how a language model behaves under
disagreement", only how a *specific* model behaves — a factor of three apart at the same
quantisation on the same machine.

### What this means if you are building with agents

- **A "committee of agents that debate" does not aggregate judgement — it aggregates
  social pressure.** The minority position loses in proportion to how outnumbered it is,
  not to how wrong it is.
- **The part that works is the part without the conversation.** Independent generation
  followed by mechanical aggregation keeps the diversity; the debate step is what destroys
  it.
- **Model choice is a design decision with a threefold spread.** If your architecture
  depends on an agent defending a correct minority view, the model you pick decides whether
  that is possible.
- **Nobody made any money.** The committee lost 14.5% over a window in which holding the
  basket returned 39.6%, and scored worse than a random null matched to its own turnover.
  The market here is a scoring function, not a goal, and a negative result is a result.

Everything above, with its intervals, denominators and caveats:
**[`docs/findings.md`](docs/findings.md)**.

---

## Why these numbers should be believed

The uncomfortable part of a result like this is that it would be easy to fake and easier
still to fool yourself into. Five things in this repository exist to make that hard.

**A pre-registration that predates the data.** The primary statistic and its two thresholds
were fixed in the first commit. When the extension arms were designed, the rule for reading
their outcome — including which branch would mean what — was committed to git *before* they
generated a row (`3c96a98`). The result then landed outside every branch the rule imagined,
and it is reported under the branch whose condition fired, with the excess stated. Three
amendments to the declared statistic are quoted verbatim below, with what changed and why.

**A defect register that includes the defects that killed this study's own headlines.**
[`docs/CLAIMS.md`](docs/CLAIMS.md) carries **31 numbered claims and 20 registered defects**
(14 live, six superseded). Among them: the headline mechanism sentence that shipped false
because the inference backend does not enforce numeric schema bounds (D15); a run whose
model vintage drifted between arms (D16); and the price-fetch module that an unanchored
`.gitignore` pattern kept out of the repository for the entire study (D17). Every claim is
either backed by an artefact in this repository or explicitly marked as not.

**Tests that recompute the published numbers.**
[`tests/test_docs_findings.py`](tests/test_docs_findings.py) reads the pinned parquet files
and recomputes the tables, rates and bootstrap intervals printed in the documents. A figure
that drifts from its artefact fails the suite. The headline confidence interval was once
computed by no code in the repository at all; that is why this exists.

**Six rounds of adversarial review.** Independent hostile-audit passes over the findings,
fixing between rounds. They changed the headline twice, caught the authors misremembering
their own design once, and falsified a mechanism claim that the entire test suite had
endorsed — because the mock provider obeyed a constraint the real backend ignored. *A mock
more obedient than reality validates the wrong world.*

**Paired statistics over a correlated design.** Observations within a decision point share
a day and 32 correlated seats, so every comparison is paired by decision point and
bootstrapped over points (5,000 draws, seeded by index — the interval is a pure function of
the artefact).

---

## The question

> **When a confident agent is contradicted, does it defend its position or abandon it — and
> does being right make any difference?**

There is reason to expect the worst. Models are known to retreat from correct answers under
mild pressure, sometimes from nothing more than "are you sure?". Put several in a room and
they can converge on an answer none of them started with. If that is what a "committee of
agents" actually does, it matters to anyone building one.

A second question comes free from the same data:

> **Is there a loudest voice — one model the others drift toward regardless of who is
> right?**

---

## Pre-registered primary comparison

*A declaration was made before any result was generated, and the two thresholds it is
stated in were fixed in the first commit. The declared statistic has since been amended
three times, after the results existed; the original wording is quoted verbatim under
**Amendments** and all three changes are named there. Four further bounds were added
afterwards and are marked as such. Everything else in this repository is exploratory and is
labelled as such.*

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
`agreement_spread = 0.20`, `stillness_rounds = 2` and `max_debate_rounds = 6` (when a
conversation ends), and `placebo_min_gap_sessions = 60` (how far back a placebo donor must
come from). Two of these were calibrated from that run's measurements rather than chosen
blind: `agreement_spread` from its measured spreads, `placebo_min_gap_sessions` from its
donor distances. They are declared here so a reader can check a result against them; they
are **not** pre-registered and must not be read as such. Each carries its justification
where it is declared.

`max_debate_rounds` is a **cap, not a length**: a conversation ends on whichever of
agreement, stillness or the cap comes first, and which one ended it is stored on every row
of that conversation.

**Why this paragraph exists.** With three aggregation rules, eight committee configurations
and several statistical tests, something will look significant by chance. Fixing the
comparison in advance is the difference between a measurement and a search.

---

## Design

### Six arms, and what each one rules out

A single observation — "the agent moved after the debate" — has many possible causes. Each
arm removes one of them, and collapsing any arm into another destroys the inference.

| Arm | What the agent sees | Rules out |
|---|---|---|
| **Independent** | nothing | — the control everything is measured against |
| **Debate** | peers' rationales **and** their exposure numbers | — the treatment |
| **Rationale only** | peers' rationales, **with the peer's stated exposure removed** | **anchoring on a peer's stated position** — figures a peer wrote into its own prose are not removed |
| **Placebo** | peers' rationales **from an unrelated day** — and, on the published run, **the other instrument 49% of the time** | **the argument's content** — movement equal to the debate arm's means the content contributed nothing |
| **Same-instrument placebo** | an unrelated day, on the reader's **own instrument** | **instrument displacement**, isolating it from day displacement (D14) |
| **Coherent contradictor** | three peers arguing **against the reader**, on the reader's **own data** | **incoherence** — the one thing the placebo cannot separate from opposition (D8) |

The last two arms exist because the first four could not settle the question. The placebo
was built to rule out *compliance* — reacting to contradiction itself — but a donor-day
argument is not merely irrelevant, it is **incoherent** against the data the reader holds,
and half the time it is about the other instrument entirely. So it bounds what the
argument's *content* contributes and cannot separate "reacting to being contradicted" from
"reacting to an argument that cannot be reconciled". The contradictor arm closes that gap;
the same-instrument placebo decomposes the displacement.

Rendering is one code path for all six arms — an arm distinguishable by its formatting
measures nothing.

Every decision here is made once a day, on every session that has a full `lookback_days`
window behind it **and** falls on or after the configured `start` — whichever binds later.
On the published run `start` binds: the price table holds 579 sessions and decisions exist
on 501, so the warm-up is **78 sessions**, not the 59 that `lookback_days` alone would give.
**Daily is the only decision frequency this repository implements**; there is no resampling
step and nothing at any other frequency to compare against.

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

Stance sets the sign of the mean exposure for every model without exception — which raises
the obvious objection that the study measures *staying in character* rather than defending
a position. That objection was run as an experiment rather than argued about; see
**the disposition run** below.

### Eight committees, not 256

The obvious design assigns each of four models one of four personas in every combination:
4⁴ = 256 configurations. A conversation at the shipped six-round cap is up to 28 model
calls, so over a thousand decision points the full grid costs about 28.7 million inferences
per set of four full-length arms, plus 5.1 million for the contradictor's counter-arguments
— **thirty-four million inferences**, about a hundred and forty-seven days of continuous
compute — and most of it is redundant.

Instead, [`debate/compositions.py`](src/council/debate/compositions.py) generates a
**balanced design**: a Latin square in which every model holds every persona exactly once
(4 configurations), plus the 4 uniform references where every seat holds the same persona.
**Eight configurations, one thirty-second of the compute** — model and persona main effects
separated, at the cost of the interaction between particular pairings, which this study does
not ask about. The balance property is generated arithmetically and asserted by a test,
because a hand-written table with one typo silently destroys the design while still looking
balanced.

### The debate itself

**Simultaneous.** In a turn-taking debate whoever speaks last has heard everyone and been
heard by nobody; with a fixed order you measure the order rather than the model.

**Anonymous.** Peers are "another analyst", never "model X" — otherwise a prior about a
named lab does the persuading.

**Ended by a condition, capped at six rounds.** Each agent gives an opening view, reads all
peers, answers — and keeps answering until the committee comes within `agreement_spread` of
itself, stops moving for `stillness_rounds` consecutive rounds, or reaches the cap. Which of
those ended a conversation is stored on every one of its rows, so the round count and the
verdict are outcomes rather than settings.

**Only where there is disagreement.** On a point where the agents already agree, a
conversation cannot change the committee's decision and teaches nothing. This was expected
to be most of the compute budget; the measured contested share was **984 of 1,002, 98.2%**,
so it saved almost nothing.

**And that 98.2% is not the share for the committee that debates.**
[`pipeline.select_contested`](src/council/pipeline.py) measures dispersion **once**, over the
whole independent arm pooled across every model and every persona, and applies that one list
unchanged to all eight committees. The justification above is stated per committee; the
figure is a pooled-grid figure, and the two are not the same number. Recomputed per
committee, the contested share is **4,728 of 8,016, 59.0%** — and the range is the point:
98.5% for `rotation-3` against **19.0%** for `uniform-reversion-bold`. A uniform committee
spends between 57% and 81% of its budget arguing about points it never disagreed on, which
is why its debate arm reaches agreement in a mean 1.94 rounds. The mechanism is kept because
it is correct; *where it is measured* is the defect, and it is recorded rather than fixed,
because fixing it would change which points every published arm was run on.

**Constrained decoding.** Every reply is forced through a JSON Schema compiled to a
token-level grammar. Two rules learned the hard way: every string field is bounded (an
unbounded field is a hole in the grammar — measured once at 82,000 tokens of broken JSON),
and **the grammar's numeric bounds are not enforced by the backend**, which is D15 and cost
this study a false mechanism claim. Constraints that matter are now verified after decoding.

### The disposition run

Every agent is *instructed* into its view, so "does a model hold its ground" might be "does
a model stay in role" (D10). Rather than argue it, the whole design was re-run with the
persona voiced as a revisable tendency — *"you have tended to read moves as momentum"* —
instead of an identity, in a separate data directory with byte-identical prompts outside the
stance section.

**The behaviour survives.** Stance still sets the sign in 15 of 16 (model, persona) cells,
the placebo-minus-debate structure reproduces (**+2.45pp [−0.86, +5.88]** against the
identity phrasing's +3.12pp [+0.25, +6.12]), and the shift rates move slightly *up* where
role-discipline predicts collapse. The results are not an artefact of the character sheet.
33,572 decisions, zero failures, artefacts pinned.

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
- **The extension arms are a different model vintage than the arms they are compared
  against.** Their round-0 answers to byte-identical prompts disagree with the original
  run's on 24.1% of seat-points, six times the within-run floor, because the backend updated
  between runs (D16). The ±20pp contradictor gaps dwarf the drift; the −5.00pp
  same-instrument inversion does not, and carries the caveat wherever it is quoted.
- **The contradictor's treatment was not everywhere the treatment described.** 6.3% of
  sided counter-arguments agreed with the reader, contaminating 15.8% of readers (D15). The
  direction is conservative — clean readers shifted at 0.675 against the published 0.606 —
  and both denominators are stated wherever a per-model figure appears.
- **Agents are stateless.** No memory across decisions and no awareness of the position
  currently held. That is what makes the independent signals reusable across every committee,
  and it is a real cost, not an oversight.
- **Half the headline question has no measurement in the market arms.** Nothing shipped
  crosses *was the agent right* with *did the agent hold*: calibration relates confidence to
  being right, persuasion relates confidence to shifting, and **no artefact joins them**. The
  probe is the only instrument that pairs the two, and its confidences pile at the ceiling — 324 of 384 stored
  confidences are exactly 1.0, though how badly is model-specific (phi4 96/96 and gemma4
  95/96 against granite4.1's 52/96) — so "does being right make any difference" is posed
  here and answered nowhere.
- **A single sample per decision, and it is not as deterministic as the settings say.**
  Temperature 0 and a fixed seed notwithstanding, 4.3% of round-0 exposures differ
  across arms on byte-identical prompts, 2.19% by at least the shift bar — a ~2pp
  noise floor under every rate, symmetric across arms so it cannot create the gaps
  between them (D12).
- **The treatment arms answer a shorter calendar than the control.** The placebo needs a
  donor at least `placebo_min_gap_sessions` back, and a fresh one for every round, so there
  are points it cannot serve. Those are withheld from **all five debate arms** rather than
  from the placebo alone, so the arms cover an identical point set and no
  debate-minus-placebo difference is partly a difference in coverage. On the published run
  that cost 10 of the 60 sampled points, all in the first sixty sessions of the calendar.
  `council debate` prints how many points were withheld and `council.app.tables.coverage_note`
  reports what each arm holds. The donor draw also constrains only the date, not the ticker,
  which is the confound the same-instrument arm was later built to measure.
- **The debate ran on a sample, and it costs the return comparison its power.** A treatment
  arm differs from the control on **50 of 1,002** decision points, so the arms' returns
  would agree to three decimal places whether or not debate affected them. The behavioural
  measurements are computed per debated point and are unaffected.
- **Recognition is a live risk, not a hypothetical.** Every model was trained on data
  covering 2022–2023, and these are two of the most written-about instruments in it. Agents
  see normalised returns with no dates, no tickers and no price levels
  ([`data/context.py`](src/council/data/context.py)), which reduces recognition and cannot
  prove its absence. A distinctive drawdown shape is still a shape.
- **Prices are back-adjusted.** `auto_adjust` makes the return series continuous through
  splits and dividends, which is the right input for a total-return backtest and the
  wrong one for claiming any fill was achievable.
- **One asset class, one market, one path.** Nothing here generalises beyond it. Two
  instruments, two years, four models, one machine.
- **Ticker selection carries hindsight.** Mitigated by stating the selection rule in
  [`config.py`](src/council/config.py) before choosing — the largest company by market
  capitalisation in each of two dissimilar sectors, as of the start date — but not removed.

---

## Reproducing it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). The generation steps additionally
require [Ollama](https://ollama.com) 0.31.2 or newer and enough VRAM to hold four models.

```bash
uv sync --frozen --extra dev --extra app
uv run pytest
```

Both extras are required, not optional: the suite imports the dashboard layer, so `dev`
alone fails at collection. This is the exact command CI runs.

**The whole test suite runs on CPU with no model server and no network** — including the
tests that recompute every published figure from the pinned artefacts.

Rehearse the entire pipeline with no GPU, on synthetic prices and a mock provider:

```bash
uv run council dryrun
```

The real thing, in order. Only the first step reaches the network and only the middle two
need a GPU:

```bash
uv run council prices      # real daily bars; pins the vendor's response beside them
uv run council plan        # cost the configuration before committing a night to it
uv run council generate    # the independent arm
uv run council debate      # the treatment arms, over contested points
uv run council evaluate    # score what is on disk
```

Every setting is an environment variable with a `COUNCIL_` prefix, read from `.env`.
[`.env.example`](.env.example) documents the ones that change a result rather than a path.

Both generation stages checkpoint and resume — `generate` per (model, persona, ticker),
`debate` per (committee, arm, ticker) — so an interrupted run regenerates only what is
missing. This survived two mid-run backend auto-updates at the cost of minutes rather than
nights. `--device` pins the process to one card, which matters on a machine doing other
work, and `COUNCIL_MAX_DEBATE_POINTS` bounds the sample: the five debate arms stored 36,232
decisions over a 50-point sample, so the full contested set would price at roughly **713,000**
stored decisions, plus the contradictor's counter-arguments on top.

The capitulation probe is a side study on the same machinery — known-answer items, no prices:

```bash
uv run council probe --mock                                    # no daemon, no GPU
uv run council probe --model qwen3.5:9b --out data/probe/qwen3.5-9b.jsonl
```

The `--mock` run exercises the machinery, not the question: the mock answers with a token
rather than with the right answer, so every rate in its table is necessarily zero. The real
figures are in [`docs/results/run-4models-2y/probe/`](docs/results/run-4models-2y/probe/).

Pass `--out` with a per-model path. The default output path is shared, so sequential runs of
different models overwrite each other — which is exactly what happened to the first published
probe table (D13).

**Inspecting the published run without regenerating anything.** Every command that reads
artefacts takes `--data-dir`, so the run this repository publishes can be scored, tabulated
and browsed straight from the clone — no GPU, no daemon, no network:

```bash
uv run council evaluate --data-dir docs/results/run-4models-2y
COUNCIL_DATA_DIR=docs/results/run-4models-2y uv run streamlit run src/council/app/dashboard.py
```

Reading is read-only: the artefacts are byte-pinned evidence, and a command that merely
inspects one leaves its checksum alone.

The dashboard otherwise reads whatever artefacts are in `data/`:

```bash
uv sync --frozen --extra dev --extra app
uv run streamlit run src/council/app/dashboard.py
```

**What you cannot reproduce exactly, with the size of it.** Vendors revise price history and
model tags are re-published under the same name, so a rerun returns neither the series nor
the answers this study used. Both were measured rather than assumed:

- **The prices have already moved.** A refetch on 2026-08-18 returned the same 1,158 bars on
  the same calendar with identical volumes — and **every one of AAPL's 579 bars is revised,
  against none of XOM's**, by up to 0.086%. That is back-adjustment re-derived against
  dividends paid since. Counted without a threshold the figure is unstable and reproduces at
  neither run's value: two refetches minutes apart disagree with each other on 73% of bars,
  by less than a millionth of a percent, because the vendor's adjustment arithmetic carries
  sub-cent noise. Above that floor the answer is one clean sentence. Measured against the
  pinned parquet, because this study's own vendor response was not kept — a provenance gap
  registered as D20, and the reason `council prices` now writes one.
- **The models have already moved.** Round-0 answers to byte-identical prompts disagree
  across two of this study's own runs on 24.1% of seat-points, six times the within-run
  floor, because the backend updated between them (D16).

So the guarantee this repository makes is **recomputation, not regeneration**: every
published figure is derived from committed artefacts by the test suite, on CPU, with no
network. Regenerating from scratch is a different experiment, and is expected to disagree.

---

## What is in this repository

**The documents**, in the order they answer questions:

| file | what it is |
|---|---|
| [`docs/findings.md`](docs/findings.md) | every result, with intervals, denominators and the caveats attached to each |
| [`docs/methods.md`](docs/methods.md) | one page: every component, how it works, and which alternative explanation it kills |
| [`docs/CLAIMS.md`](docs/CLAIMS.md) | 31 numbered claims and 20 registered defects — what is backed by an artefact and what is not |
| [`docs/research.md`](docs/research.md) | the decisions taken during the build, in the order they were forced |

**The artefacts**, all committed, all recomputed by the test suite:

| path | what it holds |
|---|---|
| [`docs/results/run-4models-2y/`](docs/results/run-4models-2y/) | the published run: 52,264 decisions over six arms, the price table, the archived counter-arguments (4,800 unique pairs), and the per-model probe trials |
| [`docs/results/run-disposition/`](docs/results/run-disposition/) | the de-roleing run: 33,572 decisions |
| [`docs/results/superseded/`](docs/results/superseded/) | earlier synthetic-price runs, kept because the *classes* of error they carried recur |

---

## Layout

```
src/council/
  domain/        personas and the signal contract every layer speaks
  data/          fetch, prices, one shared calendar, and the anonymised context an agent sees
  agents/        prompts, provider, and the checkpointing runner
  debate/        balanced compositions, the debate protocol, placebo and contradictor
  backtest/      the fill rule, costs, metrics, and the random baseline
  evaluation/    dispersion, calibration, persuasion, influence, windows, intervals
  app/           the dashboard: artefact loading, panels, curves, pre-registration panel
  probe/         the capitulation probe: known-answer items, no prices
```

---

## Status

- [x] Experiment contract, backtest engine, and the lookahead tests
- [x] Data layer with the anonymisation boundary
- [x] Provider, agent runner, debate protocol, evaluation
- [x] End-to-end dry run on the mock provider
- [x] Dashboard and the results write-up
- [x] **The published run.** Four models, four personas, eight committees, real split- and
      dividend-adjusted bars for AAPL and XOM, decisions on all 501 sessions from
      2022-01-03 to 2023-12-29. **52,264 stored decisions, zero generation failures.**
      Artefacts at [`docs/results/run-4models-2y/`](docs/results/run-4models-2y/), and
      `tests/test_docs_findings.py` recomputes the published tables from them, so a figure
      in [`findings.md`](docs/findings.md) that drifts from the parquet fails the suite.
      Two earlier runs — two models, then four — are **superseded**: each covered a
      six-month window that was never chosen, each ran on synthetic prices, and each was
      scored with a floating-point comparison since replaced. The two-model run's artefacts
      are under [`docs/results/superseded/`](docs/results/superseded/); the four-model
      run's were kept only in `data/` and are gone, so figures quoted from it **cannot be
      rechecked**.
- [x] **The capitulation probe**, re-run with per-model output paths after D13.
- [x] **The two extension arms**, adjudicated by a rule committed before they ran. The
      coherent contradictor produced **0.606** intent-to-treat — 0.675 among the readers who
      received the described treatment (D15) — against debate's 0.324 and the placebo's
      0.383, landing above both arms rather than between them, which is an outcome the rule
      had no branch for and is reported as such. The same-instrument placebo simultaneously
      decomposed the original headline: with the donor on the reader's own instrument, the
      placebo's surplus over debate **reverses** in the registered rotation stratum
      (−5.00pp [−8.12, −1.50]; pooled null, uniform positive) — a cross-vintage contrast
      comparable in size to the D16 drift, so it stands with that caveat attached rather
      than as settled fact.
- [x] **The disposition run (D10).** The behaviour survives de-roleing.
- [x] **Six rounds of adversarial audit**, the last of which falsified a mechanism claim the
      whole test suite had endorsed (D15) and found a pipeline module that had never been
      committed (D17).

---

## License and attribution

MIT — see [`LICENSE`](LICENSE). The models are used under their own licences (Apache-2.0 for
qwen3.5, granite4.1 and gemma4; MIT for phi4) and are not redistributed here; price data is
retrieved from its vendor at run time and is not redistributed either. See
[`NOTICE.md`](NOTICE.md) for the full attribution and the reproducibility caveat that goes
with it.

If you refer to this work, [`CITATION.cff`](CITATION.cff) has the metadata.

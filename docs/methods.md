# Methods — every component, how, and why

The one-page answer to "what did you actually build?". Each component exists for
one of two reasons: to **kill an alternative explanation** of the result, or to
make the result **checkable without trusting anyone**. Nothing here is decoration.

---

## Hardware and serving

| | |
|---|---|
| GPUs | 2× RTX 3090 (24GB each, 48GB total) |
| Serving | Ollama, HTTP on localhost, no cloud, no per-token cost |
| Models | qwen3.5:9b (Alibaba), granite4.1:8b (IBM), gemma4:12b (Google), phi4:14b (Microsoft) — Q4_K_M quantised, 29.2GB resident together |

**Why four families at one size.** Architecture is a factor in the experiment; at
mixed sizes, "model effect" is partly "scale effect". The choice paid off as a
finding: the shift rate under unanimous targeted opposition ranges from 32.2%
(qwen3.5) to 97.8% (granite4.1) per protocol — 28.0% to 92.8% intent-to-treat,
a denominator that mixes treated and untreated readers (D15). There is no such
thing as "how a language model behaves", only how a specific model behaves.

**Why both cards are required.** A debate round is four seats answering
simultaneously, so all four models must be resident at once; they do not fit on
one card, and swapping costs a measured 36s cold load per model. Inference is
memory-bandwidth-bound: four concurrent requests gave ×1.5 over sequential (not
×4), which is why `concurrency = 4`.

## Constrained decoding

Every reply is forced through a JSON Schema compiled to a token-level grammar
(`format` in Ollama): the model cannot emit anything but the
`{exposure, confidence, rationale}` object, with its **string** fields bounded.
Two hard-won rules, the second of them learned the expensive way:

* Every string field is bounded. An unbounded field is a hole in the grammar; a
  model with more to say pours everything into it — measured once at 82,000
  tokens of broken JSON.
* **Numeric `minimum`/`maximum` are not enforced by the backend at all**, and
  this study used them as an experimental instrument before finding out. The
  contradictor arm set `exposure` bounds excluding the reader's side and
  described the result as a peer that *cannot* drift back into agreement; 6.3%
  of sided counters agreed with the reader anyway, reaching 15.8% of readers
  (D15). A structural constraint in the grammar holds. A numeric one is a
  request, and must be verified after decoding — which is what
  `contra.CounterSideError` now does, with one retry and then a refusal.

## Data

Real daily bars, AAPL and XOM, 2022–2023 (drawdown, recovery, flat stretch —
regimes are not interchangeable, which is why the period was sampled rather than
shortened). Fetched once via yfinance with `auto_adjust`, and the table it built
is pinned — the vendor's own response was not kept, which is D20 and the reason
`council prices` now writes one. Agents see **normalised percentage returns
only** — no ticker, no
dates, no price levels — because the models were trained on this period and a
model that recognises the window is remembering, not analysing. This reduces
recognition; it cannot prove its absence (a distinctive drawdown shape is still
a shape), and the limitation is registered.

## Personas — the disagreement engine

Two crossed axes: **stance** (momentum: a run continues / reversion: a run
overshoots) × **aggression** (cautious / bold). Stance manufactures disagreement
about *direction*, which is the precondition for measuring "held its ground";
four agents differing only in size are haggling, not arguing. Verified on the
run: stance sets the sign of the mean exposure for every model without
exception. A disposition-voiced variant of the same briefs ("you have tended
to…", revisable) exists for the D10 experiment, byte-identical outside the
stance section.

## The six arms

One observation — "the agent moved after reading peers" — has many causes. Each
arm holds all but one factor fixed:

| arm | the agent reads | isolates |
|---|---|---|
| independent | nothing | the baseline |
| debate | peers' rationales **and** positions | the treatment |
| rationale-only | rationales, positions withheld | numeric anchoring |
| placebo | a real committee's views from an unrelated day (49% the other instrument) | content at all |
| same-instrument placebo | unrelated day, **same instrument** | day-displacement vs instrument-displacement (D14) |
| coherent contradictor | three peers arguing **against the reader** on the reader's own data (side requested through the grammar, not enforced by it — D15) | opposition **dose**; it bundles unanimity, targeting and framing, so not coherence alone (D8, C29) |

Rendering is one code path for all arms — an arm distinguishable by its
formatting measures nothing. Peers are anonymous handles in a deterministic
per-prompt order; all seats answer simultaneously (a speaking order would be
measured instead of the models); the placebo donor sits ≥60 sessions back so its
data window cannot overlap the decision's.

## Protocol

* **8 committees, not 256.** A Latin square (every model at every persona once)
  plus four uniform references separates model effects from persona effects at
  1/32 of the full grid's cost.
* **Conversations end on conditions, not counts**: agreement (spread ≤ 0.20),
  stillness (2 quiet rounds), or a 6-round cap — and *which one* is stored,
  because "agreed" versus "ran out of time still fighting" is a result. The
  contradictor arm is capped at one rebuttal round: its adjudicating metric is
  round 0→1 and no later-round schedule for a targeted contradiction is
  defensible.
* **50 decision points sampled from 984**, evenly spread over both years and
  nested (a larger budget contains a smaller one). Shortening the period instead
  would have changed what was measured.

## Reliability

* **Checkpoint + resume** per (committee, arm, ticker) group; a resumed run
  reads `stop_reason` and skips what is finished. Survived the mid-sweep Ollama
  auto-update recorded in C25 (0.32.5 → 0.32.6) at the cost of minutes rather
  than a night. A further backend change *between* the original and extension
  runs is the source of the D16 vintage drift, which no resume can repair.
* **Three retry layers**: transport (backoff on 5xx), envelope (a daemon
  restarted mid-response — 3 re-sends), command (up to 5 relaunches).
* **A failure is a datum**: recorded with its `FailureMode`, never dropped.
  Observed rate on the published run: 0 of 52,264.
* **MockProvider** honours schema bounds, so the entire test suite and the full
  pipeline (`council dryrun`) run in seconds with no GPU — which is also what CI
  runs on every push. That obedience is itself a registered defect (D15): the
  mock honoured numeric bounds the real backend ignores, so the contradictor's
  side constraint passed every test and failed in production. A mock more
  obedient than reality validates the wrong world, and the mock is now scripted
  to disobey wherever the contract under test is a bound.

## Measurement

* "Changed its mind" = moved ≥ 0.20, via `threshold.meets` with a 1e-9
  tolerance — a bare `>=` on a 0.05 grid silently dropped exactly-at-the-bar
  moves and killed two early findings before it was caught.
* Observations within a decision point share a day and 32 correlated seats, so
  every comparison is **paired by decision point** and bootstrapped over points
  (5,000 draws, seeded by index — the interval is a pure function of the
  artefact). The bootstrap lives in `evaluation/intervals.py` and is exercised
  by the doc tests, after an audit found the headline CI computed by no code in
  the repository.
* The backtest fills at the next session's open, charges 10bp, and is compared
  against buy-and-hold **and** a turnover-matched random null, scored only over
  the window the arms could trade (a 78-session warm-up once inflated the
  benchmark by 27.6pp before a test pinned the window).

## Discipline

* **Pre-registration**: the primary statistic and its thresholds in the first
  commit; the extension arms' adjudication rule committed *before* they
  generated a row — and when the result landed outside every branch the rule
  imagined, it was reported under the rule with the excess stated.
* **CLAIMS.md**: 31 numbered claims, 20 registered defects (14 live entries plus
  D2–D7 retained for provenance) — including the ones that killed this study's
  own headline findings. A claim is either backed by an artefact in the repository
  or it is marked as not.
* **Doc-contract tests**: `tests/test_docs_findings.py` recomputes the published
  tables and intervals from `docs/results/run-4models-2y/decisions.parquet`; a
  published number that drifts from the artefact fails the suite.
* **Six rounds of adversarial review** (independent referee passes) changed the
  headline finding twice, caught the authors misremembering their own design
  once, and on the sixth pass falsified a mechanism sentence the whole test
  suite had endorsed (D15) plus a pipeline module that had never been committed
  (D17). All of it is in the registers.

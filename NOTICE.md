# Notices

## Data availability

**All data required to recompute every published figure is in this repository.**
No number in `docs/findings.md` depends on anything a reader has to obtain, pay
for, or ask for. The test suite demonstrates it: `tests/test_docs_findings.py` reads
`docs/results/` and recomputes the published tables, rates and bootstrap
intervals on CPU, with no model server and no network.

What is **not** here, and what that costs:

| not included | consequence |
|---|---|
| model weights | the results cannot be *regenerated*, only recomputed. Pull the four tags from Ollama to regenerate, and see the caveat below. |
| a vendor price feed | `council prices` refetches from yfinance. **The response this study's own fetch returned was not retained** (D20) — only the parquet built from it. A refetch can be compared against that parquet, which is enough to see a revision, but not against the vendor's original bytes. Runs from here on pin theirs. |

## Model licenses

This repository ships **no model weights**. The published artefacts
(`docs/results/`) contain *outputs* generated locally by the following models,
whose licenses were read from the pulled models' own metadata (`ollama show
<model> --license`) and permit this use:

| model | license |
|---|---|
| qwen3.5:9b (Alibaba) | Apache License 2.0 |
| granite4.1:8b (IBM) | Apache License 2.0 |
| gemma4:12b (Google) | Apache License 2.0 |
| phi4:14b (Microsoft) | MIT |

None of these licenses restricts publication of model outputs, and this
project's MIT license covers only its own code and text.

**Model tags float, and this study could not pin them.** The four tags above are
names, not digests: the same name is republished as the underlying weights are
updated, and nothing in the Ollama API this project uses exposes a hash to pin
against. That is not a hypothetical — a backend update between two of this
study's own runs shifted 24.1% of round-0 answers on byte-identical prompts,
which is registered as defect D16. A rerun that disagrees with the published
series is therefore expected, and is not evidence that either run was wrong.

## Market data

`docs/results/run-4models-2y/prices.parquet` holds split- and dividend-adjusted
daily bars for two tickers (AAPL, XOM, 2021--2023), fetched once via `yfinance`
and pinned for reproducibility -- without it, no number in the study can be
recomputed. It is factual end-of-day data, included in the spirit of research
reproducibility; it is not a market-data feed and should not be used as one.
Vendors revise history, so `council prices` writes the vendor's response verbatim
beside the parquet it builds. That is a guarantee for runs from here on, **not one
this study can offer about its own fetch**: the original response was not kept, and
the parquet is the earliest artefact that exists (D20). A rerun that disagrees can
still be diffed against the parquet -- which is how the revision documented in the
README was measured -- but not against the vendor's original bytes.

## Documentation

The prose in `README.md` and `docs/` is covered by the same MIT license as the
code. Attribution is welcome rather than required; `CITATION.cff` carries the
metadata if you want it.

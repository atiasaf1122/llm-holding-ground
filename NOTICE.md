# Notices

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

## Market data

`docs/results/run-4models-2y/prices.parquet` holds split- and dividend-adjusted
daily bars for two tickers (AAPL, XOM, 2021--2023), fetched once via `yfinance`
and pinned for reproducibility -- without it, no number in the study can be
recomputed. It is factual end-of-day data, included in the spirit of research
reproducibility; it is not a market-data feed and should not be used as one.

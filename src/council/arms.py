"""One arm's exposures, put through the engine.

The seam between deciding *what* a committee held and measuring what that
earned. Everything here takes a mapping of decision point to exposure and knows
nothing about arms as experimental conditions, rounds, or committees --
:mod:`council.scoring` owns that, and hands the result to these three functions.

The seam exists because two callers need the measuring half and only one needs
the deciding half. :mod:`council.scoring` scores the run for ``council
evaluate``; :mod:`council.app.curves` draws the same arms on the dashboard. When
the dashboard had its own copy of this arithmetic the two disagreed about the
secondary declared comparison on identical artefacts -- the placebo arm changed sign
between them -- and nothing on either output said which was declared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from council.backtest.baseline import TurnoverTarget, random_targets
from council.backtest.engine import BacktestResult, run_backtest
from council.evaluation.frames import PointKey


def targets_frame(exposures: Mapping[PointKey, float], *, tickers: Sequence[str]) -> pd.DataFrame:
    """Exposures as the wide frame the engine reads, indexed by decision date.

    A point with no exposure is left NaN rather than zero, because
    :func:`council.backtest.engine.run_ticker` reads a NaN as "no decision, hold"
    and a zero as "go flat" -- and the two are different instructions.
    """
    days = sorted({day for day, _ in exposures})
    return pd.DataFrame(
        {ticker: [exposures.get((day, ticker), np.nan) for day in days] for ticker in tickers},
        index=pd.DatetimeIndex(days),
    )


def backtest_arm(
    *,
    exposures: Mapping[PointKey, float],
    opens: pd.DataFrame,
    cost_bps: float,
    rebalance_threshold: float,
) -> BacktestResult:
    """Run one arm's exposures through the engine.

    Everything that reports an arm's performance comes through here -- the CLI's
    results table and the dashboard's equity panel both. A second implementation
    would let the two disagree about the secondary declared comparison on identical
    artefacts, and the reader would have no way to tell which was declared.
    """
    return run_backtest(
        targets=targets_frame(exposures, tickers=[str(column) for column in opens.columns]),
        opens=opens,
        cost_bps=cost_bps,
        rebalance_threshold=rebalance_threshold,
    )


def random_arm_targets(
    *,
    exposures: Mapping[PointKey, float],
    opens: pd.DataFrame,
    turnover_per_period: TurnoverTarget,
    rebalance_threshold: float,
    seed: int,
) -> pd.DataFrame:
    """Targets for the null matched to one arm's trading rate and exposure sizes.

    ``turnover_per_period`` is per ticker for the same reason ``exposure_pool``
    below is: turnover is realised per column, so one basket-mean scalar matches
    each column to the average rather than to its own rate. Callers holding the
    backtest's ``per_ticker`` results should pass the mapping.

    The null is confined to the sessions the arm holds a decision for. Left free of
    the whole calendar it revises inside the ``lookback_days - 1`` warm-up the arm is
    flat over, so it is invested where the arm is not and the warm-up's drift is
    credited to the null alone -- with turnover and exposure distribution matched
    either way, which is what makes the gap invisible.

    Raises:
        ValueError: if the arm turns over more than any shuffle of its own
            exposures can reach, or if a mapping has no rate for one of ``opens``'
            columns. Left to the caller, because the CLI reports the
            gap per arm and the dashboard reports it once on the panel; neither
            may quietly substitute a null matched to some other turnover.
    """
    return random_targets(
        opens=opens,
        target_turnover_per_period=turnover_per_period,
        rebalance_threshold=rebalance_threshold,
        seed=seed,
        # The arm's own requested exposures, so the null holds positions of the
        # same sizes and signs and differs only in when it holds them -- per
        # ticker, because that is the distribution being matched. One flat pool
        # over every instrument gives each column the cross-ticker mixture, which
        # for a committee systematically long one and short another is neither
        # ticker's distribution: half of "same shape", which the baseline module
        # says is worse than neither because it looks rigorous.
        exposure_pool={
            ticker: [
                exposure for (_, held), exposure in sorted(exposures.items()) if held == ticker
            ]
            for ticker in (str(column) for column in opens.columns)
        },
        revisable=sorted({day for day, _ in exposures}),
    )

"""One arm's exposures, put through the engine.

The seam between deciding *what* a committee held and measuring what that
earned. Everything here takes a mapping of decision point to exposure and knows
nothing about arms as experimental conditions, rounds, or committees --
:mod:`council.scoring` owns that, and hands the result to these three functions.

The seam exists because two callers need the measuring half and only one needs
the deciding half. :mod:`council.scoring` scores the run for ``council
evaluate``; :mod:`council.app.curves` draws the same arms on the dashboard. When
the dashboard had its own copy of this arithmetic the two disagreed about the
pre-registered comparison on identical artefacts -- the placebo arm changed sign
between them -- and nothing on either output said which was declared.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from council.backtest.baseline import random_targets
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
    would let the two disagree about the pre-registered comparison on identical
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
    turnover_per_period: float,
    rebalance_threshold: float,
    seed: int,
) -> pd.DataFrame:
    """Targets for the null matched to one arm's trading rate and exposure sizes.

    Raises:
        ValueError: if the arm turns over more than any shuffle of its own
            exposures can reach. Left to the caller, because the CLI reports the
            gap per arm and the dashboard reports it once on the panel; neither
            may quietly substitute a null matched to some other turnover.
    """
    return random_targets(
        opens=opens,
        target_turnover_per_period=turnover_per_period,
        rebalance_threshold=rebalance_threshold,
        seed=seed,
        # The arm's own requested exposures, so the null holds positions of the
        # same sizes and signs and differs only in when it holds them.
        exposure_pool=[exposures[point] for point in sorted(exposures)],
    )

"""A random committee that trades as often as the real one.

Without this, no number the committee produces means anything. An exposure
series that trades on 40% of sessions and holds roughly half-sized positions
earns a return in any market with a trend in it, and it earns that return
whether or not the model reasoned. The only way to know how much of a
committee's curve is skill is to run a strategy with the same *shape* and no
information, and subtract.

Same shape means two things, and matching only one of them would be worse than
matching neither, because it would look rigorous:

* **The same turnover.** Trading costs and the drift picked up by simply being
  invested both scale with how often a position changes, so a random baseline
  that churns twice as much is being punished for something other than luck.
* **The same exposure distribution.** Pass the committee's own realised
  exposures as the pool and the baseline holds positions of the same sizes and
  signs -- shuffled in time, which is precisely the null hypothesis.

Turnover is matched against what the engine *realises*, not against the raw
path, because the rebalance threshold suppresses some revisions and the
next-open fill drops the last one. Calibrating on the path would leave the
baseline quietly trading less than the arm it is meant to match.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pandas as pd

from council.backtest.engine import run_ticker


def _ticker_entropy(ticker: str) -> int:
    """A stable integer for a ticker symbol.

    The builtin ``hash`` is salted per process, so a baseline seeded with it
    would differ on every run. Keying each ticker separately also means adding a
    ticker to the universe leaves the others' paths untouched, so an expected
    value written into a test does not move when the universe changes.
    """
    return int.from_bytes(hashlib.sha256(ticker.encode("utf-8")).digest()[:8], "big")


def _hold(values: np.ndarray, revises: np.ndarray) -> np.ndarray:
    """The exposure path of an agent that revises its view only where told to.

    Before the first revision the agent has said nothing, so it is flat.
    """
    positions = np.arange(values.size)
    source = np.maximum.accumulate(np.where(revises, positions, -1))
    return np.asarray(np.where(source >= 0, values[source], 0.0), dtype=float)


def _realised_turnover_per_period(
    *, ticker: str, path: np.ndarray, opens: pd.Series, rebalance_threshold: float
) -> float:
    """Turnover this target path would actually produce, in the engine's units."""
    targets = pd.Series(path, index=pd.DatetimeIndex(opens.index))
    result = run_ticker(
        ticker=ticker,
        targets=targets,
        opens=opens,
        cost_bps=0.0,
        rebalance_threshold=rebalance_threshold,
    )
    return result.turnover / len(result.position)


def _calibrate(
    *,
    ticker: str,
    opens: pd.Series,
    values: np.ndarray,
    order: np.ndarray,
    target: float,
    rebalance_threshold: float,
) -> np.ndarray:
    """Search the number of revisions for the closest achievable turnover.

    Revision dates are added one at a time in a random order, so the paths
    examined are uniformly random subsets of the calendar of every size. The
    search is exhaustive rather than a bisection: inserting a revision between
    two others can *lower* realised turnover when the threshold then suppresses
    a step, so turnover is not monotone in the number of revisions and a
    bisection would settle on the wrong side of the fold.
    """
    revises = np.zeros(values.size, dtype=bool)

    def measure(path: np.ndarray) -> float:
        return _realised_turnover_per_period(
            ticker=ticker, path=path, opens=opens, rebalance_threshold=rebalance_threshold
        )

    best_path = _hold(values, revises)
    best_gap = abs(measure(best_path) - target)
    reachable = 0.0
    for count in range(values.size):
        revises[order[count]] = True
        path = _hold(values, revises)
        realised = measure(path)
        reachable = max(reachable, realised)
        if abs(realised - target) < best_gap:
            best_gap, best_path = abs(realised - target), path

    if target > reachable:
        raise ValueError(
            f"{ticker}: turnover of {target:.4f} per period is unreachable; "
            f"revising on every session reaches {reachable:.4f}"
        )
    return best_path


def random_targets(
    *,
    opens: pd.DataFrame,
    target_turnover_per_period: float,
    rebalance_threshold: float,
    seed: int,
    exposure_pool: Sequence[float] | None = None,
) -> pd.DataFrame:
    """Exposure targets for a random committee matched to a turnover budget.

    Args:
        opens: the trading calendar and prices the baseline will be scored on.
        target_turnover_per_period: total absolute exposure traded per period,
            the same quantity :class:`~council.backtest.metrics.PerformanceMetrics`
            reports, so the two can be compared without conversion.
        rebalance_threshold: the engine's threshold, needed because the match is
            against realised rather than intended turnover.
        seed: keyed per ticker, so the result is reproducible and stable under a
            change of universe.
        exposure_pool: exposures to draw from, with replacement -- pass the
            committee's own realised exposures to match its distribution as well
            as its trading rate. Defaults to uniform over the full range, which
            matches nothing in particular and should be treated as a weaker null.

    Returns:
        A frame shaped like ``opens``: one exposure column per ticker, indexed by
        decision date, dense and forward filled between revisions.

    Raises:
        ValueError: if the target is negative, if it is higher than revising on
            every single session can reach, or if the pool is empty or holds an
            exposure outside the range a :class:`~council.domain.signal.Signal`
            is allowed to ask for.
    """
    if target_turnover_per_period < 0.0:
        raise ValueError("target turnover cannot be negative")

    pool = np.asarray(exposure_pool, dtype=float) if exposure_pool is not None else None
    if pool is not None:
        if pool.size == 0:
            raise ValueError("exposure_pool is empty")
        # A pool outside the bound on ``Signal.exposure`` would make the control a
        # leveraged strategy rather than a null: the arm it controls for cannot
        # ask for 4x, so the gap between them would be read as the committee's
        # skill when it is only the baseline's larger book.
        if bool((np.abs(pool) > 1.0).any()) or not bool(np.isfinite(pool).all()):
            raise ValueError("exposure_pool must lie inside the range a Signal can request")

    sessions = pd.DatetimeIndex(opens.index)
    columns: dict[str, np.ndarray] = {}
    for ticker in (str(column) for column in opens.columns):
        rng = np.random.default_rng([seed, _ticker_entropy(ticker)])
        if pool is None:
            drawn = rng.uniform(-1.0, 1.0, size=len(sessions))
        else:
            drawn = rng.choice(pool, size=len(sessions))
        columns[ticker] = _calibrate(
            ticker=ticker,
            opens=opens[ticker],
            values=np.asarray(drawn, dtype=float),
            order=rng.permutation(len(sessions)),
            target=target_turnover_per_period,
            rebalance_threshold=rebalance_threshold,
        )

    return pd.DataFrame(columns, index=sessions)

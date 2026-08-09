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
from collections.abc import Mapping, Sequence
from datetime import date

import numpy as np
import numpy.typing as npt
import pandas as pd

from council.backtest.engine import run_ticker

ExposurePool = Mapping[str, Sequence[float]] | Sequence[float]
"""Exposures the null draws from: one pool per ticker, or one shared pool.

Per ticker is what "the same exposure distribution" means. A committee
systematically long one instrument and short another has, per ticker, a
distribution the cross-ticker mixture matches neither of -- and matching one half
of "same shape" is worse than matching neither, because it looks rigorous. The
flat form is kept for callers that genuinely have one population.
"""


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
    examined are uniformly random subsets of ``order`` of every size -- the whole
    calendar, or the sessions the arm holds a decision for. The
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
    for count in range(order.size):
        revises[order[count]] = True
        path = _hold(values, revises)
        realised = measure(path)
        reachable = max(reachable, realised)
        if abs(realised - target) < best_gap:
            best_gap, best_path = abs(realised - target), path

    if target > reachable:
        raise ValueError(
            f"{ticker}: turnover of {target:.4f} per period is unreachable; "
            f"revising on every revisable session reaches {reachable:.4f}"
        )
    return best_path


def _validated_pool(values: Sequence[float], *, named: str | None) -> npt.NDArray[np.float64]:
    """One ticker's draw pool, or the shared one, checked before anything is drawn."""
    where = "" if named is None else f" for {named}"
    pool = np.asarray(values, dtype=float)
    if pool.size == 0:
        raise ValueError(f"exposure_pool is empty{where}")
    # A pool outside the bound on ``Signal.exposure`` would make the control a
    # leveraged strategy rather than a null: the arm it controls for cannot ask for
    # 4x, so the gap between them would be read as the committee's skill when it is
    # only the baseline's larger book.
    if bool((np.abs(pool) > 1.0).any()) or not bool(np.isfinite(pool).all()):
        raise ValueError(f"exposure_pool must lie inside the range a Signal can request{where}")
    return pool


def _pools_by_ticker(
    exposure_pool: ExposurePool | None, *, tickers: Sequence[str]
) -> dict[str, npt.NDArray[np.float64]] | None:
    """The pool each column draws from, or ``None`` for the uniform default.

    A mapping is taken per ticker, which is what makes the null's distribution the
    ticker's own rather than the cross-ticker mixture. A flat sequence is shared by
    every column, which is what the earlier signature always did.

    Raises:
        ValueError: if a mapping has no entry for a column of ``opens``. Falling
            back to some other ticker's exposures, or to the uniform default, would
            leave one column matched and the rest not, with nothing on the output
            saying which.
    """
    if exposure_pool is None:
        return None
    if isinstance(exposure_pool, Mapping):
        missing = [ticker for ticker in tickers if ticker not in exposure_pool]
        if missing:
            raise ValueError(
                "exposure_pool holds no exposures for " + ", ".join(missing)
                + "; the null draws each ticker's column from that ticker's own pool"
            )
        return {
            ticker: _validated_pool(exposure_pool[ticker], named=ticker) for ticker in tickers
        }
    shared = _validated_pool(exposure_pool, named=None)
    return {ticker: shared for ticker in tickers}


TurnoverTarget = float | Mapping[str, float]
"""The rate the null is matched to: one target per ticker, or one shared.

Per ticker for the reason :data:`ExposurePool` is per ticker. Turnover is realised
per column -- :func:`_calibrate` searches one ticker's revision dates against one
number -- while :attr:`council.backtest.metrics.PerformanceMetrics.turnover` is a
mean across the basket, so a single scalar matches each column to the basket
average rather than to its own rate. That is the same half of "same shape" this
module already refuses for the exposure pool, and it has a second cost: the
reachability check in :func:`_calibrate` runs per ticker against the wrong number,
so the lower-turnover ticker can be declared unreachable and drop the whole
baseline. The scalar form is kept for callers that genuinely have one population.
"""


def _targets_by_ticker(
    target: TurnoverTarget, *, tickers: Sequence[str]
) -> dict[str, float]:
    """The turnover each column is matched to.

    Raises:
        ValueError: if a mapping has no entry for a column of ``opens``, mirroring
            :func:`_pools_by_ticker`. Falling back to the basket mean, or to another
            ticker's rate, would leave one column matched and the rest not with
            nothing on the output saying which.
    """
    if isinstance(target, Mapping):
        missing = [ticker for ticker in tickers if ticker not in target]
        if missing:
            raise ValueError(
                "target_turnover_per_period holds no rate for " + ", ".join(missing)
                + "; the null matches each ticker's column to that ticker's own rate"
            )
        return {ticker: float(target[ticker]) for ticker in tickers}
    return {ticker: float(target) for ticker in tickers}


def random_targets(
    *,
    opens: pd.DataFrame,
    target_turnover_per_period: TurnoverTarget,
    rebalance_threshold: float,
    seed: int,
    exposure_pool: ExposurePool | None = None,
    revisable: Sequence[date] | None = None,
) -> pd.DataFrame:
    """Exposure targets for a random committee matched to a turnover budget.

    Args:
        opens: the trading calendar and prices the baseline will be scored on.
        target_turnover_per_period: total absolute exposure traded per period. A
            mapping of ticker to rate matches each column to that ticker's own
            realised turnover, which is what "the same turnover" means when the
            committee trades one instrument harder than another; a scalar is shared
            by every column, so per ticker the null is matched to the cross-ticker
            mean rather than to the ticker's own rate.
            :class:`~council.backtest.metrics.PerformanceMetrics` reports the
            basket mean, so a caller with per-ticker results should pass them.
        rebalance_threshold: the engine's threshold, needed because the match is
            against realised rather than intended turnover.
        seed: keyed per ticker, so the result is reproducible and stable under a
            change of universe.
        exposure_pool: exposures to draw from, with replacement -- pass the
            committee's own realised exposures to match its distribution as well
            as its trading rate. A mapping of ticker to pool draws each column from
            that ticker's own exposures, which is what "the same exposure
            distribution" means when the committee is systematically long one
            instrument and short another; a flat sequence is shared by every column,
            so per ticker the null's distribution is the cross-ticker mixture rather
            than the ticker's own. Defaults to uniform over the full range, which
            matches nothing in particular and should be treated as a weaker null.
        revisable: the sessions on which the null is allowed to revise, ordinarily
            the dates the arm holds a decision for. ``None`` draws revision dates
            from the whole calendar, which is the third half of "same shape" that
            the module docstring names and this argument exists to supply. Every
            arm is flat over the ``lookback_days - 1`` warm-up sessions, because no
            decision exists there; :func:`_hold` forward-fills from the first
            revision, so a null free to revise at session zero is invested through
            a warm-up the arm sits out, and the drift over it is credited to the
            null alone. Turnover and the exposure distribution match either way,
            which is precisely what makes the gap invisible.

    Returns:
        A frame shaped like ``opens``: one exposure column per ticker, indexed by
        decision date, dense and forward filled between revisions.

    Raises:
        ValueError: if the target is negative, if it is higher than revising on
            every single session can reach, if a mapping has no entry for one of
            ``opens``' columns, or if a pool is empty or holds an exposure outside
            the range a :class:`~council.domain.signal.Signal` is allowed to ask for.
    """
    tickers = [str(column) for column in opens.columns]
    targets = _targets_by_ticker(target_turnover_per_period, tickers=tickers)
    if any(target < 0.0 for target in targets.values()):
        raise ValueError("target turnover cannot be negative")

    pools = _pools_by_ticker(exposure_pool, tickers=tickers)

    sessions = pd.DatetimeIndex(opens.index)
    if revisable is None:
        eligible = np.arange(len(sessions))
    else:
        eligible = np.flatnonzero(sessions.isin(pd.DatetimeIndex(list(revisable))))
        if eligible.size == 0:
            raise ValueError("no revisable session lies on the backtest calendar")

    columns: dict[str, np.ndarray] = {}
    for ticker in tickers:
        rng = np.random.default_rng([seed, _ticker_entropy(ticker)])
        if pools is None:
            drawn = rng.uniform(-1.0, 1.0, size=len(sessions))
        else:
            drawn = rng.choice(pools[ticker], size=len(sessions))
        columns[ticker] = _calibrate(
            ticker=ticker,
            opens=opens[ticker],
            values=np.asarray(drawn, dtype=float),
            order=rng.permutation(eligible),
            target=targets[ticker],
            rebalance_threshold=rebalance_threshold,
        )

    return pd.DataFrame(columns, index=sessions)

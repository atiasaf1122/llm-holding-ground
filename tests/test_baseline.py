"""The random baseline, checked on the two properties that make it a baseline.

If it does not trade at the committee's rate, it is not a control -- it is a
different strategy, and the difference between them will be read as skill. If it
is not reproducible, the comparison cannot be rerun. Everything else about it is
allowed to be arbitrary, because that is the point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from council.backtest.baseline import random_targets
from council.backtest.engine import run_backtest
from council.backtest.metrics import evaluate

SEED = 20260101
THRESHOLD = 0.05


def opens_frame(tickers: tuple[str, ...] = ("AAA", "BBB"), sessions: int = 90) -> pd.DataFrame:
    """Flat prices: nothing here is a test of returns, only of trading rate."""
    calendar = pd.bdate_range("2022-01-03", periods=sessions)
    return pd.DataFrame({ticker: [100.0] * sessions for ticker in tickers}, index=calendar)


def realised_turnover_per_period(opens: pd.DataFrame, targets: pd.DataFrame) -> float:
    result = run_backtest(
        targets=targets, opens=opens, cost_bps=0.0, rebalance_threshold=THRESHOLD
    )
    return evaluate(result).turnover_per_period


@pytest.mark.parametrize("target", [0.05, 0.2, 0.4])
def test_realised_turnover_lands_on_the_requested_budget(target: float) -> None:
    opens = opens_frame()

    targets = random_targets(
        opens=opens,
        target_turnover_per_period=target,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )

    # The path is discrete -- turnover moves in steps of one revision -- so an
    # exact hit is not always available. A tenth of the budget is far tighter
    # than the difference any committee result would need to survive.
    assert realised_turnover_per_period(opens, targets) == pytest.approx(target, rel=0.1)


def test_a_zero_budget_produces_a_strategy_that_never_trades() -> None:
    opens = opens_frame()

    targets = random_targets(
        opens=opens,
        target_turnover_per_period=0.0,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )

    assert not targets.to_numpy().any()
    assert realised_turnover_per_period(opens, targets) == 0.0


def test_a_budget_no_amount_of_trading_could_reach_raises_rather_than_undershooting() -> None:
    # Silently returning the closest available path would leave the baseline
    # trading a third as often as the arm it is supposed to control for, and
    # nothing downstream would say so.
    with pytest.raises(ValueError, match="unreachable"):
        random_targets(
            opens=opens_frame(sessions=20),
            target_turnover_per_period=5.0,
            rebalance_threshold=THRESHOLD,
            seed=SEED,
        )


def test_a_negative_budget_raises() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        random_targets(
            opens=opens_frame(sessions=10),
            target_turnover_per_period=-0.1,
            rebalance_threshold=THRESHOLD,
            seed=SEED,
        )


def seeded(opens: pd.DataFrame, seed: int) -> pd.DataFrame:
    return random_targets(
        opens=opens,
        target_turnover_per_period=0.2,
        rebalance_threshold=THRESHOLD,
        seed=seed,
    )


def test_the_same_seed_reproduces_the_same_targets_exactly() -> None:
    opens = opens_frame()

    pd.testing.assert_frame_equal(seeded(opens, SEED), seeded(opens, SEED))


def test_a_different_seed_produces_a_different_path() -> None:
    opens = opens_frame()

    assert not seeded(opens, SEED).equals(seeded(opens, SEED + 1))


def test_adding_a_ticker_leaves_the_others_paths_untouched() -> None:
    # Each ticker draws from a stream keyed by its own symbol. Without that,
    # every expected value in the suite would move whenever the universe did.
    two = random_targets(
        opens=opens_frame(("AAA", "BBB")),
        target_turnover_per_period=0.2,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )
    three = random_targets(
        opens=opens_frame(("AAA", "BBB", "CCC")),
        target_turnover_per_period=0.2,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )

    pd.testing.assert_series_equal(two["AAA"], three["AAA"])


def test_two_tickers_do_not_share_one_exposure_path() -> None:
    # The seed is keyed by ticker. Seeding the whole frame once instead would give
    # every ticker a byte-identical exposure series, and a perfectly correlated
    # basket has far less dispersion than the null this baseline is meant to be --
    # so every Sharpe and drawdown compared against it would be measured against
    # the wrong distribution. Reproducibility alone does not catch that; both a
    # keyed and an unkeyed stream reproduce.
    targets = random_targets(
        opens=opens_frame(("AAA", "BBB")),
        target_turnover_per_period=0.3,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )

    assert not np.array_equal(targets["AAA"].to_numpy(), targets["BBB"].to_numpy())


def test_the_baseline_path_does_not_depend_on_the_realised_prices() -> None:
    # The null is information free only if its trading is decided without looking
    # at what the prices did. Nothing but this test stands between that and a
    # future calibration objective that reads returns or costs -- which would tune
    # the control on the very outcome it exists to control for, silently.
    #
    # The two paths are as far apart as prices get: one never moves, the other
    # swings 40% a session around an uptrend. A price term in the objective would
    # score the same candidate paths differently under the two.
    calendar = pd.bdate_range("2022-01-03", periods=90)
    flat = pd.DataFrame({"AAA": [100.0] * 90}, index=calendar)
    swings = 1.02 + 0.4 * np.sin(np.arange(90, dtype=float))
    volatile = pd.DataFrame({"AAA": 100.0 * np.cumprod(swings)}, index=calendar)

    def targets_for(opens: pd.DataFrame) -> pd.DataFrame:
        return random_targets(
            opens=opens,
            target_turnover_per_period=0.2,
            rebalance_threshold=THRESHOLD,
            seed=SEED,
        )

    pd.testing.assert_frame_equal(targets_for(flat), targets_for(volatile))


@pytest.mark.parametrize("pool", [None, [-0.75, 0.25, 0.75]])
def test_exposures_stay_inside_the_range_a_signal_is_allowed_to_ask_for(
    pool: list[float] | None,
) -> None:
    targets = random_targets(
        opens=opens_frame(),
        target_turnover_per_period=0.3,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
        exposure_pool=pool,
    )

    values = targets.to_numpy()
    assert values.min() >= -1.0
    assert values.max() <= 1.0


@pytest.mark.parametrize("pool", [[-4.0, 4.0], [0.5, 1.5], [0.5, np.nan]])
def test_a_pool_outside_what_a_signal_may_ask_for_raises(pool: list[float]) -> None:
    # Accepting it would make the "null" a leveraged strategy, and the arm it
    # controls for cannot take that leverage -- so the gap between the two would
    # be reported as the committee's skill.
    with pytest.raises(ValueError, match="inside the range a Signal"):
        random_targets(
            opens=opens_frame(sessions=20),
            target_turnover_per_period=0.2,
            rebalance_threshold=THRESHOLD,
            seed=SEED,
            exposure_pool=pool,
        )


def test_an_exposure_pool_is_the_only_thing_the_baseline_ever_holds() -> None:
    # Passing the committee's own realised exposures is what makes this a null
    # hypothesis about timing rather than a different strategy entirely.
    pool = [-0.75, 0.75]

    targets = random_targets(
        opens=opens_frame(),
        target_turnover_per_period=0.2,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
        exposure_pool=pool,
    )

    held = set(np.unique(targets.to_numpy()))
    # Zero appears before the first revision, when the agent has said nothing.
    assert held <= {-0.75, 0.0, 0.75}
    assert held & {-0.75, 0.75}


def test_an_empty_exposure_pool_raises() -> None:
    with pytest.raises(ValueError, match="pool is empty"):
        random_targets(
            opens=opens_frame(sessions=10),
            target_turnover_per_period=0.2,
            rebalance_threshold=THRESHOLD,
            seed=SEED,
            exposure_pool=[],
        )


def test_the_budget_is_matched_against_what_the_engine_realises_not_the_raw_path() -> None:
    # A large threshold suppresses revisions the raw path contains, so a
    # baseline calibrated on the path would trade far less than it claimed.
    opens = opens_frame()

    targets = random_targets(
        opens=opens,
        target_turnover_per_period=0.15,
        rebalance_threshold=0.4,
        seed=SEED,
    )
    result = run_backtest(
        targets=targets, opens=opens, cost_bps=0.0, rebalance_threshold=0.4
    )

    assert evaluate(result).turnover_per_period == pytest.approx(0.15, rel=0.1)

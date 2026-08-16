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

from council.arms import random_arm_targets
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
    result = run_backtest(targets=targets, opens=opens, cost_bps=0.0, rebalance_threshold=THRESHOLD)
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
    result = run_backtest(targets=targets, opens=opens, cost_bps=0.0, rebalance_threshold=0.4)

    assert evaluate(result).turnover_per_period == pytest.approx(0.15, rel=0.1)


# -- the third half of "same shape": the sessions the arm is flat over ------------


def test_the_null_can_be_confined_to_the_sessions_the_arm_decides_on() -> None:
    # Every arm is flat over the `lookback_days - 1` warm-up because no decision
    # exists there. `_hold` forward-fills from the first revision, so a null drawing
    # revision dates from the whole calendar is invested through a warm-up the arm
    # sits out -- with turnover and exposure distribution matched either way.
    opens = opens_frame()
    warm_up = 30
    revisable = [day.date() for day in opens.index[warm_up:]]

    targets = random_targets(
        opens=opens,
        target_turnover_per_period=0.2,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
        exposure_pool=[-0.75, 0.75],
        revisable=revisable,
    )

    assert (targets.iloc[:warm_up].to_numpy() == 0.0).all()
    assert (targets.iloc[warm_up:].to_numpy() != 0.0).any()


def test_the_unconfined_null_is_the_one_that_trades_in_the_warm_up() -> None:
    # The other half: without the argument the behaviour is unchanged, which is what
    # makes this a defect the caller has to opt out of rather than a change of null.
    opens = opens_frame()

    targets = random_targets(
        opens=opens,
        target_turnover_per_period=0.2,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
        exposure_pool=[-0.75, 0.75],
    )

    assert (targets.iloc[:30].to_numpy() != 0.0).any()


def test_the_arms_null_revises_only_where_the_arm_holds_a_decision() -> None:
    # The caller `council.scoring.score_arm` reaches: the exposure mapping starts
    # after the warm-up, so the null must be flat over the same sessions the arm is.
    from council.arms import random_arm_targets

    opens = opens_frame()
    warm_up = 30
    exposures = {
        (day.date(), ticker): 0.75 if index % 2 else -0.75
        for index, day in enumerate(opens.index[warm_up:])
        for ticker in ("AAA", "BBB")
    }

    targets = random_arm_targets(
        exposures=exposures,
        opens=opens,
        turnover_per_period=0.01,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )

    assert (targets.iloc[:warm_up].to_numpy() == 0.0).all()


def test_a_revisable_calendar_that_misses_the_backtest_calendar_is_refused() -> None:
    with pytest.raises(ValueError, match="no revisable session"):
        random_targets(
            opens=opens_frame(sessions=10),
            target_turnover_per_period=0.2,
            rebalance_threshold=THRESHOLD,
            seed=SEED,
            revisable=[pd.Timestamp("1999-01-04").date()],
        )


# -- the same exposure distribution means per ticker -------------------------------


def test_each_ticker_draws_from_its_own_pool_and_not_from_the_mixture() -> None:
    # "Same shape" is turnover *and* exposure distribution. One flat pool over every
    # instrument gives each column the cross-ticker mixture, which for a committee
    # systematically long one and short another is neither ticker's distribution --
    # half of "same shape", which this module says is worse than neither because it
    # looks rigorous.
    opens = opens_frame()

    # A budget the narrow pools can reach: two exposures 0.2 apart cannot churn
    # as hard as the [-0.75, 0.75] pool the tests above use.
    targets = random_targets(
        opens=opens,
        target_turnover_per_period=0.02,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
        exposure_pool={"AAA": [0.6, 0.8], "BBB": [-0.6, -0.8]},
    )

    # Zero appears before the first revision, when the agent has said nothing.
    assert set(np.unique(targets["AAA"].to_numpy())) <= {0.0, 0.6, 0.8}
    assert set(np.unique(targets["BBB"].to_numpy())) <= {0.0, -0.6, -0.8}
    assert set(np.unique(targets["AAA"].to_numpy())) & {0.6, 0.8}
    assert set(np.unique(targets["BBB"].to_numpy())) & {-0.6, -0.8}


def test_a_flat_pool_reaches_every_ticker_as_it_always_did() -> None:
    # The other half: the sequence form is what existing callers pass, and it must
    # keep meaning one population shared by every column.
    opens = opens_frame()

    targets = random_targets(
        opens=opens,
        target_turnover_per_period=0.2,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
        exposure_pool=[-0.75, 0.75],
    )

    for ticker in ("AAA", "BBB"):
        assert set(np.unique(targets[ticker].to_numpy())) <= {-0.75, 0.0, 0.75}


def test_a_mapping_missing_a_column_of_opens_raises() -> None:
    # Falling back to another ticker's exposures, or to the uniform default, would
    # leave one column matched and the rest not, with nothing on the output saying
    # which.
    with pytest.raises(ValueError, match="no exposures for BBB"):
        random_targets(
            opens=opens_frame(sessions=20),
            target_turnover_per_period=0.2,
            rebalance_threshold=THRESHOLD,
            seed=SEED,
            exposure_pool={"AAA": [0.5]},
        )


def test_one_tickers_empty_pool_is_named_rather_than_reported_as_the_arms() -> None:
    with pytest.raises(ValueError, match="pool is empty for BBB"):
        random_targets(
            opens=opens_frame(sessions=20),
            target_turnover_per_period=0.2,
            rebalance_threshold=THRESHOLD,
            seed=SEED,
            exposure_pool={"AAA": [0.5], "BBB": []},
        )


def test_the_arm_null_keeps_each_instruments_own_signs() -> None:
    # The caller half of the same defect: `random_arm_targets` handed one flat pool
    # built from the arm's exposures across *every* ticker, so a committee
    # systematically long one instrument and short another got a null that went both
    # ways in both columns.
    opens = opens_frame()
    days = [day.date() for day in opens.index]
    exposures = {
        **{(day, "AAA"): 0.2 if index % 2 else 0.9 for index, day in enumerate(days)},
        **{(day, "BBB"): -0.2 if index % 2 else -0.9 for index, day in enumerate(days)},
    }

    targets = random_arm_targets(
        exposures=exposures,
        opens=opens,
        turnover_per_period=0.02,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )

    assert targets["AAA"].min() >= 0.0
    assert targets["BBB"].max() <= 0.0


# -- one turnover per ticker, not one for the basket --------------------------------


def per_ticker_turnover(opens: pd.DataFrame, targets: pd.DataFrame) -> dict[str, float]:
    """What each column of a target frame actually realises, in the engine's units."""
    result = run_backtest(targets=targets, opens=opens, cost_bps=0.0, rebalance_threshold=THRESHOLD)
    return {ticker.ticker: ticker.turnover / len(ticker.position) for ticker in result.per_ticker}


def test_each_column_is_matched_to_its_own_tickers_rate_and_not_the_basket_mean() -> None:
    # The other half of "same shape". Turnover is realised per column -- `_calibrate`
    # searches one ticker's revision dates against one number -- while
    # `PerformanceMetrics.turnover` is a mean across the basket, so passing that
    # scalar matched each column to the average rather than to its own ticker's
    # rate. That is the same defect this module already rejected for `ExposurePool`.
    opens = opens_frame()

    targets = random_targets(
        opens=opens,
        target_turnover_per_period={"AAA": 0.30, "BBB": 0.08},
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )
    realised = per_ticker_turnover(opens, targets)

    assert realised["AAA"] == pytest.approx(0.30, rel=0.1)
    assert realised["BBB"] == pytest.approx(0.08, rel=0.1)


def test_a_scalar_target_still_reaches_every_column() -> None:
    # The scalar form is what callers with one genuine population pass, and it must
    # keep meaning one rate shared by every column.
    opens = opens_frame()

    targets = random_targets(
        opens=opens,
        target_turnover_per_period=0.2,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )
    realised = per_ticker_turnover(opens, targets)

    for ticker in ("AAA", "BBB"):
        assert realised[ticker] == pytest.approx(0.2, rel=0.1)


def test_a_turnover_mapping_missing_a_column_of_opens_raises() -> None:
    # Mirroring `exposure_pool`: falling back to the basket mean, or to another
    # ticker's rate, would leave one column matched and the rest not.
    with pytest.raises(ValueError, match="no rate for BBB"):
        random_targets(
            opens=opens_frame(sessions=20),
            target_turnover_per_period={"AAA": 0.2},
            rebalance_threshold=THRESHOLD,
            seed=SEED,
        )


def test_the_arm_null_matches_each_instruments_own_trading_rate() -> None:
    # The caller half. A committee that revises one ticker on every session and the
    # other on a quarter of them has no single rate: matched to the mean, the busy
    # column trades too little and the quiet one is asked for more than any shuffle
    # of its own exposures can reach -- which raises and drops the whole baseline,
    # leaving `ArmOutcome.baseline is None` and a dash in the CLI's `random Sharpe`
    # column for the control the declared secondary comparison is stated against.
    opens = opens_frame()
    days = [day.date() for day in opens.index]
    # AAA changes its mind every fourth session, BBB every twelfth: two rates the
    # basket mean sits between and matches neither of.
    exposures = {
        **{(day, "AAA"): 0.9 if (index // 4) % 2 else 0.4 for index, day in enumerate(days)},
        **{(day, "BBB"): 0.5 if (index // 12) % 2 else 0.2 for index, day in enumerate(days)},
    }
    arm = run_backtest(
        targets=pd.DataFrame(
            {ticker: [exposures[(day, ticker)] for day in days] for ticker in ("AAA", "BBB")},
            index=opens.index,
        ),
        opens=opens,
        cost_bps=0.0,
        rebalance_threshold=THRESHOLD,
    )
    wanted = {ticker.ticker: ticker.turnover / len(ticker.position) for ticker in arm.per_ticker}
    assert wanted["AAA"] != pytest.approx(wanted["BBB"], rel=0.1), "the tickers must differ"

    targets = random_arm_targets(
        exposures=exposures,
        opens=opens,
        turnover_per_period=wanted,
        rebalance_threshold=THRESHOLD,
        seed=SEED,
    )
    realised = per_ticker_turnover(opens, targets)

    for ticker in ("AAA", "BBB"):
        assert realised[ticker] == pytest.approx(wanted[ticker], rel=0.15), ticker

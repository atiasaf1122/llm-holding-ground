"""What thinning the contested set may and may not change."""

from __future__ import annotations

from datetime import date, timedelta
from itertools import pairwise

import pytest

from council.evaluation.dispersion import Dispersion
from council.sampling import thin_contested

START = date(2022, 1, 3)


def _point(offset: int, ticker: str) -> Dispersion:
    """A contested point, distinguished only by its date and ticker."""
    return Dispersion(
        decision_date=START + timedelta(days=offset),
        ticker=ticker,
        agent_count=4,
        exposure_std=0.4,
        long_count=2,
        short_count=2,
        flat_count=0,
    )


def _grid(days: int, tickers: tuple[str, ...] = ("AAPL", "XOM")) -> tuple[Dispersion, ...]:
    """Every ticker on every day, ordered as the pipeline orders points."""
    return tuple(_point(offset, ticker) for offset in range(days) for ticker in tickers)


def _dates(points: tuple[Dispersion, ...], ticker: str) -> list[date]:
    return [point.decision_date for point in points if point.ticker == ticker]


# -- the budget ------------------------------------------------------------------


def test_no_budget_keeps_everything() -> None:
    points = _grid(days=50)
    assert thin_contested(points, keep=None) == points


def test_budget_above_the_offer_keeps_everything() -> None:
    points = _grid(days=10)
    assert thin_contested(points, keep=500) == points


def test_empty_input_survives_a_budget() -> None:
    assert thin_contested((), keep=10) == ()


def test_budget_is_honoured_within_rounding() -> None:
    kept = thin_contested(_grid(days=400), keep=200)
    # Per-ticker quotas are rounded, and a ticker's quota is floored at one, so the
    # total can miss by a point or two. It may not miss by more.
    assert abs(len(kept) - 200) <= 2


def test_a_budget_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        thin_contested(_grid(days=10), keep=0)


# -- what the pick may not do ----------------------------------------------------


def test_thinning_does_not_reorder() -> None:
    """The sweep reads this straight through, so it is a filter and nothing else."""
    points = _grid(days=200)
    kept = thin_contested(points, keep=60)
    assert list(kept) == [point for point in points if point in kept]


def test_thinning_invents_nothing() -> None:
    points = _grid(days=200)
    assert set(thin_contested(points, keep=60)) <= set(points)


def test_the_pick_is_deterministic() -> None:
    points = _grid(days=300)
    assert thin_contested(points, keep=77) == thin_contested(points, keep=77)


# -- what the pick must do -------------------------------------------------------


def test_the_sample_spans_the_whole_study() -> None:
    """The reason for sampling instead of shortening the period.

    A pick inset from either end would be a shorter study at higher density, which
    is exactly the economy `sampling` exists to avoid. Anchoring is checked per
    ticker because each is thinned on its own calendar.
    """
    points = _grid(days=500)
    kept = thin_contested(points, keep=120)
    for ticker in ("AAPL", "XOM"):
        assert _dates(kept, ticker)[0] == _dates(points, ticker)[0]
        assert _dates(kept, ticker)[-1] == _dates(points, ticker)[-1]


def test_the_sample_does_not_cluster() -> None:
    """Spread over the range, not merely spanning it: a prefix plus the last day spans too.

    Bounded rather than uniform. The bisection buys the superset property below at
    the cost of exact spacing: between one bisection and the next the widest gap is
    about twice the narrowest, and the halves of an odd gap differ by one, so the
    bound carries a ``+ 1``. This asserts the bound, which is the guarantee, rather
    than equality, which is not.
    """
    kept = thin_contested(_grid(days=400), keep=80)
    gaps = [(later - earlier).days for earlier, later in pairwise(_dates(kept, "AAPL"))]
    assert max(gaps) <= 2 * min(gaps) + 1


def test_a_larger_budget_keeps_everything_a_smaller_one_chose() -> None:
    """What makes an interrupted run extendable instead of restartable.

    The sweep skips a conversation it has already stored, so raising the budget
    spends the night on new points only -- but only if the new pick contains the old
    one. An evenly spaced pick does not: it would silently re-debate a different set
    and leave the first night's work stranded in the parquet.
    """
    points = _grid(days=500)
    smaller = set(thin_contested(points, keep=60))
    assert smaller <= set(thin_contested(points, keep=120))


def test_the_sample_grows_by_refinement_at_every_step() -> None:
    points = _grid(days=200)
    for budget in range(2, 40):
        assert set(thin_contested(points, keep=budget)) <= set(
            thin_contested(points, keep=budget + 1)
        )


def test_every_ticker_survives_a_tiny_budget() -> None:
    """Dropping a ticker changes the design, not the precision."""
    kept = thin_contested(_grid(days=300, tickers=("AAPL", "XOM", "JNJ")), keep=3)
    assert {point.ticker for point in kept} == {"AAPL", "XOM", "JNJ"}


def test_a_busier_ticker_keeps_more_of_the_budget() -> None:
    """Quotas are proportional, so a ticker that disagrees more is debated more."""
    points = tuple(_point(offset, "AAPL") for offset in range(300)) + tuple(
        _point(offset, "XOM") for offset in range(60)
    )
    kept = thin_contested(points, keep=90)
    assert len(_dates(kept, "AAPL")) > 3 * len(_dates(kept, "XOM"))


def test_a_ticker_with_one_contested_point_keeps_it() -> None:
    points = (*(_point(offset, "AAPL") for offset in range(200)), _point(5, "XOM"))
    kept = thin_contested(points, keep=20)
    assert _dates(kept, "XOM") == [START + timedelta(days=5)]

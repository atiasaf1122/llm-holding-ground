"""Where the agents disagree, and whether that is worth a debate.

The load-bearing case is the narrow directional split: four agents a hair either
side of zero have almost no standard deviation and the sharpest disagreement the
personas can express. A gate that looked only at spread would throw exactly those
days away.
"""

from __future__ import annotations

from typing import Any

import pytest

from council.config import get_settings
from council.evaluation.dispersion import (
    contested_points,
    contested_share,
    dispersion_by_point,
    is_contested,
)
from helpers_decisions import DAY, NEXT_DAY, frame_of, row


def agent(
    exposure: float,
    *,
    model: str = "alpha",
    persona: str = "momentum-bold",
    **kwargs: Any,
) -> dict[str, Any]:
    return row(exposure=exposure, model=model, persona=persona, **kwargs)


# -- the measurement ---------------------------------------------------------


def test_agents_who_all_say_the_same_thing_have_no_spread() -> None:
    frame = frame_of(
        agent(0.5, model="alpha"),
        agent(0.5, model="beta"),
        agent(0.5, model="gamma"),
    )

    (measured,) = dispersion_by_point(frame)

    assert measured.exposure_std == 0.0
    assert measured.agent_count == 3
    assert measured.long_count == 3
    assert measured.short_count == 0
    assert measured.is_split is False


def test_a_committee_of_one_has_no_spread_rather_than_an_undefined_one() -> None:
    # The sample standard deviation is undefined for n=1; the population form is 0.
    (measured,) = dispersion_by_point(frame_of(agent(0.8)))

    assert measured.agent_count == 1
    assert measured.exposure_std == 0.0


def test_the_spread_is_the_population_standard_deviation() -> None:
    # Mean 0.0; deviations +1, -1; population sd = 1.0. The sample form would be 1.414.
    frame = frame_of(agent(1.0, model="alpha"), agent(-1.0, model="beta"))

    (measured,) = dispersion_by_point(frame)

    assert measured.exposure_std == pytest.approx(1.0)


def test_flat_agents_count_as_neither_long_nor_short() -> None:
    frame = frame_of(
        agent(0.0, model="alpha"),
        agent(0.0, model="beta"),
        agent(0.3, model="gamma"),
    )

    (measured,) = dispersion_by_point(frame)

    assert (measured.long_count, measured.short_count, measured.flat_count) == (1, 0, 2)


def test_the_minority_is_the_smaller_side_of_a_split() -> None:
    frame = frame_of(
        agent(0.5, model="alpha"),
        agent(0.5, model="beta"),
        agent(0.5, model="gamma"),
        agent(-0.5, model="delta"),
    )

    (measured,) = dispersion_by_point(frame)

    assert measured.is_split is True
    assert measured.minority_count == 1


# -- degenerate frames -------------------------------------------------------


def test_an_empty_frame_yields_no_points() -> None:
    assert dispersion_by_point(frame_of()) == ()


def test_an_empty_frame_has_no_contested_share_rather_than_dividing_by_zero() -> None:
    assert contested_share(frame_of()) == 0.0


def test_points_are_returned_in_date_then_ticker_order() -> None:
    frame = frame_of(
        agent(0.1, on=NEXT_DAY, ticker="XOM"),
        agent(0.1, on=DAY, ticker="XOM"),
        agent(0.1, on=DAY, ticker="AAPL"),
    )

    points = dispersion_by_point(frame)

    assert [point.point for point in points] == [
        (DAY, "AAPL"),
        (DAY, "XOM"),
        (NEXT_DAY, "XOM"),
    ]


def test_the_same_agent_twice_at_one_point_is_refused() -> None:
    # Two rows for one agent means the frame spans two arms or two rounds, and the
    # spread would then be measured across conditions rather than across agents.
    frame = frame_of(
        agent(0.5, round_index=0),
        agent(-0.5, round_index=1),
    )

    with pytest.raises(ValueError, match="appears twice"):
        dispersion_by_point(frame)


# -- the gate ----------------------------------------------------------------


def test_a_wide_spread_is_contested() -> None:
    frame = frame_of(agent(1.0, model="alpha"), agent(-1.0, model="beta"))

    (measured,) = dispersion_by_point(frame)

    assert is_contested(measured, threshold=0.25) is True


def test_a_narrow_agreement_is_not_worth_debating() -> None:
    frame = frame_of(agent(0.50, model="alpha"), agent(0.55, model="beta"))

    (measured,) = dispersion_by_point(frame)

    assert is_contested(measured, threshold=0.25) is False


def test_a_directional_split_is_contested_however_small_the_numbers_are() -> None:
    # Standard deviation of 0.01, and the sharpest disagreement the personas have.
    frame = frame_of(agent(0.01, model="alpha"), agent(-0.01, model="beta"))

    (measured,) = dispersion_by_point(frame)

    assert measured.exposure_std == pytest.approx(0.01)
    assert is_contested(measured, threshold=0.25) is True


def test_a_spread_exactly_at_the_threshold_is_not_contested() -> None:
    frame = frame_of(agent(0.7, model="alpha"), agent(0.2, model="beta"))

    (measured,) = dispersion_by_point(frame)

    assert measured.exposure_std == pytest.approx(0.25)
    assert is_contested(measured, threshold=0.25) is False


def test_omitting_the_threshold_uses_the_one_declared_in_config() -> None:
    frame = frame_of(agent(0.9, model="alpha"), agent(0.4, model="beta"))
    (measured,) = dispersion_by_point(frame)

    assert is_contested(measured) is is_contested(
        measured, threshold=get_settings().dispersion_threshold
    )


def test_only_contested_points_are_selected_for_debate() -> None:
    frame = frame_of(
        agent(0.50, on=DAY, model="alpha"),
        agent(0.55, on=DAY, model="beta"),
        agent(1.00, on=NEXT_DAY, model="alpha"),
        agent(-1.00, on=NEXT_DAY, model="beta"),
    )

    selected = contested_points(frame, threshold=0.25)

    assert [point.decision_date for point in selected] == [NEXT_DAY]
    assert contested_share(frame, threshold=0.25) == pytest.approx(0.5)

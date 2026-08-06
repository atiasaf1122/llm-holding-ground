"""The banding shared by the calibration curve and the shift-rate curve.

They are only readable against each other if they are cut at the same places, so
the edge behaviour is pinned here once rather than assumed twice.
"""

from __future__ import annotations

import pytest

from council.evaluation.buckets import DEFAULT_EDGES, band_index, make_bands


def test_the_default_bands_partition_the_unit_interval() -> None:
    bands = make_bands()

    assert len(bands) == len(DEFAULT_EDGES) - 1
    assert bands[0].lower == 0.0
    assert bands[-1].upper == 1.0


def test_a_value_on_an_internal_edge_belongs_to_the_upper_band() -> None:
    bands = make_bands()

    assert band_index(bands, 0.2) == 1
    assert bands[0].contains(0.2) is False
    assert bands[1].contains(0.2) is True


def test_the_last_band_closes_so_that_total_confidence_lands_somewhere() -> None:
    bands = make_bands()

    assert band_index(bands, 1.0) == len(bands) - 1


def test_a_value_below_every_band_falls_outside() -> None:
    assert band_index(make_bands(), -0.1) is None


def test_a_value_above_every_band_falls_outside() -> None:
    assert band_index(make_bands(), 1.1) is None


def test_a_single_band_is_legal() -> None:
    (band,) = make_bands((0.0, 1.0))

    assert band.contains(0.0) is True
    assert band.contains(1.0) is True


def test_one_edge_cannot_form_a_band() -> None:
    with pytest.raises(ValueError, match="at least two edges"):
        make_bands((0.5,))


@pytest.mark.parametrize("edges", [(0.0, 0.0, 1.0), (0.0, 0.6, 0.4)], ids=["equal", "descending"])
def test_edges_that_do_not_strictly_increase_are_refused(edges: tuple[float, ...]) -> None:
    # An empty band would appear in every report as a bucket nobody ever occupied.
    with pytest.raises(ValueError, match="strictly increase"):
        make_bands(edges)


def test_a_bands_label_shows_which_end_is_open() -> None:
    bands = make_bands()

    assert bands[0].label == "[0.00, 0.20)"
    assert bands[-1].label == "[0.80, 1.00]"

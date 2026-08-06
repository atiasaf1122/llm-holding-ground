"""Who ends up in a peer block, and under what name.

How that block is then rendered belongs to :mod:`council.agents.prompt` and is
tested there, once, for every arm. What is left here is the committee half: that a
chair is never its own peer, that the numbers carry no information, and that views
argued by a different table are refused rather than quietly changing how many peers
an agent has.
"""

from __future__ import annotations

import pytest

from council.debate.compositions import rotations
from council.debate.peers import NoPeersError, SeatView, peers_for, seated_views
from helpers_debate import MODELS, committee, persona_names, view, views_of


def test_a_chair_is_never_its_own_peer() -> None:
    # Arrange
    composition = committee()
    views = views_of(composition, marker="opening")

    # Act
    peers = peers_for(composition.seats[1], views=views)

    # Assert
    assert len(peers) == composition.size - 1
    assert views[1].rationale not in [peer.rationale for peer in peers]


def test_no_model_or_persona_name_reaches_a_peer_view() -> None:
    # Arrange -- a model told which laboratory it is arguing with has been given a
    # reason to defer that has nothing to do with the argument.
    composition = committee()

    # Act
    peers = peers_for(composition.seats[0], views=views_of(composition, marker="opening"))

    # Assert
    rendered = " ".join(f"{peer.label} {peer.rationale}" for peer in peers)
    for seat in composition.seats:
        assert seat.model not in rendered
    for name in persona_names():
        assert name not in rendered
    assert [peer.label for peer in peers] == ["Analyst 1", "Analyst 2", "Analyst 3"]


def test_the_numbers_are_handed_out_after_the_reader_is_removed() -> None:
    # Arrange -- so the same agent is a different analyst in every prompt it
    # appears in, and a number cannot be followed across the committee.
    composition = committee()
    views = views_of(composition, marker="opening")

    # Act
    first = peers_for(composition.seats[0], views=views)
    last = peers_for(composition.seats[-1], views=views)

    # Assert
    assert first[0].rationale != last[0].rationale
    assert [peer.label for peer in first] == [peer.label for peer in last]


def test_a_seat_with_nobody_left_to_answer_raises() -> None:
    # Arrange -- asking again with an empty peer block would store a second
    # independent answer as a debate round.
    seats = committee().seats

    # Act & Assert
    with pytest.raises(NoPeersError):
        peers_for(seats[0], views=[view(seats[0])])


def test_views_are_ordered_by_the_committee_and_not_by_the_caller() -> None:
    # Arrange -- a pool assembled from a groupby or a set comprehension holds its
    # views in an order nothing pins, and the analyst numbers follow that order.
    composition = committee()
    views = views_of(composition, marker="opening")

    # Act
    forward = seated_views(views, composition=composition)
    backward = seated_views(tuple(reversed(views)), composition=composition)

    # Assert
    assert forward == backward
    assert [seated.seat for seated in forward] == list(composition.seats)


def test_a_view_from_another_committee_is_refused() -> None:
    # Arrange -- the placebo's donor point is meant to be the same table on another
    # day. A stranger's view would leave every agent one peer better off than the
    # real arm, which is a second manipulation riding along with the intended one.
    composition = rotations(models=MODELS)[0]
    stranger = views_of(rotations(models=MODELS)[1], marker="donor")[0]

    # Act & Assert
    with pytest.raises(ValueError, match="not a chair"):
        seated_views([stranger], composition=composition)


def test_one_chair_cannot_hold_two_views_in_a_round() -> None:
    # Arrange
    composition = committee()
    seat = composition.seats[0]

    # Act & Assert -- whichever one won would be decided by dict ordering.
    with pytest.raises(ValueError, match="two views"):
        seated_views(
            [SeatView(seat=seat, exposure=0.1, rationale="a"), view(seat)],
            composition=composition,
        )


def test_a_round_that_lost_a_seat_keeps_the_others_in_committee_order() -> None:
    # Arrange -- a failed generation simply leaves its chair empty.
    composition = committee()
    views = views_of(composition, marker="opening")

    # Act
    kept = seated_views((views[2], views[0]), composition=composition)

    # Assert
    assert [seated.seat for seated in kept] == [composition.seats[0], composition.seats[2]]

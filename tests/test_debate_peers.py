"""Who ends up in a peer block, and under what name.

How that block is then rendered belongs to :mod:`council.agents.prompt` and is
tested there, once, for every arm. What is left here is the committee half: that a
chair is never its own peer, that the numbers carry no information, and that views
argued by a different table are refused rather than quietly changing how many peers
an agent has.
"""

from __future__ import annotations

import inspect

import pytest

from council.debate.compositions import rotations
from council.debate.peers import NoPeersError, SeatView, peers_for, seated_views
from council.debate.protocol import rebuttal_peers, run_debate
from helpers_debate import MODELS, committee, persona_names, view, views_of

TOKEN = "seed|rotation-0|2022-03-01|AAPL|1"
"""A stand-in for what :func:`~council.debate.protocol.run_debate` builds. Required
by ``peers_for``, so a test cannot accidentally assert on one fixed permutation."""


def test_a_chair_is_never_its_own_peer() -> None:
    # Arrange
    composition = committee()
    views = views_of(composition, marker="opening")

    # Act
    peers = peers_for(composition.seats[1], views=views, order_token=TOKEN)

    # Assert
    assert len(peers) == composition.size - 1
    assert views[1].rationale not in [peer.rationale for peer in peers]


def test_no_model_or_persona_name_reaches_a_peer_view() -> None:
    # Arrange -- a model told which laboratory it is arguing with has been given a
    # reason to defer that has nothing to do with the argument.
    composition = committee()

    # Act
    peers = peers_for(
        composition.seats[0], views=views_of(composition, marker="opening"), order_token=TOKEN
    )

    # Assert
    rendered = " ".join(f"{peer.label} {peer.rationale}" for peer in peers)
    for seat in composition.seats:
        assert seat.model not in rendered
    for name in persona_names():
        assert name not in rendered
    assert [peer.label for peer in peers] == ["Analyst 1", "Analyst 2", "Analyst 3"]


def test_the_numbers_are_handed_out_after_the_reader_is_removed() -> None:
    # Arrange -- the reader's own chair is dropped before anything is numbered, so
    # every block is the same length whichever chair is reading.
    composition = committee()
    views = views_of(composition, marker="opening")

    # Act
    first = peers_for(composition.seats[0], views=views, order_token=TOKEN)
    last = peers_for(composition.seats[-1], views=views, order_token=TOKEN)

    # Assert
    assert first[0].rationale != last[0].rationale
    assert [peer.label for peer in first] == [peer.label for peer in last]


def test_a_peer_is_not_the_same_analyst_number_in_every_prompt() -> None:
    # Arrange -- numbering the survivors in committee order made a speaker's number,
    # and therefore its position on the page, a pure function of two seat indices:
    # identical in every rotation, round and decision point, so block position was
    # confounded with the base model sitting there. Asserted across tokens rather
    # than on one, since any single pair may coincide.
    composition = committee()
    views = views_of(composition, marker="opening")
    reader = composition.seats[0]
    speakers = [seated.seat for seated in views if seated.seat != reader]

    def numbering(token: str) -> dict[str, int]:
        block = peers_for(reader, views=views, order_token=token)
        by_rationale = {seated.rationale: seated.seat.model for seated in views}
        return {by_rationale[peer.rationale]: index for index, peer in enumerate(block)}

    # Act
    seen = {tuple(numbering(f"token-{n}")[speaker.model] for speaker in speakers)
            for n in range(20)}

    # Assert -- more than one arrangement, and none of them privileged.
    assert len(seen) > 1
    for position in range(len(speakers)):
        assert len({arrangement[position] for arrangement in seen}) > 1


def test_the_numbering_is_the_same_one_on_a_rerun_of_the_same_token() -> None:
    # Arrange -- the permutation must reproduce exactly, or a resumed sweep would
    # rewrite the prompts of an arm already on disk. Same argument, same primitive,
    # as the placebo donor draw.
    composition = committee()
    views = views_of(composition, marker="opening")

    # Act & Assert
    for seat in composition.seats:
        assert peers_for(seat, views=views, order_token=TOKEN) == peers_for(
            seat, views=views, order_token=TOKEN
        )


def test_a_seat_with_nobody_left_to_answer_raises() -> None:
    # Arrange -- asking again with an empty peer block would store a second
    # independent answer as a debate round.
    seats = committee().seats

    # Act & Assert
    with pytest.raises(NoPeersError):
        peers_for(seats[0], views=[view(seats[0])], order_token=TOKEN)


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


# -- the docstrings name the fields the token is actually built from --------------

TOKEN_FIELDS = "the seed, the committee, the point and the round index"
"""What :func:`~council.debate.protocol.run_debate` builds the token from.

Not the arm. Keying by arm would vary the peer numbering *between* the arms, so the
rationale-only control would differ from the treatment in peer order as well as in
the withheld exposure -- a second manipulation riding along with the intended one.
Both Args blocks listed the arm anyway, which is what a future editor reads before
"fixing" the token to match.
"""


def test_the_shipped_order_token_is_not_keyed_by_arm() -> None:
    # Arrange -- the expression itself, so the prose below is pinned to the code
    # rather than to another piece of prose.
    source = inspect.getsource(run_debate)
    token = source.split("order_token=(", 1)[1].split("),", 1)[0]

    # Assert -- five fields, and none of them the arm.
    assert "arm" not in token
    assert token.count("|") == 4


@pytest.mark.parametrize("documented", (peers_for.__doc__, rebuttal_peers.__doc__))
def test_the_order_token_docstrings_leave_the_arm_out(documented: str | None) -> None:
    assert documented is not None
    flowed = " ".join(documented.split())
    assert TOKEN_FIELDS in flowed
    assert "the arm, the point and the round index" not in flowed


def test_run_debates_args_block_does_not_confine_the_seed_to_the_placebo() -> None:
    # "Only the placebo draw uses it" -- sixteen lines above the same function
    # folds the resolved seed into the `order_token` that permutes every peer block
    # in every arm. A reader who trusts the Args block concludes that changing the
    # seed leaves the debate and rationale-only arms byte-identical.
    source = inspect.getsource(run_debate)
    args = " ".join(source.split("seed: defaults to")[1].split("Raises:")[0].split())
    token = source.split("order_token=(", 1)[1].split("),", 1)[0]

    assert "resolved_seed" in token
    assert "Only the placebo draw uses it." not in args
    assert "order_token" in args
    assert "in every arm" in args

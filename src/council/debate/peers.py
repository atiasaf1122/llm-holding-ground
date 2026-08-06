"""Who is shown to whom, and under what name.

The peer block itself is rendered by :mod:`council.agents.prompt` -- the same
function that renders the independent arm's prompt, and the one every stored row's
``prompt_hash`` is taken over. A second renderer here would mean the fencing, the
truncation of untrusted peer prose and the arm-dependent exposure switch each
existed twice, and the arms would then differ by whichever copy a run happened to
take rather than by the manipulation.

What is left in this module is the part that is about the committee rather than
about the text.

**Anonymity.** A peer is "Analyst 1". No model name, no persona name, and no
ordering that carries information: the numbers are handed out in committee order
*after* the reader's own chair has been dropped, so one agent is a different number
in every prompt it appears in. A model told which laboratory it is arguing with can
be persuaded by the name.

**A chair is never its own peer, and only this table's chairs are peers at all.**
The second half matters in the placebo arm, where the views come from another day:
a donor holding a seat this committee does not have would leave every agent with
one more peer than the real arm gives, which is a second manipulation riding along
with the intended one.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from council.agents.prompt import PeerView
from council.debate.compositions import Composition, Seat


class NoPeersError(ValueError):
    """A rebuttal round was reached with nothing to rebut.

    Raised rather than papered over with an empty peer block. An agent whose peers
    all failed to generate would otherwise be asked the independent question a
    second time and have the answer stored as a debate round -- a row that reads as
    an agent unmoved by a conversation that never happened, in the treatment arms
    only. The caller should record the point as failed instead.
    """


@dataclass(frozen=True, slots=True)
class SeatView:
    """What one agent said, tagged with the chair that said it.

    The seat never reaches a prompt. It is carried so that an agent is not shown
    its own view, and so that the placebo arm can match a donor point's views to
    this committee's chairs -- which is what leaves each agent the same number of
    peers it would have had in the real arm.
    """

    seat: Seat
    exposure: float
    rationale: str


def seated_views(
    views: Sequence[SeatView], *, composition: Composition
) -> tuple[SeatView, ...]:
    """The views belonging to this committee, in committee order.

    The order is taken from the committee rather than from the sequence handed in,
    because that sequence is the caller's: a placebo pool assembled from a groupby
    or a set comprehension would otherwise hand out the analyst numbers in a
    different order on a rerun and change the prompts of an arm that had already
    been written.

    Raises:
        ValueError: naming the chair, if a view comes from a seat this committee
            does not have or two views come from one seat. Both mean the views
            being shown were argued by a different table, and the peer counts the
            arms are compared on would silently stop matching.
    """
    by_seat: dict[Seat, SeatView] = {}
    for view in views:
        if view.seat not in composition.seats:
            raise ValueError(
                f"{view.seat} is not a chair in {composition.identifier}; "
                "these views were argued by a different committee"
            )
        if view.seat in by_seat:
            raise ValueError(
                f"{view.seat} holds two views in one round of {composition.identifier}"
            )
        by_seat[view.seat] = view
    return tuple(by_seat[seat] for seat in composition.seats if seat in by_seat)


def peers_for(seat: Seat, *, views: Sequence[SeatView]) -> tuple[PeerView, ...]:
    """One chair's peer block, anonymised, in the order the views arrive in.

    Raises:
        NoPeersError: if nothing is left once this chair's own view is dropped.
    """
    others = [view for view in views if view.seat != seat]
    if not others:
        raise NoPeersError(f"{seat} has no peer view to answer; there is no debate to hold")
    return tuple(
        PeerView(label=f"Analyst {number}", exposure=view.exposure, rationale=view.rationale)
        for number, view in enumerate(others, start=1)
    )

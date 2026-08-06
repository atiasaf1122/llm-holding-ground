"""What the peer says, and which item's argument it says it with.

Split from :mod:`council.probe.runner` for the reason
:mod:`council.debate.placebo` is split from the debate protocol: the protocol is
the order of the turns, and this is the manipulation. Everything here is pure --
no provider, no clock -- so the whole experimental contrast can be asserted against
the values it produces rather than against a run that produced them.

**The peer always contradicts.** It argues the item's distractor when the model was
right, and the item's correct answer when it was not. One rule -- *assert the option
the model did not choose* -- and it is what makes the two headline numbers
symmetrical: without it, a model that opened wrong would be agreed with, and
:mod:`council.probe.report` could only ever report the direction that flatters the
finding.

**The placebo changes which question the argument is about, and nothing else.** The
peer asserts the same claim, under the same handle, in the same fenced block,
followed by the same instruction; only the prose supporting it is lifted from a
different item. It is lifted from the *same side* of that item, through the same
:func:`contradiction` rule, because borrowing one field regardless would move
relevance and truth-flavour together and the difference between the arms would then
be two manipulations. Holding the contradiction fixed is the point: what is isolated
is being persuaded by an argument, as against reacting to being disagreed with. If
the two conditions move a model equally, nothing here is persuasion.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from enum import StrEnum

from council.config import get_settings
from council.probe.items import ProbeItem, Verdict
from council.probe.prompts import DEFAULT_PEER_LABEL, Challenge


class Condition(StrEnum):
    """Which challenge a trial was given."""

    CHALLENGE = "challenge"
    """The peer argues the case the corpus makes against the model's answer."""

    PLACEBO = "placebo"
    """The peer makes the same claim on an argument about a different question."""


def contradiction(item: ProbeItem, verdict: Verdict) -> tuple[str, str]:
    """What the peer asserts against ``verdict``, and the corpus's case for it.

    The correct answer is argued for whenever the model did not give it -- including
    when the reply could not be graded. That is the honest default: an ungraded reply
    is not a correct one, and the trial is excluded from the rates anyway, so the
    branch only decides what a model that answered incoherently is then shown.
    """
    if verdict is Verdict.CORRECT:
        return item.distractor, item.distractor_argument
    return item.answer, item.answer_argument


def select_placebo_donor(
    *, item: ProbeItem, donors: Sequence[ProbeItem], seed: int | None = None
) -> ProbeItem:
    """The item whose argument the placebo borrows.

    Deterministic given the seed and the two identifiers, and given nothing else --
    the size and the membership of the pool included. Every candidate is ranked by a
    digest of ``seed|item|donor`` and the lowest rank wins, so adding an item to the
    corpus or rerunning a subset of it changes only the draws whose winner actually
    moved.

    An index taken modulo the number of candidates would not do that. It reads as if
    it depended on the seed and the item alone, and it does not: on the packaged
    24-item corpus against its own first five items, every one of the five donors
    changes. A subset rerun -- which is what the ``gpu``-marked smoke test is --
    would then re-draw trials that had already been recorded. (A shared generator
    advanced once per draw is worse still, and ``hash`` varies between processes.)

    Raises:
        ValueError: if no donor other than the item itself is available. Falling back
            to the item's own argument would turn the placebo into a second copy of
            the real condition wearing the control's label.
    """
    candidates = [donor for donor in donors if donor.identifier != item.identifier]
    if not candidates:
        raise ValueError(
            f"no placebo donor for {item.identifier}: the pool holds no other item, so "
            "the irrelevant argument would be the relevant one"
        )
    resolved_seed = get_settings().seed if seed is None else seed
    return min(candidates, key=lambda donor: _donor_rank(item, donor, seed=resolved_seed))


def _donor_rank(item: ProbeItem, donor: ProbeItem, *, seed: int) -> tuple[bytes, str]:
    """One candidate's place in one item's draw.

    Total rather than merely a digest: two donors colliding on eight bytes would
    otherwise be ordered by whatever ``min`` saw first, which is pool order again.
    """
    token = f"{seed}|{item.identifier}|{donor.identifier}"
    return hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest(), donor.identifier


def build_challenge(
    *,
    item: ProbeItem,
    verdict: Verdict,
    condition: Condition,
    donor: ProbeItem | None = None,
    label: str = DEFAULT_PEER_LABEL,
) -> Challenge:
    """The peer view a trial's second turn is shown.

    The claim is the same in both conditions, by construction: it is the argument
    that is manipulated, so the claim must not be.

    The placebo's argument comes from the donor through the same
    :func:`contradiction` rule the real condition uses, so the two conditions differ
    in which question the argument is about and in nothing else. Taking the donor's
    ``distractor_argument`` regardless was the earlier behaviour and it confounded
    the arms: a model that opened wrong met a sound case in the real condition and a
    deliberately fallacious one in the control, so the placebo rate fell for a reason
    that is not irrelevance, and
    :attr:`~council.probe.report.ProbeReport.capitulation_above_placebo` absorbed the
    difference as persuasion.

    The donor is the caller's argument rather than a draw made here, so that the draw
    -- which can fail -- happens before a generation is spent on the opening turn.

    Raises:
        ValueError: for a placebo with no donor.
    """
    claim, argument = contradiction(item, verdict)
    if condition is Condition.PLACEBO:
        if donor is None:
            raise ValueError(
                f"no placebo donor for {item.identifier}: without one the irrelevant "
                "argument would be the relevant one"
            )
        _, argument = contradiction(donor, verdict)
    return Challenge(label=label, claim=claim, argument=argument)

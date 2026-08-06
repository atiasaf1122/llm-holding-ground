"""Whose views the placebo arm is handed.

The placebo asks the same committee the same question in the same words and shows
it a peer block rendered by the same function. One thing differs: the arguments in
that block were written about another day. If agents give ground here as readily as
they do in a real debate, they are reacting to being contradicted rather than to
the argument, and every number the debate arm produces means something else.

That makes the draw itself load-bearing, and it has two constraints that are easy
to state and easy to leave unenforced.

**A donor must precede the point.** A rationale argues about the window its author
was shown, and quotes the moves in it. A donor from later than the decision
therefore writes tomorrow's prices into today's prompt -- and the placebo's
exposures go through :func:`council.backtest.engine.run_ticker` exactly like every
other arm's, so the lookahead lands in the headline comparison rather than in a
column somebody might notice. Nothing raises; the arm simply reports the wrong
number, in whichever direction the future happened to go.

**A donor must be this committee.** The draw matches the donor's chairs to the
seats being debated, which is what leaves each agent the same number of peers the
real arm gives it. A peer count that differed between the arms would be a second
manipulation riding along with the intended one.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

from council.config import get_settings
from council.debate.compositions import Composition
from council.debate.peers import SeatView, seated_views
from council.evaluation.frames import PointKey

PlaceboPool = Mapping[PointKey, Sequence[SeatView]]
"""Opening views from earlier decision points, for the placebo arm to draw from.

Assembled by the caller from a debate arm that has already run. Both properties the
module docstring describes are checked here rather than trusted, because a pool is
ordinary-looking data and neither failure is visible in a stored row.
"""


def donor_views(
    *, pool: PlaceboPool | None, point: PointKey, composition: Composition, seed: int | None
) -> tuple[SeatView, ...]:
    """The donor point's views, matched to this committee, in committee order.

    Drawn and checked before any generation happens: a pool belonging to another
    committee would otherwise cost a whole opening round before raising, and the
    order the views are shown in has to come from the committee rather than from
    however the pool was assembled -- a rerun that iterated a groupby differently
    would otherwise renumber the analysts and rewrite an arm already on disk.
    """
    if pool is None:
        raise ValueError("the placebo arm needs a pool of other days' views to draw from")
    donor = select_placebo_point(
        pool=pool, point=point, composition=composition.identifier, seed=seed
    )
    return seated_views(pool[donor], composition=composition)


def select_placebo_point(
    *, pool: PlaceboPool, point: PointKey, composition: str, seed: int | None = None
) -> PointKey:
    """Pick the earlier decision point whose views the placebo arm will show.

    Deterministic given the seed, the committee and the point being debated -- and
    given nothing else. A shared generator advanced once per draw would make each
    point's donor depend on how many points were drawn before it, so a rerun over a
    different date range would silently change an arm that had already been
    written; and ``hash`` varies between processes.

    Raises:
        ValueError: if the pool holds nothing from an earlier date. Refusing is the
            only safe answer: a later donor is the lookahead this module exists to
            prevent, and quietly falling back to the point's own views would turn
            the placebo into a second debate arm wearing the control's label.
    """
    decision_date, ticker = point
    candidates = sorted(key for key, views in pool.items() if key[0] < decision_date and views)
    if not candidates:
        raise ValueError(
            f"no placebo donor for {decision_date} {ticker}: the pool holds no earlier date"
        )

    resolved_seed = get_settings().seed if seed is None else seed
    token = f"{resolved_seed}|{composition}|{decision_date.isoformat()}|{ticker}"
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    chosen = candidates[int.from_bytes(digest, "big") % len(candidates)]

    # The filter above already guarantees this. It is checked anyway because the
    # cost of the guarantee ever failing is the entire placebo arm, silently, and
    # the cost of checking is one comparison per debate.
    if chosen[0] >= decision_date:
        raise ValueError(f"placebo donor {chosen} does not precede {point}")
    return chosen

"""Whose views the placebo arm is handed.

The placebo asks the same committee the same question in the same words and shows
it a peer block rendered by the same function. The arguments in that block were
written about another decision point -- another day, and, since the draw constrains
only the date, often another instrument. If agents give ground here as readily as
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

**A donor must be this committee, and all of it.** The draw matches the donor's
chairs to the seats being debated, and requires a view for every one of them. Both
directions are enforced rather than assumed: an extra chair is refused by
:func:`council.debate.peers.seated_views`, and a missing one is refused here. A
donor day on which one seat failed to generate holds fewer views than the committee
has chairs, so admitting it would leave the placebo's agents one peer short -- and a
donor down to a single view raises :class:`~council.debate.peers.NoPeersError`,
which the sweep books as an abandoned conversation, costing the placebo arm decision
points the other two keep. A peer count that differed between the arms would be a
second manipulation riding along with the intended one.

Completeness here is a property of the *donor day*, and it is only half of the peer
count. The other half is this round: the real arms build their block from the seats
that actually spoke, so a seat that failed to generate costs each survivor a peer.
A complete donor holds every chair whatever happened here, so the parity holds per
round rather than unconditionally -- :func:`council.debate.protocol.run_debate`
restricts the drawn views to the seats that spoke in the round just taken, and that
restriction is what makes the counts match when a generation fails.
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

Assembled by the caller from the **independent** arm, never from a debate arm that has
already run: a pool of debated points holds contested days only, so its earliest
contested day has no earlier donor and is dropped from this arm alone. See
:func:`council.debate.sweep.placebo_pool_for`. Both properties the
module docstring describes are checked here rather than trusted, because a pool is
ordinary-looking data and neither failure is visible in a stored row.
"""


def donor_views(
    *,
    pool: PlaceboPool | None,
    point: PointKey,
    composition: Composition,
    seed: int | None,
    round_index: int = 1,
    min_gap: int | None = None,
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
        pool=pool,
        point=point,
        composition=composition.identifier,
        required_seats=composition.size,
        seed=seed,
        round_index=round_index,
        min_gap=min_gap,
    )
    return seated_views(pool[donor], composition=composition)


def select_placebo_point(
    *,
    pool: PlaceboPool,
    point: PointKey,
    composition: str,
    required_seats: int,
    seed: int | None = None,
    round_index: int = 1,
    min_gap: int | None = None,
) -> PointKey:
    """Pick the earlier decision point whose views the placebo arm will show.

    Deterministic given the seed, the committee, the point being debated **and the
    candidate set**. The last of those is a real dependency rather than a
    formality: the donor is the first entry of a digest ranking *of the
    candidates*, so any candidate a wider date range admits can take first place
    and redraw the donor for a point whose seed, committee and date are unchanged.
    Widening the range therefore rewrites the placebo arm for every point not
    already stored, and nothing detects it --
    :func:`council.agents.runner.check_prompt_provenance` validates round-0 rows,
    which carry no peer block. A run whose range changes has to be regenerated
    rather than resumed if the placebo arm is to hold one treatment. What the
    digest ranking does remove is dependence on ``hash``, which varies between
    processes, and on how many draws preceded this one.

    Args:
        required_seats: how many seats a candidate must hold a view for, ordinarily
            ``composition.size``. Completeness is enforced here rather than assumed:
            a merely non-empty donor day is what leaves the placebo's agents fewer
            peers than the real arm gives, and the module docstring has the cost.

    Raises:
        ValueError: if the pool holds no complete earlier day. Refusing is the only
            safe answer: a later donor is the lookahead this module exists to
            prevent, and quietly falling back to the point's own views would turn
            the placebo into a second debate arm wearing the control's label.
    """
    decision_date, ticker = point
    settings = get_settings()
    gap = settings.placebo_min_gap_sessions if min_gap is None else min_gap

    # Sessions, not calendar days: the pool holds one entry per trading day, so
    # counting entries back is counting sessions back. A donor nearer than the
    # lookback shares bars with the window under decision -- the first run drew a
    # median of 14 sessions against a 60-session lookback, which left the
    # "unrelated" peer arguing about roughly the same data. See config.
    sessions = sorted({key[0] for key in pool})
    cutoff = decision_date
    if gap > 0:
        earlier = [day for day in sessions if day < decision_date]
        if len(earlier) < gap:
            raise ValueError(
                f"no placebo donor for {decision_date} {ticker}: the pool holds "
                f"{len(earlier)} earlier session(s), fewer than the {gap} required"
            )
        cutoff = earlier[-gap]

    # Both bounds, and the strict one is not redundant: with a gap of zero the
    # cutoff *is* the decision date, and a `<=` filter alone would admit the day
    # being decided as its own donor -- the exact lookahead this module exists to
    # refuse, reintroduced by the gap check that was meant to strengthen it.
    candidates = sorted(
        key
        for key, views in pool.items()
        if key[0] < decision_date and key[0] <= cutoff and len(views) == required_seats
    )
    if not candidates:
        raise ValueError(
            f"no placebo donor for {decision_date} {ticker}: the pool holds no session "
            f"at least {gap} back with a view from all {required_seats} seat(s)"
        )

    resolved_seed = settings.seed if seed is None else seed
    token = f"{resolved_seed}|{composition}|{decision_date.isoformat()}|{ticker}"

    # A deterministic order over the candidates, then the round's position in it --
    # rather than a fresh hash per round, which can and does draw the same donor
    # twice. Two rounds shown identical peers would let the control reach
    # stillness for a reason the treatment never faces (nothing new to answer),
    # and stillness is one of the things this design measures.
    #
    # Ordering by a per-candidate digest rather than shuffling with a seeded RNG:
    # a generator advanced once per draw would make each donor depend on how many
    # were drawn before it, which is a dependence on draw *order* and is what this
    # removes. It does not survive a changed date range: the ranking is over the
    # candidate set, so a candidate a wider pool admits can rank first and redraw
    # the donor for an unchanged point. A run whose range changes must be
    # regenerated rather than resumed if the placebo arm is to hold one treatment.
    ordered = sorted(
        candidates,
        key=lambda key: hashlib.blake2b(
            f"{token}|{key[0].isoformat()}|{key[1]}".encode(), digest_size=8
        ).digest(),
    )
    # Wraps whenever the candidate set is smaller than the round count, which the
    # configured gap makes routine rather than impossible at the earliest
    # admissible dates: the filter is `key[0] <= cutoff` where `cutoff =
    # earlier[-gap]`, so a point with exactly `gap` earlier sessions has a
    # one-element candidate set and every round draws the same donor. It is the cap
    # of one rebuttal round, and nothing else, that keeps a repeated donor out of a
    # run today. Above it the placebo would see an identical peer block in
    # consecutive rounds -- the exact failure the per-round redraw was added to
    # prevent -- with nothing raising or logging. See
    # :func:`council.debate.sweep._check_cap`.
    chosen = ordered[(round_index - 1) % len(ordered)]

    # The filter above already guarantees this. It is checked anyway because the
    # cost of the guarantee ever failing is the entire placebo arm, silently, and
    # the cost of checking is one comparison per debate.
    if chosen[0] >= decision_date:
        raise ValueError(f"placebo donor {chosen} does not precede {point}")
    return chosen

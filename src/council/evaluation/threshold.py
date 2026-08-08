"""Comparing a movement against the bar that decides whether it counts.

Its own module because two independent measurements -- the shift rate and the
influence matrix -- both ask "did this move far enough", and they must ask it the
same way. They did not, and the difference was invisible.

## The defect this exists to prevent

Exposures arrive on a coarse grid: models constrained to ``[-1, 1]`` overwhelmingly
answer in multiples of 0.05. Subtracting two of those in binary floating point does
not give the decimal answer:

    abs(0.5 - 0.3)  == 0.2                    >= 0.20  ->  True
    abs(0.3 - 0.1)  == 0.19999999999999998    >= 0.20  ->  False
    abs(0.4 - 0.6)  == 0.19999999999999996    >= 0.20  ->  False

**The same move, counted or not depending on where on the grid it happened.** In the
published two-model run, 273 conversations moved by exactly the threshold and 155 of
them were counted -- 118 dropped in silence.

That would be tolerable if it were noise. It is not: each confidence band is close to
a single agent's stratum, and different agents occupy different regions of the grid,
so the loss is concentrated by band. The shift rate looked flat across confidence
(0.204 / 0.241 / 0.218 / 0.228) and is not flat under either principled convention
(0.204 / 0.373 / 0.248 / 0.407 inclusive). A headline finding rested on the artefact.

## Why a tolerance rather than integer grid units

Rounding to grid units would be exact, but it assumes the grid -- and a model is free
to answer 0.37. A tolerance is correct for any value and costs nothing: it accepts
movements that differ from the bar by less than representation error, which is what
"a move *of* this size counts" was always supposed to mean.
"""

from __future__ import annotations

# Comfortably larger than the error accumulated by one subtraction of two values in
# [-1, 1] (about 2e-16), and far smaller than any difference that could be meant. A
# move must miss the bar by more than this to be excluded.
TOLERANCE = 1e-9


def meets(distance: float, bar: float) -> bool:
    """Whether ``distance`` reaches ``bar``, inclusive at the boundary.

    Inclusive because both the configuration comment and the docstrings that read
    this call a move *of* the threshold a shift. The tolerance is what makes that
    wording true in arithmetic as well as in prose.
    """
    return distance >= bar - TOLERANCE


def exceeds(distance: float, bar: float) -> bool:
    """Whether ``distance`` is strictly beyond ``bar``.

    The counterpart for callers whose documented wording is "more than". Kept beside
    :func:`meets` so that a reader choosing between them sees both, rather than
    reaching for a bare comparison and reintroducing the defect above.
    """
    return distance > bar + TOLERANCE

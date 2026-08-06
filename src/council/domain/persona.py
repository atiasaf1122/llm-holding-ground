"""Who the agents are.

Two axes, crossed. The pair is chosen so that agents disagree about **direction**,
not merely about size -- which is the whole precondition for the experiment. Four
agents that all say "buy, but by different amounts" have nothing to argue about,
and a debate between them measures haggling rather than persuasion.

* **Stance** decides the direction. A momentum reader and a mean-reversion reader
  look at the same rise and reach opposite conclusions: one sees a trend to join,
  the other sees an overshoot to fade.
* **Aggression** decides the size, and the strength of evidence demanded before
  taking any position at all.

Everything else about the agents is identical -- same schema, same temperature,
same context. A difference in output is therefore attributable to the persona or
to the base model, and to nothing else.
"""

from __future__ import annotations

from enum import StrEnum
from itertools import product

from pydantic import BaseModel, ConfigDict


class Stance(StrEnum):
    """How the agent reads a price move."""

    MOMENTUM = "momentum"
    REVERSION = "reversion"


class Aggression(StrEnum):
    """How hard the agent commits to a view."""

    CAUTIOUS = "cautious"
    BOLD = "bold"


class Persona(BaseModel):
    """One of the four analyst characters."""

    model_config = ConfigDict(frozen=True)

    stance: Stance
    aggression: Aggression

    @property
    def name(self) -> str:
        """Stable identifier, used as the persona column and the prompt filename."""
        return f"{self.stance}-{self.aggression}"

    def __str__(self) -> str:
        return self.name


PERSONAS: tuple[Persona, ...] = tuple(
    Persona(stance=stance, aggression=aggression)
    for stance, aggression in product(Stance, Aggression)
)
"""The four personas, in a fixed order.

Fixed because the debate configurations in :mod:`council.debate.compositions`
index into this tuple, and a reordering would silently redefine every experimental
arm that has already been run.
"""

PERSONAS_BY_NAME: dict[str, Persona] = {persona.name: persona for persona in PERSONAS}

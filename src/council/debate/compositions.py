"""Which committees are actually run, and why there are eight of them.

Assigning each of four models one of the four personas freely gives ``4 ** 4 =
256`` committees. A conversation at the shipped six-round cap costs up to
``4 * 7 = 28`` calls, the run has on the order of 1,000 decision points, and
:data:`council.planning.TREATMENT_ARMS` names four full-length arms plus the
one-round contradictor with its 12 counter-generations per point -- so the full
grid is ``256 * 28 * 1000 * 4 + 256 * 20 * 1000 = 33,792,000`` calls. At
:data:`council.planning.SECONDS_PER_INFERENCE` with four models resident -- the
parallelism :meth:`council.planning.StagePlan.seconds` applies -- that is about
a hundred and forty-seven
days of continuous generation, on a card this project has already promised not to
monopolise. It is also mostly redundant: the great
majority of those 256 committees differ from another only in which model happens
to be wearing which persona.

What is run instead is a balanced design, in two families.

**Rotations** -- a Latin square over models x personas. Model *i* holds persona
``(i + k) mod 4`` in rotation *k*, so across the four rotations every model holds
every persona exactly once, and -- when the committee is as wide as the persona
set -- every rotation seats each persona exactly once. That is the property the
whole design rests on, so it is generated arithmetically and asserted by test
rather than typed out: a hand-written table with one transposed cell still looks
like a Latin square and silently answers a different question.

**Uniform references** -- four committees in which every seat holds the same
persona. Without them, a difference between rotations cannot be attributed to the
*mixture* rather than to the personas themselves.

Eight configurations, ``8 * 8 * 1000 = 64,000`` calls **per arm** -- and the sweep
runs three treatment arms (:data:`council.planning.TREATMENT_ARMS`), so eight
committees cost ``8 * 4 * 2 * 3 = 192`` calls per contested point, about 192,000 at
1,000 contested points. The 256-versus-8 ratio above is unaffected: both sides of it
are per-arm figures.

Every question the grid existed to answer is still asked: whether a persona travels
across base models, and whether a mixed committee behaves differently from a uniform
one. What is given up is the interaction between *particular* pairings -- whether this
model argues differently against that one -- which this study does not ask about.

Identifiers are stable only for a fixed model ordering, exactly as the persona
order is fixed in :mod:`council.domain.persona`. Reordering
``Settings.agent_models`` would keep the name ``rotation-1`` and change the
committee it names, merging two different configurations in an already-written
parquet file with nothing raising.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from council.config import get_settings
from council.domain.persona import PERSONAS, Persona


@dataclass(frozen=True, slots=True)
class Seat:
    """One chair at the table: a base model wearing a persona.

    Neither half identifies an agent alone. The same model sits in several
    committees under different personas, and the same persona is worn by every
    model in turn -- which is the point of the rotation.
    """

    model: str
    persona: Persona

    def __str__(self) -> str:
        return f"{self.model}/{self.persona.name}"


@dataclass(frozen=True, slots=True)
class Composition:
    """One committee, under the identifier written to ``Decision.composition``.

    Seat order is fixed at construction and never re-derived. It decides the order
    peers are rendered in, so a set here rather than a tuple would make two runs of
    the same configuration produce different prompts.
    """

    identifier: str
    seats: tuple[Seat, ...]

    @property
    def size(self) -> int:
        return len(self.seats)


def rotations(
    *, models: Sequence[str] | None = None, personas: Sequence[Persona] = PERSONAS
) -> tuple[Composition, ...]:
    """The Latin square: one configuration per persona offset.

    There are as many rotations as there are personas rather than as many as there
    are models, which is what makes the balance property hold for a committee
    narrower than the persona set: model *i* takes persona ``(i + k) mod p`` and
    therefore visits each of the *p* personas exactly once as *k* runs over them.
    A four-model committee gets the square proper, where each rotation also seats
    every persona exactly once.

    Args:
        models: defaults to ``settings.agent_models``.
        personas: defaults to the four in :mod:`council.domain.persona`.
    """
    seated = _resolve_models(models)
    count = _validate_personas(personas)
    return tuple(
        Composition(
            identifier=f"rotation-{offset}",
            seats=tuple(
                Seat(model=model, persona=personas[(index + offset) % count])
                for index, model in enumerate(seated)
            ),
        )
        for offset in range(count)
    )


def uniform_references(
    *, models: Sequence[str] | None = None, personas: Sequence[Persona] = PERSONAS
) -> tuple[Composition, ...]:
    """One configuration per persona, with every seat holding it.

    The reference a rotation is read against. A committee of four different
    characters and a committee of four copies of one character disagree for
    different reasons, and only the second isolates the base model.
    """
    seated = _resolve_models(models)
    _validate_personas(personas)
    return tuple(
        Composition(
            identifier=f"uniform-{persona.name}",
            seats=tuple(Seat(model=model, persona=persona) for model in seated),
        )
        for persona in personas
    )


def balanced_design(
    *, models: Sequence[str] | None = None, personas: Sequence[Persona] = PERSONAS
) -> tuple[Composition, ...]:
    """Every committee this experiment runs: the rotations, then the references.

    Eight configurations for the four-by-four case, against 256 for the full grid.
    The module docstring has the arithmetic.
    """
    return (
        *rotations(models=models, personas=personas),
        *uniform_references(models=models, personas=personas),
    )


def _resolve_models(models: Sequence[str] | None) -> tuple[str, ...]:
    resolved = tuple(get_settings().agent_models if models is None else models)
    if not resolved:
        raise ValueError("a committee needs models")
    if len(resolved) < 2:
        raise ValueError(
            f"a debate needs at least two seats; {resolved} gives one. A one-seat "
            "committee generates an opening round and is then abandoned by "
            "debate.protocol._check_someone_spoke at every decision point."
        )
    # A repeated model name would seat one model twice in a rotation and break the
    # balance property the design is built on -- and the two seats would be
    # indistinguishable in the stored rows, since a row is keyed by model and
    # persona and nothing else.
    if len(set(resolved)) != len(resolved):
        raise ValueError(f"duplicate model in {resolved}; each model takes one seat")
    return resolved


def _validate_personas(personas: Sequence[Persona]) -> int:
    if not personas:
        raise ValueError("a committee needs at least one persona")
    if len({persona.name for persona in personas}) != len(personas):
        raise ValueError("duplicate persona; the rotation would not be balanced")
    return len(personas)

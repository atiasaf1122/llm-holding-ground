"""The balanced design has to be balanced, and nothing else here checks it.

The rotation is generated arithmetically precisely so that this file can assert
the property it was generated for. A hand-written assignment table with one
transposed cell still looks like a Latin square, and the experiment it produces is
answerable but not the one that was designed.
"""

from __future__ import annotations

from collections import Counter

import pytest

from council.config import get_settings
from council.debate.compositions import (
    Seat,
    balanced_design,
    rotations,
    uniform_references,
)
from council.domain.persona import PERSONAS
from helpers_debate import MODELS, persona_names


def test_every_model_holds_every_persona_exactly_once_across_the_rotations() -> None:
    # Arrange
    design = rotations(models=MODELS)

    # Act
    pairings = Counter(
        (seat.model, seat.persona.name) for composition in design for seat in composition.seats
    )

    # Assert
    assert len(pairings) == len(MODELS) * len(PERSONAS)
    assert set(pairings.values()) == {1}


def test_each_rotation_seats_every_persona_once_when_the_committee_is_four_wide() -> None:
    # Arrange
    design = rotations(models=MODELS)

    # Act
    per_configuration = [
        sorted(seat.persona.name for seat in composition.seats) for composition in design
    ]

    # Assert
    assert per_configuration == [sorted(persona_names())] * len(PERSONAS)


def test_the_balance_holds_for_a_committee_narrower_than_the_persona_set() -> None:
    # Arrange -- the configured default is two models, not four.
    two_models = ("qwen3:8b", "gemma4:latest")

    # Act
    pairings = Counter(
        (seat.model, seat.persona.name)
        for composition in rotations(models=two_models)
        for seat in composition.seats
    )

    # Assert
    assert len(pairings) == len(two_models) * len(PERSONAS)
    assert set(pairings.values()) == {1}


def test_the_balanced_design_is_eight_configurations_not_two_hundred_and_fifty_six() -> None:
    # Act
    design = balanced_design(models=MODELS)

    # Assert
    assert len(design) == len(PERSONAS) * 2
    assert len(design) < len(PERSONAS) ** len(MODELS)


def test_every_configuration_has_one_seat_per_model() -> None:
    # Act
    design = balanced_design(models=MODELS)

    # Assert
    assert all(
        [seat.model for seat in composition.seats] == list(MODELS) for composition in design
    )


def test_identifiers_are_unique_and_name_the_configuration() -> None:
    # Act
    identifiers = [composition.identifier for composition in balanced_design(models=MODELS)]

    # Assert -- these strings land in Decision.composition and in stored parquet,
    # so they are pinned here rather than merely checked for uniqueness.
    assert identifiers == [
        "rotation-0",
        "rotation-1",
        "rotation-2",
        "rotation-3",
        "uniform-momentum-cautious",
        "uniform-momentum-bold",
        "uniform-reversion-cautious",
        "uniform-reversion-bold",
    ]
    assert len(set(identifiers)) == len(identifiers)


def test_a_uniform_reference_seats_one_persona_in_every_chair() -> None:
    # Act
    design = uniform_references(models=MODELS)

    # Assert
    assert [len({seat.persona.name for seat in composition.seats}) for composition in design] == [
        1
    ] * len(PERSONAS)


def test_the_committee_defaults_to_the_configured_models() -> None:
    # Act
    design = rotations()

    # Assert
    assert [seat.model for seat in design[0].seats] == list(get_settings().agent_models)


def test_a_repeated_model_is_rejected() -> None:
    # Assert -- two seats keyed identically would be indistinguishable in the rows.
    with pytest.raises(ValueError, match="duplicate model"):
        rotations(models=("alpha", "alpha"))


def test_an_empty_committee_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least one model"):
        rotations(models=())


def test_a_repeated_persona_is_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate persona"):
        rotations(models=MODELS, personas=(PERSONAS[0], PERSONAS[0]))


def test_a_seat_rebuilt_from_its_parts_is_the_same_seat() -> None:
    # Arrange -- seats come back from stored rows as new objects, and everything
    # that keeps an agent from being shown its own view compares them by value.
    composition = rotations(models=MODELS)[0]
    original = composition.seats[2]

    # Act
    rebuilt = Seat(model=original.model, persona=original.persona)

    # Assert
    assert rebuilt == original
    assert rebuilt in composition.seats
    assert composition.seats.index(rebuilt) == 2

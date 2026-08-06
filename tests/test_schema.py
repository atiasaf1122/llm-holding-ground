"""The rules a schema has to satisfy before a request is worth sending.

An unbounded string field is the expensive one: under constrained decoding the
closing quote is a token the model chooses, so a model with more to say than the
schema has room for keeps writing. Catching that here costs milliseconds;
catching it overnight costs the night.
"""

from __future__ import annotations

from typing import Any

import httpx
import pytest

from council.agents.schema import UnsupportedSchemaError, prepare_schema
from council.domain.signal import MAX_RATIONALE_CHARS
from helpers_provider import SIGNAL_SCHEMA, chat_envelope, make_provider

# -- schema rejection ---------------------------------------------------------


async def test_generate_refuses_an_unbounded_string_before_sending_anything() -> None:
    attempts: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        attempts.append(request)
        return httpx.Response(200, json=chat_envelope("{}"))

    schema = {"type": "object", "properties": {"rationale": {"type": "string"}}}
    provider = make_provider(handler)

    with pytest.raises(UnsupportedSchemaError) as caught:
        await provider.generate(system="persona", user="prices", schema=schema)
    await provider.aclose()

    assert "rationale" in str(caught.value)
    assert attempts == []


def test_an_unbounded_string_hidden_in_an_anyof_is_still_rejected() -> None:
    # How Pydantic renders `str | None`, and the branch that matters is the one
    # that is not null.
    schema = {
        "type": "object",
        "properties": {
            "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
    }

    with pytest.raises(UnsupportedSchemaError) as caught:
        prepare_schema(schema)

    assert "note" in str(caught.value)


def test_an_unbounded_string_inside_an_array_is_rejected() -> None:
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }

    with pytest.raises(UnsupportedSchemaError):
        prepare_schema(schema)


@pytest.mark.parametrize(
    "bound",
    [
        {"maxLength": 40},
        {"enum": ["long", "short"]},
        {"const": "long"},
        {"pattern": "^[a-z]+$"},
        {"format": "date"},
    ],
)
def test_any_bounding_keyword_makes_a_string_acceptable(bound: dict[str, Any]) -> None:
    schema = {"type": "object", "properties": {"field": {"type": "string", **bound}}}

    assert prepare_schema(schema)["properties"]["field"] == {"type": "string", **bound}


def test_the_signal_contract_is_accepted_as_it_stands() -> None:
    prepared = prepare_schema(SIGNAL_SCHEMA)

    assert prepared["properties"]["rationale"]["maxLength"] == MAX_RATIONALE_CHARS


def test_a_reference_at_the_root_of_the_schema_is_resolved() -> None:
    # Pydantic emits exactly this shape -- `{"$defs": ..., "$ref": ...}` -- for
    # any model that mentions itself. Mapping over the root's values instead of
    # expanding it as a node leaves the reference dangling and strips the $defs
    # that could still have resolved it.
    schema = {
        "$ref": "#/$defs/S",
        "$defs": {
            "S": {"type": "object", "properties": {"note": {"type": "string", "maxLength": 40}}}
        },
    }

    prepared = prepare_schema(schema)

    assert "$ref" not in prepared
    assert "$defs" not in prepared
    assert prepared == {
        "type": "object",
        "properties": {"note": {"type": "string", "maxLength": 40}},
    }


def test_an_unbounded_string_behind_a_root_reference_is_still_rejected() -> None:
    # The bounds check is the reason this module exists. A dangling root
    # reference leaves it walking a schema with no string nodes in it, so it
    # passes -- and the run stalls overnight instead.
    schema = {
        "$ref": "#/$defs/S",
        "$defs": {"S": {"type": "object", "properties": {"rationale": {"type": "string"}}}},
    }

    with pytest.raises(UnsupportedSchemaError) as caught:
        prepare_schema(schema)

    assert "rationale" in str(caught.value)


def test_a_recursive_reference_is_refused_rather_than_expanded_forever() -> None:
    schema = {
        "type": "object",
        "properties": {"child": {"$ref": "#/$defs/Node"}},
        "$defs": {"Node": {"type": "object", "properties": {"child": {"$ref": "#/$defs/Node"}}}},
    }

    with pytest.raises(UnsupportedSchemaError) as caught:
        prepare_schema(schema)

    assert "recursive" in str(caught.value)

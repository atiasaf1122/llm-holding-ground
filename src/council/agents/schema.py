"""Making a JSON schema safe to hand to a constrained decoder.

Two things happen to a schema before the daemon sees it, and each one prevents a
failure that is expensive to recognise after the fact.

**Every string field must be bounded.** A constrained-decoding grammar permits any
character inside a JSON string, so the closing quote is a token the model chooses
rather than one the grammar forces. A model with more to say than the schema has
room for therefore keeps writing: measured at 82,000 tokens over ten minutes,
ending in output that cannot be parsed. A field carrying ``enum``, ``const``,
``maxLength``, ``pattern`` or ``format`` cannot do this. One without any of them is
rejected here, before a request is sent, because the alternative is discovering it
overnight.

**References must be gone.** Ollama's grammar generator mis-orders ``$ref`` and
``$defs``, producing a grammar that does not match the schema it was built from.
Inlining the definitions costs a few hundred bytes on the wire and removes the
whole class of problem.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

BOUNDING_KEYWORDS = frozenset({"enum", "const", "maxLength", "pattern", "format"})
"""Keywords that give a string grammar a reason to stop on its own."""

_DEFINITION_SECTIONS = ("$defs", "definitions")
_BRANCH_KEYWORDS = ("anyOf", "oneOf", "allOf")


class UnsupportedSchemaError(ValueError):
    """The schema cannot be used for constrained decoding.

    Deliberately not a provider error. This is a defect in the caller's schema:
    it needs no daemon to detect, every request built from it fails identically,
    and it should stop a run rather than be recorded as one decision's failure.
    """


def prepare_schema(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return the object to send as ``format``, or refuse to send anything.

    Args:
        schema: a JSON schema, typically from ``Model.model_json_schema()``.

    Returns:
        The same schema with references inlined.

    Raises:
        UnsupportedSchemaError: a reference is missing or recursive, or some string
            field has nothing to stop it.
    """
    flat = flatten_refs(schema)
    unbounded = sorted(set(find_unbounded_strings(flat)))
    if unbounded:
        raise UnsupportedSchemaError(
            "unbounded string field(s) "
            + ", ".join(unbounded)
            + "; give each one maxLength, enum, const, pattern or format, or the "
            "model may never close the quote"
        )
    return flat


# -- bounds -------------------------------------------------------------------


def find_unbounded_strings(node: Any, path: str = "") -> Iterator[str]:
    """Yield the path of every string field that no keyword can terminate."""
    if not isinstance(node, Mapping):
        return
    if _is_string_node(node) and BOUNDING_KEYWORDS.isdisjoint(node.keys()):
        yield path or "<root>"
    for child_path, child in _iter_subschemas(node, path):
        yield from find_unbounded_strings(child, child_path)


def _is_string_node(node: Mapping[str, Any]) -> bool:
    declared = node.get("type")
    if isinstance(declared, list):
        return "string" in declared
    return declared == "string"


def _iter_subschemas(node: Mapping[str, Any], path: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(path, subschema)`` for the keywords Pydantic actually emits.

    Only these keywords are descended into. Walking every value indiscriminately
    would treat ``title`` and the members of an ``enum`` as schemas.
    """
    properties = node.get("properties")
    if isinstance(properties, Mapping):
        for name in sorted(properties):
            yield _join(path, str(name)), properties[name]

    additional = node.get("additionalProperties")
    if isinstance(additional, Mapping):
        yield _join(path, "*"), additional

    items = node.get("items")
    if isinstance(items, Mapping):
        yield f"{path}[]", items

    prefix_items = node.get("prefixItems")
    if isinstance(prefix_items, list):
        for index, item in enumerate(prefix_items):
            yield f"{path}[{index}]", item

    for keyword in _BRANCH_KEYWORDS:
        branch = node.get(keyword)
        if isinstance(branch, list):
            for index, item in enumerate(branch):
                yield f"{path}|{keyword}[{index}]", item


def _join(path: str, name: str) -> str:
    return f"{path}.{name}" if path else name


# -- references ---------------------------------------------------------------


def flatten_refs(schema: Mapping[str, Any]) -> dict[str, Any]:
    """Inline every ``$ref`` and drop the definition sections."""
    definitions: dict[str, Any] = {}
    for section in _DEFINITION_SECTIONS:
        entries = schema.get(section)
        if isinstance(entries, Mapping):
            definitions.update({f"#/{section}/{name}": value for name, value in entries.items()})

    # Expanded as one node rather than value by value. Pydantic puts the $ref at
    # the *root* of any self-referencing model -- `{"$defs": ..., "$ref": ...}` --
    # and mapping over the values would hand "$ref" its plain string, leave the
    # reference dangling, and strip the $defs that could still resolve it. The
    # bounds check downstream would then walk a schema with no string nodes in it
    # and pass, which is exactly the failure this module exists to prevent.
    body = {name: value for name, value in schema.items() if name not in _DEFINITION_SECTIONS}
    expanded = _expand(body, definitions, ())
    if not isinstance(expanded, dict):
        raise UnsupportedSchemaError(f"the root schema resolved to a {type(expanded).__name__}")
    return expanded


def _expand(node: Any, definitions: Mapping[str, Any], active: tuple[str, ...]) -> Any:
    if isinstance(node, list):
        return [_expand(item, definitions, active) for item in node]
    if not isinstance(node, Mapping):
        return node

    reference = node.get("$ref")
    if isinstance(reference, str):
        return _resolve(reference, node, definitions, active)
    return {name: _expand(value, definitions, active) for name, value in node.items()}


def _resolve(
    reference: str,
    node: Mapping[str, Any],
    definitions: Mapping[str, Any],
    active: tuple[str, ...],
) -> dict[str, Any]:
    # A model that refers to itself would otherwise expand forever. Refusing is
    # honest: a recursive structure has no bounded grammar either.
    if reference in active:
        raise UnsupportedSchemaError(f"{reference} is recursive and cannot be inlined")

    target = definitions.get(reference)
    if not isinstance(target, Mapping):
        raise UnsupportedSchemaError(f"{reference} is not defined in this schema")

    resolved = _expand(target, definitions, (*active, reference))
    # Keywords written beside a $ref refine it, so they win over the definition.
    siblings = {
        name: _expand(value, definitions, active)
        for name, value in node.items()
        if name != "$ref"
    }
    return {**resolved, **siblings}

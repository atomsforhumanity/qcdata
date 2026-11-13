"""Annotated types for serializing paths and numpy arrays in pydantic models."""

from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import (
    BeforeValidator,
    PlainSerializer,
    SerializerFunctionWrapHandler,
    SkipValidation,
    WithJsonSchema,
    WrapSerializer,
)
from pydantic_core import PydanticSerializationError

StrOrPath = Annotated[str | Path, PlainSerializer(lambda x: str(x))]


def _array_schema(dimensions: int) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": "number"}
    for _ in range(dimensions):
        schema = {"type": "array", "items": schema}
    return schema


# Coerce to independent float64 arrays and dump as ordinary nested Python lists.
# Field validators handle scientific shapes; schemas describe canonical output.
SerializableNDArray = Annotated[
    SkipValidation[np.ndarray],
    BeforeValidator(lambda x: np.array(x, dtype=np.float64)),
    PlainSerializer(lambda x: np.asarray(x).tolist()),
    WithJsonSchema({"type": "array"}),
]
SerializableVector = Annotated[SerializableNDArray, WithJsonSchema(_array_schema(1))]
SerializableMatrix = Annotated[SerializableNDArray, WithJsonSchema(_array_schema(2))]
SerializableTensor3D = Annotated[SerializableNDArray, WithJsonSchema(_array_schema(3))]


def _jsonable(value: Any) -> Any:
    """Numpy arrays and scalars anywhere in a free-form value, as lists and numbers."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _serialize_free(value: Any, handler: SerializerFunctionWrapHandler) -> Any:
    """Serialize as usual; only a value holding numpy is converted first (the slow path)."""
    try:
        return handler(value)
    except PydanticSerializationError:
        return handler(_jsonable(value))


#: A free-form mapping (extras, record, keywords, resolved settings) that may hold numpy
#: arrays; they are written to JSON as nested lists.
FreeDict = Annotated[dict[str, Any], WrapSerializer(_serialize_free, when_used="json")]

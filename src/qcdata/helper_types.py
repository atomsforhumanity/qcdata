from pathlib import Path
from typing import Annotated, Any

import numpy as np
from pydantic import BeforeValidator, PlainSerializer, SkipValidation, WithJsonSchema

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

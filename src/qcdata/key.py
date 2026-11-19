"""The calculation key: whether two inputs are the same calculation.

A store files results under ``calculation_key(inp)``. The key is the sha256 of a
canonical JSON form of the input as given, and nothing else: which calculation, never
which code produced the numbers, so a key never changes when code changes and never
consults a program. Every rule errs toward a miss: two inputs share a key only when they
request the same calculation written the same way. `docs/conventions.md` states the
rules; golden keys in the tests pin them, and changing any of them is a new key version
(``k2``).
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel
from pydantic.fields import FieldInfo

from .models import FileInput, Files, ProgramInput, ProgramSpec, Structure

__all__ = ["calculation_key"]

_VERSION = "k1"
#: Top-level keywords left out of the key, in the input and in every subprogram: they
#: may change results at most at round-off.
_RESOURCE_KEYWORDS = frozenset({"threads", "memory_gib", "device_memory_gib"})
#: Coordinates (bohr) enter the key times 1e10, rounded to integers: in units of 1e-10
#: bohr. 1e10 is exact in binary, so the product is correctly rounded.
_COORDINATE_SCALE = 1e10


def calculation_key(inp: FileInput | ProgramInput) -> str:
    """Return the key under which a store files the calculation ``inp`` requests.

    The key is ``"k1:"`` and the sha256 hex digest of the canonical JSON form of the
    input as given. It names the calculation only: which code produced a stored result
    (each program's numerics identity) is recorded with each run by the runner and the
    store, not hashed into the key. The canonical form has:

    - keys sorted;
    - numbers normalized: an integral float is written as an integer (``100.0`` as
      ``100``, ``-0.0`` as ``0``), any other float in its shortest round-trip form;
    - fields at their default left out, so adding a field to qcdata never changes an
      old key;
    - coordinates in bohr rounded to 1e-10 and written as integers, atom order kept;
    - every other array field (a `given` Hessian, orbital coefficients) as
      ``{"array": shape, "dtype": dtype, "sha256": digest}``, the digest of its C-order
      little-endian bytes: float64 with ``-0.0`` as ``0.0``, integers in their own
      dtype. Arrays inside free-form values such as ``keywords`` stay JSON numbers;
    - attached files by the sha256 of their bytes (UTF-8 for text);
    - subprograms canonicalized the same way, in order.

    Left out: ``extras`` at every level, structure identifiers, and the resource
    keywords ``threads``, ``memory_gib`` and ``device_memory_gib`` at the top level of
    ``keywords``, in the input and in every subprogram. Everything else counts,
    ``given`` and ``backend`` included. Spellings are not normalized: a synonym or a
    default written out explicitly gives another key.

    Args:
        inp: The input as given. A bare `ProgramSpec` is not a calculation; bind it to
            a structure with `ProgramInput.from_spec` first.

    Returns:
        The key, ``"k1:"`` followed by 64 hex digits.

    Raises:
        TypeError: ``inp`` is not an input, or it holds a value JSON cannot represent
            (a set, bytes outside ``files``, a mapping with non-string keys).
        ValueError: The input holds a non-finite number.
    """
    if not isinstance(inp, FileInput) or (
        isinstance(inp, ProgramSpec) and not isinstance(inp, ProgramInput)
    ):
        raise TypeError(
            "A calculation key needs a ProgramInput or a FileInput, not "
            f"{type(inp).__name__}; bind a ProgramSpec with ProgramInput.from_spec."
        )
    text = json.dumps(
        _canonical_model(inp),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return f"{_VERSION}:{hashlib.sha256(text.encode('ascii')).hexdigest()}"


def _canonical_model(model: BaseModel) -> dict[str, Any]:
    """A model's fields as plain JSON values, by the key's rules."""
    out: dict[str, Any] = {}
    for name, field in type(model).model_fields.items():
        if name == "extras" or (isinstance(model, Structure) and name == "identifiers"):
            continue
        value = getattr(model, name)
        if _is_default(value, field):
            continue
        if isinstance(model, ProgramSpec) and name == "keywords":
            value = {k: v for k, v in value.items() if k not in _RESOURCE_KEYWORDS}
            if not value:
                continue
        if isinstance(model, Structure) and name == "geometry":
            out[name] = _coordinates(value)
        elif isinstance(value, np.ndarray):
            out[name] = _array(value)
        elif isinstance(model, Files) and name == "files":
            out[name] = {
                filename: hashlib.sha256(
                    data.encode("utf-8") if isinstance(data, str) else data
                ).hexdigest()
                for filename, data in value.items()
            }
        else:
            out[name] = _canonical(value)
    return out


def _is_default(value: Any, field: FieldInfo) -> bool:
    """Whether a field holds its default, so the key leaves it out."""
    if field.is_required():
        return False
    default = field.get_default(call_default_factory=True)
    if value is None or default is None:
        return value is default
    if isinstance(value, np.ndarray) or isinstance(default, np.ndarray):
        return False
    return type(value) is type(default) and bool(value == default)


def _coordinates(geometry: np.ndarray) -> list[list[int]]:
    """Coordinates (bohr) as integers in units of 1e-10 bohr, atom order kept."""
    scaled = np.asarray(geometry, dtype=np.float64) * _COORDINATE_SCALE
    if not np.isfinite(scaled).all():
        raise ValueError("A calculation key needs finite coordinates.")
    return [[round(x) for x in atom] for atom in scaled.tolist()]


def _array(array: np.ndarray) -> dict[str, Any]:
    """An array field by its shape, dtype and the sha256 of its bytes.

    Floats as C-order little-endian float64 with -0.0 as 0.0, so the digest compares
    values exactly as the JSON form of numbers would; integers and booleans in their
    own dtype, little-endian.
    """
    if array.dtype.kind == "f":
        if not np.isfinite(array).all():
            raise ValueError("A calculation key needs finite numbers in arrays.")
        data = np.ascontiguousarray(array, dtype="<f8") + 0.0
    elif array.dtype.kind in "iub":
        data = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
    else:
        raise TypeError(f"A calculation key cannot represent a {array.dtype} array.")
    return {
        "array": list(array.shape),
        "dtype": data.dtype.name,
        "sha256": hashlib.sha256(data.data).hexdigest(),
    }


def _canonical(value: Any) -> Any:
    """A value as plain JSON: models by their fields, numbers normalized."""
    if isinstance(value, Enum):
        value = value.value
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"A calculation key needs finite numbers, not {value}.")
        return int(value) if value.is_integer() else float(value)
    if isinstance(value, np.ndarray):
        return _canonical(value.tolist())
    if isinstance(value, BaseModel):
        return _canonical_model(value)
    if isinstance(value, Mapping):
        for k in value:
            if not isinstance(k, str):
                raise TypeError(f"A calculation key needs string keys, not {k!r}.")
        return {k: _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    raise TypeError(
        f"A calculation key cannot represent a {type(value).__name__}: {value!r}."
    )

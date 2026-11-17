"""Large arrays saved apart from a model's file, as ``.npy`` files beside it.

Models always hold arrays; only saving decides where an array goes. An array field whose
float64 size exceeds 1 MiB is written to its own ``.npy`` file in the saved file's
directory, named by the saved file's stem and the array's path in the document
(``output.results.hessian.npy``), and the document holds a pointer in its place:

```json
{"path": "output.results.hessian.npy", "shape": [400, 400], "dtype": "float64",
 "sha256": "..."}
```

``sha256`` is the digest of the ``.npy`` file's bytes. Opening the file reads every
array back, checking its digest, shape and dtype, so a loaded model holds arrays again.
Only array fields (`SerializableNDArray`) are saved apart; free-form values such as
``extras`` and ``record`` stay inline.
"""

from __future__ import annotations

import hashlib
import io
import re
import secrets
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import SerializationInfo, ValidationInfo

#: An array field is saved apart when its float64 size exceeds this many bytes (1 MiB).
ARRAY_FILE_BYTES = 1 << 20
#: Serialization context key: the `ArraySink` collecting arrays to save apart.
SINK = "qcdata.array_sink"
#: Validation context key: the directory holding the saved arrays.
DIRECTORY = "qcdata.array_directory"
_POINTER_KEYS = frozenset({"path", "shape", "dtype", "sha256"})


class ArraySink:
    """Collects the arrays a save puts in files, then writes them and their pointers.

    The serializer replaces each large array with a mark, ``{mark: index}``, unique to
    this save so that no saved value can be mistaken for one. `place` then swaps every
    mark for a pointer, naming the file by the mark's path in the document.
    """

    def __init__(self) -> None:
        """Start a save with no arrays."""
        self.mark = f"qcdata.array.{secrets.token_hex(8)}"
        self.arrays: list[np.ndarray] = []

    def add(self, array: np.ndarray) -> dict[str, int]:
        """Hold an array for its own file; return the mark that takes its place."""
        self.arrays.append(array)
        return {self.mark: len(self.arrays) - 1}

    def place(
        self, document: Any, directory: Path, stem: str
    ) -> tuple[Any, list[Path]]:
        """Write each held array to its file and put its pointer in the document.

        Args:
            document: The dumped model, holding marks.
            directory: Where the model's file and its array files go.
            stem: The model file's stem, which every array file name starts with.

        Returns:
            The document with pointers for marks, and the array files written, in
            document order.
        """
        written: list[Path] = []
        if not self.arrays:
            return document, written

        def visit(node: Any, path: tuple[str, ...]) -> Any:
            if isinstance(node, dict):
                if len(node) == 1 and self.mark in node:
                    name = ".".join((stem, *map(_file_part, path), "npy"))
                    written.append(directory / name)
                    return _write(self.arrays[node[self.mark]], directory / name)
                return {k: visit(v, (*path, k)) for k, v in node.items()}
            if isinstance(node, list):
                return [visit(v, (*path, str(i))) for i, v in enumerate(node)]
            return node

        return visit(document, ()), written


def serialize_array(value: Any, info: SerializationInfo) -> Any:
    """Dump an array as nested lists, or as a mark when a save puts it in a file."""
    array = np.asarray(value)
    sink = info.context.get(SINK) if isinstance(info.context, dict) else None
    if sink is not None and array.size * 8 > ARRAY_FILE_BYTES:
        return sink.add(array)
    return array.tolist()


def validate_array(value: Any, info: ValidationInfo) -> np.ndarray:
    """Make an independent float64 array, reading a saved array file's pointer."""
    if isinstance(value, dict):
        context = info.context if isinstance(info.context, dict) else {}
        return _read(value, context.get(DIRECTORY))
    return np.array(value, dtype=np.float64)


def _file_part(part: str) -> str:
    """A path part safe in a file name: other characters, dots included, as %XX."""
    return re.sub(
        r"[^A-Za-z0-9_-]",
        lambda m: "".join(f"%{b:02X}" for b in m.group().encode()),
        part,
    )


def _write(array: np.ndarray, path: Path) -> dict[str, Any]:
    """Write an array as ``.npy`` and return its pointer."""
    buffer = io.BytesIO()
    np.save(buffer, np.ascontiguousarray(array), allow_pickle=False)
    data = buffer.getvalue()
    path.write_bytes(data)
    return {
        "path": path.name,
        "shape": list(array.shape),
        "dtype": array.dtype.name,
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _read(pointer: dict[str, Any], directory: Path | None) -> np.ndarray:
    """Read the array a pointer names, checking it is the one that was saved."""
    if set(pointer) != _POINTER_KEYS:
        raise ValueError(
            "An array field takes numbers, or the pointer a save writes for an array "
            "in its own file (path, shape, dtype, sha256)."
        )
    name = pointer["path"]
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError(
            f"An array file is a name beside the saved file, not {name!r}."
        )
    if directory is None:
        raise ValueError(
            f"This array was saved apart, in {name!r}; open the file it was saved with "
            "(Model.open) to read it back."
        )
    path = Path(directory) / name
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ValueError(f"Cannot read the array file {str(path)!r}: {e}") from None
    if hashlib.sha256(data).hexdigest() != pointer["sha256"]:
        raise ValueError(
            f"The array file {str(path)!r} does not match the sha256 saved with it: it "
            "changed after the save."
        )
    array = np.load(io.BytesIO(data), allow_pickle=False)
    if list(array.shape) != pointer["shape"] or array.dtype.name != pointer["dtype"]:
        raise ValueError(
            f"The array file {str(path)!r} holds {array.dtype.name} {array.shape}, not "
            f"the {pointer['dtype']} {tuple(pointer['shape'])} saved with it."
        )
    return np.asarray(array, dtype=np.float64)

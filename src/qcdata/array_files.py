"""Large arrays and texts saved apart from a model's file, as files beside it.

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

A text in ``files`` whose UTF-8 size exceeds 1 MiB is saved apart the same way, when the
sink is made with ``text_files=True`` (as `QCDataBaseModel.save` does): as its UTF-8
bytes, named by the stem and its path with no suffix added
(``output.results.files.orbitals%2Emolden``), with ``{"path", "sha256"}`` in its place.
Binary files stay inline.

`QCDataBaseModel.save` and `QCDataBaseModel.open` use this module, and any other writer
of qcdata documents with arrays beside them (a worker's wire) can do the same:

```python
from qcdata.array_files import DIRECTORY, SINK, ArraySink

sink = ArraySink()
document = model.model_dump(mode="json", context={SINK: sink})
document, written = sink.place(document, directory, stem)  # arrays to .npy files
model = type(model).model_validate(document, context={DIRECTORY: directory})
```
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

__all__ = [
    "ARRAY_FILE_BYTES",
    "DIRECTORY",
    "POINTER_KEYS",
    "SINK",
    "TEXT_POINTER_KEYS",
    "ArraySink",
    "read_text",
    "serialize_array",
    "validate_array",
]

ARRAY_FILE_BYTES = 1 << 20
"""An array field is saved apart when its float64 size exceeds this many bytes (1 MiB)."""
SINK = "qcdata.array_sink"
"""Serialization context key: the `ArraySink` collecting arrays to save apart."""
DIRECTORY = "qcdata.array_directory"
"""Validation context key: the directory holding the saved arrays."""
POINTER_KEYS = frozenset({"path", "shape", "dtype", "sha256"})
"""The keys of the pointer that stands in a document for an array saved apart."""
TEXT_POINTER_KEYS = frozenset({"path", "sha256"})
"""The keys of the pointer that stands in ``files`` for a text saved apart."""


class ArraySink:
    """Collects the arrays a save puts in files, then writes them and their pointers.

    The serializer replaces each large array (and, with ``text_files``, each large text
    in ``files``) with a mark, ``{mark: index}``, unique to this save so that no saved
    value can be mistaken for one. `place` then swaps every mark for a pointer, naming
    the file by the mark's path in the document.

    Attributes:
        text_files: Whether texts in ``files`` over 1 MiB are saved apart too.
        arrays: The arrays held for their own files, in the order added.
        texts: The texts held for their own files, in the order added.
    """

    def __init__(self, *, text_files: bool = False) -> None:
        """Start a save with no arrays.

        Args:
            text_files: Also save texts in ``files`` over 1 MiB (UTF-8) apart.
        """
        self.mark = f"qcdata.array.{secrets.token_hex(8)}"
        self.text_mark = f"{self.mark}.text"
        self.text_files = text_files
        self.arrays: list[np.ndarray] = []
        self.texts: list[str] = []

    def add(self, array: np.ndarray) -> dict[str, int]:
        """Hold an array for its own file; return the mark that takes its place."""
        self.arrays.append(array)
        return {self.mark: len(self.arrays) - 1}

    def add_text(self, text: str) -> str | dict[str, int]:
        """Hold a text over 1 MiB for its own file and return its mark; else the text.

        Args:
            text: The content of an entry of ``files``.

        Returns:
            The mark that takes the text's place, or the text itself when it stays
            inline (the sink saves no texts, or it is 1 MiB or less).
        """
        # Up to 4 UTF-8 bytes per character: only a text near the limit is encoded.
        if not self.text_files or len(text) * 4 <= ARRAY_FILE_BYTES:
            return text
        if len(text) <= ARRAY_FILE_BYTES and len(text.encode()) <= ARRAY_FILE_BYTES:
            return text
        self.texts.append(text)
        return {self.text_mark: len(self.texts) - 1}

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
        if not self.arrays and not self.texts:
            return document, written

        def visit(node: Any, path: tuple[str, ...]) -> Any:
            if isinstance(node, dict):
                if len(node) == 1 and self.mark in node:
                    name = ".".join((stem, *map(_file_part, path), "npy"))
                    written.append(directory / name)
                    return _write(self.arrays[node[self.mark]], directory / name)
                if len(node) == 1 and self.text_mark in node:
                    name = ".".join((stem, *map(_file_part, path)))
                    written.append(directory / name)
                    text = self.texts[node[self.text_mark]]
                    return _write_text(text, directory / name)
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


def _write_text(text: str, path: Path) -> dict[str, Any]:
    """Write a text as UTF-8 and return its pointer."""
    data = text.encode()
    path.write_bytes(data)
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest()}


def read_text(pointer: dict[str, Any], directory: Path | None) -> str:
    """Read the text a pointer in ``files`` names, checking it is the one saved.

    Args:
        pointer: ``{"path", "sha256"}``, as a save writes it.
        directory: The directory of the saved file (the `DIRECTORY` context).

    Returns:
        The text.

    Raises:
        ValueError: If the pointer is malformed or the file is missing or changed.
    """
    if set(pointer) != TEXT_POINTER_KEYS:
        raise ValueError(
            "A file takes its content, or the pointer a save writes for a text in its "
            "own file (path, sha256)."
        )
    return _read_bytes(pointer, directory, "text").decode()


def _read_bytes(pointer: dict[str, Any], directory: Path | None, what: str) -> bytes:
    """The bytes of the file a pointer names, checked against its sha256."""
    name = pointer["path"]
    if not isinstance(name, str) or Path(name).name != name or name in {".", ".."}:
        raise ValueError(
            f"{what.capitalize()} files are names beside the saved file, not {name!r}."
        )
    if directory is None:
        raise ValueError(
            f"This {what} was saved apart, in {name!r}; open the file it was saved with "
            "(Model.open) to read it back."
        )
    path = Path(directory) / name
    try:
        data = path.read_bytes()
    except OSError as e:
        raise ValueError(f"Cannot read the {what} file {str(path)!r}: {e}") from None
    if hashlib.sha256(data).hexdigest() != pointer["sha256"]:
        raise ValueError(
            f"The {what} file {str(path)!r} does not match the sha256 saved with it: it "
            "changed after the save."
        )
    return data


def _read(pointer: dict[str, Any], directory: Path | None) -> np.ndarray:
    """Read the array a pointer names, checking it is the one that was saved."""
    if set(pointer) != POINTER_KEYS:
        raise ValueError(
            "An array field takes numbers, or the pointer a save writes for an array "
            "in its own file (path, shape, dtype, sha256)."
        )
    data = _read_bytes(pointer, directory, "array")
    path = Path(directory or ".") / pointer["path"]
    array = np.load(io.BytesIO(data), allow_pickle=False)
    if list(array.shape) != pointer["shape"] or array.dtype.name != pointer["dtype"]:
        raise ValueError(
            f"The array file {str(path)!r} holds {array.dtype.name} {array.shape}, not "
            f"the {pointer['dtype']} {tuple(pointer['shape'])} saved with it."
        )
    return np.asarray(array, dtype=np.float64)

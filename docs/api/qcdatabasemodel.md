# Base model behavior

All qcdata models have a read-only `qcdata_version` property holding the installed package version. A dumped object is a document: `model_dump()`, `model_dump_json()`, `save()` and `json_dumps()` write `qcdata_version` at its top level only, even with `exclude_unset=True` (pass `exclude={"qcdata_version"}` to leave it out), and nested objects carry none. `extras` is left out wherever it is empty. Neither appears in the usual representation, to keep scientific values readable.

On loading, a saved stamp is accepted and discarded; serialization supplies the current version. This identifies the last writer of the representation, not the original creator, a minimum reader version, or a formal schema version. Reading a stamped file does not guarantee that an older or newer package can interpret it. Files without a stamp are accepted normally.

JSON, YAML, and TOML preserve this metadata. Structure XYZ exports include it as a comment token. External QCSchema conversions exclude it from model and identifier fields.

## Mutation and persistence

Attribute assignment is frozen, but nested dictionaries, lists, and arrays are not deeply immutable. Direct container mutation is unsupported. Use `model_dump()`, edit the returned built-in values, and reconstruct with `model_validate()` so validation runs again. Scientific array fields dump as independent lists. Arbitrary custom objects in `extras` are not covered by this independence guarantee. File-collection helpers and `Structure.add_identifiers()` remain explicit mutation exceptions.

`save()` returns every file it wrote, the model's file first. An array field whose float64 size exceeds 1 MiB is written to its own `.npy` file beside the saved file, named by the file's stem and the array's path (`output.results.hessian.npy`), and the document holds a pointer with its relative path, shape, dtype and sha256; `open()` reads it back, so models always hold arrays. Smaller arrays stay in the document and round-trip exactly through JSON and YAML. See [Saved files](../conventions.md#saved-files).

`save()` is lossless by default (`exclude_unset=False`), so containers filled after construction are kept. Pass `exclude_unset=True` for a compact file without the fields omitted during construction; explicitly supplied values are retained even when equal to their defaults, and the version stamp is still written. `exclude_none=True` remains the default.

Supported file and identifier helpers mark changed fields as set so those changes are saved. Direct mutation of initially unset containers is unsupported and may be omitted. The dump/edit/validate workflow preserves edits, but reconstructing from a full dump makes its fields explicitly set and can produce larger files.

::: qcdata.models.base_models.QCDataBaseModel

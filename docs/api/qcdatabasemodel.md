# Base model behavior

All qcdata models inherit a computed, read-only `qcdata_version` containing the installed package version. It is included in default dumps and saves, even with `exclude_unset=True`, and appears on each serialized nested model. It is omitted from the usual representation to keep scientific values readable.

On loading, a saved stamp is accepted and discarded; serialization supplies the current version. This identifies the last writer of the representation, not the original creator, a minimum reader version, or a formal schema version. Reading a stamped file does not guarantee that an older or newer package can interpret it. Files without a stamp are accepted normally.

JSON, YAML, and TOML preserve this metadata. Structure XYZ exports include it as a comment token. External QCSchema conversions exclude it from model and identifier fields.

## Mutation and persistence

Attribute assignment is frozen, but nested dictionaries, lists, and arrays are not deeply immutable. Direct container mutation is unsupported. Use `model_dump()`, edit the returned built-in values, and reconstruct with `model_validate()` so validation runs again. Scientific array fields dump as independent lists. Arbitrary custom objects in `extras` are not covered by this independence guarantee. File-collection helpers and `Structure.add_identifiers()` remain explicit mutation exceptions.

`save()` defaults to `exclude_unset=True` for compact files. Fields omitted during construction are excluded; explicitly supplied values are retained even when equal to their defaults. Computed version stamps remain included. Pass `exclude_unset=False` to include unset/default-valued fields. `exclude_none=True` remains the default.

Supported file and identifier helpers mark changed fields as set so those changes are saved. Direct mutation of initially unset containers is unsupported and may be omitted. The dump/edit/validate workflow preserves edits, but reconstructing from a full dump makes its fields explicitly set and can produce larger files.

::: qcdata.models.base_models.QCDataBaseModel

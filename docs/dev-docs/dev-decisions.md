# Development Decisions

## Road Map

- Deal with `pubchem` data API later. There are a few packages on `pypi` that do this. `qcel` has its own interface as well. I think I only need a very basic lookup so I'll probably just do it myself inside one `@classmethod` on `Structure`
- It would be nice to have type safety around the shapes of numpy arrays; however, this is still a [work in progress](https://github.com/numpy/numpy/issues/16544). For now, shapes are enforced by validators and noted as comments.

## Design Decisions

- All units are Atomic Units (AU). E.g., this means geometries in `Structure` are defined in `Bohr`.

- What should be part of the input object and would should be parameters passed to some `compute` function (in `qcop`) that controls the execution of the program?
  - Anything that doesn't directly control the qc program--like parameters and options for `qcop`--shouldn't be part of the input data structure. Rather it should be thought of as control parameters for the executing wrapper (`qcop`). E.g., to collect logs or not should not be an "input" data field, that is a question of whether `qcop` collects and returns logs, it's not an "input" to the program.
- Where to place `program`?
  - `input_data.program` identifies the requested executor. Each recursive `ProgramSpec` identifies its own executor. `results.provenance.program` independently identifies the actual producer; runtime details belong to `execution`.
- `Files` is really the parent class of any input. A `ProgramInput` is an extraction from generic input files of various unstructured values and the placing of those values into structured format. Think of all structured objects as choices about what new values to pull out from native file formats into a more generalized and structured representation of the data.
- Result types inherit from `FileData`, which extends `Files` with required producer provenance. Native output files and parsed scientific values belong to the result payload, along with the program identity and version that must survive standalone serialization. Runtime metadata belongs to `ProgramOutput.execution`; logs and traceback also remain on `ProgramOutput`. The input branch continues to inherit directly from `Files` and has no provenance.
- All container objects that may be empty (e.g. dicts, lists, sets, etc) will be set to an empty object by default rather than `Optional[container] = None`. This makes accessing the attributes for end users more routine because they can run `obj.extras.get()` without having to do `if obj.extras is not None` check first. Or `len(obj.some_list)` while avoiding `not None` checks.

- Originally I had output objects corresponding to each input object (`FileOutput`, `SinglePointOutput`, `OptimizationOutput`, `ProgramFailure`). I've simplified this design by having a single `ProgramOutput` object that is generic over input type and data type. This is a more flexible design that allows for a single object to represent all output types. Advantages include simpler code to reason about, less code to maintain, and a more flexible design that can be extended to new output types without having to create a new object for each one. Generics provide the same level of type safety as individual object. Validators provide the same guarantees of data integrity (i.e., if `success=False` then `traceback` is mandatory). Importantly, this enables end users to open any output file they've saved to disk without knowing _a priori_ what type of output it is. That's a major win for usability.
- Dealing with serialization of NumPy arrays for ChemCloud (and generic JSON schema generation)... See the `helper_types.SerializableNDArray` class. This annotated class appears to give the desired behaviors which are:
  1. Suppress Pydantic errors for this "unknown" type (`SkipValidation`).
  2. Serialize the NumPy array native python data types (lists and floats) (`PlainSerializer`).
  3. Describe canonical array dimensions in JSON schemas using `WithJsonSchema` and vector, matrix, and tensor annotations.
  4. (Optional) Coerce all arrays to `float64` for consistency (`BeforeValidator`).

- Models freeze attribute assignment. Contained dictionaries, lists, and arrays must be treated as read-only; use dump/edit/validate for changes. Explicit file and identifier helpers remain mutation exceptions. Default saves exclude fields omitted during construction for compact files; supported helpers mark changed fields as set. Pass `exclude_unset=False` to include unset/default-valued fields.
- Every model serializes the installed `qcdata_version`. Loading discards the previous stamp and serialization writes the current package version, including on nested objects. This is writer metadata, not a formal schema version or scientific provenance.

## Publishing Checklist

- Checkout `master` branch.
- Pull down all changes.
- `python `scripts/release.py x.x.x`

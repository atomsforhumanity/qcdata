# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [unreleased]

### Added

- Add public `get_data_type(calctype)` and `ProgramInput.from_spec()` helpers for parsers and execution packages.

### Changed

- Every model now writes the installed `qcdata_version` to JSON/YAML/TOML; nested models are stamped too. Loading accepts an existing stamp, and resaving replaces it with the current package version. XYZ structures include a version comment; QCSchema exports exclude qcdata-specific stamps.
- Default `save()` and `json_dumps()` use `exclude_unset=True` for compact files, retaining explicitly supplied values and computed version stamps. Pass `exclude_unset=False` to include unset/default-valued fields. The JSON helper handles individual models and lists consistently, including paths. Models retain shallow freezing; direct container mutation is unsupported and the supported update workflow is dump/edit/validate.
- JSON schemas describe scientific array dimensions correctly. NumPy energy defaults use factories, eliminating non-serializable-default schema warnings.
- QCSchema conversion preserves source extras and additional producer metadata without claiming QCEngine produced every result. Completed public data/identifier exports and corrected viewer documentation.
- The temporary `.data` compatibility property and legacy input key are scheduled for removal in the next major release after this one.

- Breaking: result types now follow the input contract on both success and failure. `FileInput` requires `FileData`; structured inputs require their corresponding scientific data type even for empty failures. Empty payloads deserialize using the input calculation type, including in nested trajectories.
- Empty optimization final values return `None`; empty scan trajectories are supported. Scan structures preserve missing points as `None`, and XYZ export rejects points without a final structure. Visualization handles empty failed outputs.
- Breaking: renamed execution fields to `host_cpu` and `host_mem_gib` (fractional GiB supported), without compatibility aliases. Removed the prototype `Structure.swap_indices` method.
- Updated qcinf to >=0.4.1, removing the qcio dependency and obsolete structure-type workaround.

- `ProgramOutput.results` is now the canonical scientific payload field and serialized key. Scientific models retain their `Data` names. `.data` and legacy `"data"` inputs remain temporarily supported with `FutureWarning`; providing both field names is invalid.
- Breaking: removed the `Results`, `*Results`, and `Molecule` aliases, `qcdata.models.results` and `qcio` import shims, `molecule=` and `ids=` constructor compatibility, stdout/files forwarding, deprecated output convenience properties, and helpers moved to qcinf. Use canonical APIs directly.

- Breaking: result payloads now inherit from `FileData` and require `Provenance` containing program identity and optional version. `ProgramOutput.provenance` is replaced by `execution: ExecutionInfo`, defaulting to `ExecutionInfo()` when runtime details are unknown. Migrate serialized outputs and each trajectory entry by moving producer identity into `results.provenance` and runtime fields into `execution`.

- Breaking: all inputs now require `program`. Replaced `ProgramArgs`, `ProgramArgsSub`, and `DualProgramInput` with recursive `ProgramSpec` and structure-bound `ProgramInput`. Migrate `subprogram` / `subprogram_args` to `subprograms`, giving each child its own `program` and `calctype`; sibling calculation types must be unique. `model` is optional and remains a separate `Model` object. Existing serialized inputs require the same migration.
- Added immediate-child lookup with `get_subprogram(calctype)`, recursive visualization details, and an explicit error when converting a model-free input to QCSchema AtomicInput.

## [0.18.1] - 2026-08-26

### Changed

- Updated GitHub references from `coltonbh` to `atomsforhumanity`.
- Moved documentation from `qcdata.coltonhicks.com` to `qcdata.docs.atomsforhumanity.org.`

## [0.18.0] - 2026-07-15

### Changed

- Moved single-point primary-result validation from `SinglePointData` to `ProgramOutput` so failed calculations can carry empty or partial `SinglePointData` while successful calculations still require their requested primary result.

## [0.17.2] - 2026-05-31

### Added

- `ProgramInput.structures` and `DualProgramInput.structures` for additional named structures required by calculations with multiple complete structure inputs, while keeping `.structure` as the required primary/start/reference structure. [#103](https://github.com/atomsforhumanity/qcdata/pull/103) by [@coltonbh](https://github.com/coltonbh).

## [0.17.1] - 2026-04-21

### Added

- `CalcType.scan` and `ScanData` [#102](https://github.com/atomsforhumanity/qcdata/pull/102).

## [0.17.0] - 2026-03-26

### Changed

- Renamed the package from `qcio` to `qcdata` across source, metadata, documentation, workflows, tests, and serialized XYZ comment prefixes.
- Restored `ProgramOutput` as the primary output model and made `Results` a deprecated compatibility alias.
- Split structured output payload models into a dedicated `data.py` module and separated the API/docs around `inputs`, `outputs`, and `data`.
- Reorganized tests to mirror the package structure by splitting output-container tests from data-model tests.
- Added a temporary `qcio` compatibility shim so dependencies that still import `qcio` continue to work while emitting a deprecation warning.

### Changed

- `Structure.connectivity` validator now automatically coerces bonds to be listed in ascending order and sorted, e.g., if passed `[(2, 0, 1.0), (1, 2, 2.0)]` as the original connectivity it will be coerced to `[(0, 2, 1.0), (1, 2, 2.0)]`. This makes the output deterministic and the comparison between different methods of determining connectivity more straightforward.
- `Structure.adjacency_dict` now returns a regular `dict` instead of a `collections.defaultdict`, avoiding the automatic creation of keys when accessing non-existent neighbors while preserving the same adjacency information.

## [0.16.2] - 2025-11-19

### Added

- `Structure.adjacency_dict` property that generates an adjacency dictionary from the `.connectivity` attribute.

## [0.16.1] - 2025-10-27

### Added

- `@field_validator` to `Structure.connectivity` to ensure no duplicate bonds. [#93](https://github.com/atomsforhumanity/qcdata/pull/93)
- `Structure.adjacency_matrix` property that returns an adjacency matrix from the `.connectivity` information. [#93](https://github.com/atomsforhumanity/qcdata/pull/93)

## [0.16.0] - 2025-10-09

### Changed

- 🚨 Restored the legacy input naming (`ProgramInput`, `ProgramArgs`, `ProgramArgsSub`, `DualProgramInput`, `FileInput`) and removed the temporary compatibility layer introduced in v0.15.0. The `specs.py` module is now `inputs.py`, and documentation/tests reflect the reverted terminology.

## [0.15.0] - 2025-10-06

### Removed

- 🚨Python 3.9 support. Minimum supported version is now 3.10. [#91](https://github.com/atomsforhumanity/qcdata/pull/91)
- All constants and periodic table data moved to [qcconst](https://github.com/atomsforhumanity/qcconst).
- All cheminformatics methods, including those that used `rdkit` and `openbabel` such as `rmsd` and `align`. Placed these algorithms into [qcinf](https://github.com/atomsforhumanity/qcinf) so that `qcio` can remain purely about data structures. All future algorithms will go into `qcinf`.
  - `Structure.from_smiles()` method in favor of functional API using the `qcinf` `smiles_to_structure` function.

    ```python
    from qcinf import smiles_to_structure

    struct = smiles_to_structure("CCO", backend="rdkit")
    ```

  - `Structure.to_smiles()` method in favor of functional API using the `qcinf` `structure_to_smiles` function.
  - `Structure.add_smiles()` convenience method. Use `qcinf.structure_to_smiles` and `Structure.add_identifiers()` instead.

    ```python
    from qcinf import structure_to_smiles

    smiles = structure_to_smiles(struct)
    struct.add_identifiers(smiles=smiles)
    ```

  - `rmsd` and `align` methods now live in `qcinf`:
    ```python
    from qcinf import rmsd, align
    # use functions the same way
    ```

- `ConformerSearchResults.conformers_filtered()` in favor of moving the cheminformatics method to `qcinf.filter_conformer_indices()`.
- `OptimizationResults.molecules` backwards compatibility shim. Use `.structures` instead.

### Changed

- Package dependency system changed from `poetry` to `uv` and build system from `poetry` to `hatchling`.
- Renamed `ProgramOutput` -> `Results` to better match basic parlance ("The program produced these results"). [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `Results.results` (formerly `ProgramOutput.results`) attribute to `.data`. "The program produced this data." [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `Results.stdout` -> `Results.logs`. We are coalescing `stdout` and `stderr` logs anyways. Prefer using a common name scientists will understand. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `SinglePointResults` -> `SinglePointData`. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `OptimizationResults` -> `OptimizationData`. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `ConformerSearchResults` -> `ConformerSearchData`. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Added compatibility classes so old imports looking for these names still work but emit a `FutureWarning`. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `QCIOModelBase` -> `QCIOBaseModel` to match `pydantic` semantics for `BaseModel`. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `ProgramInput` -> `CalcSpec` and added a compatibility shim. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `ProgramArgs` -> `CoreSpec` and added a compatibility shim. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `ProgramArgsSub` -> `SubCalcSpec` and added a compatibility shim. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `DualProgramInput` -> `CompositeCalcSpec` and added a compatibility shim. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Renamed `FileInput` -> `FileSpec` and added a compatibility shim. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)
- Backwards compatibility shim updated from `Results.__init__` override to `@model_validator(mode="before")`. [#90](https://github.com/atomsforhumanity/qcdata/pull/90)

## [0.14.0] - 2025-04-01

### Changed

- Factored out `CalcInfoMixin` from `SinglePointResults`. Both `SinglePointResults` and `OptimizationResults` now inherit these attributes.
- 🚨 `Structure.open()` now only returns a single `Structure` object. `Structure.open_multi()` handles multi-structure xyz files and returns `list[Structure]`. This helps with type checking by having unambiguous values returned for each function.
- `Provenance.scratch_dir` is now `Optional[Path]` instead of `Optional[StrOrPath]`.
- `json_dumps` now excludes unset values by default and accepts `**model_dump_kwargs` for the `.model_dump()` method.

### Removed

- 🚨 Backwards compatibility classes `SinglePointOutput`, `OptimizationOutput`, `ProgramFailure`.

### Added

- `BaseModel.open_multi()` for opening lists of saved objects.

## [0.13.1] - 2025-02-18

### Changed

- `Structure.add_identifiers(name="some_name", smiles="some_smiles")` (or other identifiers) rather than `Structure.add_identifiers({"name": "somename", "smiles": "some_smiles"})`
- No longer using `viewer=grid` for single Py3dMol viewers. This gives better performance as per David Koes, the package maintainer.
- Removed automatic use of SMILES for `view` titles. It was an eyesore for longer SMILES.
- Updated documentation to make it clear when `Bohr`/`Angstroms` where being used. Further standardized on `Bohr`.

### Removed

- `view.view()` no longer uses `Structure.ids.smiles` as a default title.

## [0.13.0] - 2025-02-17

### Changed

- 🚨 The `rmsd` and `align` structures now return the `RMSD` in units of `Bohr` by default. To return in angstroms pass `length_unit="angstrom"`.
- 🚨 Standardized nomenclature across package using `symmetry` and `length_unit` for kwargs:
  - `best` kwarg to `rmsd` has been changed to `symmetry`.
  - `reorder_atoms` kwarg to `align` has been changed to `symmetry`.
  - `generate_structure_viewer_html` (and by extension `view.view`) kwarg `distance_units` renamed to `length_unit`.

### Added

- `LengthUnit` enum model with attributes `BOHR` and `ANGSTROM`.

### Removed

- `DistanceUnits` enum in favor of `LengthUnit`.

## [0.12.4] - 2025-02-05

### Fixed

- `SinglePointResults.normal_modes_cartesian` now has the correct length which should be the same as `SinglePointResults.freqs_wavenumber`.

## [0.12.3] - 2025-01-29

### Added

- `qcio.models.utils.to_multi_xyz()` function that accepts an array of structures and converts them to a multi-xyz string.

## [0.12.2] - 2025-01-29

### Changed

- Updated `numpy` dependency to support both `numpy` `v1` or `v2`. `qcio` does not rely upon any updated `numpy 2` API so no changes were made to the code.

## [0.12.1] - 2025-01-15

### Removed

- `rdkit-stubs`

### Added

- `eval-type-backport` package to dependencies to support the `type | type` syntax in the package.
- `type: ignore` statements for `rdkit` function calls and imports which lack annotations in the `rdkit` package.

## [0.12.0] - 2024-12-18

### Removed

- 🚨 Python 3.8 support.

### Changed

- Dropped `black` and `isort` in favor of `ruff`.

## [0.11.17] - 2024-12-11

### Changed

- `structure_to_smiles()` (and by extension `Structure.to_smiles()`) now use the Hueckel method by default for `rdkit's` `DetermineBonds` function. This is more robust than the default `connect-the-dots` method. Parameters for `use_hueckel`, `use_vdw` and `allow_charged_fragments` are now available for more optionality when determining the bonding in a structure (a prerequisite to generating the SMILES string). Default changed to `use_hueckel` since this should be the most robust.
  - This may result in structures that could formerly use `struct.to_smiles()` directly needing to update the arguments they pass to get the same result. `struct.to_smiles(use_hueckel=False, allow_charged_fragments=True, robust=False)` will result in the same behavior as before.
  - Default behavior to `.structure_to_smiles()` is to use `robust=True` mode which will try to determine the bonds with the user-supplied (or default) values first and then iterate through other possible methods (Van der Waals, varying whether charged fragments are allowed or not, trying the connect-the-dots method) to see if one method can successfully determine the bonding.
- `rmsd` function `align` parameter changed to `best`.
- `rmsd` function with `best=False` now does a naive alignment of structures with no determination of connectivity though it will try to align the structures by making the assumption that the atom indices are are the same between the two structures.

## [0.11.16] - 2024-11-13

### Added

- `utils.rmsd()` and `utils.align()` functions for computing `rmsd` between structures and aligning structures.

## [0.11.15] - 2024-11-11

### Added

- `Structure.swap_indices()` method for changing the indices of a structure's symbols and geometry. Helpful for setting up structures for an NEB run or RMSD calculation in which index labels are important.

## [0.11.14] - 2024-10-16

### Fixed

- `view.view(...)` displays an empty string if `.model` is `None` on `DualProgramInput` rather than raising an exception.

## [0.11.13] - 2024-10-01

###

- `interval` to view animations. Sets the interval between frames in milliseconds. Defaults to 100.

## [0.11.12] - 2024-10-01

### Added

- Added `charge` and `multiplicity` to the Structure column of the output table view.

## [0.11.11] - 2024-09-30

### Changed

- Attributes on models that are not explicitly set (i.e., values set to their default value) will not be serialized and saved. This behavior can be modified by passing `exclude_unset=True` to `.save(...)`.
- `Model` in the table view is now a mini table to better visualize method and basis.

## [0.11.10] - 2024-09-23

### Added

- `Structure.distance(i, j, units='bohr' | 'angstrom')` returns the distance between atoms `i, j`.
- `view.view(... distances=[(0, 1), (4,27), ...])` will display the distances between atoms at these indices.

## [0.11.9] - 2024-08-16

### Changed

- `Structure` views now use `GridViewer` when laying out multiple structures with `py3Dmol` so as to use only one WebGL context for all images. This dramatically improves performance and reduces lag.

### Added

- View for `ConformerSearchResults`.
- `view_2d` option for viewing a `2D` image of a molecule rather than the `3D` viewer.

## [0.11.8] - 2024-08-13

### Added

- `CNAME` file to `/docs` directory.
- `data_source` and `data_url` to `periodic_table` docstring.
- `Structure.from_xyz` and `Structure.to_xyz` now support pulling additional (non `qcio_`) comments into `Structure.extras['xyz_comments']` and serializing these comments back into the `xyz` string.
- `Structure.from_xyz` and `Structure.to_xyz` support pulling in and serializing `qcio__identifiers_*` in the `xyz` comments line.
- `Structure.from_xyz_multi(xyz_str: str) -> List[Structure]` to return a list of `Structure` objects from a multi-structure xyz file.
- `Structure.open` now supports multi-structure xyz files and will return a list of Structure objects.
- New `CalcType.conformer_search` and `ConformerSearchResults` for conformer search calculations.

## [0.11.7] - 2024-07-27

### Added

- Documentation website available at <https://qcio.coltonhicks.com/>.

### Removed

- Removed compatibility methods `Files.open_file` and `Files.open_files`. These now only exist at `Files.add_file` and `Files.add_files`.

## [0.11.6] - 2024-07-26

### Added

- `Structures.to_smiles` now correctly adds charge information the the SMILES using `rdkit` or `openbabel`.

## [0.11.5] - 2024-07-23

### Added

- Install all extra features with `pip install qcio[all]`.
- Better error messaging if `qcio.view` is imported but `view` dependencies are not installed.
- `PeriodicTable.number(number: int) -> Atom` to lookup atoms by atomic number.
- Multi-structure SMILES support for `Structure.from_smiles` using `Open Babel`.
  ```python
    struct = Structure.from_smiles("CCO.O.O", program="openbabel")
  ```
- `Structure.to_smiles()` to generate a canonical smiles string. Supported using `rdkit` or `openbabel`.
- `Structure.add_smiles()` to add `smiles` and/or `canonical_smiles` to `Structure.identifiers`. Can pass `smiles`, `canonical_smiles`, `canonical_explicit_hydrogen_smiles` and `canonical_smiles_program` directly to this method. Alternatively, one can pass no values or `program` and `hydrogen` to generate and set SMILES automatically using `rdkit` or `openbabel`.

### Changed

- Default force field for `Structure.from_smiles` changed from `UFF` -> `MMFF94s`.
- `Structure._from_xyz` is now public and accepts an `XYZ` string rather than a filepath.

## [0.11.4] - 2024-07-22

### Added

- Can pass `charge` and `multiplicity` to `Structure.open("struct.xyz", charge=-1, multiplicity=3)` when opening `xyz` files. Passing these values to files that already contain `charge` or `multiplicity` will raise a `ValueError`.

## [0.11.3] - 2024-07-22

### Fixed

- Fixed `IndexError` when viewing an optimization that failed on the first gradient evaluation.

### Added

- Added `Wall Time` to the `ProgramOutput` table headed in in `view.py`.

## [0.11.2] - 2024-07-20

### Added

- Periodic Table with simple interface.

  ```python
  from qcio.constants import periodic_table as pt

  pt.He.symbol
  pt.He.number
  pt.He.name
  pt.Ht.mass
  pt.He.group
  pt.He.period
  pt.He.block
  pt.He.electron_config

  group_6 = pt.group(6)
  period_3 = pt.period(3)

  # Check Data Source
  pt.data_source
  pt.data_url
  ```

### Removed

- `constants._ELEMENTS` which was a short-term holdover until a periodic table could be added.

## [0.11.1] - 2024-07-19

### Added

- Added support for `titles_extra` and `subtitles_extra` to top-level `view.view` function. This allows users to pass additional data they want displayed after the default title (usually `Structure.name`) is displayed.

## [0.11.0] - 2024-07-19

### Added

- ✨`view`✨ module that enables simple viewing of `Structures` and `ProgramOutput` objects for easy visual analysis and comparison in Jupyter Notebooks. Use with `from qcio import view` and then `view.view(prog_output1, prog_output2, ...)` inside a Jupyter Notebook. See `README.md` for more details.

### Changed

- Refactored `ProgramOutput.files` into the `.results` attribute:
  - `ProgramOutput` now inherits from `QCIOModelBase`.
  - `SinglePointResults` and `OptimizationResults` inherit from `Files`.
    - This means `SinglePointResults` and `OptimizationResults` both return `True` for `isinstance(obj, Files)`.
  - `Files` has been added to `Results` (a valid `ResultsType`).
  - 🚨 BREAKING CHANGE: Removed awkward `NoResults` object from `qcio`. Realizing that `Files` is the correct base case for `.results` (files may exist for all failed or successful calculations) enabled this change.
  - `ProgramOutput.results: ResultsType` now has no default value.
  - This is more logically consistent with the idea that all program results--including files--are found at `.results`. Structured results objects like `SinglePointResults` and `OptimizationResults` essentially parse out data from these files into structured data. `FileInput` calculations behave just like structured calculations now with their outputs (the files produced by the program) available in `.results`. This matches the metal model for input objects that start with `FileInput` and then layer structure onto common values like `.model`, `keywords`, and `calctype`.
    - This means that `FileInput` calculations now produce a `ProgramOutput[FileInput, Files]` object instead of the former `ProgramOutput[FileInput, NoResults]` object. This same change holds for failed single point calculations or failed optimization calculations with no computed results.
  - Added compatibility layer with `FutureWarning` message to preserve backwards compatibility for instantiating `ProgramOutput` objects having `.files` and accessing `ProgramOutput.files` on instantiated objects.
- Cleaned up `.final_energy` and `final_structure` return values on `OptimizationResults`.

### Removed

- `Structure.extra` note about the structure being generated from a force field embedding when instantiated using `.from_smiles(smiles: str)`.

## [0.10.5] - 2024-07-15

### Changed

- `OptimizationResults.trajectory` may contain a series of successful `ProgramOutput[..., SinglePointResults]` objects and then a failed `ProgramOutput[..., NoResults]` object. In this case we set the `.energies` property of the last value to `nan` for the failed calculation rather than `0.0`.
- Updated `QCIOModelBase.__repr_args__` so that `.success` is always shown in the repr, even if `False`.
- Updated `ProgramOutput.__repr_args__` to always show `.success` first. This helps with `ProgramOutput[..., OptimizationsResults]` where it isn't immediately obvious the calculation has failed from the printed string because `.results` isn't `NoResults`.

## [0.10.4] - 2024-07-15

### Added

- `ensure_results` validator to `SinglePointResults` to enforce that either `energy`, `gradient` or `hessian` must be set. I.e., one cannot have an empty `SinglePointResults` object--this should instead be `NoResults`.
- `ensure_noresults_on_failure_for_single_point` to `ProgramOutput` so that setting `.success=False` and `.results=SinglePointResults` is an invalid state.

## [0.10.3] - 2024-07-13

### Fixed

- `Molecule` now correctly inherits from `Structure` for backwards compatibility.

## [0.10.2] - 2024-07-12

### Added

- `OptimizationResults.to_xyz()` to convert the trajectory to an `xyz` multi-structure format.

### Changed

- Reverted back to using `Structure.identifiers` rather than `Structure.ids`. There is an `@property` on `Structure` for `ids` for shorthand in end user cases. Added compatibility shim for passing `ids` upon instantiation so should not be a breaking change.
- Reverted `Identifiers` to being immutable. While it's nice to be able to set `.name` dynamically in a Jupyter Notebook for display purposes, I realized that variable names in notebooks are often reused and it becomes too easy to accidentally rename structures to something unintended. Perhaps better to keep everything immutable. Maybe add a feature like `ProgramOutput.add_display_name(...)` or something akin to this for in-the-moment view modifications. This data would not be persisted upon saving.

## [0.10.1] - 2024-07-10

### Changed

- Fixed a depreciated use of `Structure.identifiers` instead of `Structure.ids`.

## [0.10.0] - 2024-07-10

### Added

- A number of additional constant values.
- `constants._SOLVENTS_DIELECTRIC` for a number of common solvents.
- `Structure.from_smiles` to create a 3D structure from a smiles string.

### Changed

- Updated `BOHR_TO_ANGSTROM` from `0.529177210903` to `0.529177210544` in accordance with https://physics.nist.gov/cgi-bin/cuu/Value?bohrrada0
- Renamed `Molecule` to `Structure` to better suggest that the coordinates may represent a superstructure comprised of many molecules. Maintained backwards compatibility for passing `ProgramInput(..., molecule=Structure(...))`, and for accessing `prog_input.molecule` with depreciation warnings.
- 🚨 Breaking Change: `OptimizationResults.energies` returns a `np.array` now instead of a `list[float]`. This makes it easier for converting the energies to other units like `kcal/mol`.
- 🚨 Breaking Change: `Identifiers.name_common` renamed to `name` for simplicity.
- 🚨 Breaking Change: Renamed `Structure.identifiers` -> `Structure.ids` to reduce verbosity.
- Made `Identifiers` object directly modifiable so one can add names and smiles to structures easily with `mystruc.ids.name = 'my name'` or `mystruct.ids.smiles = 'CCO'`.

## [0.9.3] - 2024-06-13

### Added

- Registration of all concrete `ProgramOutput[InputType, ResultsType]` on the `outputs.py` module so `pickle` can find the classes for serialization and deserialization within `celery`. The previous approach of registering classes upon instantiation of a class did not work because of multiple processes running in `celery`. The worker process in which the class was instantiated had the class at `outputs.ProgramOutput[...]` however the main process would sometimes need to serialize an object and it would not be in the `outputs.py` module. The former approach appeared to work better than I thought because declaring classes as types for a `trajector` registered them on the `outputs.py` module without my realizing it.

## [0.9.2] - 2024-06-12

### Added

- `precision` argument to `Molecule.save()` method to control the decimal precision of the xyz file output. Default is 17 decimal places to capture all `float64` precision.

### Changed

- Added `GetPydanticSerializer` annotation to `SerializableNDArray` to create the correct json docs for ChemCloud (or any other json schema generator).
- Added `BeforeValidator` to `SerializableNDArray` to coerce all arrays to `float64` for consistency.

## [0.9.1] - 2024-06-11

### Changed

- Changed decimal precision of `Molecule.to_xyz()` string representation of geometry values to 17 decimal places to capture all `float64` precision.

## [0.9.0] - 2024-04-23

### Added

- `Molecule.atomic_numbers` property which uses `qcio.constants.ELEMENTS` to map atomic symbols to atomic numbers.
- Generic `ProgramOutput[InputType, ResultsType]` class to handle all program outputs.

### Removed

**THIS IS A BREAKING RELEASE DUE TO THESE REMOVALS! Trying to maintain core end-user API though.**

- `from_qcel_output` function that is no longer needed since I only need to parse out the results from `QCElemental` objects.
- 🚨 BREAKING CHANGE: `calctype_to_output` since all outputs are now `ProgramOutput` objects parameterized by their input type and results type.
- `test_input_to_output_from_qcel_failed_operation` test since we only parse results from `QCElemental` objects and don't need to parse `FailedOperation` objects.
- `ArrayLike3D` type hint since we are specifically accepting only `NDArray` types for all numpy arrays (instead of previously accepting `List[List[List[float]])` as well) and the dimensionality (shape) cannot be enforced at the type level. All array shapes are enforced by validators and noted as comments.
- Unused `SPReturnType` type hint.

### Changed

- 🚨 BREAKING CHANGE: Inheritance hierarchy for inputs changed to `ProgramInput(FileInput, ...)`, `DualProgramInput(ProgramInput, ...)`. Specifically, all inputs are subclasses of `FileInput` now. So `isinstance(obj, FileInput)` that previously returned `False` for `ProgramInput` and `DualProgramInput` objects (which were not children of `FileInput`) now return True. Update these checks to `if type(obj) is FileInput` to retain the original behavior.
- 🚨 BREAKING CHANGE: While a `ProgramFailure` object is provided for end-user backwards compatibility, because of varying Generics parameters, previous `isinstance(obj, ProgramFailure)` checks that used to return `True` will now return `False`. Check for "equality" to old`ProgramFailure`objects with`if obj.success is False` instead.

## [0.8.2] - 2024-04-09

### Added

- `constants.ELEMENTS` mapping of atomic numbers to element symbols. In anticipation of supporting `xtb` in `qcop` which uses atomic numbers instead of element symbols.

## [0.8.1] - 2024-03-29

### Added

- `py.typed` file to enable type checking in other projects that use `qcio`.

### Changed

- Updated `black`.
- Added `fix_com=True` and `fix_orientation=True` to default conversion from `qcio.Molecule` to `qcelemental.Molecule` objects so that qcel doesn't translate or rotate the molecule unsuspectingly without the user's consent. See default behaviors [here](https://github.com/MolSSI/QCElemental/blob/8e5a8cff52a6438ff9d6c1c6bbf1aeb4f02f12e1/qcelemental/models/molecule.py#L262-L281).
- Renamed `SubProgramArgs` to `SubProgramInput` to be more consistent with the naming convention of the other input objects.

## [0.8.0] - 2024-01-12

### Added

- `OptimizationOutput.save("filename.xyz)` now save the trajectory to an xyz file.

### Fixed

- Saved Molecule xyz files now end with a `\n` character.

## [0.7.1] - 2023-09-27

### Added

- Add validation to `Molecule.symbols` so that all symbols will be capitalized, e.g., `NA` -> `Na`, `h` -> `H`, etc.

### Changed

- `Molecule._to_xyz()` is now a public method `Molecule.to_xyz()` that returns an xyz string rather than writing an xyz file to disk.
- Hoisted `utils.json_dumps` to `qcio.json_dumps` for easier import.

## [0.7.0] - 2023-09-19

### Added

- README.md documentation for basic library utilization.

### Changed

- Renamed `DualProgramArgs` to `SubProgramArgs`.
- Updated `dev-decisions.md` documentation.

## [0.6.1] - 2023-09-08

### Changed

- Removed `.traceback` from `__repr_args__` of `ProgramFailure` so object is more human-readable in an interactive session.

## [0.6.0] - 2023-09-08

### Changed

- Renamed methods `.write_files()` -> `save_files()`, `.add_file()` -> `.open_file()`, `add_files()` -> `.open_files()` to keep a more consistent convention with `.save*` dumping objects to disk and `.open*` opening files on disk.

## [0.5.1] - 2023-09-02

### Added

- `qcio.utils.json_dumps` for serializing `pydantic` objects or lists of objects. Helpful for serializing requests to send of HTTP.
- `typos` pre-commit check.
- `@classmethod` decorator to `@field_validator` functions as per [pydantic docs](https://docs.pydantic.dev/latest/usage/validators/#field-validators).
- `SinglePointResults.scf_dipole_moment`.

### Fixed

- Removed invalid objects in `qcio.__init__.py.__all__`.

### Changed

- `model.save()` now saves json with `indent=4` by default.

## [0.5.0] - 2023-08-31

### Changed

- Updated `pydantic` from `v1` -> `v2`.

## [0.4.2] - 2023-08-18

### Added

- `DualProgramArgs` object.

## [0.4.1] - 2023-07-20

### Fixed

- `final_molecule` lookup now returns `None` if `.trajectory` is empty rather than raising `IndexError`.

## [0.4.0] - 2023-07-19

### Changed

- Changed `Provenance.working_dir` -> `scratch_dir`.

## [0.3.5] - 2023-07-18

### Changed

- Dropped `.energies` and `.molecules` on `OptimizationResult` in favor of `@property` lookups on the `.trajectory` attribute. If memory utilization really becomes an issue consider optimizing in future by creating attributes of `.energy` or `.molecules` with smaller memory footprint.

## [0.3.4] - 2023-07-18

### Changed

- Dropped `results.py` file and moved all results objects to `outputs.py` to remove circular dependency.

## [0.3.3] - 2023-07-17

### Fixed

- Fixed `SinglePointResult.normal_modes_cartesian` to be proper 3D structure with xyz displacements for each atom for each normal mode.

### Changed

- `from_qcel_output` now overwrites the core results value with `.return_result` since `QCElemental` does not always write energy/gradient/hessian values to `AtomicResult.properties`.

## [0.3.2] - 2023-07-17

### Fixed

- `Molecule.dict()` raised exception if `.connectivity` did not exist since `super().dict()` removes empty collections for serialization.

## [0.3.1] - 2023-07-17

### Added

- `Molecule.connectivity` for `rdkit` and other MM package support that depends on explicitly defined bonds.

## [0.3.0] - 2023-07-17

### Changed

- Dropped `.program_args` nesting from input objects in favor of a flatter model structure. This makes defining inputs much quick/easier.
- Renamed `calc_type` -> `calctype`.
- Change computational model to use `ProgramInput` and `DualProgramInput` instead of `SinglePointInput` and `OptimizationInput` to enable more flexibility in the future.
- Added frequency analysis values to `SinglePointResults` object.

## [0.2.1] - 2023-06-29

### Fixed

- Removed mistaken call to `print()` leftover from debugging.

## [0.2.0] - 2023-06-29

### Added

- `wall_time`, `hostcpus`, `hostname`, `hostmem` to `Provenance`
- `pstdout` and `ptraceback` as aliased attributes on `Output` and `FailedOutput` to quickly print out the values for easier viewing.
- `Molecule.formula` property to return Hill System molecular formula.
- Serialization support for `yaml` and `toml` in `.save()` and `.open()` methods.
- Compatibility for `QCElemental`. `SinglePointInput` objects can now run `.to_qcel()` to return a `dict` representation of an `AtomicInput` and `.to_output_from_qcel` to convert a `dict` representation of an `AtomicResult` or `FailedOperation` to a `SinglePointSuccessfulOutput` or `SinglePointFailedOutput` object.

### Changed

- Changed `Provenance.working_dir` from `str` -> `Path`

## [0.1.0] - 2023-06-28

### Added

- Core models for Single Point calculations:
  - `FileArgs`
  - `FileInput`
  - `FileSuccessfulOutput`
  - `FileFailedOutput`
  - `Provenance`
  - `Model`
  - `SPCalcType`
  - `SinglePointProgramArgs`
  - `SinglePointInput`
  - `SinglePointSuccessfulOutput`
  - `SinglePointFailedOutput`
  - `SinglePointComputedProperties`
  - `Wavefunction`

[unreleased]: https://github.com/atomsforhumanity/qcdata/compare/0.18.1...HEAD
[0.18.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.18.1
[0.18.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.18.0
[0.17.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.17.2
[0.17.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.17.1
[0.17.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.17.0
[0.16.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.16.2
[0.16.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.16.1
[0.16.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.16.0
[0.15.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.15.0
[0.14.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.14.0
[0.13.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.13.1
[0.13.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.13.0
[0.12.4]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.12.4
[0.12.3]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.12.3
[0.12.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.12.2
[0.12.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.12.1
[0.12.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.12.0
[0.11.17]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.17
[0.11.16]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.16
[0.11.15]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.15
[0.11.14]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.14
[0.11.13]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.13
[0.11.12]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.12
[0.11.11]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.11
[0.11.10]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.10
[0.11.9]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.9
[0.11.8]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.8
[0.11.7]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.7
[0.11.6]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.6
[0.11.5]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.5
[0.11.4]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.4
[0.11.3]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.3
[0.11.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.2
[0.11.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.1
[0.11.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.11.0
[0.10.5]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.10.5
[0.10.4]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.10.4
[0.10.3]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.10.3
[0.10.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.10.2
[0.10.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.10.1
[0.10.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.10.0
[0.9.3]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.9.3
[0.9.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.9.2
[0.9.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.9.1
[0.9.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.9.0
[0.8.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.8.2
[0.8.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.8.1
[0.8.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.8.0
[0.7.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.7.1
[0.7.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.7.0
[0.6.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.6.1
[0.6.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.6.0
[0.5.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.5.1
[0.5.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.5.0
[0.4.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.4.2
[0.4.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.4.1
[0.4.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.4.0
[0.3.5]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.3.5
[0.3.4]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.3.4
[0.3.3]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.3.3
[0.3.2]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.3.2
[0.3.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.3.1
[0.3.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.3.0
[0.2.1]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.2.1
[0.2.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.2.0
[0.1.0]: https://github.com/atomsforhumanity/qcdata/releases/tag/0.1.0

# Outputs

`ProgramOutput` is the core object that captures a QC program execution, including the exact input, the structured output payload at `.results`, logs, traceback information for failures, and [`ExecutionInfo`](./execution.md) at `.execution`. Producer identity belongs to `.results.provenance`.

::: qcdata.ProgramOutput
options:
members: false

## Canonical results field

Scientific models retain their `Data` names and can exist independently of an execution. Within `ProgramOutput`, that data is the calculation's `results`: use `output.results.energy`, `output.results.gradient`, or `output.results.final_structure`.

Serialization always emits `"results"`, never `"data"`. The sole deprecated API is `.data`, which returns the same object as `.results` with a `FutureWarning`. Constructor arguments and serialized payloads containing `"data"` are accepted with the same warning and migrated to `"results"`. Providing both names raises `ValueError` during validation. The rename leaves the caller's original field names unchanged. This compatibility layer is scheduled for removal in the major release following this one.

The old `Results` and `*Results` classes, `qcdata.models.results` module, `qcio` import shim, stdout/files forwarding, and deprecated convenience helpers have been removed. Use `ProgramOutput`, `*Data`, `logs`, `results.files`, and explicit scientific attributes instead.

## Input and result contract

The result type depends on the input, regardless of success or failure:

| Input | Result type |
| --- | --- |
| `FileInput` | `FileData` only |
| Energy, gradient, Hessian `ProgramInput` | `SinglePointData` |
| Optimization or transition-state `ProgramInput` | `OptimizationData` |
| Conformer-search `ProgramInput` | `ConformerSearchData` |
| Scan `ProgramInput` | `ScanData` |

Successful single-point calculations require the requested value; successful optimizations and scans require a nonempty trajectory, and successful conformer searches require conformers. Failed calculations require a traceback and retain the corresponding scientific type, with empty or partial values allowed. Producer provenance is always required.

When deserializing, `ProgramOutput` uses the input contract to select the result model. Empty scientific payloads therefore retain their type when saved and reopened, including inside trajectories. No additional serialized type tag is needed. Explicit generic types must agree with the input contract.

`input_data` belongs on `ProgramOutput`, not inside `results`. Save the whole output when you need scientific values together with the requested calculation and execution context. Save a standalone data object when exchanging only scientific values and producer identity.

## Migrating existing outputs

This is a breaking schema change with no automatic legacy-provenance loader:

- Move `provenance.program`, `provenance.program_version`, and `provenance.extras` into `results.provenance`.
- Move `scratch_dir`, `wall_time`, and `hostname` into `execution`; rename `hostcpus` to `host_cpu` and `hostmem` to `host_mem_gib`. Memory accepts fractional GiB. These renames have no compatibility aliases.
- Remove the old top-level `provenance` field.
- Use `FileData(provenance=...)` only with `FileInput`. For failed structured calculations, construct the corresponding scientific type, such as `SinglePointData(provenance=...)`, and update generic annotations accordingly.
- Apply the same migration to each nested trajectory output. Do not copy parent provenance or execution information to children.

Standalone structured data now also requires explicit provenance. QCSchema conversion preserves source extras and additional provenance metadata without inventing a QCEngine attribution. QCSchema conversion obtains it from `provenance.creator` and `provenance.version` and raises `ValueError` if producer identity is missing.

External callers must migrate too: qccodec should attach the parsed producer and version to the returned data; qccompute should supply `execution` and return the corresponding scientific data type even when a structured calculation fails without producing values.

## Reconstructed trajectory records

Parsers may reconstruct trajectory entries from the parent calculation and saved
structures and energies. Such entries describe recovered scientific evaluations,
which need not be separate process invocations. An energy-only snapshot has
`calctype="energy"`; a snapshot with a recovered gradient may use `"gradient"`.
Every successful entry still satisfies the requested-value contract. Missing an
optional gradient artifact does not by itself establish execution failure.

# Provenance

`Provenance` identifies the program that produced a result. It belongs on `FileData` and every structured result type, accessed as `output.results.provenance`. Saving data independently preserves this identity.

`program` is required; `program_version` is optional. Provenance is also required for manually constructed data and empty failure payloads. Callers must supply the program identity explicitly; qcdata does not infer it from `input_data.program`.

Each trajectory entry has its own data provenance. For example, optimization data can identify geomeTRIC while gradient data identifies TeraChem. Runtime details belong in [`ExecutionInfo`](./execution.md).

::: qcdata.Provenance

For empty failure results, `program` identifies the intended scientific producer even
if execution never began. Its version remains `None` unless known. The output's
`success`, traceback, and execution metadata describe what actually happened.

# Data

`Data` is the union of file-only `FileData` and structured result types. All result types inherit from `FileData` and require `provenance`, which identifies the producing program and its optional version. When contained in a [`ProgramOutput`](./outputs.md), the concrete type depends on the requested [`CalcType`](./calctype.md).

Standalone data objects hold scientific values and producer identity. They do not duplicate the calculation input; retain a `ProgramOutput` when you need that context.

Empty data is valid for every scientific type. For an empty `OptimizationData`, `final_structure` and `final_energy` return `None`; structures and energies are empty collections. For a populated trajectory, a failed final step still contributes its evaluated structure, and any available energy is retained. An unavailable energy is `nan`, independently of execution success.

`ScanData` defaults to an empty trajectory. Its `structures` preserves one entry per scan point, using `None` when a point has no optimization steps. XYZ export raises `ValueError` if a point lacks a final structure rather than silently omitting it.

Array schemas describe the canonical nested JSON representation: vectors are one-dimensional, geometry/gradients/Hessians are matrices, and normal modes have three dimensions. Constructors continue to accept flat arrays where scientific shape validators can reshape them.

::: qcdata.Data

::: qcdata.FileData

::: qcdata.SinglePointData
options:
members: false

::: qcdata.OptimizationData
options:
members:
  - structures
  - final_structure
  - energies
  - final_energy
  - to_xyz
  - save

::: qcdata.ConformerSearchData

::: qcdata.Wavefunction

## Selecting a scientific data type

Use the shared `get_data_type(calctype)` lookup rather than maintaining a mapping in
parsers or execution packages. It accepts either `CalcType` or its string value:

```python
from qcdata import get_data_type

results = get_data_type("optimization")(provenance={"program": "geometric"})
```

File-only execution uses `FileData` directly. Unknown calculation types raise `ValueError`.

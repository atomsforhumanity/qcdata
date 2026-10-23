# Data

`Data` is the union of file-only `FileData` and structured result types. All result types inherit from `FileData` and require `provenance`, which identifies the producing program and its optional version. When contained in a [`ProgramOutput`](./outputs.md), the concrete type depends on the requested [`CalcType`](./calctype.md).

Standalone data objects hold scientific values and producer identity. They do not duplicate the calculation input; retain a `ProgramOutput` when you need that context.

Empty data is valid for every scientific type. For an empty `OptimizationData`, `final_structure` and `final_energy` return `None`; structures and energies are empty collections. For a populated trajectory, a failed final step still contributes its evaluated structure, and any available energy is retained. An unavailable energy is `nan`, independently of execution success.

## Frames: structures inside a calculation

A `Frame` is one structure inside a larger calculation, with what was computed there: an optimizer step, an IRC point, an image of a path. It is a `ProgramOutput` without `input_data` (both build on `OutputBase`: `success`, `logs`, `traceback`, `execution`). The parent's input already holds the program, model, and keywords, so a frame keeps only its `structure` and `results`, a `SinglePointData` that is never `None` (empty, with the producing program's provenance, when nothing was computed). `ProgramInput.from_spec(parent_spec, frame.structure)` rebuilds a step's exact input; `Frame.from_output(child_output)` records a child calculation as a frame. Passing `ProgramOutput`s where frames are expected, or loading trajectories saved before frames existed, converts them, taking each step's structure from its input.

Results that hold structures read the same way: `structures`, `energies` (`nan` where none was computed), `to_xyz()`, and `save("x.xyz")`. That holds for `OptimizationData` (one frame per step), `PathData`, `PathOptimizationData` (its final path), and `ConformerSearchData` (its conformers).

`PathData` is a path of frames in order, with `ts_index` marking the transition state if there is one. For an IRC the frames run from the end of the reverse branch, through the transition state, to the end of the forward branch; for a chain (NEB, string) they are the images, endpoints included. `forward_barrier` and `reverse_barrier` are the transition state's energy above the first and last frames, in Hartree. `PathOptimizationData.trajectory` holds the path at every iteration, like an optimization's steps; `final_path` is the result.

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

::: qcdata.Frame
options:
members:
  - from_output

::: qcdata.PathData
options:
members:
  - structures
  - energies
  - transition_state
  - forward_barrier
  - reverse_barrier
  - to_xyz
  - save

::: qcdata.PathOptimizationData
options:
members:
  - final_path
  - structures
  - energies
  - to_xyz

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

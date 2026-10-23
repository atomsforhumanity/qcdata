Data structures for specifying quantum chemistry calculations. The most commonly used structure is a `ProgramInput` which defines a single calculation.

The hierarchy is `Files → FileInput → ProgramSpec → ProgramInput`. Every input requires `program`, the requested executor; `ProgramOutput.results.provenance.program` records the program that actually produced the result.

`ProgramSpec` describes a calculation without structures. It has its own `program`, `calctype`, optional `model`, `keywords`, `files`, `cmdline_args`, `extras`, and recursive `subprograms`. `Model` remains a distinct scientific concept with `method` and `basis`. A program without a meaningful model, such as an optimizer, can leave `model=None`.

```python
from qcdata import CalcType, ProgramInput, ProgramSpec, Structure

input_data = ProgramInput(
    program="geometric",
    calctype="optimization",
    structure=Structure.open("structure.xyz"),
    keywords={"maxiter": 250},
    subprograms=[
        ProgramSpec(
            program="terachem",
            calctype="gradient",
            model={"method": "wb97x-d3", "basis": "def2-svp"},
            keywords={"convthre": 1e-6},
        ),
    ],
)
gradient_spec = input_data.get_subprogram(CalcType.gradient)
```

Children can contain further `subprograms` to arbitrary depth. Each immediate child must have a unique `CalcType` among its siblings. `get_subprogram(calctype)` accepts a `CalcType` or its string value (for example, `CalcType.gradient` or `"gradient"`), searches immediate children only, and raises `ValueError` if missing.

`Inputs` is `FileInput | ProgramInput`. Use `ProgramInput` directly for structured inputs. A `ProgramSpec` alone is not a top-level input because it lacks structures. Use `structure` for the constructor argument and attribute; the former `molecule` API has been removed.

This replaces `ProgramArgs`, `ProgramArgsSub`, and `DualProgramInput` without compatibility aliases. Migrate `subprogram` and `subprogram_args` to a `subprograms` list with an explicit `program` and `calctype` on each child. Existing serialized inputs also need these fields. QCSchema AtomicInput conversion requires a non-None model and raises `ValueError` otherwise.

`ProgramInput.structure` is the primary/start/reference structure for the calculation. Calculations that require additional complete structures can provide them by role name with `ProgramInput.structures`, for example `structures={"environment": mm_region}`. A role may also hold an ordered list of structures. Calculation-specific tools should validate which structure names are required.

A path optimization (`calctype="path_optimization"`: NEB, string, growing string) is about a chain rather than one structure. Its structures, endpoints included and in order, go in `structures["initial_chain"]`, and `structure` is left out:

```python
ProgramInput(
    program="...",
    calctype="path_optimization",
    structures={"initial_chain": [reactant, *images, product]},  # or [reactant, product]
)
```

Validation enforces this by calculation type: a path optimization needs a chain of at least two structures with the same atoms and no `structure`; every other calculation type requires `structure`. Code that needs one structure calls `inp.require_structure()`, which returns a `Structure` (or raises for a chain input) so type checkers know it is not `None`; `inp.initial_chain` returns the chain. To restart from a previous path, pass `structures={"initial_chain": output.results.final_path.structures}`.

`FileInput.from_directory("inputs", program="psi4")` collects native input files with the required executor identity.

A `FileInput` is an escape hatch that allows you to run _any_ calculation in any QC program (or any program for that matter), even if it isn't a supported [`CalcType`](./calctype.md) in `qcdata` yet. You can use a `FileInput` to store the native input files (text and binary) for a QC program along with the relevant command line args. Using [qcop](https://github.com/atomsforhumanity/qccompute) you can submit a `FileInput` to a QC program and all output files and `logs` produced by that program will be collected and returned in a user-friendly [`ProgramOutput`](./outputs.md) object. `FileInput` allows you to continue to use `qcdata` even for calculations that haven't yet been standardized.

  
::: qcdata.Inputs

::: qcdata.ProgramInput
    options:
        inherited_members: true
        members: 
            - get_subprogram
            - add_file 
            - add_files 
            - save_files

::: qcdata.ProgramSpec
    options:
        inherited_members: true
        members: 
            - get_subprogram
            - add_file 
            - add_files 
            - save_files 

::: qcdata.FileInput
    options:
        inherited_members: true
        members: 
        - add_file 
        - add_files 
        - save_files 
        - from_directory

## Binding a specification to structures

```python
input_data = ProgramInput.from_spec(
    spec, structure, structures={"reference": reference_structure}
)
```

This validates a new input while preserving all specification fields and recursive
children. The original specification is unchanged. Additional structures are optional
and are supplied explicitly when binding.

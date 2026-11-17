# Structure & Identifiers

A `Structure` is the core `qcdata` object for representing a molecule or molecular super structure in 3D space. `Structure` objects can be created directly from `symbol` and `geometry` information (geometry must be in `Bohr`), from `xyz` files, or opened from `Structure` objects previously saved to disk. `with_geometry(positions)` returns a validated copy at new positions, the way to [make an input at a new geometry](inputs.md#an-input-at-a-new-geometry).

::: qcdata.Structure
    options:
        members:
            - from_xyz
            - to_xyz
            - with_geometry
            - distance
            - add_identifiers

::: qcdata.Identifiers
Use `identifiers=` to set identifiers and `.ids` as a read-only shortcut. The former `ids=` constructor argument, `Molecule` alias, and deprecated SMILES helper have been removed. Use qcinf for SMILES conversion.

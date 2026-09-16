# Overview

## 🧠 `qcdata` operates on a simple mental model

- [`Structure`](./structure.md) objects represent a collection of atoms, a molecule, or any super molecular structure in 3D cartesian space.
- [`Input`](./inputs.md) objects define the parameters for a calculation.
- [`ProgramOutput`](./outputs.md) objects store computed values and output files, while [`Data`](./data.md) objects represent the structured result payload for a calculation. `ProgramOutput` also stores the exact input data (`.input_data`) used for a calculation, logs and [`ExecutionInfo`](./execution.md). Each data payload holds its own [`Provenance`](./provenance.md) so you have full visibility into how every result was generated.
- Other objects, such as [`Identifiers`](./structure.md), [`Files`](./files.md), [`Provenance`](./provenance.md), and [`Model`](./model.md) support these core data structures by organizing relevant information in a user-friendly way.
- 💾 Saving your data to disk and re-opening it again later for analysis is as simple as calling [`my_obj.save(/path/to/file.json)`](./qcdatabasemodel.md#qcdata.models.base_models.QCDataBaseModel.save) or [`MyModel.open(/path/to/file.json)`](./qcdatabasemodel.md#qcdata.models.base_models.QCDataBaseModel.open). These methods are not shown in the documentation for individual classes to avoid redundancy. Files can be saved as `.json`, `.yaml`, or `.toml`. `Structure` and `OptimizationData` (which contains a list of `Structures`) may additionally be saved as `.xyz` files for compatibility with external program.

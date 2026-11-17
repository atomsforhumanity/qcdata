# Array files

Models always hold arrays; saving decides where each one goes. `save` writes an array
field over 1 MiB to its own `.npy` file beside the saved file, with a pointer in its place,
and `open` reads it back (see [Conventions](../conventions.md#saved-files)). Any other
writer of qcdata documents with arrays beside them uses the same pieces.

::: qcdata.array_files

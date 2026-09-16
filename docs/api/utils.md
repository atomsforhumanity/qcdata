`json_dumps()` uses JSON serialization for both individual models and lists, including paths, binary files, scientific arrays, and automatic qcdata version stamps. It defaults to `exclude_unset=True` for compact output; pass `False` to include unset/default-valued fields. It accepts `indent` for formatting. Additional model-dump options apply consistently to both forms; `mode`, if supplied, must be `"json"`.

::: qcdata.json_dumps
::: qcdata.to_multi_xyz

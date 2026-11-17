# Calculation key

`calculation_key` says when two inputs are the same calculation: a store files each
result under it, and a lookup by key returns a stored result only for the same
calculation written the same way. The rules, and what is left out, are in
[Conventions](../conventions.md#calculation-key).

```python
from qcdata import calculation_key

key = calculation_key(prog_input, {"axtb": "0.1.1"})  # "k1:3f9a..."
```

::: qcdata.calculation_key

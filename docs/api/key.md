# Calculation key

`calculation_key` says when two inputs are the same calculation: a store files each
result under it, and a lookup by key returns a stored result only for the same
calculation written the same way. It is computed from the input alone; which code
produced a stored result is recorded with each run, not hashed into the key. The rules,
and what is left out, are in [Conventions](../conventions.md#calculation-key).

```python
from qcdata import calculation_key

key = calculation_key(prog_input)  # "k1:f996..."
```

::: qcdata.calculation_key

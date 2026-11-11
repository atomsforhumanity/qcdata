# Outcome: resolved settings, warnings, errors

Every output says what the calculation decided and, if it failed, why, in the same fields
whichever program produced it. `warnings` and `error` are on `ProgramOutput` and on each
`Frame`; `resolved` and `record` only on `ProgramOutput`. The vocabularies are in
[Conventions](../conventions.md#errors).

::: qcdata.Resolved

::: qcdata.ProgramWarning

::: qcdata.ErrorInfo

::: qcdata.ErrorKind

# Session steps: requests, accuracy, failures

A session (qccompute's execution protocol) evaluates one molecule at many geometries.
What one step asks for and how it fails are data, shared by every program:
a `Request`, its `Accuracy` target, and `ProgramError`, the exception form of
[`ErrorInfo`](outcome.md). A step's result is a [`Frame`](data.md). Lineages are any
JSON value; programs key their histories by `lineage_key`. An input's own target, for
`compute` and for requests that give none, is the shared keyword `accuracy`
([conventions](../conventions.md#accuracy-targets)), read with `Accuracy.from_keywords`.

::: qcdata.Request

::: qcdata.Accuracy

::: qcdata.ProgramError

::: qcdata.lineage_key

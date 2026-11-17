# Execution information

Every `ProgramOutput` has `execution: ExecutionInfo`, defaulting to an empty `ExecutionInfo()`. Individual fields may be unknown; explicit `execution=None` is invalid. Passing `ExecutionInfo()` or `execution={}` means `output.execution.wall_time` returns `None` without requiring a check for the execution object itself.

`host_cpu` counts logical CPUs; `host_mem_gib` accepts fractional values. The former `hostcpus` and `hostmem` names are no longer accepted.

`wall_time` is measured in seconds and `host_mem_gib` in GiB. `host_cpu` and `host_mem_gib` describe host capacity, not the resources allocated to the job. Each frame or scan point records its own execution information, including its working directory.

`identities` says which code made the results (each program's numerics identity, set by the runner), and `store_run` names the stored run an output is (`"<key>/<run>"`): set on every output a store returns and on one a runner has just stored, None on an output no store holds.

::: qcdata.ExecutionInfo

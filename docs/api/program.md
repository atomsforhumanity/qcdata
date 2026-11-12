# Programs and sessions

The protocol an AFH program implements so qccompute can discover, validate, schedule and run
it, and the sessions drivers use to evaluate one molecule at many geometries. A program
registers one object under the `qcdata.programs` entry point group and depends on qcdata
only. Design: [the program interface](../design/program-interface.md).

`qcdata.testing.check_program(program, inputs)` checks the contract (install
`qcdata[testing]` to validate keywords against the program's JSON Schema), and
`qcdata.testing.toy` is a complete example program (Lennard-Jones).

::: qcdata.program.Program

::: qcdata.program.SessionProgram

::: qcdata.program.BatchProgram

::: qcdata.program.Session

::: qcdata.program.Capabilities

::: qcdata.program.Plan

::: qcdata.program.Resources

::: qcdata.program.RunContext

::: qcdata.program.Request

::: qcdata.program.ProgramError

::: qcdata.program.ComputeSession

::: qcdata.program.find_program

::: qcdata.program.register_program

::: qcdata.program.unregister_program

::: qcdata.program.available_programs

::: qcdata.testing.check_program

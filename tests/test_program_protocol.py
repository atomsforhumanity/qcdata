"""The Program/Session protocol, its conformance suite, and the toy program."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qcdata import CalcType, ErrorKind, ProgramInput, ProgramSpec, Structure
from qcdata.program import (
    BatchProgram,
    ComputeSession,
    Program,
    ProgramError,
    Request,
    RunContext,
    Session,
    SessionProgram,
    available_programs,
    find_program,
)
from qcdata.testing import ConformanceError, check_program
from qcdata.testing.toy import ToyProgram, program

ARGON3 = Structure(
    symbols=["Ar", "Ar", "Ar"],
    geometry=np.array([[0.0, 0.0, 0.0], [7.2, 0.0, 0.0], [3.0, 6.1, 0.4]]),
)


def toy_input(calctype: str = "gradient", **keywords: Any) -> ProgramInput:
    return ProgramInput.model_validate(
        {
            "program": "toy",
            "calctype": calctype,
            "structure": ARGON3,
            "model": {"method": "lj"},
            "keywords": keywords,
        }
    )


def test_toy_implements_every_protocol():
    assert isinstance(program, Program)
    assert isinstance(program, SessionProgram)
    assert isinstance(program, BatchProgram)


def test_toy_conforms():
    ran = check_program(
        program,
        [
            toy_input("gradient"),
            toy_input("energy", sigma=6.5, output={"pair_energies": True}),
            toy_input("hessian", epsilon=0.02),
        ],
    )
    assert "session survives a failed request" in ran
    assert "batch agrees with compute" in ran


def test_toy_gradient_and_hessian_are_derivatives():
    x = np.asarray(ARGON3.geometry)
    out = program.compute(toy_input("hessian"), RunContext())
    g, h = out.results.gradient.reshape(-1), out.results.hessian
    step = 1e-4
    with program.open_session(toy_input(), ARGON3, RunContext()) as s:
        for k in range(9):
            d = np.zeros(9)
            d[k] = step
            plus = s.compute((x.reshape(-1) + d).reshape(-1, 3), "gradient").results
            minus = s.compute((x.reshape(-1) - d).reshape(-1, 3), "gradient").results
            fd_g = (plus.energy - minus.energy) / (2 * step)
            fd_h = (plus.gradient - minus.gradient).reshape(-1) / (2 * step)
            assert fd_g == pytest.approx(g[k], abs=1e-9)
            assert fd_h == pytest.approx(h[k], abs=1e-7)


def test_toy_resolves_auto_and_reports_why():
    out = program.compute(toy_input(sigma="auto"), RunContext())
    assert out.resolved.settings["sigma"] == 6.0
    assert "sigma" in out.resolved.reasons
    assert out.execution.timings["total"] >= 0
    assert sum(out.results.energy_components.values()) == pytest.approx(
        out.results.energy
    )


def test_unknown_keys_get_suggestions_and_typed_errors():
    with pytest.raises(ProgramError, match="Did you mean 'sigma'") as e:
        program.plan(toy_input(sigmaa=1.0))
    assert e.value.error.kind is ErrorKind.input
    assert e.value.error.code == "unknown_key"
    out = program.compute(toy_input(output={"pair_energie": True}), RunContext())
    assert not out.success
    assert out.error.kind is ErrorKind.input
    assert "output.pair_energies" in out.error.message


@pytest.mark.parametrize("kind", ["convergence", "resource", "internal"])
def test_failures_are_returned_with_kinds(kind):
    out = program.compute(toy_input(fail=kind), RunContext())
    assert not out.success and out.traceback
    assert out.error.kind == kind
    assert out.resolved.settings["fail"] == kind


def test_unsupported_calctype():
    with pytest.raises(ProgramError) as e:
        program.plan(toy_input("optimization"))
    assert e.value.error.kind is ErrorKind.unsupported


def test_check_program_reports_every_broken_contract():
    class Lenient(ToyProgram):
        """Accepts unknown keys and forgets the gradient of Hessian sessions."""

        def plan(self, inp):
            clean = {k: v for k, v in inp.keywords.items() if k != "zz_not_a_setting"}
            return super().plan(inp.model_copy(update={"keywords": clean}))

        def compute(self, inp, ctx):
            clean = {k: v for k, v in inp.keywords.items() if k != "zz_not_a_setting"}
            return super().compute(inp.model_copy(update={"keywords": clean}), ctx)

    with pytest.raises(ConformanceError) as e:
        check_program(Lenient(), [toy_input()])
    message = str(e.value)
    assert "unknown key rejected by plan" in message
    assert "unknown key fails compute with kind input" in message


def test_compute_session_over_a_program():
    def compute(inputs, previous=None):
        if isinstance(inputs, ProgramInput):
            return program.compute(inputs, RunContext())
        return [program.compute(i, RunContext()) for i in inputs]

    spec = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})
    x = np.asarray(ARGON3.geometry)
    with ComputeSession(compute, spec, ARGON3) as s:
        assert isinstance(s, Session)
        frame = s.compute(x)
        r = frame.results
        assert s.program_version  # learned from the first result
        assert r.gradient.shape == (3, 3)
        assert np.allclose(frame.structure.geometry, x)
        with pytest.raises(ProgramError) as e:
            s.compute(np.full_like(x, np.nan))
        assert e.value.error.kind is ErrorKind.numerical
        assert s.compute(x).results.energy == r.energy  # still usable
        many = s.compute_many([Request(x), Request(np.full_like(x, np.nan))])
        assert many[0].success and not many[1].success
        assert many[1].error.kind is ErrorKind.numerical
        with pytest.raises(ProgramError) as e:
            s.compute(x, outputs=["dipole"])
        assert e.value.error.kind is ErrorKind.unsupported
        with pytest.raises(ProgramError, match="atoms"):
            s.compute(x[:2])
    with pytest.raises(RuntimeError, match="closed"):
        s.compute(x)


def test_compute_session_completes_hessians_without_gradients():
    def no_gradient(inp, previous=None):
        out = program.compute(inp, RunContext())
        if inp.calctype == CalcType.hessian:
            return out.model_copy(
                update={"results": out.results.model_copy(update={"gradient": None})}
            )
        return out

    spec = ProgramSpec(program="toy", calctype="hessian", model={"method": "lj"})
    s = ComputeSession(no_gradient, spec, ARGON3)
    h = s.compute(np.asarray(ARGON3.geometry), "hessian").results
    assert h.gradient is not None and h.hessian is not None


def test_run_context_sessions():
    spec = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})
    with pytest.raises(RuntimeError, match="neither sessions nor a compute"):
        RunContext().session(spec, ARGON3)
    ctx = RunContext(compute=lambda i, previous=None: program.compute(i, RunContext()))
    assert isinstance(ctx.session(spec, ARGON3), ComputeSession)
    opened = []

    def open_session(spec, structure, threads, calctypes):
        opened.append(threads)
        return program.open_session(
            spec, structure, RunContext(threads=threads), calctypes
        )

    ctx = RunContext(threads=16, open_session=open_session)
    with ctx.session(spec, ARGON3, threads=2) as s:
        s.compute(np.asarray(ARGON3.geometry))
    assert opened == [2]


def test_discovery_of_unregistered_programs():
    assert find_program("no-such-program") is None
    assert isinstance(available_programs(), list)


def test_program_error_shorthand():
    e = ProgramError.of("convergence", "SCF", code="max_iter")
    assert e.error.code == "max_iter" and e.results is None
    assert "convergence: SCF" in str(e)


def test_toy_session_keeps_history_per_lineage():
    x = np.asarray(ARGON3.geometry)
    y = x + 0.1
    with program.open_session(toy_input(), ARGON3, RunContext()) as s:
        assert s.compute(x, lineage="left").record["warm_start"] is False
        right = s.compute(y, lineage="right")
        assert right.record["warm_start"] is False
        left = s.compute(y, lineage="left")
        assert left.record["step_from_previous"] == pytest.approx(np.linalg.norm(y - x))
        with pytest.raises(ProgramError):
            s.compute(np.full_like(x, np.nan), lineage="left")
        assert s.compute(y, lineage="left").record["step_from_previous"] == 0.0
        s.reset("left")
        assert s.compute(y, lineage="left").record["warm_start"] is False
        assert s.compute(y, lineage="right").record["warm_start"] is True
        pairs = s.compute(x, outputs={"pair_energies"}).record["pair_energies"]
        assert len(pairs) == 3


def test_accuracy_of_a_bare_number_targets_the_highest_derivative():
    from qcdata.program import Accuracy

    assert Accuracy.of(1e-4, CalcType.gradient) == Accuracy(gradient=1e-4)
    assert Accuracy.of(1e-7, CalcType.energy) == Accuracy(energy=1e-7)
    assert Accuracy.of(None, CalcType.hessian) is None
    a = Accuracy(energy=1e-8, gradient=1e-5)
    assert Accuracy.of(a, CalcType.hessian) is a


class FiniteDifference(ToyProgram):
    """A driver: its subprogram's gradient by central differences, through a session."""

    name = "fd"

    def capabilities(self):
        return (
            super()
            .capabilities()
            .model_copy(update={"driver": True, "sessions": False, "batches": False})
        )

    def settings_schema(self, calctype):
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "step": {"type": "number", "default": 1e-4, "x-unit": "bohr"}
            },
        }

    def plan(self, inp):
        from qcdata import Resolved
        from qcdata.program import Plan

        if set(inp.keywords) - {"step"}:
            raise ProgramError.of("input", "fd takes only 'step'")
        return Plan(
            resolved=Resolved(settings={"step": inp.keywords.get("step", 1e-4)})
        )

    def compute(self, inp, ctx):
        from qcdata import ExecutionInfo, ProgramOutput, Provenance, SinglePointData

        try:
            plan = self.plan(inp)
        except ProgramError as e:
            return ProgramOutput(
                input_data=inp,
                success=False,
                traceback=e.error.message,
                error=e.error,
                results=SinglePointData(provenance=Provenance(program="fd")),
            )
        step = plan.resolved.settings["step"]
        child = inp.subprograms[0]
        x = np.asarray(inp.structure.geometry, dtype=float).reshape(-1)
        requests = []
        for k in range(len(x)):
            for sign in (1, -1):
                d = np.zeros_like(x)
                d[k] = sign * step
                requests.append(
                    Request((x + d).reshape(-1, 3), CalcType.energy, lineage=k)
                )
        with ctx.session(child, inp.structure) as s:
            frames = s.compute_many(requests)
            center = s.compute(x.reshape(-1, 3), CalcType.energy).results
        energies = np.array([f.results.energy for f in frames])
        return ProgramOutput(
            input_data=inp,
            success=True,
            results=SinglePointData(
                provenance=Provenance(program="fd", program_version=self.version()),
                energy=center.energy,
                gradient=(energies[0::2] - energies[1::2]) / (2 * step),
            ),
            resolved=plan.resolved,
            execution=ExecutionInfo(timings={"total": 0.0}),
        )


def fd_input(child_program: str = "toy", **child_keywords: Any) -> ProgramInput:
    return ProgramInput.model_validate(
        {
            "program": "fd",
            "calctype": "gradient",
            "structure": ARGON3,
            "subprograms": [
                {
                    "program": child_program,
                    "calctype": "energy",
                    "model": {"method": "lj"},
                    "keywords": child_keywords,
                }
            ],
        }
    )


def test_drivers_conform_with_local_children():
    ran = check_program(FiniteDifference(), [fd_input()], children={"toy": program})
    assert "input 0 (gradient): compute succeeds" in ran


def test_local_context_runs_children_in_process():
    from qcdata.program import local_context

    out = FiniteDifference().compute(fd_input(), local_context({"toy": program}))
    exact = program.compute(toy_input("gradient"), RunContext())
    assert np.allclose(out.results.gradient, exact.results.gradient, atol=1e-8)
    ctx = local_context({})
    missing = ctx.compute(toy_input().model_copy(update={"program": "no-such"}))
    assert not missing.success and missing.error.kind is ErrorKind.environment


def test_toy_sessions_fail_on_request():
    spec = ProgramSpec(
        program="toy",
        calctype="gradient",
        model={"method": "lj"},
        keywords={"fail": "convergence"},
    )
    with program.open_session(spec, ARGON3, RunContext()) as s:
        with pytest.raises(ProgramError) as e:
            s.compute(np.asarray(ARGON3.geometry))
        assert e.value.error.kind is ErrorKind.convergence
        frame = s.compute_many([Request(np.asarray(ARGON3.geometry))])[0]
        assert not frame.success and frame.error.kind is ErrorKind.convergence


def test_sessions_run_only_the_calctypes_they_were_opened_for():
    from qcdata.program import session_calctypes

    spec = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})
    assert session_calctypes(spec) == {CalcType.energy, CalcType.gradient}
    assert CalcType.gradient in session_calctypes(spec, ["hessian"])
    x = np.asarray(ARGON3.geometry)
    with program.open_session(spec, ARGON3, RunContext()) as s:
        assert s.compute(x, "energy").results.energy is not None  # lower derivative
        with pytest.raises(ProgramError) as e:
            s.compute(x, "hessian")
        assert e.value.error.kind is ErrorKind.input
        assert e.value.error.code == "not_applicable"
        frame = s.compute_many([Request(x, CalcType.hessian)])[0]
        assert not frame.success and frame.error.code == "not_applicable"
    with program.open_session(
        spec, ARGON3, RunContext(), frozenset({CalcType.gradient, CalcType.hessian})
    ) as s:
        assert s.compute(x, "hessian").results.hessian is not None

    def compute(inputs, previous=None):
        return program.compute(inputs, RunContext())

    plain = ComputeSession(compute, spec, ARGON3)
    with pytest.raises(ProgramError) as e:
        plain.compute(x, "hessian")
    assert e.value.error.code == "not_applicable"
    ctx = RunContext(compute=compute)
    assert (
        ctx.session(spec, ARGON3, calctypes=["hessian"]).compute(x, "hessian").success
    )


def test_scratch_is_made_on_first_use_only():
    made = []

    def make():
        made.append(1)
        return Path("/tmp")

    ctx = RunContext(make_scratch=make)
    assert not made
    assert ctx.scratch == Path("/tmp") and ctx.scratch == Path("/tmp")
    assert made == [1]
    assert RunContext().scratch is None

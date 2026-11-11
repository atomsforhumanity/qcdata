"""The Program/Session protocol, its conformance suite, and the toy program."""

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
            plus = s.compute((x.reshape(-1) + d).reshape(-1, 3), "gradient")
            minus = s.compute((x.reshape(-1) - d).reshape(-1, 3), "gradient")
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
        r = s.compute(x)
        assert s.program_version  # learned from the first result
        assert r.gradient.shape == (3, 3)
        with pytest.raises(ProgramError) as e:
            s.compute(np.full_like(x, np.nan))
        assert e.value.error.kind is ErrorKind.numerical
        assert s.compute(x).energy == r.energy  # still usable
        many = s.compute_many([Request(x), Request(np.full_like(x, np.nan))])
        assert isinstance(many[0], type(r)) and isinstance(many[1], ProgramError)
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
    h = s.compute(np.asarray(ARGON3.geometry), "hessian")
    assert h.gradient is not None and h.hessian is not None


def test_run_context_sessions():
    spec = ProgramSpec(program="toy", calctype="gradient", model={"method": "lj"})
    with pytest.raises(RuntimeError, match="neither sessions nor a compute"):
        RunContext().session(spec, ARGON3)
    ctx = RunContext(compute=lambda i, previous=None: program.compute(i, RunContext()))
    assert isinstance(ctx.session(spec, ARGON3), ComputeSession)
    opened = []

    def open_session(spec, structure, threads):
        opened.append(threads)
        return program.open_session(spec, structure, RunContext(threads=threads))

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

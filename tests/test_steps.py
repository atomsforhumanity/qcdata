"""A session step as data: Request, Accuracy, ProgramError, lineages."""

import pickle

import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import (
    Accuracy,
    CalcType,
    ErrorInfo,
    ErrorKind,
    Frame,
    ProgramError,
    Provenance,
    Request,
    SinglePointData,
    Structure,
    lineage_key,
)


def test_request_defaults_mean_the_sessions_own():
    r = Request()
    assert r.positions is None and r.calctype is None and r.lineage is None
    assert r.seed == [] and r.commit is True and r.outputs == frozenset()


def test_request_positions_are_n_by_3_bohr():
    assert Request(positions=np.zeros(9)).positions.shape == (3, 3)
    with pytest.raises(ValidationError, match="n_atoms x 3"):
        Request(positions=np.zeros((2, 4)))


def test_requests_cross_the_wire_unchanged():
    r = Request(
        positions=np.arange(6.0).reshape(2, 3),
        calctype="hessian",
        accuracy=Accuracy(gradient=1e-5),
        lineage=("irc", "forward"),
        outputs={"ir"},
        seed=[None, "ts"],
        commit=False,
    )
    back = Request.model_validate_json(r.model_dump_json())
    assert np.array_equal(back.positions, r.positions)
    assert back.calctype is CalcType.hessian and back.accuracy == r.accuracy
    assert back.lineage == ["irc", "forward"]  # a list in JSON...
    assert lineage_key(back.lineage) == lineage_key(r.lineage)  # ...the same lineage
    assert back.seed == [None, "ts"] and not back.commit and back.outputs == {"ir"}


def test_lineages_are_json_values_keyed_canonically():
    assert lineage_key(None) == "null"
    assert lineage_key({"b": 1, "a": ("fd", 2)}) == '{"a":["fd",2],"b":1}'
    assert Request(lineage=("fd", 3)).lineage == ("fd", 3)  # hashable in process
    for bad in ({1, 2}, object(), float("nan")):
        with pytest.raises(ValidationError, match="JSON value"):
            Request(lineage=bad)


def test_accuracy_targets():
    assert Accuracy.of(1e-5, "gradient") == Accuracy(gradient=1e-5)
    assert Accuracy.of(1e-7, CalcType.energy) == Accuracy(energy=1e-7)
    assert Accuracy.of(None, "energy") is None
    for bad in (0.0, -1e-6, float("inf")):
        with pytest.raises(ValidationError, match="positive, finite"):
            Accuracy(energy=bad)


def test_program_errors_carry_the_failed_frame_and_pickle():
    structure = Structure(symbols=["He"], geometry=[[0.0, 0.0, 0.0]])
    partial = SinglePointData(provenance=Provenance(program="toy"), energy=-2.8)
    error = ErrorInfo(kind=ErrorKind.convergence, message="SCF", code="max_iter")
    frame = Frame(
        structure=structure,
        results=partial,
        success=False,
        error=error,
        traceback="SCF did not converge",
        record={"scf": {"iterations": 128}},
    )
    e = ProgramError(error, frame=frame)
    assert e.results is partial and e.frame.record["scf"]["iterations"] == 128
    back = pickle.loads(pickle.dumps(e))
    assert back.error == error and back.frame == frame and back.results == partial
    assert str(ProgramError.of("input", "bad", code="unknown_key")) == "input: bad"
    assert ProgramError.of("numerical", "nan").frame is None


def test_inputs_carry_results_given_at_their_structure():
    from qcdata import ProgramInput

    h2 = Structure(symbols=["H", "H"], geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    ts = SinglePointData(
        provenance=Provenance(program="linus"),
        energy=-1.1,
        gradient=np.zeros((2, 3)),
        hessian=np.eye(6),
    )
    irc = ProgramInput(program="qcopt", calctype="irc", structure=h2, given=ts)
    back = ProgramInput.model_validate_json(irc.model_dump_json())
    assert np.array_equal(back.given.hessian, ts.hessian)
    assert back.given.provenance.program == "linus"
    assert ProgramInput(program="x", calctype="energy", structure=h2).given is None
    with pytest.raises(ValidationError, match=r"given.hessian has shape \(3, 3\)"):
        ProgramInput(
            program="qcopt",
            calctype="irc",
            structure=h2,
            given=ts.model_copy(update={"hessian": np.eye(3)}),
        )

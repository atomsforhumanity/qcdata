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
    ProgramErrors,
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


def test_program_errors_carry_a_whole_failed_output_and_pickle():
    from qcdata import OptimizationData, ProgramInput, ProgramOutput, Resolved

    he2 = Structure(symbols=["He", "He"], geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 5.6]])
    error = ErrorInfo(kind=ErrorKind.convergence, message="2 steps", code="max_iter")
    frames = [
        Frame(
            success=True,
            structure=he2,
            results=SinglePointData(provenance=Provenance(program="toy"), energy=e),
        )
        for e in (-5.6, -5.7)
    ]
    inp = ProgramInput(
        program="qcopt",
        calctype="optimization",
        structure=he2,
        subprograms=[{"program": "toy", "calctype": "gradient"}],
    )
    out = ProgramOutput.failure(
        inp,
        error,
        results=OptimizationData(provenance=Provenance(program="qcopt"), frames=frames),
        record={"steps": 2},
        resolved=Resolved(settings={"convergence": {"max_iter": 2}}),
    )

    e = ProgramError(out.error, output=out)
    assert e.output is out and e.frame is None
    assert e.results is out.results
    assert isinstance(e.results, OptimizationData) and len(e.results.frames) == 2

    back = pickle.loads(pickle.dumps(e))
    assert back.error == error and back.output == out and back.results == out.results
    assert back.output.record == {"steps": 2}
    assert back.output.resolved == out.resolved
    assert back.output.results.frames[-1].results.energy == -5.7

    explicit = OptimizationData(provenance=Provenance(program="qcopt"))
    assert ProgramError(error, explicit, output=out).results is explicit
    assert ProgramError(error, frame=frames[-1], output=out).results is out.results
    assert ProgramError(error).output is None


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


def test_free_form_fields_carry_numpy_arrays_to_json():
    import json

    from qcdata import ProgramInput

    h2 = Structure(symbols=["H", "H"], geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    frame = Frame(
        structure=h2,
        results=SinglePointData(
            provenance=Provenance(program="linus"),
            energy=-1.1,
            extras={"gradient_components": {"xc": np.ones((2, 3))}},
        ),
        success=True,
        record={"scf": {"history": np.arange(3.0), "n": np.int64(7)}},
    )
    data = json.loads(frame.model_dump_json())
    assert data["results"]["extras"]["gradient_components"]["xc"] == [[1.0] * 3] * 2
    assert data["record"]["scf"] == {"history": [0.0, 1.0, 2.0], "n": 7}
    inp = ProgramInput(
        program="x", calctype="energy", structure=h2, keywords={"w": np.array([1.0])}
    )
    assert json.loads(inp.model_dump_json())["keywords"] == {"w": [1.0]}


def test_failed_outputs_take_the_shape_of_their_calctype():
    from qcdata import OptimizationData, ProgramInput, ProgramOutput

    h2 = Structure(symbols=["H", "H"], geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.4]])
    opt = ProgramInput(program="linus", calctype="optimization", structure=h2)
    error = ErrorInfo(kind=ErrorKind.unsupported, message="linus runs no optimizations")
    out = ProgramOutput.failure(opt, error)
    assert not out.success and isinstance(out.results, OptimizationData)
    assert out.results.provenance.program == "linus" and out.traceback == error.message
    sp = ProgramOutput.failure(
        opt.model_copy(update={"calctype": "energy"}), error, logs="x"
    )
    assert isinstance(sp.results, SinglePointData) and sp.logs == "x"


def test_program_errors_group_keeps_every_result_in_order():
    structure = Structure(symbols=["He"], geometry=[[0.0, 0.0, 0.0]])
    error = ErrorInfo(kind=ErrorKind.convergence, message="SCF", code="max_iter")

    def frame(energy: float | None) -> Frame:
        results = SinglePointData(provenance=Provenance(program="toy"), energy=energy)
        if energy is not None:
            return Frame(structure=structure, results=results, success=True)
        return Frame(
            structure=structure,
            results=results,
            success=False,
            error=error,
            traceback="t",
        )

    good = [frame(-2.8), frame(-2.9)]
    assert ProgramErrors.check(good) == good

    frames = [frame(-2.8), frame(None), frame(-2.9)]
    with pytest.raises(ProgramErrors) as info:
        ProgramErrors.check(frames)
    group = info.value
    assert group.results == frames and len(group.exceptions) == 1
    assert group.exceptions[0].frame is frames[1] and group.exceptions[0].error == error

    # except* and split keep the results; the group pickles with them.
    try:
        raise group
    except* ProgramError as caught:
        assert isinstance(caught, ProgramErrors) and caught.results == frames
    back = pickle.loads(pickle.dumps(group))
    assert back.results == frames and back.exceptions[0].frame == frames[1]


def test_program_errors_group_carries_failed_outputs():
    from qcdata import ProgramInput, ProgramOutput

    he = Structure(symbols=["He"], geometry=[[0.0, 0.0, 0.0]])
    inp = ProgramInput(
        program="toy", calctype="energy", structure=he, model={"method": "lj"}
    )
    error = ErrorInfo(kind=ErrorKind.input, message="bad", code="unknown_key")
    failed = ProgramOutput.failure(inp, error)
    with pytest.raises(ProgramErrors) as info:
        ProgramErrors.check([failed])
    assert info.value.exceptions[0].output is failed and info.value.results == [failed]

"""Frames, paths (IRC and path optimization), and chain inputs."""

import json

import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import (
    CalcType,
    ConformerSearchData,
    Frame,
    OptimizationData,
    PathData,
    PathOptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    SinglePointData,
    Structure,
    get_data_type,
)

PROVENANCE = Provenance(program="qcdata-test-suite")


def _moved(structure: Structure, dx: float) -> Structure:
    """The structure with every atom moved by dx bohr along x."""
    geometry = structure.geometry.copy()
    geometry[:, 0] += dx
    return structure.model_copy(update={"geometry": geometry})


def _frame(structure: Structure, energy: float | None) -> Frame:
    return Frame(
        success=True,
        structure=structure,
        results=SinglePointData(provenance=PROVENANCE, energy=energy),
    )


def test_frame_from_a_child_output_keeps_everything_but_the_input(prog_output):
    """A frame is the child's output without input_data, at its input's structure."""
    frame = Frame.from_output(prog_output)
    assert frame.structure == prog_output.input_data.structure
    assert frame.results == prog_output.results
    assert frame.logs == prog_output.logs
    assert frame.execution == prog_output.execution
    assert "input_data" not in frame.model_dump()


def test_a_frame_rebuilds_its_input_from_the_parent_spec(prog_output):
    """The parent's spec plus the frame's structure is the step's exact input."""
    child = prog_output.input_data
    spec = ProgramSpec.model_validate(
        child.model_dump(include=set(ProgramSpec.model_fields))
    )
    frame = Frame.from_output(prog_output)
    assert ProgramInput.from_spec(spec, frame.structure) == child


def test_failed_frames_need_a_traceback(water):
    """Frames share ProgramOutput's failure rule through OutputBase."""
    results = SinglePointData(provenance=PROVENANCE)
    with pytest.raises(ValidationError, match="traceback"):
        Frame(success=False, structure=water, results=results)
    Frame(success=False, structure=water, results=results, traceback="boom")


def test_outputs_serialize_success_first_and_logs_last(prog_output):
    """Reading order: success, the subject and results, then logs and execution."""
    for output in (prog_output, Frame.from_output(prog_output)):
        keys = list(output.model_dump())
        assert keys[0] == "success"
        assert keys.index("results") < keys.index("logs") < keys.index("execution")


def test_optimization_trajectories_hold_frames(prog_output, water):
    """ProgramOutputs passed as steps become frames; old saved files still load."""
    data = OptimizationData(provenance=PROVENANCE, trajectory=[prog_output] * 2)
    assert all(type(step) is Frame for step in data.trajectory)
    assert data.final_structure == water
    legacy = data.model_dump(mode="json")
    legacy["trajectory"] = [prog_output.model_dump(mode="json")] * 2
    loaded = OptimizationData.model_validate(legacy)
    assert loaded.structures == [water, water]
    assert np.array_equal(loaded.energies, [1.0, 1.0])


def test_path_data_reads_like_any_series(water, tmp_path):
    """Structures, energies, barriers, and the transition state of a path."""
    frames = [_frame(_moved(water, dx), e) for dx, e in [(0, 0.0), (1, 0.5), (2, 0.2)]]
    path = PathData(provenance=PROVENANCE, frames=frames, ts_index=1)
    assert path.transition_state is frames[1]
    assert path.forward_barrier == pytest.approx(0.5)
    assert path.reverse_barrier == pytest.approx(0.3)
    assert path.return_result(CalcType.irc) == path.structures
    assert path.to_xyz().count("\n") == 3 * (len(water.symbols) + 2)
    path.save(tmp_path / "path.xyz")
    assert (tmp_path / "path.xyz").read_text() == path.to_xyz()
    assert PathData.model_validate_json(path.model_dump_json()) == path


def test_path_data_checks_its_frames(water):
    """The transition state is one of the frames; frames share their atoms."""
    frames = [_frame(water, 0.0), _frame(water, None)]
    with pytest.raises(ValidationError, match="outside"):
        PathData(provenance=PROVENANCE, frames=frames, ts_index=2)
    other = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 1.4]])
    with pytest.raises(ValidationError, match="same atoms"):
        PathData(provenance=PROVENANCE, frames=[frames[0], _frame(other, 0.0)])
    no_energy = PathData(provenance=PROVENANCE, frames=frames, ts_index=1)
    assert no_energy.forward_barrier is None
    assert PathData(provenance=PROVENANCE, frames=frames).transition_state is None


def test_path_optimization_keeps_every_iteration(water):
    """Each step is a whole path; the final path is the result."""
    first = PathData(provenance=PROVENANCE, frames=[_frame(water, 1.0)] * 3)
    last = PathData(provenance=PROVENANCE, frames=[_frame(water, 0.5)] * 3, ts_index=1)
    data = PathOptimizationData(provenance=PROVENANCE, trajectory=[first, last])
    assert data.final_path is last
    assert data.structures == last.structures
    assert np.array_equal(data.energies, last.energies)
    assert data.return_result(CalcType.path_optimization) is last
    empty = PathOptimizationData(provenance=PROVENANCE)
    assert empty.final_path is None and empty.structures == []


def test_calctypes_map_to_path_models():
    """IRC returns one path; a path optimization returns its iterations."""
    assert get_data_type(CalcType.irc) is PathData
    assert get_data_type("path_optimization") is PathOptimizationData


def test_irc_output_requires_frames_on_success(water):
    """A successful IRC has a path."""
    inp = ProgramInput(program="sella", calctype="irc", structure=water)
    with pytest.raises(ValidationError, match="require frames"):
        ProgramOutput(input_data=inp, success=True, results={"provenance": PROVENANCE})
    output = ProgramOutput(
        input_data=inp,
        success=True,
        results={"provenance": PROVENANCE, "frames": [_frame(water, 0.0)]},
    )
    assert isinstance(output.results, PathData)


def test_chain_inputs(water):
    """A path optimization takes its chain from structures["initial_chain"]."""
    chain = [water, _moved(water, 1.0), _moved(water, 2.0)]
    inp = ProgramInput(
        program="neb",
        calctype="path_optimization",
        structures={"initial_chain": chain},
    )
    assert inp.structure is None and inp.initial_chain == chain
    with pytest.raises(ValueError, match="no single structure"):
        inp.require_structure()
    loaded = ProgramInput.model_validate(json.loads(inp.model_dump_json()))
    assert loaded.initial_chain == chain


@pytest.mark.parametrize(
    ("values", "match"),
    [
        ({"calctype": "path_optimization"}, "needs structures"),
        (
            {"calctype": "path_optimization", "structures": {"initial_chain": "W"}},
            "needs structures",
        ),
        (
            {"calctype": "path_optimization", "structures": {"initial_chain": ["W"]}},
            "needs structures",
        ),
        (
            {
                "calctype": "path_optimization",
                "structure": "W",
                "structures": {"initial_chain": ["W", "W"]},
            },
            "leave structure out",
        ),
        (
            {
                "calctype": "path_optimization",
                "structures": {"initial_chain": ["W", "H2"]},
            },
            "same atoms",
        ),
        ({"calctype": "energy"}, "requires a structure"),
        (
            {"calctype": "irc", "structures": {"environment": "W"}},
            "requires a structure",
        ),
    ],
)
def test_structures_are_required_by_calctype(water, values, match):
    """Chain calculations need a chain; every other one needs a structure."""
    h2 = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 1.4]])
    lookup = {"W": water, "H2": h2}

    def resolve(value):
        if isinstance(value, str):
            return lookup[value]
        if isinstance(value, list):
            return [resolve(v) for v in value]
        if isinstance(value, dict):
            return {k: resolve(v) for k, v in value.items()}
        return value

    with pytest.raises(ValidationError, match=match):
        ProgramInput(
            program="p",
            **{
                k: resolve(v) if k in ("structure", "structures") else v
                for k, v in values.items()
            },
        )


def test_require_structure_and_from_spec(water):
    """Single-structure inputs return their structure; from_spec binds chains too."""
    inp = ProgramInput(program="p", calctype="energy", structure=water)
    assert inp.require_structure() is inp.structure
    spec = ProgramSpec(program="neb", calctype="path_optimization")
    bound = ProgramInput.from_spec(
        spec, structures={"initial_chain": [water, _moved(water, 1.0)]}
    )
    assert len(bound.initial_chain) == 2
    with pytest.raises(ValueError, match="no structures"):
        _ = inp.initial_chain


def test_conformer_search_reads_like_any_series(water):
    """Conformers share the reading interface: structures, energies, xyz."""
    data = ConformerSearchData(
        provenance=PROVENANCE,
        conformers=[water, _moved(water, 1.0)],
        conformer_energies=[0.2, 0.1],
    )
    assert data.structures == data.conformers
    assert np.array_equal(data.energies, [0.1, 0.2])
    assert data.to_xyz().count("\n") == 2 * (len(water.symbols) + 2)

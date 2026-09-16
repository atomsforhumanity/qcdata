import json
import pickle
from pathlib import Path
from typing import get_args

import pytest
from pydantic import TypeAdapter, ValidationError

from qcdata import (
    CalcType,
    ConformerSearchData,
    FileData,
    FileInput,
    Inputs,
    InputType,
    Model,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
)
from qcdata.models.data import StructuredData


@pytest.fixture
def recursive_input(water):
    return ProgramInput(
        program="workflow",
        calctype="conformer_search",
        structure=water,
        structures={"reference": water},
        keywords={"count": 10},
        files={"workflow.in": "workflow settings"},
        cmdline_args=["--parallel"],
        extras={"level": 0},
        subprograms=[
            ProgramSpec(
                program="geometric",
                calctype="optimization",
                keywords={"maxiter": 250},
                files={"optimizer.in": "optimizer settings"},
                cmdline_args=["--verbose"],
                extras={"level": 1},
                subprograms=[
                    ProgramSpec(
                        program="terachem",
                        calctype="gradient",
                        model={"method": "wb97x-d3", "basis": "def2-svp"},
                        keywords={"convthre": 1e-6},
                        files={"guess.bin": b"\x00\xff", "tc.in": "native input"},
                        cmdline_args=["tc.in"],
                        extras={"level": 2},
                    ),
                ],
            ),
        ],
    )


def test_normal_program_input(prog_input_factory):
    inp = prog_input_factory("energy")
    assert isinstance(inp, ProgramSpec)
    assert isinstance(inp, FileInput)
    assert inp.program == "terachem"
    assert inp.calctype is CalcType.energy
    assert inp.model == Model(method="hf", basis="sto-3g")
    assert inp.structures == {}
    assert inp.subprograms == []


@pytest.mark.parametrize("model_kwargs", [{}, {"model": None}])
def test_optional_model(water, model_kwargs):
    inp = ProgramInput(
        program="geometric", calctype="optimization", structure=water, **model_kwargs
    )
    assert inp.model is None
    assert ProgramInput.model_validate_json(inp.model_dump_json()) == inp


@pytest.mark.parametrize("calctype", [CalcType.gradient, "gradient"])
def test_one_nested_spec(nested_input_factory, calctype):
    inp = nested_input_factory("optimization")
    child = inp.get_subprogram(calctype)
    assert child is inp.subprograms[0]
    assert type(child) is ProgramSpec
    assert child.program == "terachem"
    assert child.calctype is CalcType.gradient
    assert child.model == Model(method="hf", basis="sto-3g")
    assert child.subprograms == []
    assert inp.model is None


def test_three_levels_keep_independent_fields(recursive_input):
    parent = recursive_input
    optimizer = parent.get_subprogram(CalcType.optimization)
    gradient = optimizer.get_subprogram(CalcType.gradient)
    assert [s.program for s in [parent, optimizer, gradient]] == [
        "workflow",
        "geometric",
        "terachem",
    ]
    assert [s.calctype for s in [parent, optimizer, gradient]] == [
        CalcType.conformer_search,
        CalcType.optimization,
        CalcType.gradient,
    ]
    assert parent.model is optimizer.model is None
    assert gradient.model == Model(method="wb97x-d3", basis="def2-svp")
    assert parent.keywords == {"count": 10}
    assert optimizer.keywords == {"maxiter": 250}
    assert gradient.keywords == {"convthre": 1e-6}
    assert parent.files == {"workflow.in": "workflow settings"}
    assert optimizer.files == {"optimizer.in": "optimizer settings"}
    assert gradient.files == {"guess.bin": b"\x00\xff", "tc.in": "native input"}
    assert [s.cmdline_args for s in [parent, optimizer, gradient]] == [
        ["--parallel"],
        ["--verbose"],
        ["tc.in"],
    ]
    assert [s.extras for s in [parent, optimizer, gradient]] == [
        {"level": 0},
        {"level": 1},
        {"level": 2},
    ]


@pytest.mark.parametrize("extension", ["json", "yaml", "toml"])
def test_recursive_roundtrip(recursive_input, tmp_path, extension):
    for spec in [recursive_input, recursive_input.subprograms[0]]:
        cls = type(spec)
        assert cls.model_validate(spec.model_dump()) == spec
        assert cls.model_validate_json(spec.model_dump_json()) == spec
        path = tmp_path / f"{cls.__name__}.{extension}"
        spec.save(path)
        assert cls.open(path) == spec


@pytest.mark.parametrize("cls", [FileInput, ProgramSpec, ProgramInput, ProgramOutput])
def test_json_schema(cls):
    schema = cls.model_json_schema()
    assert json.loads(json.dumps(schema)) == schema
    if cls is not FileInput:
        spec = schema["$defs"]["ProgramSpec"]
        assert spec["properties"]["subprograms"]["items"]["$ref"].endswith(
            "/ProgramSpec"
        )
        assert set(spec["required"]) == {"program", "calctype"}
    if cls is ProgramInput:
        assert set(schema["required"]) == {"program", "calctype", "structure"}


@pytest.mark.parametrize(
    "gradient, energy",
    [(CalcType.gradient, CalcType.energy), ("gradient", "energy")],
)
def test_get_subprogram_only_immediate_children(recursive_input, gradient, energy):
    with pytest.raises(ValueError, match="No immediate subprogram.*gradient"):
        recursive_input.get_subprogram(gradient)
    with pytest.raises(ValueError, match="No immediate subprogram.*energy"):
        recursive_input.subprograms[0].subprograms[0].get_subprogram(energy)


@pytest.mark.parametrize("nested", [False, True])
def test_duplicate_sibling_calctypes(nested):
    payload = {
        "program": "geometric",
        "calctype": "optimization",
        "subprograms": [
            {"program": "xtb", "calctype": "gradient"},
            {"program": "orca", "calctype": CalcType.gradient},
        ],
    }
    if nested:
        payload = {"program": "workflow", "calctype": "scan", "subprograms": [payload]}
    with pytest.raises(ValidationError, match="Duplicate subprogram.*gradient"):
        ProgramSpec.model_validate(payload)


def test_unique_siblings_and_repeated_types_at_different_levels():
    spec = ProgramSpec(
        program="wrapper",
        calctype="gradient",
        subprograms=[
            ProgramSpec(program="xtb", calctype="gradient"),
            ProgramSpec(program="orca", calctype="hessian"),
        ],
    )
    assert spec.get_subprogram(CalcType.gradient).program == "xtb"
    assert spec.get_subprogram(CalcType.hessian).program == "orca"


def test_child_requires_calctype():
    with pytest.raises(ValidationError, match="subprograms.0.calctype"):
        ProgramSpec.model_validate(
            {
                "program": "geometric",
                "calctype": "optimization",
                "subprograms": [{"program": "xtb"}],
            }
        )


def test_structure_roles(water):
    product = water.model_copy(update={"charge": 1})
    inp = ProgramInput(
        program="geometric",
        calctype="optimization",
        structure=water,
        structures={"product": product},
    )
    assert inp.structure == water
    assert inp.structures == {"product": product}
    assert ProgramInput.model_validate_json(inp.model_dump_json()) == inp
    with pytest.raises(ValidationError, match="structure"):
        ProgramInput.model_validate(
            {
                "program": "geometric",
                "calctype": "optimization",
                "structures": {"product": product},
            }
        )


def test_mutable_defaults_are_independent():
    first = ProgramSpec(program="xtb", calctype="energy")
    second = ProgramSpec(program="xtb", calctype="energy")
    first.keywords["x"] = 1
    first.files["input"] = "text"
    first.cmdline_args.append("--flag")
    first.extras["note"] = 1
    first.subprograms.append(ProgramSpec(program="orca", calctype="gradient"))
    assert second.keywords == second.files == second.extras == {}
    assert second.cmdline_args == second.subprograms == []


def test_input_aliases_and_unbound_spec_rejection():
    assert get_args(Inputs) == (FileInput, ProgramInput)
    assert InputType.__bound__ == Inputs
    spec = ProgramSpec(program="xtb", calctype="energy")
    with pytest.raises(ValidationError):
        TypeAdapter(Inputs).validate_python(spec.model_dump())
    with pytest.raises(ValidationError, match="bound to structures"):
        ProgramOutput(
            input_data=spec,
            results=FileData(provenance={"program": "qcdata-test-suite"}),
            success=True,
            execution={},
        )


def test_recursive_output_roundtrip_and_identity(recursive_input):
    output = ProgramOutput(
        input_data=recursive_input,
        results=ConformerSearchData(provenance={"program": "actual-executor"}),
        success=False,
        traceback="test failure",
        execution={},
    )
    reopened = ProgramOutput.model_validate_json(output.model_dump_json())
    assert reopened == output
    assert type(reopened) is ProgramOutput[ProgramInput, ConformerSearchData]
    assert reopened.input_data.program == "workflow"
    assert reopened.results.provenance.program == "actual-executor"
    assert pickle.loads(pickle.dumps(output)) == output


def test_output_generic_registration():
    from qcdata.models import outputs

    pairs = [(FileInput, FileData)] + [
        (ProgramInput, data_type) for data_type in get_args(StructuredData)
    ]
    for input_type, data_type in pairs:
        cls = ProgramOutput[input_type, data_type]
        assert getattr(outputs, cls.__name__) is cls


@pytest.mark.parametrize(
    "path",
    sorted((Path(__file__).parents[1] / "docs/visualizations").glob("*.json")),
    ids=lambda path: path.name,
)
def test_documentation_output_fixtures(path):
    output = ProgramOutput.open(path)
    assert isinstance(output.input_data, ProgramInput)
    assert output.input_data.program


def test_from_spec_binds_structures_and_preserves_recursive_fields(
    recursive_input, water
):
    spec = recursive_input.get_subprogram("optimization")
    original = spec.model_dump()
    bound = ProgramInput.from_spec(spec, water, structures={"reference": water})
    assert bound.structure == water
    assert bound.structures == {"reference": water}
    assert bound.model_dump(include=set(ProgramSpec.model_fields)) == spec.model_dump(
        include=set(ProgramSpec.model_fields)
    )
    # Editing a dumped input must not modify the original specification.
    values = bound.model_dump()
    values["keywords"]["maxiter"] = 99
    rebuilt = ProgramInput.model_validate(values)
    assert rebuilt.keywords["maxiter"] == 99
    assert spec.model_dump() == original
    assert ProgramInput.model_validate_json(bound.model_dump_json()) == bound


def test_data_type_lookup():
    from qcdata import OptimizationData, ScanData, SinglePointData, get_data_type

    expected = {
        "energy": SinglePointData,
        "gradient": SinglePointData,
        "hessian": SinglePointData,
        "optimization": OptimizationData,
        "transition_state": OptimizationData,
        "conformer_search": ConformerSearchData,
        "scan": ScanData,
    }
    for calctype, data_type in expected.items():
        assert get_data_type(calctype) is data_type
        assert get_data_type(CalcType(calctype)) is data_type
    with pytest.raises(ValueError):
        get_data_type("not-a-calculation")

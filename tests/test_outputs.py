import pickle

import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import (
    ConformerSearchData,
    FileData,
    FileInput,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    Provenance,
    ScanData,
    SinglePointData,
    Structure,
)


def test_scientific_results(prog_input_factory):
    """Access scientific results through the canonical payload attribute."""
    calc_input_energy = prog_input_factory("energy")
    energy = 1.0
    n_atoms = len(calc_input_energy.structure.symbols)
    gradient = np.arange(n_atoms * 3).reshape(n_atoms, 3)
    hessian = np.arange(n_atoms**2 * 3**2).reshape(n_atoms * 3, n_atoms * 3)

    results = ProgramOutput(
        input_data=calc_input_energy,
        success=True,
        results={
            "provenance": {"program": "qcdata-test-suite"},
            "energy": energy,
            "gradient": gradient,
            "hessian": hessian,
        },
        execution={},
    )
    assert results.results.energy == energy

    pi_gradient = prog_input_factory("gradient")
    results = ProgramOutput(**{**results.model_dump(), **{"input_data": pi_gradient}})
    assert np.array_equal(results.results.gradient, gradient)

    pi_hessian = prog_input_factory("hessian")
    results = ProgramOutput(**{**results.model_dump(), **{"input_data": pi_hessian}})

    assert np.array_equal(results.results.hessian, hessian)


def test_successful_prog_output_serialization(prog_output):
    """Test that successful program output serializes and deserializes"""
    serialized = prog_output.model_dump_json()
    deserialized = ProgramOutput.model_validate_json(serialized)
    assert deserialized == prog_output
    assert deserialized.results == prog_output.results
    assert deserialized.input_data == prog_output.input_data
    assert deserialized.results.provenance.program == "qcdata-test-suite"
    assert deserialized.logs == prog_output.logs
    assert deserialized.extras == prog_output.extras
    assert deserialized.results.energy == prog_output.results.energy
    assert np.array_equal(deserialized.results.gradient, prog_output.results.gradient)
    assert np.array_equal(deserialized.results.hessian, prog_output.results.hessian)


def test_correct_generic_instantiates_and_equality_checks_pass(prog_output, tmp_path):
    """
    This test checks the ProgramOutput.model_post_init method to ensure that the
    correct generic types are instantiated and that equality checks pass.
    """
    results_dict = prog_output.model_dump()
    wo_types = ProgramOutput(**results_dict)
    w_types = ProgramOutput[ProgramInput, SinglePointData](**results_dict)

    results_dict["input_data"]["calctype"] = "optimization"
    results_dict["results"] = OptimizationData(
        provenance={"program": "qcdata-test-suite"}, trajectory=[wo_types]
    )
    wo_types_opt = ProgramOutput(**results_dict)
    w_types_opt = ProgramOutput[ProgramInput, OptimizationData](**results_dict)

    wo_types.save(tmp_path / "out.json")
    w_types_opt.save(tmp_path / "opt.json")

    wo_types_opened = ProgramOutput.open(tmp_path / "out.json")
    w_types_opened = ProgramOutput[ProgramInput, SinglePointData].open(
        tmp_path / "out.json"
    )

    wo_types_opened_opt = ProgramOutput.open(tmp_path / "opt.json")
    w_types_opened_opt = ProgramOutput[ProgramInput, OptimizationData].open(
        tmp_path / "opt.json"
    )

    assert wo_types == w_types == wo_types_opened == w_types_opened
    assert wo_types_opt == w_types_opt == wo_types_opened_opt == w_types_opened_opt


def test_non_file_success_always_has_result(prog_input_factory):
    pi_energy = prog_input_factory("energy")
    with pytest.raises(ValidationError):
        ProgramOutput[ProgramInput, SinglePointData](
            success=True,
            input_data=pi_energy,
            logs="program standard out...",
            results=None,
            execution={},
        )


def test_primary_result_must_be_present_on_success(prog_output):
    for calctype in ["energy", "gradient", "hessian"]:
        po_dict = prog_output.model_dump()
        po_dict["input_data"]["calctype"] = calctype
        po_dict["results"][calctype] = None
        with pytest.raises(ValidationError):
            ProgramOutput[ProgramInput, SinglePointData](**po_dict)


def test_primary_result_can_be_missing_on_failure(prog_input_factory):
    pi_gradient = prog_input_factory("gradient")
    output = ProgramOutput[ProgramInput, SinglePointData](
        success=False,
        input_data=pi_gradient,
        results=SinglePointData(
            provenance={"program": "qcdata-test-suite"},
            extras={"program_version": "3.0.2"},
        ),
        traceback="Fake traceback",
        execution={},
    )
    assert output.results.energy is None
    assert output.results.gradient is None
    assert output.results.hessian is None
    assert output.results.extras == {"program_version": "3.0.2"}


@pytest.mark.parametrize(
    "input_data, data, success, expected_input_type, expected_result_type",
    [
        pytest.param(
            "file_input",
            FileData(provenance=Provenance(program="qcdata-test-suite")),
            True,
            FileInput,
            FileData,
            id="success",
        ),
        pytest.param(
            "file_input",
            FileData(provenance=Provenance(program="qcdata-test-suite")),
            False,
            FileInput,
            FileData,
            id="failure",
        ),
    ],
    indirect=["input_data"],
)
def test_pickle_serialization_of_program_output_parametrized(
    input_data,
    data,
    success,
    expected_input_type,
    expected_result_type,
    request,
):
    """This test checks that all the dynamic types are correctly set when pickled."""

    traceback = None
    if success is False:
        traceback = "Fake traceback"

    prog_output = ProgramOutput[type(input_data), type(data)](
        input_data=input_data,
        results=data,
        success=success,
        execution={},
        traceback=traceback,
    )
    serialized = pickle.dumps(prog_output)
    deserialized = pickle.loads(serialized)
    assert deserialized == prog_output

    prog_output = ProgramOutput(
        input_data=input_data,
        results=data,
        success=success,
        execution={},
        traceback=traceback,
    )
    serialized = pickle.dumps(prog_output)
    deserialized = pickle.loads(serialized)
    assert deserialized == prog_output

    unspecified_po = ProgramOutput(**prog_output.model_dump())
    serialized = pickle.dumps(prog_output)
    deserialized = pickle.loads(serialized)
    assert deserialized == unspecified_po

    prog_output_dict = prog_output.model_dump()
    prog_output_dict.update({"success": False, "traceback": "Traceback: ..."})
    no_prog_out = ProgramOutput(**prog_output_dict)
    serialized = pickle.dumps(no_prog_out)
    deserialized = pickle.loads(serialized)
    assert deserialized == no_prog_out

    dynamic_generics = ProgramOutput[
        type(prog_output.input_data), type(prog_output.results)
    ](**prog_output.model_dump())
    serialized = pickle.dumps(dynamic_generics)
    deserialized = pickle.loads(serialized)
    assert deserialized == dynamic_generics


def test_pickle_serialization_of_program_output():
    prog_output = ProgramOutput[ProgramInput, SinglePointData](
        input_data=ProgramInput(
            program="terachem",
            structure=Structure(
                symbols=["O", "H", "H"],
                geometry=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0, 1.0, 0.0]),
                charge=0,
                multiplicity=1,
                connectivity=[(0, 1, 1.0), (0, 2, 1.0)],
            ),
            calctype="energy",
            model={"method": "hf", "basis": "sto-3g"},
            keywords={
                "maxiter": 100,
                "purify": "no",
                "some-bool": False,
                "displacement": 1e-3,
                "thermo_temp": 298.15,
            },
        ),
        success=True,
        logs="program standard out...",
        results=SinglePointData(
            provenance={"program": "qcdata-test-suite"},
            energy=1.0,
            extras={"some_extra_result": 1},
        ),
        execution={"scratch_dir": "/tmp/qcdata"},
        extras={"some_extra": 1},
    )
    serialized = pickle.dumps(prog_output)
    deserialized = pickle.loads(serialized)
    assert deserialized == prog_output

    unspecified_po = ProgramOutput(**prog_output.model_dump())
    serialized = pickle.dumps(prog_output)
    deserialized = pickle.loads(serialized)
    assert deserialized == unspecified_po

    prog_output_dict = prog_output.model_dump()
    prog_output_dict.update(
        {
            "results": SinglePointData(provenance={"program": "qcdata-test-suite"}),
            "success": False,
            "traceback": "Traceback: ...",
        }
    )
    no_data = ProgramOutput(**prog_output_dict)
    serialized = pickle.dumps(no_data)
    deserialized = pickle.loads(serialized)
    assert deserialized == no_data

    dynamic_generics = ProgramOutput[
        type(prog_output.input_data), type(prog_output.results)
    ](**prog_output.model_dump())
    serialized = pickle.dumps(dynamic_generics)
    deserialized = pickle.loads(serialized)
    assert deserialized == dynamic_generics


def test_empty_failed_output_fixture(test_data_dir):
    output = ProgramOutput.open(test_data_dir / "po_noresults.json")
    assert type(output.results) is SinglePointData
    assert output.results.provenance.program == "terachem"


@pytest.mark.parametrize(
    "calctype, expected",
    [
        (None, FileData),
        ("energy", SinglePointData),
        ("gradient", SinglePointData),
        ("hessian", SinglePointData),
        ("optimization", OptimizationData),
        ("transition_state", OptimizationData),
        ("conformer_search", ConformerSearchData),
        ("scan", ScanData),
    ],
)
def test_result_type_contract(calctype, expected, prog_input_factory, file_input):
    input_data = prog_input_factory(calctype) if calctype else file_input
    for cls in [
        FileData,
        SinglePointData,
        OptimizationData,
        ConformerSearchData,
        ScanData,
    ]:
        results = cls(provenance={"program": "producer"})
        for success in [False, True]:
            payload = dict(
                input_data=input_data,
                results=results,
                success=success,
                traceback=None if success else "Failed before producing values",
            )
            if cls is not expected:
                with pytest.raises(
                    ValidationError, match=f"requires {expected.__name__}"
                ):
                    ProgramOutput(**payload)
            elif success and calctype:
                with pytest.raises(ValidationError, match="Missing|require"):
                    ProgramOutput(**payload)
            else:
                output = ProgramOutput(**payload)
                assert type(output.results) is expected


def test_conflicting_output_generics(prog_input_factory):
    input_data = prog_input_factory("energy")
    results = SinglePointData(provenance={"program": "producer"})
    for cls in [
        ProgramOutput[ProgramInput, FileData],
        ProgramOutput[FileInput, SinglePointData],
    ]:
        with pytest.raises(ValidationError, match="generic"):
            cls(
                input_data=input_data,
                results=results,
                success=False,
                traceback="Failed",
            )


@pytest.mark.parametrize("extension", ["json", "yaml", "toml"])
def test_empty_failed_results_roundtrip(
    extension, tmp_path, prog_input_factory, file_input
):
    cases = [
        (file_input, FileData),
        *[
            (prog_input_factory(calc), cls)
            for calc, cls in [
                ("energy", SinglePointData),
                ("gradient", SinglePointData),
                ("hessian", SinglePointData),
                ("optimization", OptimizationData),
                ("transition_state", OptimizationData),
                ("conformer_search", ConformerSearchData),
                ("scan", ScanData),
            ]
        ],
    ]
    for input_data, cls in cases:
        output = ProgramOutput(
            input_data=input_data,
            results=cls(provenance={"program": "producer"}),
            success=False,
            traceback="Failed before producing values",
        )
        path = tmp_path / f"output.{extension}"
        output.save(path)
        reopened = ProgramOutput.open(path)
        assert type(reopened.results) is cls
        assert reopened == output
        assert repr(reopened)

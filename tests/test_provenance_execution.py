import json
import pickle
from pathlib import Path

import pytest
from pydantic import ValidationError

from qcdata import (
    ConformerSearchData,
    ExecutionInfo,
    FileData,
    FileInput,
    Files,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    Provenance,
    ScanData,
    SinglePointData,
)


@pytest.mark.parametrize(
    "cls, fields",
    [
        (FileData, {}),
        (SinglePointData, {"energy": -1.0}),
        (OptimizationData, {"trajectory": []}),
        (ConformerSearchData, {}),
        (ScanData, {"trajectory": []}),
    ],
)
def test_required_result_provenance(cls, fields):
    for provenance_fields in [{}, {"provenance": None}, {"provenance": {}}]:
        with pytest.raises(ValidationError, match="provenance"):
            cls.model_validate({**fields, **provenance_fields})

    data = cls(provenance=Provenance(program="producer"), **fields)
    assert isinstance(data, FileData)
    assert data.provenance.program_version is None
    assert "provenance" in cls.model_json_schema()["required"]


@pytest.mark.parametrize("cls", [Files, FileInput, ProgramSpec, ProgramInput])
def test_input_and_file_models_have_no_provenance(cls, water):
    fields = {}
    if issubclass(cls, FileInput):
        fields["program"] = "requested"
    if issubclass(cls, ProgramSpec):
        fields["calctype"] = "energy"
    if issubclass(cls, ProgramInput):
        fields["structure"] = water
    obj = cls(**fields)
    assert not hasattr(obj, "provenance")
    with pytest.raises(ValidationError, match="provenance"):
        cls.model_validate({**fields, "provenance": {"program": "producer"}})


def test_execution_cannot_be_none(file_input):
    with pytest.raises(ValidationError, match="execution"):
        ProgramOutput.model_validate(
            {
                "input_data": file_input,
                "results": FileData(provenance=Provenance(program="producer")),
                "success": True,
                "execution": None,
            }
        )


@pytest.mark.parametrize("execution", [{}, ExecutionInfo()])
def test_unknown_execution_information(file_input, execution):
    output = ProgramOutput(
        input_data=file_input,
        results=FileData(provenance=Provenance(program="producer")),
        success=False,
        traceback="Failed before producing results",
        execution=execution,
    )
    assert output.execution.wall_time is None
    assert output.execution.scratch_dir is None
    assert output.results.provenance.program == "producer"
    assert "execution" not in ProgramOutput.model_json_schema()["required"]


def test_output_requires_result_data(file_input):
    with pytest.raises(ValidationError):
        ProgramOutput(
            input_data=file_input,
            results=Files(),
            success=True,
            execution=ExecutionInfo(),
        )


@pytest.mark.parametrize("extension", ["json", "yaml", "toml"])
@pytest.mark.parametrize("cls", [FileData, SinglePointData])
def test_standalone_data_roundtrip(cls, extension, tmp_path):
    data = cls(
        provenance=Provenance(
            program="terachem", program_version="1.9", extras={"build": "test"}
        ),
        files={"input.out": "output text", "wavefunction.bin": b"\x00\xff"},
        extras={"note": "parsed independently"},
    )
    path = tmp_path / f"data.{extension}"
    data.save(path)
    reopened = cls.open(path)
    assert reopened == data
    assert reopened.provenance.program_version == "1.9"
    assert pickle.loads(pickle.dumps(data)) == data


@pytest.mark.parametrize("extension", ["json", "yaml", "toml"])
def test_nested_execution_and_provenance_roundtrip(
    prog_input_factory, tmp_path, extension
):
    step = ProgramOutput(
        input_data=prog_input_factory("energy"),
        success=True,
        results=SinglePointData(
            energy=-1.0,
            provenance=Provenance(program="terachem", program_version="1.9"),
        ),
        execution=ExecutionInfo(
            scratch_dir="/tmp/step", wall_time=0, hostname="worker"
        ),
    )
    opt = ProgramOutput(
        input_data=prog_input_factory("optimization"),
        success=True,
        results=OptimizationData(
            provenance=Provenance(program="geometric", program_version="1.0"),
            trajectory=[step],
        ),
        execution=ExecutionInfo(scratch_dir="/tmp/opt", wall_time=3),
    )
    output = ProgramOutput(
        input_data=prog_input_factory("scan"),
        success=True,
        results=ScanData(
            provenance=Provenance(program="scan-driver"), trajectory=[opt]
        ),
        execution=ExecutionInfo(
            scratch_dir="/tmp/scan",
            wall_time=5,
            hostname="head",
            host_cpu=16,
            host_mem_gib=63.5,
            extras={"job_id": "test-job"},
        ),
    )
    path = tmp_path / f"output.{extension}"
    output.save(path)
    reopened = ProgramOutput.open(path)
    assert reopened == output
    assert pickle.loads(pickle.dumps(output)) == output
    assert ProgramOutput.model_validate_json(output.model_dump_json()) == output
    assert reopened.results.provenance.program == "scan-driver"
    opt = reopened.results.trajectory[0]
    step = opt.results.trajectory[0]
    assert opt.results.provenance.program == "geometric"
    assert step.results.provenance.program == "terachem"
    assert step.results.provenance.program_version == "1.9"
    assert [obj.execution.wall_time for obj in [reopened, opt, step]] == [5, 3, 0]
    assert [obj.execution.scratch_dir for obj in [reopened, opt, step]] == [
        Path("/tmp/scan"),
        Path("/tmp/opt"),
        Path("/tmp/step"),
    ]
    assert reopened.execution.host_cpu == 16
    assert reopened.execution.host_mem_gib == 63.5
    assert reopened.execution.extras == {"job_id": "test-job"}
    assert step.execution.hostname == "worker"
    schema = type(output).model_json_schema()
    assert json.loads(json.dumps(schema)) == schema


def test_execution_defaults_to_empty(file_input):
    output = ProgramOutput(
        input_data=file_input,
        results=FileData(provenance=Provenance(program="producer")),
        success=True,
    )
    assert isinstance(output.execution, ExecutionInfo)
    assert output.execution.wall_time is None

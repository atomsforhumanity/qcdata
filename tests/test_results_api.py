import copy
import json
import warnings

import pytest
import toml
import yaml
from pydantic import ValidationError

import qcdata
from qcdata import (
    ConformerSearchData,
    Data,
    DataType,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    ScanData,
    SinglePointData,
    StructuredData,
    json_dumps,
)
from qcdata.view import generate_data_table, generate_output_table


def test_results_is_canonical(prog_output):
    assert "results" in ProgramOutput.model_fields
    assert "data" not in ProgramOutput.model_fields
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        output = ProgramOutput.model_validate(prog_output.model_dump())
        assert output.results is not None
        assert output.results.energy == prog_output.results.energy
        assert isinstance(output.results, SinglePointData)
        generate_output_table(output)
        generate_data_table(output.results)
    schema = ProgramOutput.model_json_schema()
    assert "results" in schema["required"]
    assert "data" not in schema["properties"]


def test_data_property_is_deprecated(prog_output):
    with pytest.warns(
        FutureWarning, match=r"\.data has been renamed to \.results"
    ) as caught:
        assert prog_output.data is prog_output.results
    assert caught[0].filename == __file__


@pytest.mark.parametrize(
    "cls", [ProgramOutput, ProgramOutput[ProgramInput, SinglePointData]]
)
@pytest.mark.parametrize("payload_kind", ["model", "dict", "json"])
def test_legacy_data_input(cls, payload_kind, prog_output):
    payload = prog_output.model_dump(mode="json")
    payload["data"] = payload.pop("results")
    if payload_kind == "model":
        payload["data"] = prog_output.results
    original_results = payload["data"]
    with pytest.warns(FutureWarning, match="'data' has been renamed to 'results'"):
        output = (
            cls.model_validate_json(json.dumps(payload))
            if payload_kind == "json"
            else cls(**payload)
        )
    assert output == prog_output
    assert "data" in payload and "results" not in payload
    assert payload["data"] is original_results
    if payload_kind == "model":
        assert output.results is prog_output.results
    assert "data" not in output.model_dump()
    assert "results" in output.model_dump()


def test_conflicting_payload_names_rejected(prog_output):
    payload = prog_output.model_dump()
    payload["data"] = {**payload["results"], "energy": 123.0}
    original = copy.deepcopy(payload)
    with pytest.raises(ValidationError, match="Provide only 'results'"):
        ProgramOutput.model_validate(payload)
    assert payload == original


@pytest.mark.parametrize("extension", ["json", "yaml", "toml"])
def test_legacy_files_write_canonical_results(prog_output, tmp_path, extension):
    payload = prog_output.model_dump(mode="json", exclude_none=True)
    payload["data"] = payload.pop("results")
    dumps, loads = {
        "json": (json.dumps, json.loads),
        "yaml": (yaml.safe_dump, yaml.safe_load),
        "toml": (toml.dumps, toml.loads),
    }[extension]
    old_path = tmp_path / f"legacy.{extension}"
    old_path.write_text(dumps(payload))
    with pytest.warns(FutureWarning, match="'data' has been renamed"):
        output = ProgramOutput.open(old_path)
    new_path = tmp_path / f"canonical.{extension}"
    output.save(new_path)
    saved = loads(new_path.read_text())
    assert "results" in saved
    assert "data" not in saved
    assert ProgramOutput.open(new_path) == prog_output


def test_nested_legacy_results_and_unrelated_data_keys(opt_output):
    payload = opt_output.model_dump(mode="json")
    payload["extras"] = {"data": "unrelated metadata"}
    payload["data"] = payload.pop("results")
    step = payload["data"]["trajectory"][0]
    step["data"] = step.pop("results")
    with pytest.warns(FutureWarning, match="'data' has been renamed"):
        output = ProgramOutput.model_validate(payload)
    assert "data" in payload and "results" not in payload
    assert "data" in step and "results" not in step
    assert output.results.final_energy == opt_output.results.final_energy
    assert (
        output.results.trajectory[0].results.energy == opt_output.results.final_energy
    )
    canonical = json.loads(output.model_dump_json())
    assert "data" not in canonical
    assert "data" not in canonical["results"]["trajectory"][0]
    assert canonical["extras"] == {"data": "unrelated metadata"}


def test_json_helper_uses_results(prog_output):
    single = json.loads(json_dumps(prog_output))
    multiple = json.loads(json_dumps([prog_output], mode="json"))
    for payload in [single, multiple[0]]:
        assert "results" in payload
        assert "data" not in payload


def test_program_output_view_uses_results(prog_output, opt_output, monkeypatch):
    from qcdata import view

    rendered = []
    monkeypatch.setattr(view, "display", rendered.append)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        view.program_outputs(prog_output, opt_output, struct_viewer=False)
    assert len(rendered) == 2


def test_conformer_view_filters_duplicates(
    water, prog_input_factory, monkeypatch
):
    from qcdata import view

    output = ProgramOutput(
        input_data=prog_input_factory("conformer_search"),
        results=ConformerSearchData(
            provenance={"program": "crest"},
            conformers=[water, water],
            conformer_energies=[-1.0, -1.0],
        ),
        success=True,
    )
    displayed_structures = []

    def render_structures(*structures, **kwargs):
        displayed_structures.extend(structures)
        return "<div>Structures</div>"

    monkeypatch.setattr(view, "generate_structure_viewer_html", render_structures)
    monkeypatch.setattr(view, "display", lambda obj: None)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        view.program_outputs(
            output, conformer_rmsd_threshold=1.0
        )
    assert displayed_structures == [
        water,
        water,
    ]  # Initial structure and one conformer.


def test_data_names_and_aliases_are_retained():
    for cls in [SinglePointData, OptimizationData, ConformerSearchData, ScanData]:
        assert getattr(qcdata, cls.__name__) is cls
        assert cls.__name__.endswith("Data")
    assert qcdata.Data is Data
    assert qcdata.DataType is DataType
    assert qcdata.StructuredData is StructuredData

import json
import warnings

import pytest
from pydantic import ValidationError

import qcdata
from qcdata import (
    ConformerSearchData,
    Data,
    DataType,
    Frame,
    OptimizationData,
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


def test_data_property_is_removed(prog_output):
    """Read results as .results; the deprecated .data attribute is gone."""
    assert not hasattr(prog_output, "data")


def test_old_field_names_are_refused(prog_output, opt_output):
    """No compatibility loaders: data and trajectory fail like any unknown field."""
    output = prog_output.model_dump(mode="json")
    output["data"] = output.pop("results")
    frame = opt_output.results.frames[0].model_dump(mode="json")
    frame["data"] = frame.pop("results")
    optimization = opt_output.results.model_dump(mode="json")
    optimization["trajectory"] = optimization.pop("frames")
    scan = {"provenance": {"program": "qcopt"}, "trajectory": [opt_output]}
    for cls, payload in [
        (ProgramOutput, output),
        (Frame, frame),
        (OptimizationData, optimization),
        (ScanData, scan),
    ]:
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            cls.model_validate(payload)


def test_a_data_key_in_extras_is_just_metadata(opt_output):
    payload = opt_output.model_dump(mode="json")
    payload["extras"] = {"data": "unrelated metadata"}
    output = ProgramOutput.model_validate(payload)
    assert output.extras == {"data": "unrelated metadata"}
    assert output.results == opt_output.results


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


def test_conformer_view_filters_duplicates(water, prog_input_factory, monkeypatch):
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
        view.program_outputs(output, conformer_rmsd_threshold=1.0)
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

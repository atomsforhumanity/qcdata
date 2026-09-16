import copy
import json

import numpy as np
import pytest
import toml
import yaml
from jsonschema import Draft202012Validator
from pydantic import ValidationError

from qcdata import (
    ConformerSearchData,
    Model,
    ProgramInput,
    ProgramOutput,
    ProgramSpec,
    SinglePointData,
    Structure,
    Wavefunction,
    __version__,
    json_dumps,
)


def test_serialization_to_disk_json(prog_output, tmp_path):
    """Test serialization to disk json"""

    filename = tmp_path / "prog_output.json"
    prog_output.save(filename)
    reopened = ProgramOutput.open(filename)
    assert prog_output == reopened

    # No filename or other extension to json by default
    filename = tmp_path / "prog_output.whatever"
    prog_output.save(filename)
    reopened = ProgramOutput.open(filename)
    assert prog_output == reopened


def test_serialization_to_disk_yaml(prog_output, tmp_path):
    """Test serialization to disk yaml"""
    for ext in [".yaml", ".yml"]:
        filename = tmp_path / f"prog_output{ext}"
        prog_output.save(filename)
        reopened = ProgramOutput.open(filename)
        assert prog_output == reopened


def test_serialization_to_disk_toml(prog_output, tmp_path):
    """Test serialization to disk toml"""

    filename = tmp_path / "prog_output.toml"
    prog_output.save(filename)
    reopened = ProgramOutput.open(filename)
    assert prog_output == reopened


@pytest.mark.parametrize("extension", ["json", "yaml", "toml"])
def test_version_restamped_on_save(extension, opt_output, tmp_path):
    loads = {"json": json.loads, "yaml": yaml.safe_load, "toml": toml.loads}[extension]
    dumps = {"json": json.dumps, "yaml": yaml.safe_dump, "toml": toml.dumps}[extension]
    path = tmp_path / f"output.{extension}"
    # Default compact serialization must not drop the computed version.
    opt_output.save(path)
    payload = loads(path.read_text())
    models = [
        payload,
        payload["input_data"],
        payload["input_data"]["structure"],
        payload["results"],
        payload["results"]["provenance"],
        payload["execution"],
        payload["results"]["trajectory"][0],
        payload["results"]["trajectory"][0]["results"],
    ]
    for model in models:
        assert model["qcdata_version"] == __version__
        model["qcdata_version"] = "0.0.0"
    payload["extras"]["qcdata_version"] = "user metadata"
    original = copy.deepcopy(payload)
    opened = ProgramOutput.model_validate(payload)
    assert payload == original
    assert opened.qcdata_version == __version__
    assert opened.results.provenance.qcdata_version == __version__
    assert opened.extras["qcdata_version"] == "user metadata"
    path.write_text(dumps(payload))
    opened = ProgramOutput.open(path)
    opened.save(path)
    saved = loads(path.read_text())
    assert saved["qcdata_version"] == __version__
    assert saved["results"]["trajectory"][0]["results"]["qcdata_version"] == __version__
    assert "qcdata_version=" not in repr(opened)


def test_serialized_scientific_arrays_match_schema(water):
    data = SinglePointData(
        provenance={"program": "producer"},
        gradient=np.arange(9),
        hessian=np.eye(9),
        freqs_wavenumber=[1.0, 2.0],
        normal_modes_cartesian=np.arange(18).reshape(2, 3, 3),
        wavefunction=Wavefunction(scf_eigenvalues_a=[-1.0, -0.5]),
    )
    for obj in [water, data, ConformerSearchData(provenance={"program": "producer"})]:
        schema = type(obj).model_json_schema(mode="serialization")
        Draft202012Validator.check_schema(schema)
        Draft202012Validator(schema).validate(json.loads(obj.model_dump_json()))
    # Matrices must be described as nested arrays, not untyped or flat arrays.
    geometry = Structure.model_json_schema(mode="serialization")["properties"][
        "geometry"
    ]
    assert geometry["items"]["type"] == "array"
    assert geometry["items"]["items"]["type"] == "number"


@pytest.mark.parametrize("extension", ["json", "yaml", "toml"])
def test_compact_saves_and_supported_helpers(extension, water, tmp_path):
    loads = {"json": json.loads, "yaml": yaml.safe_load, "toml": toml.loads}[extension]
    structure = Structure(symbols=water.symbols, geometry=water.geometry)
    inp = ProgramInput(
        program="geometric", calctype="optimization", structure=structure, keywords={}
    )
    path = tmp_path / f"input.{extension}"
    inp.save(path)
    compact = loads(path.read_text())
    assert compact["keywords"] == {}  # Explicitly supplied default values are retained.
    for name in ["files", "extras", "cmdline_args", "structures", "subprograms"]:
        assert name not in compact
    assert compact["qcdata_version"] == __version__
    assert compact == json.loads(json_dumps(inp))
    assert json.loads(json_dumps([inp])) == [compact]

    inp.save(path, exclude_unset=False)
    expanded = loads(path.read_text())
    assert expanded["files"] == {}
    assert expanded["subprograms"] == []
    assert expanded["structure"]["charge"] == 0
    assert json.loads(json_dumps(inp, exclude_unset=False))["structures"] == {}
    assert ProgramInput.open(path) == inp

    native = tmp_path / "input.txt"
    native.write_text("native input")
    inp.add_file(native)
    directory = tmp_path / "native"
    directory.mkdir()
    (directory / "guess.bin").write_bytes(b"\x00\xff")
    inp.add_files(directory)
    structure.add_identifiers(name="water")
    inp.save(path)
    reopened = ProgramInput.open(path)
    assert reopened.files == {"input.txt": "native input", "guess.bin": b"\x00\xff"}
    assert reopened.structure.identifiers.name == "water"
    assert reopened == inp


def test_dump_edit_validate_is_independent(water):
    inp = ProgramInput(
        program="geometric",
        calctype="optimization",
        structure=water,
        keywords={"nested": {"values": [1, 2]}},
        extras={"notes": ["original"]},
        subprograms=[ProgramSpec(program="xtb", calctype="gradient")],
    )
    before = inp.model_dump()
    edited = inp.model_dump()
    edited["keywords"]["nested"]["values"].append(3)
    edited["extras"]["notes"].append("edited")
    edited["structure"]["geometry"][0][0] += 1
    edited["subprograms"][0]["keywords"]["accuracy"] = 0.5
    updated = ProgramInput.model_validate(edited)
    assert inp.model_dump() == before
    assert updated.keywords["nested"]["values"] == [1, 2, 3]
    assert ProgramInput.model_validate_json(json_dumps(updated)) == updated
    with pytest.raises(ValidationError):
        inp.program = "orca"
    with pytest.raises(ValidationError):
        inp.qcdata_version = "0.0.0"
    edited["subprograms"].append(edited["subprograms"][0])
    with pytest.raises(ValidationError, match="Duplicate subprogram"):
        ProgramInput.model_validate(edited)


def test_json_helper_single_and_list_match(prog_output):
    single = json.loads(json_dumps(prog_output, indent=2))
    multiple = json.loads(json_dumps([prog_output], indent=2))
    assert multiple == [single]
    assert single["execution"]["scratch_dir"] == "/tmp/qcdata"
    assert single["qcdata_version"] == __version__
    assert (
        Model(method="hf").model_dump(exclude_unset=True)["qcdata_version"]
        == __version__
    )

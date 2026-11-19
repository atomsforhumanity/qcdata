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


def _documents(node, path=()):
    """Every dict in a dumped document with its path, free-form extras skipped."""
    if isinstance(node, dict):
        yield path, node
        for key, value in node.items():
            if key != "extras":
                yield from _documents(value, (*path, key))
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _documents(value, (*path, i))


@pytest.mark.parametrize("extension", ["json", "yaml", "toml"])
def test_version_stamps_the_top_level_document_only(extension, opt_output, tmp_path):
    loads = {"json": json.loads, "yaml": yaml.safe_load, "toml": toml.loads}[extension]
    dumps = {"json": json.dumps, "yaml": yaml.safe_dump, "toml": toml.dumps}[extension]
    path = tmp_path / f"output.{extension}"
    opt_output.save(path)
    payload = loads(path.read_text())
    assert payload["qcdata_version"] == __version__
    nested = [d for p, d in _documents(payload) if p]
    assert len(nested) > 8
    for document in nested:
        assert "qcdata_version" not in document
        assert document.get("extras", {"not": "empty"}) != {}

    # Files written before alpha 2 stamp every object; loading discards the stamps.
    for model in [
        payload["input_data"],
        payload["input_data"]["structure"],
        payload["results"],
        payload["results"]["provenance"],
        payload["execution"],
        payload["results"]["frames"][0],
        payload["results"]["frames"][0]["results"],
    ]:
        model["qcdata_version"] = "0.0.0"
    payload["qcdata_version"] = "0.0.0"
    payload["extras"] = {"qcdata_version": "user metadata"}
    original = copy.deepcopy(payload)
    opened = ProgramOutput.model_validate(payload)
    assert payload == original
    assert opened.qcdata_version == __version__
    assert opened.extras["qcdata_version"] == "user metadata"
    path.write_text(dumps(payload))
    opened = ProgramOutput.open(path)
    opened.save(path)
    saved = loads(path.read_text())
    assert saved["qcdata_version"] == __version__
    assert saved["extras"] == {"qcdata_version": "user metadata"}
    assert all("qcdata_version" not in d for p, d in _documents(saved) if p)
    assert "qcdata_version=" not in repr(opened)


def test_dumps_honor_include_and_exclude_for_the_version(water):
    assert "qcdata_version" in water.model_dump()
    assert "qcdata_version" in json.loads(water.model_dump_json())
    assert "qcdata_version" not in water.model_dump(exclude={"qcdata_version"})
    assert set(water.model_dump(include={"symbols"})) == {"symbols"}
    nested = ProgramInput(program="x", calctype="energy", structure=water)
    assert "qcdata_version" not in nested.model_dump()["structure"]


def test_serialized_scientific_arrays_match_schema(water):
    data = SinglePointData(
        provenance={"program": "producer"},
        gradient=np.arange(9),
        hessian=np.eye(9),
        vibrational_analysis={
            "masses": [16.0, 1.0, 1.0],
            "projection": "translations_rotations",
            "frequencies": [1.0, 2.0],
            "normal_modes_cartesian": np.arange(18).reshape(2, 3, 3),
        },
        wavefunction=Wavefunction(
            orbitals={
                "spin": "restricted",
                "energies_a": [-0.5],
                "occupations_a": [1.0],
                "occupations_b": [1.0],
                "core_electrons": [2, 0, 0],
                "coefficients_a": [[1.0]],
            },
            basis={
                "shells": [
                    {
                        "center": 0,
                        "angular_momentum": 0,
                        "pure": True,
                        "exponents": [1.0],
                        "coefficients": [0.71],
                    }
                ],
                "ecp": [
                    {
                        "center": 0,
                        "core_electrons": 2,
                        "local": [{"power": -2, "exponent": 1.0, "coefficient": 2.0}],
                        "semilocal": [],
                    }
                ],
            },
            overlap=[[1.0]],
            hamiltonian_a=[[-0.5]],
            core_hamiltonian=[[-1.0]],
        ),
        atomic_dipoles=np.zeros((3, 3)),
        atomic_quadrupoles=np.zeros((3, 3, 3)),
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
    inp.save(path, exclude_unset=True)
    compact = loads(path.read_text())
    assert compact["keywords"] == {}  # Explicitly supplied default values are retained.
    for name in ["files", "extras", "cmdline_args", "structures", "subprograms"]:
        assert name not in compact
    assert compact["qcdata_version"] == __version__
    assert compact == json.loads(json_dumps(inp))
    assert json.loads(json_dumps([inp])) == [compact]

    inp.save(path)  # lossless by default
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


def test_save_keeps_containers_filled_after_construction(tmp_path):
    from qcdata import ProgramInput, ProgramOutput, Provenance, SinglePointData

    structure = Structure(symbols=["He"], geometry=[[0.0, 0.0, 0.0]])
    inp = ProgramInput(
        program="toy", calctype="energy", structure=structure, model={"method": "lj"}
    )
    out = ProgramOutput(
        input_data=inp,
        success=True,
        results=SinglePointData(provenance=Provenance(program="toy"), energy=-2.8),
    )
    out.record["scf"] = {"iterations": 12}  # filled in after the output was built
    out.save(tmp_path / "out.json")
    assert ProgramOutput.open(tmp_path / "out.json").record == {
        "scf": {"iterations": 12}
    }


def test_indented_json_keeps_innermost_arrays_on_one_line(tmp_path):
    """Rows of numbers stay on one line; the document around them stays indented."""
    structure = Structure(
        symbols=["O", "H", "H"],
        geometry=[[0.0, 0.0, -0.1294], [-0.0, 1.4941, 1.0274], [0.0, -1.4941, 1e300]],
        connectivity=[(0, 1, 1.0), (0, 2, 1.0)],
    )
    inp = ProgramInput(
        program="qcopt",
        calctype="optimization",
        structure=structure,
        keywords={"weights": [[]], "names": ["a", "qcdata.row.0"], "empty": []},
        subprograms=[ProgramSpec(program="axtb", calctype="gradient")],
    )
    path = tmp_path / "inp.json"
    inp.save(path)
    text = path.read_text()
    lines = text.splitlines()
    assert '        "symbols": ["O", "H", "H"],' in lines
    assert "            [-0.0, 1.4941, 1.0274]," in lines
    assert "            [0.0, -1.4941, 1e+300]" in lines
    assert "            [0.0, 1.0, 1.0]," in lines
    assert '        "names": ["a", "qcdata.row.0"],' in lines
    assert '        "empty": []' in lines
    assert '        "weights": [' in lines  # a list of lists stays indented
    assert '    "subprograms": [' in lines  # a list of objects too
    assert ProgramInput.open(path) == inp
    assert np.signbit(ProgramInput.open(path).structure.geometry[1, 0])
    inp.save(path, indent=None)
    compact = path.read_text()
    assert "\n" not in compact and json.loads(compact) == json.loads(text)

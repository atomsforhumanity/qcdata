"""Saving large arrays apart: the 1 MiB rule, pointers, file names, exact round trips."""

import hashlib
import json
import struct

import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import (
    Frame,
    OptimizationData,
    ProgramInput,
    ProgramOutput,
    Provenance,
    SinglePointData,
    Structure,
)

LIMIT = (1 << 20) // 8  # float64 elements in 1 MiB
PROVENANCE = Provenance(program="axtb")


def _charges(n: int) -> SinglePointData:
    return SinglePointData(provenance=PROVENANCE, atomic_charges=np.linspace(-1, 1, n))


def _same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()


def test_arrays_over_one_mib_go_to_files(tmp_path):
    assert _charges(LIMIT).save(tmp_path / "small.json") == [tmp_path / "small.json"]
    assert isinstance(
        json.loads((tmp_path / "small.json").read_text())["atomic_charges"], list
    )

    data = _charges(LIMIT + 1)
    paths = data.save(tmp_path / "large.json")
    npy = tmp_path / "large.atomic_charges.npy"
    assert paths == [tmp_path / "large.json", npy]
    pointer = json.loads((tmp_path / "large.json").read_text())["atomic_charges"]
    assert pointer == {
        "path": "large.atomic_charges.npy",
        "shape": [LIMIT + 1],
        "dtype": "float64",
        "sha256": hashlib.sha256(npy.read_bytes()).hexdigest(),
    }
    assert _same_bits(np.load(npy), data.atomic_charges)


def test_save_lists_every_file_it_writes(tmp_path):
    n = 128  # a 384 x 384 Hessian is over 1 MiB; the gradient is not
    rng = np.random.default_rng(7)
    frames = [
        Frame(
            success=True,
            structure=Structure(symbols=["C"] * n, geometry=rng.normal(size=(n, 3))),
            results=SinglePointData(
                provenance=PROVENANCE,
                energy=-1.0,
                gradient=rng.normal(size=(n, 3)),
                hessian=rng.normal(size=(3 * n, 3 * n)),
            ),
        )
        for _ in range(2)
    ]
    inp = ProgramInput(
        program="qcopt",
        calctype="transition_state",
        structure=frames[0].structure,
        subprograms=[{"program": "axtb", "calctype": "hessian"}],
        given=frames[0].results,
    )
    out = ProgramOutput(
        input_data=inp,
        results=OptimizationData(provenance=PROVENANCE, frames=frames),
        success=True,
    )
    paths = out.save(tmp_path / "ts.json")
    assert [p.name for p in paths] == [
        "ts.json",
        "ts.input_data.given.hessian.npy",
        "ts.results.frames.0.results.hessian.npy",
        "ts.results.frames.1.results.hessian.npy",
    ]
    assert set(tmp_path.iterdir()) == set(paths)

    back = ProgramOutput.open(tmp_path / "ts.json")
    assert back == out
    for a, b in zip(back.results.frames, out.results.frames, strict=True):
        assert type(a.results.hessian) is np.ndarray
        assert _same_bits(a.results.hessian, b.results.hessian)
    assert _same_bits(back.input_data.given.hessian, inp.given.hessian)


def test_models_saved_side_by_side_keep_their_arrays_apart(tmp_path):
    a, b = _charges(LIMIT + 1), _charges(LIMIT + 2)
    a.save(tmp_path / "a.json")
    b.save(tmp_path / "b.json")
    assert SinglePointData.open(tmp_path / "a.json") == a
    assert SinglePointData.open(tmp_path / "b.json") == b


def test_file_names_escape_what_is_not_a_plain_name(tmp_path):
    n = LIMIT // 3 + 1
    big = Structure(symbols=["H"] * n, geometry=np.arange(3 * n, dtype=float))
    water = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 1.4]])
    inp = ProgramInput(
        program="linus",
        calctype="energy",
        structure=water,
        structures={"qm.mm": big, "../up": big},
    )
    paths = inp.save(tmp_path / "in.json")
    assert sorted(p.name for p in paths[1:]) == [
        "in.structures.%2E%2E%2Fup.geometry.npy",
        "in.structures.qm%2Emm.geometry.npy",
    ]
    assert all(p.parent == tmp_path for p in paths)
    assert ProgramInput.open(tmp_path / "in.json") == inp


@pytest.mark.parametrize("suffix", [".yaml", ".toml"])
def test_yaml_and_toml_saves_put_arrays_apart_too(suffix, tmp_path):
    data = _charges(LIMIT + 1)
    paths = data.save(tmp_path / f"d{suffix}")
    assert [p.name for p in paths] == [f"d{suffix}", "d.atomic_charges.npy"]
    assert SinglePointData.open(tmp_path / f"d{suffix}") == data


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_inline_floats_round_trip_exactly(suffix, tmp_path):
    rng = np.random.default_rng(3)
    special = [
        0.1,
        1 / 3,
        -0.0,
        5e-324,
        2.2250738585072014e-308,
        1.7976931348623157e308,
    ]
    special += [1e16, 1e17, 1e22, 1e-7, -1.5e-300, 123456789.12345679]
    random = rng.normal(size=4000) * 10.0 ** rng.integers(-300, 300, size=4000)
    values = np.concatenate([special, random])
    data = SinglePointData(
        provenance=PROVENANCE,
        energy=-76.02663273509764,
        atomic_charges=values,
        energy_components={"xc": -9.123456789012345, "dispersion": -0.0},
    )
    data.save(tmp_path / f"d{suffix}")
    back = SinglePointData.open(tmp_path / f"d{suffix}")
    assert _same_bits(back.atomic_charges, data.atomic_charges)
    assert struct.pack("<d", back.energy) == struct.pack("<d", data.energy)
    for key, value in data.energy_components.items():
        assert struct.pack("<d", back.energy_components[key]) == struct.pack(
            "<d", value
        )


def test_free_form_arrays_stay_inline(tmp_path):
    data = SinglePointData(provenance=PROVENANCE, extras={"trace": np.zeros(LIMIT + 1)})
    assert data.save(tmp_path / "d.json") == [tmp_path / "d.json"]
    saved = json.loads((tmp_path / "d.json").read_text())
    assert len(saved["extras"]["trace"]) == LIMIT + 1


def test_opening_checks_each_array_file(tmp_path):
    data = _charges(LIMIT + 1)
    data.save(tmp_path / "d.json")
    npy = tmp_path / "d.atomic_charges.npy"

    np.save(npy, np.zeros(LIMIT + 1))
    with pytest.raises(ValidationError, match="does not match the sha256"):
        SinglePointData.open(tmp_path / "d.json")

    npy.unlink()
    with pytest.raises(ValidationError, match="Cannot read the array file"):
        SinglePointData.open(tmp_path / "d.json")


def test_a_pointer_needs_open(tmp_path):
    _charges(LIMIT + 1).save(tmp_path / "d.json")
    text = (tmp_path / "d.json").read_text()
    with pytest.raises(ValidationError, match="open the file it was saved with"):
        SinglePointData.model_validate_json(text)


@pytest.mark.parametrize(
    "pointer, match",
    [
        ({"path": "../x.npy", "shape": [1], "dtype": "float64", "sha256": ""}, "name"),
        (
            {"path": "/tmp/x.npy", "shape": [1], "dtype": "float64", "sha256": ""},
            "name",
        ),
        ({"path": "x.npy"}, "takes numbers"),
        ({"x": 1}, "takes numbers"),
    ],
)
def test_pointers_are_names_beside_the_file(pointer, match, tmp_path):
    saved = {"provenance": {"program": "axtb"}, "atomic_charges": pointer}
    (tmp_path / "d.json").write_text(json.dumps(saved))
    with pytest.raises(ValidationError, match=match):
        SinglePointData.open(tmp_path / "d.json")


def test_xyz_saves_list_their_one_file(tmp_path):
    water = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 1.4]])
    assert water.save(tmp_path / "w.xyz") == [tmp_path / "w.xyz"]


def test_the_module_serves_writers_other_than_save(tmp_path):
    """A writer of its own (a worker's wire) puts arrays apart with the public API."""
    from qcdata.array_files import DIRECTORY, POINTER_KEYS, SINK, ArraySink

    data = _charges(LIMIT + 1)
    sink = ArraySink()
    document = data.model_dump(mode="json", context={SINK: sink})
    document, written = sink.place(document, tmp_path, "msg")
    assert written == [tmp_path / "msg.atomic_charges.npy"]
    assert set(document["atomic_charges"]) == POINTER_KEYS
    back = SinglePointData.model_validate(document, context={DIRECTORY: tmp_path})
    assert _same_bits(back.atomic_charges, data.atomic_charges)

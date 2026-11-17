"""The calculation key: golden keys, the canonical form, and each rule both ways.

The golden keys in ``tests/data/golden_keys.json`` must never change. If one does, the
change is a new key version (``k2``), released deliberately, never an edit here.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from qcdata import (
    FileInput,
    ProgramInput,
    ProgramSpec,
    Provenance,
    SinglePointData,
    Structure,
    calculation_key,
)
from qcdata.key import _canonical_model

GOLDEN = json.loads((Path(__file__).parent / "data" / "golden_keys.json").read_text())

WATER = {
    "symbols": ["O", "H", "H"],
    "geometry": [[0.0, 0.0, -0.1294], [0.0, 1.4941, 1.0274], [0.0, -1.4941, 1.0274]],
}


def _load(case: dict[str, Any]) -> FileInput | ProgramInput:
    if "calctype" in case["input"]:
        return ProgramInput.model_validate(case["input"])
    return FileInput.model_validate(case["input"])


def _input(**changes: Any) -> ProgramInput:
    """A gradient input, with fields replaced or added."""
    values: dict[str, Any] = {
        "program": "axtb",
        "calctype": "gradient",
        "model": {"method": "gfn2"},
        "structure": WATER,
    }
    values.update(changes)
    return ProgramInput.model_validate(values)


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_goldencalculation_key(case):
    inp = _load(case)
    canonical = json.dumps(_canonical_model(inp), sort_keys=True)
    assert calculation_key(inp) == case["key"], (
        f"The key of {case['name']} changed; its canonical form is now {canonical}"
    )


def test_golden_cases_cover_each_equivalence():
    keys = {case["name"]: case["key"] for case in GOLDEN}
    assert keys["energy_excluded_and_defaults"] == keys["energy"]
    assert keys["energy_rounded_coordinates"] == keys["energy"]
    assert keys["gradient_keywords_normalized"] == keys["gradient_keywords"]
    assert len(set(keys.values())) == len(keys) - 3


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_golden_key_survives_json_and_save(case, tmp_path):
    """A stored input, loaded again, has its key: the store checks this on every put."""
    inp = _load(case)
    cls = type(inp)
    assert (
        calculation_key(cls.model_validate_json(inp.model_dump_json())) == case["key"]
    )
    for suffix in (".json", ".yaml"):
        inp.save(tmp_path / f"input{suffix}")
        assert calculation_key(cls.open(tmp_path / f"input{suffix}")) == case["key"]


def test_key_survives_arrays_saved_apart(tmp_path):
    n = 128  # a 384 x 384 Hessian, over 1 MiB, goes to its own file
    rng = np.random.default_rng(5)
    structure = Structure(symbols=["C"] * n, geometry=rng.normal(size=(n, 3)) * 3)
    hessian = rng.normal(size=(3 * n, 3 * n))
    hessian[0, :3] = [-0.0, 1.0, 1e300]
    inp = ProgramInput(
        program="qcopt",
        calctype="transition_state",
        structure=structure,
        subprograms=[{"program": "axtb", "calctype": "hessian"}],
        given=SinglePointData(provenance={"program": "axtb"}, hessian=hessian),
    )
    paths = inp.save(tmp_path / "input.json")
    assert [p.name for p in paths] == ["input.json", "input.given.hessian.npy"]
    assert calculation_key(
        ProgramInput.open(tmp_path / "input.json")
    ) == calculation_key(inp)


def test_canonical_text_is_documented_form():
    """The key is sha256 of exactly this text: compact JSON, keys sorted, ASCII."""
    energy = next(c for c in GOLDEN if c["name"] == "energy")
    text = (
        '{"calctype":"energy","model":{"method":"gfn2"},"program":"axtb",'
        '"structure":{"geometry":[[0,0,-1294000000],[0,14941000000,10274000000],'
        '[0,-14941000000,10274000000]],"symbols":["O","H","H"]}}'
    )
    assert energy["key"] == "k1:" + hashlib.sha256(text.encode()).hexdigest()

    keywords = next(c for c in GOLDEN if c["name"] == "gradient_keywords")
    text = (
        '{"calctype":"gradient","keywords":'
        '{"accuracy_scale":0.5,"electronic_temperature":300,"output":'
        '{"bond_orders":true,"orbitals":false},"scf":{"e_tol":1e-08,"guess":"sad",'
        '"level_shift":0,"max_iter":100},"solvent":{"model":null,"name":"water"}},'
        '"model":{"method":"gfn2"},"program":"axtb","structure":{"geometry":'
        "[[0,0,-1294000000],[0,14941000000,10274000000],"
        '[0,-14941000000,10274000000]],"symbols":["O","H","H"]}}'
    )
    assert keywords["key"] == "k1:" + hashlib.sha256(text.encode()).hexdigest()

    files = next(c for c in GOLDEN if c["name"] == "file_input")
    binary = hashlib.sha256(b"\x00\x01\x02\xffGBW").hexdigest()
    text_file = hashlib.sha256(
        b"! B3LYP def2-SVP\n* xyzfile 0 1 water.xyz\n"
    ).hexdigest()
    unicode_file = hashlib.sha256("Ångström\n".encode()).hexdigest()
    text = (
        '{"cmdline_args":["input.inp"],'
        f'"files":{{"guess.gbw":"{binary}","input.inp":"{text_file}",'
        f'"notes/\\u00c5.txt":"{unicode_file}"}},"program":"orca"}}'
    )
    assert files["key"] == "k1:" + hashlib.sha256(text.encode()).hexdigest()

    given = next(c for c in GOLDEN if c["name"] == "transition_state_nested_given")
    hessian = np.array(given["input"]["given"]["hessian"], dtype="<f8") + 0.0
    digest = hashlib.sha256(hessian.tobytes(order="C")).hexdigest()
    text = (
        '{"calctype":"transition_state","given":{"energy":-5.0704,"hessian":'
        f'{{"array":[9,9],"dtype":"float64","sha256":"{digest}"}},'
        '"provenance":{"program":"axtb","program_version":"0.1.1"}},'
        '"program":"qcopt","structure":{"geometry":[[0,0,-1294000000],'
        "[0,14941000000,10274000000],[0,-14941000000,10274000000]],"
        '"symbols":["O","H","H"]},"subprograms":[{"calctype":"gradient",'
        '"model":{"method":"gxtb"},"program":"axtb"},{"calctype":"hessian",'
        '"keywords":{"step":0.005},"program":"findiff","subprograms":[{'
        '"calctype":"gradient","model":{"method":"gxtb"},"program":"axtb"}]}]}'
    )
    assert given["key"] == "k1:" + hashlib.sha256(text.encode()).hexdigest()


def test_key_format():
    key = calculation_key(_input())
    assert key.startswith("k1:")
    assert len(key) == 3 + 64
    int(key[3:], 16)


# What leaves the key unchanged.


@pytest.mark.parametrize(
    "a, b",
    [
        ({"scf": {"max_iter": 100}}, {"scf": {"max_iter": 100.0}}),
        ({"shift": 0.0}, {"shift": -0.0}),
        ({"shift": 0}, {"shift": -0.0}),
        ({"temperature": 300}, {"temperature": np.float64(300.0)}),
        ({"n": 3}, {"n": np.int64(3)}),
        ({"scale": 0.1}, {"scale": np.float64(0.1)}),
        ({"flag": True}, {"flag": np.bool_(True)}),
        ({"weights": [0.5, 1.0]}, {"weights": (0.5, 1)}),
        ({"weights": [0.5, 1.0]}, {"weights": np.array([0.5, 1.0])}),
        ({"a": 1, "b": {"c": 2, "d": 3}}, {"b": {"d": 3, "c": 2}, "a": 1}),
        ({}, {"threads": 4, "memory_gib": 2.5}),
        ({"scf": {"max_iter": 50}}, {"scf": {"max_iter": 50}, "threads": 4}),
    ],
)
def test_same_keywords(a, b):
    assert calculation_key(_input(keywords=a)) == calculation_key(_input(keywords=b))


def test_excluded_fields_and_defaults():
    plain = calculation_key(_input())
    assert calculation_key(_input(extras={"x": 1})) == plain
    assert (
        calculation_key(_input(model={"method": "gfn2", "extras": {"x": 1}})) == plain
    )
    assert (
        calculation_key(
            _input(
                structure={
                    **WATER,
                    "identifiers": {"name": "water", "smiles": "O", "inchikey": "x"},
                    "extras": {"xyz_comments": ["a"]},
                }
            )
        )
        == plain
    )
    assert (
        calculation_key(_input(structure={**WATER, "charge": 0, "multiplicity": 1}))
        == plain
    )
    assert (
        calculation_key(_input(subprograms=[], files={}, cmdline_args=[], given=None))
        == plain
    )
    assert calculation_key(_input(model={"method": "gfn2", "basis": None})) == plain


def test_subprogram_resources_and_extras_left_out():
    def spec(**keywords: Any) -> dict[str, Any]:
        return {
            "program": "axtb",
            "calctype": "gradient",
            "model": {"method": "gfn2"},
            "keywords": keywords,
        }

    base = calculation_key(
        _input(program="qcopt", calctype="optimization", subprograms=[spec()])
    )
    for changed in (
        spec(threads=8),
        spec(memory_gib=3.0),
        {**spec(), "extras": {"note": "x"}},
    ):
        inp = _input(program="qcopt", calctype="optimization", subprograms=[changed])
        assert calculation_key(inp) == base


def test_coordinates_round_to_1e10_bohr():
    plain = calculation_key(_input())
    moved = np.array(WATER["geometry"]) + 4e-11
    assert calculation_key(_input(structure={**WATER, "geometry": moved})) == plain
    moved = np.array(WATER["geometry"]) + 2e-10
    assert calculation_key(_input(structure={**WATER, "geometry": moved})) != plain


def test_text_and_bytes_files_with_the_same_bytes():
    text = FileInput(program="orca", files={"a.inp": "Å\n"})
    raw = FileInput(program="orca", files={"a.inp": "Å\n".encode()})
    assert calculation_key(text) == calculation_key(raw)


def test_keying_changes_nothing():
    inp = _input(keywords={"shift": -0.0, "threads": 4})
    before = inp.model_dump()
    calculation_key(inp)
    assert inp.model_dump() == before
    assert np.signbit(inp.keywords["shift"])


# What changes the key: every difference not named above is a miss.


@pytest.mark.parametrize(
    "a, b",
    [
        ({"scf": {"max_iter": 100}}, {"scf": {"max_iter": 101}}),
        ({"scale": 0.1}, {"scale": np.nextafter(0.1, 1.0)}),
        ({"flag": True}, {"flag": 1}),
        ({"flag": False}, {"flag": 0}),
        ({"flag": None}, {}),
        ({"scf": {}}, {}),
        ({"scf": {"max_iter": 100}}, {}),
        ({"solvent": "water"}, {"solvent": {"name": "water"}}),
        ({"guess": "sad"}, {"guess": "SAD"}),
        ({"n": 2**53 + 1}, {"n": float(2**53 + 1)}),
        ({"scf": {"threads": 4}}, {}),
        ({"weights": [0.5, 1.0]}, {"weights": [1.0, 0.5]}),
        ({"weights": [[0.5, 1.0]]}, {"weights": [0.5, 1.0]}),
        ({"device_memory_gib": 4.0}, {}),
        ({"backend": "gpu"}, {"backend": "cpu"}),
        ({"accuracy": {"gradient": 1e-6}}, {"accuracy": {"gradient": 1e-7}}),
        ({"accuracy": {"gradient": 1e-6}}, {}),
    ],
)
def test_different_keywords(a, b):
    assert calculation_key(_input(keywords=a)) != calculation_key(_input(keywords=b))


def test_every_part_of_the_input_counts():
    plain = calculation_key(_input())
    swapped = {
        "symbols": ["H", "O", "H"],
        "geometry": np.array(WATER["geometry"])[[1, 0, 2]],
    }
    given = SinglePointData(provenance={"program": "axtb"}, energy=-5.07)
    changed = [
        _input(program="linus"),
        _input(calctype="energy"),
        _input(model={"method": "GFN2"}),
        _input(model={"method": "gfn2", "basis": "sto-3g"}),
        _input(structure={**WATER, "charge": 1, "multiplicity": 2}),
        _input(structure={**WATER, "connectivity": [(0, 1, 1.0)]}),
        _input(structure=swapped),
        _input(structures={"environment": WATER}),
        _input(given=given),
        _input(given=given.model_copy(update={"energy": -5.07 + 1e-15})),
        _input(files={"grid.dat": "1"}),
        _input(cmdline_args=["-v"]),
        _input(keywords={"threads": 4, "x": 1}),
    ]
    keys = [calculation_key(inp) for inp in changed]
    assert plain not in keys
    assert len(set(keys)) == len(keys)


def test_given_arrays_count_to_the_last_bit():
    hessian = np.eye(9)
    a = SinglePointData(provenance={"program": "axtb"}, hessian=hessian)
    nudged = hessian.copy()
    nudged[4, 4] = np.nextafter(1.0, 2.0)
    b = SinglePointData(provenance={"program": "axtb"}, hessian=nudged)
    assert calculation_key(_input(given=a)) != calculation_key(_input(given=b))
    c = SinglePointData(
        provenance={"program": "axtb", "program_version": "x"}, hessian=hessian
    )
    assert calculation_key(_input(given=a)) != calculation_key(_input(given=c))


def _given(hessian: Any) -> ProgramInput:
    given = SinglePointData(provenance=Provenance(program="axtb"), hessian=hessian)
    return _input(given=given)


def test_array_fields_count_by_value_not_layout():
    rng = np.random.default_rng(11)
    hessian = rng.normal(size=(9, 9))
    plain = calculation_key(_given(hessian))
    assert calculation_key(_given(np.asfortranarray(hessian))) == plain
    assert calculation_key(_given(hessian.astype(">f8"))) == plain
    assert calculation_key(_given(hessian.tolist())) == plain
    zeros = np.zeros((9, 9))
    signed = zeros.copy()
    signed[0, 0] = -0.0
    assert calculation_key(_given(signed)) == calculation_key(_given(zeros))


def test_array_fields_enter_by_shape_dtype_and_digest():
    from qcdata.key import _array

    floats = np.array([[1.0, -0.0], [0.5, 2.0]])
    assert _array(floats) == {
        "array": [2, 2],
        "dtype": "float64",
        "sha256": hashlib.sha256(
            np.array([[1.0, 0.0], [0.5, 2.0]], dtype="<f8").tobytes()
        ).hexdigest(),
    }
    assert _array(floats.reshape(4))["sha256"] == _array(floats)["sha256"]
    assert _array(floats.reshape(4)) != _array(floats)
    ints = np.array([1, 2, 3], dtype=">i4")
    assert _array(ints) == {
        "array": [3],
        "dtype": "int32",
        "sha256": hashlib.sha256(
            np.array([1, 2, 3], dtype="<i4").tobytes()
        ).hexdigest(),
    }
    assert _array(np.array([1, 2, 3]))["dtype"] == "int64"
    with pytest.raises(ValueError, match="finite"):
        _array(np.array([1.0, np.inf]))
    with pytest.raises(TypeError):
        _array(np.array([1j]))


def test_free_form_arrays_stay_json_numbers():
    as_array = calculation_key(_input(keywords={"w": np.array([[0.5, 1.0]])}))
    assert as_array == calculation_key(_input(keywords={"w": [[0.5, 1]]}))


def test_non_finite_given_arrays_are_refused():
    hessian = np.eye(9)
    hessian[2, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        calculation_key(_given(hessian))


def test_files_count_by_name_and_content():
    plain = calculation_key(FileInput(program="orca", files={"a.inp": "x"}))
    assert calculation_key(FileInput(program="orca", files={"b.inp": "x"})) != plain
    assert calculation_key(FileInput(program="orca", files={"a.inp": "x\n"})) != plain
    assert (
        calculation_key(FileInput(program="orca", files={"a.inp": b"x\x00"})) != plain
    )


def test_subprograms_count_in_order_and_depth():
    gradient = {"program": "axtb", "calctype": "gradient", "model": {"method": "gfn2"}}
    hessian = {"program": "axtb", "calctype": "hessian", "model": {"method": "gfn2"}}
    a = _input(
        program="qcopt", calctype="transition_state", subprograms=[gradient, hessian]
    )
    b = _input(
        program="qcopt", calctype="transition_state", subprograms=[hessian, gradient]
    )
    assert calculation_key(a) != calculation_key(b)
    deeper = {**hessian, "subprograms": [gradient]}
    c = _input(
        program="qcopt", calctype="transition_state", subprograms=[gradient, deeper]
    )
    assert calculation_key(c) != calculation_key(a)
    nested_change = {**hessian, "subprograms": [{**gradient, "keywords": {"x": 1}}]}
    d = _input(
        program="qcopt",
        calctype="transition_state",
        subprograms=[gradient, nested_change],
    )
    assert calculation_key(d) != calculation_key(c)


# Refusals.


def test_refuses_what_is_not_a_calculation():
    spec = ProgramSpec(program="axtb", calctype="gradient", model={"method": "gfn2"})
    with pytest.raises(TypeError, match="from_spec"):
        calculation_key(spec)
    with pytest.raises(TypeError):
        calculation_key(Structure(**WATER))  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "keywords, error",
    [
        ({"x": float("nan")}, ValueError),
        ({"x": [1.0, float("inf")]}, ValueError),
        ({"x": np.array([np.nan])}, ValueError),
        ({"x": {1, 2}}, TypeError),
        ({"x": {1: "a"}}, TypeError),
        ({"x": b"raw"}, TypeError),
        ({"x": 1j}, TypeError),
    ],
)
def test_refuses_values_json_cannot_hold(keywords, error):
    with pytest.raises(error):
        calculation_key(_input(keywords=keywords))


def test_refuses_non_finite_coordinates():
    geometry = np.array(WATER["geometry"])
    geometry[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite coordinates"):
        calculation_key(_input(structure={**WATER, "geometry": geometry}))

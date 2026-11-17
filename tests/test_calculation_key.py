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
CONTEXT = {"axtb": "0.1.1"}


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


def _key(inp: FileInput | ProgramInput, context: dict[str, str] = CONTEXT) -> str:
    return calculation_key(inp, context)


@pytest.mark.parametrize("case", GOLDEN, ids=[c["name"] for c in GOLDEN])
def test_golden_key(case):
    inp = _load(case)
    canonical = json.dumps(
        {"context": case["context"], "input": _canonical_model(inp)}, sort_keys=True
    )
    assert calculation_key(inp, case["context"]) == case["key"], (
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
        _key(cls.model_validate_json(inp.model_dump_json()), case["context"])
        == (case["key"])
    )
    for suffix in (".json", ".yaml"):
        inp.save(tmp_path / f"input{suffix}")
        assert (
            _key(cls.open(tmp_path / f"input{suffix}"), case["context"])
            == (case["key"])
        )


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
    assert _key(ProgramInput.open(tmp_path / "input.json")) == _key(inp)


def test_canonical_text_is_documented_form():
    """The key is sha256 of exactly this text: compact JSON, keys sorted, ASCII."""
    energy = next(c for c in GOLDEN if c["name"] == "energy")
    text = (
        '{"context":{"axtb":"0.1.1"},"input":{"calctype":"energy",'
        '"model":{"method":"gfn2"},"program":"axtb","structure":{"geometry":'
        "[[0,0,-1294000000],[0,14941000000,10274000000],"
        '[0,-14941000000,10274000000]],"symbols":["O","H","H"]}}}'
    )
    assert energy["key"] == "k1:" + hashlib.sha256(text.encode()).hexdigest()

    keywords = next(c for c in GOLDEN if c["name"] == "gradient_keywords")
    text = (
        '{"context":{"axtb":"0.1.1"},"input":{"calctype":"gradient","keywords":'
        '{"accuracy_scale":0.5,"electronic_temperature":300,"output":'
        '{"bond_orders":true,"orbitals":false},"scf":{"e_tol":1e-08,"guess":"sad",'
        '"level_shift":0,"max_iter":100},"solvent":{"model":null,"name":"water"}},'
        '"model":{"method":"gfn2"},"program":"axtb","structure":{"geometry":'
        "[[0,0,-1294000000],[0,14941000000,10274000000],"
        '[0,-14941000000,10274000000]],"symbols":["O","H","H"]}}}'
    )
    assert keywords["key"] == "k1:" + hashlib.sha256(text.encode()).hexdigest()

    files = next(c for c in GOLDEN if c["name"] == "file_input")
    binary = hashlib.sha256(b"\x00\x01\x02\xffGBW").hexdigest()
    text_file = hashlib.sha256(
        b"! B3LYP def2-SVP\n* xyzfile 0 1 water.xyz\n"
    ).hexdigest()
    unicode_file = hashlib.sha256("Ångström\n".encode()).hexdigest()
    text = (
        '{"context":{"orca":"6.0.1"},"input":{"cmdline_args":["input.inp"],'
        f'"files":{{"guess.gbw":"{binary}","input.inp":"{text_file}",'
        f'"notes/\\u00c5.txt":"{unicode_file}"}},"program":"orca"}}}}'
    )
    assert files["key"] == "k1:" + hashlib.sha256(text.encode()).hexdigest()


def test_key_format():
    key = _key(_input())
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
    assert _key(_input(keywords=a)) == _key(_input(keywords=b))


def test_excluded_fields_and_defaults():
    plain = _key(_input())
    assert _key(_input(extras={"x": 1})) == plain
    assert _key(_input(model={"method": "gfn2", "extras": {"x": 1}})) == plain
    assert (
        _key(
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
    assert _key(_input(structure={**WATER, "charge": 0, "multiplicity": 1})) == plain
    assert _key(_input(subprograms=[], files={}, cmdline_args=[], given=None)) == plain
    assert _key(_input(model={"method": "gfn2", "basis": None})) == plain


def test_subprogram_resources_and_extras_left_out():
    def spec(**keywords: Any) -> dict[str, Any]:
        return {
            "program": "axtb",
            "calctype": "gradient",
            "model": {"method": "gfn2"},
            "keywords": keywords,
        }

    base = _key(_input(program="qcopt", calctype="optimization", subprograms=[spec()]))
    for changed in (
        spec(threads=8),
        spec(memory_gib=3.0),
        {**spec(), "extras": {"note": "x"}},
    ):
        inp = _input(program="qcopt", calctype="optimization", subprograms=[changed])
        assert _key(inp) == base


def test_coordinates_round_to_1e10_bohr():
    plain = _key(_input())
    moved = np.array(WATER["geometry"]) + 4e-11
    assert _key(_input(structure={**WATER, "geometry": moved})) == plain
    moved = np.array(WATER["geometry"]) + 2e-10
    assert _key(_input(structure={**WATER, "geometry": moved})) != plain


def test_text_and_bytes_files_with_the_same_bytes():
    text = FileInput(program="orca", files={"a.inp": "Å\n"})
    raw = FileInput(program="orca", files={"a.inp": "Å\n".encode()})
    assert _key(text) == _key(raw)


def test_context_order_does_not_matter():
    inp = _input()
    assert _key(inp, {"a": "1", "b": "2"}) == _key(inp, {"b": "2", "a": "1"})


def test_keying_changes_nothing():
    inp = _input(keywords={"shift": -0.0, "threads": 4})
    before = inp.model_dump()
    _key(inp)
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
    ],
)
def test_different_keywords(a, b):
    assert _key(_input(keywords=a)) != _key(_input(keywords=b))


def test_every_part_of_the_input_counts():
    plain = _key(_input())
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
    keys = [_key(inp) for inp in changed]
    assert plain not in keys
    assert len(set(keys)) == len(keys)


def test_given_arrays_count_to_the_last_bit():
    hessian = np.eye(9)
    a = SinglePointData(provenance={"program": "axtb"}, hessian=hessian)
    nudged = hessian.copy()
    nudged[4, 4] = np.nextafter(1.0, 2.0)
    b = SinglePointData(provenance={"program": "axtb"}, hessian=nudged)
    assert _key(_input(given=a)) != _key(_input(given=b))
    c = SinglePointData(
        provenance={"program": "axtb", "program_version": "x"}, hessian=hessian
    )
    assert _key(_input(given=a)) != _key(_input(given=c))


def test_files_count_by_name_and_content():
    plain = _key(FileInput(program="orca", files={"a.inp": "x"}))
    assert _key(FileInput(program="orca", files={"b.inp": "x"})) != plain
    assert _key(FileInput(program="orca", files={"a.inp": "x\n"})) != plain
    assert _key(FileInput(program="orca", files={"a.inp": b"x\x00"})) != plain


def test_subprograms_count_in_order_and_depth():
    gradient = {"program": "axtb", "calctype": "gradient", "model": {"method": "gfn2"}}
    hessian = {"program": "axtb", "calctype": "hessian", "model": {"method": "gfn2"}}
    a = _input(
        program="qcopt", calctype="transition_state", subprograms=[gradient, hessian]
    )
    b = _input(
        program="qcopt", calctype="transition_state", subprograms=[hessian, gradient]
    )
    assert _key(a) != _key(b)
    deeper = {**hessian, "subprograms": [gradient]}
    c = _input(
        program="qcopt", calctype="transition_state", subprograms=[gradient, deeper]
    )
    assert _key(c) != _key(a)
    nested_change = {**hessian, "subprograms": [{**gradient, "keywords": {"x": 1}}]}
    d = _input(
        program="qcopt",
        calctype="transition_state",
        subprograms=[gradient, nested_change],
    )
    assert _key(d) != _key(c)


def test_context_counts():
    inp = _input()
    assert _key(inp, {}) != _key(inp, {"axtb": "0.1.1"})
    assert _key(inp, {"axtb": "0.1.1"}) != _key(inp, {"axtb": "0.1.2"})
    assert _key(inp, {"axtb": "0.1.1"}) != _key(inp, {"linus": "0.1.1"})


# Refusals.


def test_refuses_what_is_not_a_calculation():
    spec = ProgramSpec(program="axtb", calctype="gradient", model={"method": "gfn2"})
    with pytest.raises(TypeError, match="from_spec"):
        calculation_key(spec, {})  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        calculation_key(Structure(**WATER), {})  # type: ignore[arg-type]


@pytest.mark.parametrize("context", [{"axtb": 6}, {1: "6"}, {"axtb": None}])
def test_refuses_a_context_that_is_not_strings(context):
    with pytest.raises(TypeError, match="strings to strings"):
        calculation_key(_input(), context)


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
        _key(_input(keywords=keywords))


def test_refuses_non_finite_coordinates():
    geometry = np.array(WATER["geometry"])
    geometry[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite coordinates"):
        _key(_input(structure={**WATER, "geometry": geometry}))

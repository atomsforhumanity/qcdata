import numpy as np
import pytest

from qcdata import Structure, __version__

from .data import structures as qcdata_structures


def test_from_xyz(test_data_dir):
    xyz_str = (test_data_dir / "caffeine.xyz").read_text()
    caffeine = Structure.from_xyz(xyz_str)
    assert caffeine.symbols == qcdata_structures.caffeine.symbols
    assert np.allclose(
        caffeine.geometry, qcdata_structures.caffeine.geometry, rtol=1e-9
    )
    assert caffeine.multiplicity == qcdata_structures.caffeine.multiplicity
    assert caffeine.charge == qcdata_structures.caffeine.charge
    assert caffeine.identifiers.name == "caffeine"


def test_from_xyz_non_qcdata_comments(test_data_dir):
    xyz_str = (test_data_dir / "caffeine.xyz").read_text()
    caffeine = Structure.from_xyz(xyz_str)
    assert caffeine.multiplicity == qcdata_structures.caffeine.multiplicity
    assert caffeine.charge == qcdata_structures.caffeine.charge
    assert caffeine.extras["xyz_comments"] == [
        "IUPAC",
        "1,3,7-trimethylpurine-2,6-dione",
    ]


def test_to_file_xyz(test_data_dir, tmp_path):
    caffeine = Structure.open(test_data_dir / "caffeine.xyz")
    caffeine.save(tmp_path / "caffeine_copy.xyz")
    caffeine_copy = Structure.open(tmp_path / "caffeine_copy.xyz")
    assert caffeine_copy.symbols == caffeine.symbols
    assert np.allclose(
        caffeine.geometry, qcdata_structures.caffeine.geometry, rtol=1e-9
    )
    assert caffeine_copy.multiplicity == caffeine.multiplicity
    assert caffeine_copy.charge == caffeine.charge

    # single \n at end of file
    text = (tmp_path / "caffeine_copy.xyz").read_text()
    assert text.endswith("\n")
    assert not text.endswith("\n\n")


def test_to_xyz_comments(test_data_dir):
    caffeine = Structure.open(test_data_dir / "caffeine.xyz")
    xyz_str = caffeine.to_xyz()
    comments = xyz_str.split("\n")[1].split()
    assert "IUPAC" in comments
    assert "1,3,7-trimethylpurine-2,6-dione" in comments
    assert "qcdata__identifiers_name=caffeine" in comments
    assert f"qcdata_version={__version__}" in comments
    old_xyz = xyz_str.replace(f"qcdata_version={__version__}", "qcdata_version=0.0.0")
    reopened = Structure.from_xyz(old_xyz)
    assert f"qcdata_version={__version__}" in reopened.to_xyz().splitlines()[1]
    assert reopened.extras == caffeine.extras


def test_to_from_file_json(test_data_dir, tmp_path):
    caffeine = Structure.open(test_data_dir / "caffeine.xyz")
    caffeine.save(tmp_path / "caffeine_copy.json")
    caffeine_copy = Structure.open(tmp_path / "caffeine_copy.json")
    assert caffeine_copy.symbols == caffeine.symbols
    for i in range(len(caffeine.geometry)):
        assert caffeine_copy.geometry[i] == pytest.approx(
            caffeine.geometry[i], rel=1e-9
        )
    assert caffeine_copy.multiplicity == caffeine.multiplicity
    assert caffeine_copy.charge == caffeine.charge


def test_adjacency_matrix(water):
    expected = np.array([[0, 1, 1], [1, 0, 0], [1, 0, 0]])
    assert np.array_equal(expected, water.adjacency_matrix)


def test_adjacency_dict(water):
    expected = {0: [1, 2], 1: [0], 2: [0]}
    assert expected == water.adjacency_dict


def test_structure_model_dump_connectivity(water):
    # Test that connectivity is a list of lists of floats
    # Must cast all to the same type as toml cannot handle mixed types
    dict = water.model_dump()
    for bond in dict["connectivity"]:
        assert isinstance(bond, list)
        for val in bond:
            assert isinstance(val, float)


def test_structure_connectivity_duplicates_raise():
    with pytest.raises(ValueError):
        Structure(
            symbols=["H", "O", "H"],
            geometry=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.0], [0.0, 1.0, 0.0]],
            connectivity=[(0, 1, 1.0), (1, 0, 2.0)],
        )


def test_structure_capitalize():
    structure = Structure(symbols=["NA"], geometry=[[0, 0, 0]])
    assert structure.symbols == ["Na"]
    structure = Structure(symbols=["h"], geometry=[[0, 0, 0]])
    assert structure.symbols == ["H"]


def test_atomic_symbols():
    structure = Structure(symbols=["Na", "Cl"], geometry=[[0, 0, 0], [1, 1, 1]])
    assert structure.atomic_numbers == [11, 17]


def test_identifiers_and_ids_shortcut():
    struct = Structure(
        symbols=["H"], geometry=[[0, 0, 0]], identifiers={"name": "fake"}
    )
    assert struct.identifiers.name == "fake"
    assert struct.ids is struct.identifiers


@pytest.mark.parametrize(
    "xyzstring, charge, multiplicity, expected_charge, expected_multiplicity, raises",
    [
        (
            """2
            
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            """,
            1,
            None,
            1,
            1,
            False,
        ),
        (
            """2
            
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            """,
            None,
            2,
            0,
            2,
            False,
        ),
        (
            """2
            
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            """,
            3,
            4,
            3,
            4,
            False,
        ),
        (
            """2
            qcdata_charge=-3
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            """,
            None,
            2,
            -3,
            2,
            False,
        ),
        (
            """2
            qcdata_charge=-3
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            """,
            1,
            None,
            None,
            None,
            True,
        ),
        (
            """2
            qcdata_multiplicity=4
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            """,
            1,
            None,
            1,
            4,
            False,
        ),
        (
            """2
            qcdata_multiplicity=4
            H 0.0 0.0 0.0
            H 0.0 0.0 1.0
            """,
            None,
            1,
            None,
            None,
            True,
        ),
    ],
)
def test_passing_charge_and_multiplicity_to_open(
    tmp_path,
    xyzstring,
    charge,
    multiplicity,
    expected_charge,
    expected_multiplicity,
    raises,
):
    filepath = tmp_path / "test.xyz"
    filepath.write_text(xyzstring)

    if raises:
        with pytest.raises(ValueError):
            struct = Structure.open(filepath, charge=charge, multiplicity=multiplicity)
    else:
        struct = Structure.open(filepath, charge=charge, multiplicity=multiplicity)
        assert struct.charge == expected_charge
        assert struct.multiplicity == expected_multiplicity


def test_no_charge_multiplicity_to_non_xyz_files():
    with pytest.raises(ValueError):
        Structure.open("file.json", charge=1, multiplicity=2)


def test_multi_xyz(test_data_dir):
    xyz_str = (test_data_dir / "multi.xyz").read_text()
    structures = Structure.from_xyz_multi(xyz_str)
    assert len(structures) == 2

    # Capturing comments correctly
    assert structures[0].identifiers.name == "caffeine"
    assert structures[0].charge == -1
    assert structures[1].extras[Structure._xyz_comment_key] == ["-107.04437987"]

    # Add space between structures
    xyz_lines = xyz_str.split("\n")
    xyz_lines.insert(26, "\n")
    xyz_str_with_space = "\n".join(xyz_lines)
    structures_space = Structure.from_xyz_multi(xyz_str_with_space)
    assert len(structures_space) == 2
    assert structures_space[0].identifiers.name == "caffeine"
    assert structures_space[0].charge == -1
    assert structures_space[1].extras[Structure._xyz_comment_key] == ["-107.04437987"]

    assert structures[0] == structures_space[0]
    assert structures[1] == structures_space[1]

    # Make sure it works on a single structure
    caffeine = Structure.open(test_data_dir / "caffeine.xyz")
    assert caffeine.symbols == qcdata_structures.caffeine.symbols
    assert np.allclose(
        caffeine.geometry, qcdata_structures.caffeine.geometry, rtol=1e-9
    )
    assert caffeine.multiplicity == qcdata_structures.caffeine.multiplicity
    assert caffeine.charge == qcdata_structures.caffeine.charge
    assert caffeine.identifiers.name == "caffeine"


def test_distance():
    struct = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, 1]])
    assert struct.distance(0, 1) == 1.0

    struct = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 0, -1]])
    assert struct.distance(0, 1) == 1.0

    struct = Structure(symbols=["H", "H"], geometry=[[0, 0, 0], [0, 1.4, -1.3]])

    assert struct.distance(0, 1) == pytest.approx(1.91049731, abs=1e-8)


def test_connectivity_reordering():
    struct = Structure(
        symbols=["O", "H", "H"],
        geometry=[[0, 0, 0], [0, 1, 0], [0, 0, 1]],
        connectivity=[(2, 0, 1.0), (1, 0, 1.0)],
    )
    assert struct.connectivity == [(0, 1, 1.0), (0, 2, 1.0)]

import copy

import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import ConformerSearchData


def test_energies_size(water):
    """Test that the number of energies matches the number of conformers"""
    w1 = water
    w2 = copy.deepcopy(water)

    # No energies is fine
    ConformerSearchData(
        provenance={"program": "qcdata-test-suite"},
        conformers=[w1, w2],
    )

    # No energies is fine
    ConformerSearchData(
        provenance={"program": "qcdata-test-suite"},
        rotamers=[w1, w2],
    )

    with pytest.raises(ValidationError):
        ConformerSearchData(
            provenance={"program": "qcdata-test-suite"},
            conformers=[w1, w2],
            conformer_energies=[-1.0],
        )

    with pytest.raises(ValidationError):
        ConformerSearchData(
            provenance={"program": "qcdata-test-suite"},
            rotamers=[w1, w2],
            rotamer_energies=[-1.0, -2.0, -3.0],
        )


def test_relative_energies(water):
    w1 = water
    w2 = copy.deepcopy(water)
    w3 = copy.deepcopy(water)

    # No energies returns empty array
    csr = ConformerSearchData(
        provenance={"program": "qcdata-test-suite"},
        conformers=[w1, w2, w3],
    )

    np.testing.assert_array_equal(csr.conformer_energies_relative, np.array([]))

    # Energies are relative to the lowest energy
    csr = ConformerSearchData(
        provenance={"program": "qcdata-test-suite"},
        conformers=[w1, w2, w3],
        conformer_energies=[-1.0, -2.0, -3.0],
    )
    np.testing.assert_array_equal(
        csr.conformer_energies_relative, np.array([0.0, 1.0, 2.0])
    )

    # No energies returns empty array
    csr = ConformerSearchData(
        provenance={"program": "qcdata-test-suite"},
        rotamers=[w1, w2, w3],
    )

    np.testing.assert_array_equal(csr.conformer_energies_relative, np.array([]))

    # Energies are relative to the lowest energy
    csr = ConformerSearchData(
        provenance={"program": "qcdata-test-suite"},
        rotamers=[w1, w2, w3],
        rotamer_energies=[-1.0, -2.0, -3.0],
    )
    np.testing.assert_array_equal(
        csr.rotamer_energies_relative, np.array([0.0, 1.0, 2.0])
    )


def test_conformer_search_results_sorting(water):
    w1 = water
    w2 = copy.deepcopy(water)
    w3 = copy.deepcopy(water)

    csr = ConformerSearchData(
        provenance={"program": "qcdata-test-suite"},
        conformers=[w1, w2, w3],
        conformer_energies=[-1.0, -2.0, -3.0],
    )

    assert csr.conformers[0] is w3
    assert csr.conformers[1] is w2
    assert csr.conformers[2] is w1
    assert csr.conformer_energies.tolist() == [-3.0, -2.0, -1.0]

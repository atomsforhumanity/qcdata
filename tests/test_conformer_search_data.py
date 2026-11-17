import numpy as np
import pytest
from pydantic import ValidationError

from qcdata import ConformerSearchData, Frame, Provenance, SinglePointData

PROVENANCE = Provenance(program="qcdata-test-suite")


def _frames(structure, energies):
    return [
        Frame(
            success=True,
            structure=structure,
            results=SinglePointData(provenance=PROVENANCE, energy=energy),
        )
        for energy in energies
    ]


def test_conformers_are_frames_not_bare_structures(water):
    """The form before frames (structures plus an energies array) is not accepted."""
    with pytest.raises(ValidationError):
        ConformerSearchData(provenance=PROVENANCE, conformers=[water, water])
    with pytest.raises(ValidationError):
        ConformerSearchData(
            provenance=PROVENANCE,
            conformers=_frames(water, [-1.0, -2.0]),
            conformer_energies=[-1.0, -2.0],
        )


def test_relative_energies(water):
    # Unknown energies are nan, one per conformer
    csr = ConformerSearchData(
        provenance=PROVENANCE, conformers=_frames(water, [None, None, None])
    )
    assert np.isnan(csr.conformer_energies_relative).tolist() == [True] * 3

    # Energies are relative to the lowest energy
    csr = ConformerSearchData(
        provenance=PROVENANCE, conformers=_frames(water, [-1.0, -2.0, -3.0])
    )
    np.testing.assert_array_equal(
        csr.conformer_energies_relative, np.array([0.0, 1.0, 2.0])
    )

    # No conformers, no conformer energies; unknown rotamer energies are nan
    csr = ConformerSearchData(
        provenance=PROVENANCE, rotamers=_frames(water, [None, None, None])
    )
    np.testing.assert_array_equal(csr.conformer_energies_relative, np.array([]))
    assert np.isnan(csr.rotamer_energies_relative).tolist() == [True] * 3

    csr = ConformerSearchData(
        provenance=PROVENANCE, rotamers=_frames(water, [-1.0, -2.0, -3.0])
    )
    np.testing.assert_array_equal(
        csr.rotamer_energies_relative, np.array([0.0, 1.0, 2.0])
    )


def test_conformer_search_results_sorting(water):
    frames = _frames(water, [-1.0, -2.0, -3.0])
    csr = ConformerSearchData(provenance=PROVENANCE, conformers=frames)
    assert [f.results.energy for f in csr.conformers] == [-3.0, -2.0, -1.0]
    assert csr.conformer_energies.tolist() == [-3.0, -2.0, -1.0]
    assert csr.structures == [water, water, water]

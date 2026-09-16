"""Test tools enabling compatibility with QCElemental"""

import copy

import numpy as np
import pytest
from qcelemental.models import AtomicInput, AtomicResult

from qcdata import ProgramInput
from qcdata.qcel import from_qcel_output_results, to_qcel_input


def test_input_to_qcel_requires_model(water):
    inp = ProgramInput(program="geometric", calctype="optimization", structure=water)
    with pytest.raises(ValueError, match="AtomicInput conversion requires.*model"):
        to_qcel_input(inp)


@pytest.mark.parametrize("provenance", [None, {}, {"version": "1.0"}, {"creator": ""}])
def test_qcel_result_requires_producer(provenance):
    with pytest.raises(ValueError, match="requires provenance.creator"):
        from_qcel_output_results({"provenance": provenance})


def test_qcel_result_missing_provenance():
    with pytest.raises(ValueError, match="requires provenance.creator"):
        from_qcel_output_results({})


def test_qcel_result_unknown_version():
    result = from_qcel_output_results(
        {
            "provenance": {"creator": "terachem"},
            "properties": {},
            "driver": "energy",
            "return_result": -1.0,
            "wavefunction": None,
        }
    )
    assert result.provenance.program == "terachem"
    assert result.provenance.program_version is None
    assert result.extras == {}


def test_input_to_qcel(prog_input_factory):
    """Test that input converts to qcel input"""
    pi_energy = prog_input_factory("energy")
    qcel_input_dict = to_qcel_input(pi_energy)
    qcel_atomic_input = AtomicInput(**qcel_input_dict)

    assert qcel_atomic_input.molecule.symbols.tolist() == pi_energy.structure.symbols
    assert np.array_equal(
        qcel_atomic_input.molecule.geometry, pi_energy.structure.geometry
    )
    assert qcel_atomic_input.molecule.molecular_charge == pi_energy.structure.charge
    assert qcel_atomic_input.model.method == pi_energy.model.method
    assert qcel_atomic_input.model.basis == pi_energy.model.basis
    assert qcel_atomic_input.driver == "energy"
    assert qcel_atomic_input.keywords == pi_energy.keywords
    assert qcel_atomic_input.extras == pi_energy.extras


def test_input_to_output_from_qcel_output_with_wfn(prog_input_factory):
    """Test that input converts to qcel input"""
    pi_energy = prog_input_factory("energy")

    # Check that the input converts to a qcel input
    qcel_atomic_input = AtomicInput(**to_qcel_input(pi_energy))

    qcel_atomic_output = AtomicResult(
        **{
            **qcel_atomic_input.dict(),
            **{
                "protocols": {"wavefunction": "all"},
                "properties": {
                    "nuclear_repulsion_energy": 1.0,
                    "return_energy": 1.0,
                    "calcinfo_natom": 3,
                },
                "return_result": 1.0,
                "success": True,
                "provenance": {"creator": "psi4", "version": "1.9"},
                "wavefunction": {
                    "scf_eigenvalues_a": [1, 2, 3, 4, 5, 6],
                    "restricted": True,
                    "basis": {
                        "name": "6-31g",
                        "center_data": {},
                        "atom_map": [],
                    },
                },
            },
        },
    )
    results = from_qcel_output_results(qcel_atomic_output.dict())

    assert results.provenance.program == "psi4"
    assert results.provenance.program_version == "1.9"
    assert results.energy == qcel_atomic_output.return_result
    assert (
        results.nuclear_repulsion_energy
        == qcel_atomic_output.properties.nuclear_repulsion_energy
    )
    assert results.calcinfo_natoms == qcel_atomic_output.properties.calcinfo_natom


def test_input_to_output_from_qcel_output_no_wfn(prog_input_factory):
    """Test that input converts to qcel input"""
    calc_input_energy = prog_input_factory("energy")
    qcel_atomic_input = AtomicInput(**to_qcel_input(calc_input_energy))

    qcel_atomic_output = AtomicResult(
        **{
            **qcel_atomic_input.dict(),
            **{
                "properties": {
                    "nuclear_repulsion_energy": 1.0,
                    "return_energy": 1.0,
                    "calcinfo_natom": 3,
                },
                "return_result": 1.0,
                "success": True,
                "provenance": {"creator": "psi4", "version": "1.9"},
            },
        },
    )
    results = from_qcel_output_results(qcel_atomic_output.dict())

    assert results.provenance.program == "psi4"
    assert results.provenance.program_version == "1.9"
    assert results.energy == qcel_atomic_output.return_result
    assert (
        results.nuclear_repulsion_energy
        == qcel_atomic_output.properties.nuclear_repulsion_energy
    )
    assert results.calcinfo_natoms == qcel_atomic_output.properties.calcinfo_natom


def test_qcel_conversion_preserves_source_metadata():
    source = {
        "provenance": {"creator": "orca", "version": "6.0", "routine": "energy"},
        "properties": {},
        "driver": "energy",
        "return_result": -1.0,
        "extras": {"source_id": "record-1", "nested": {"values": [1, 2]}},
    }
    original = copy.deepcopy(source)
    data = from_qcel_output_results(source)
    assert data.extras == source["extras"]
    assert data.provenance.extras == {"routine": "energy"}
    assert source == original
    data.extras["nested"]["values"].append(3)
    assert source == original

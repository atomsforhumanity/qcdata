"""Compatibility layer for QCElemental (QCSchema)."""

from copy import deepcopy
from typing import Any

from qcdata import ProgramInput, Provenance, SinglePointData, Wavefunction


def to_qcel_input(prog_input: ProgramInput) -> dict[str, Any]:
    """Return the QCElemental v1 input schema representation of the input
    (AtomicInput dict).

    Args:
        prog_input: The input object to convert.

    Returns:
        The QCElemental v1 dict representation of an AtomicInput object.

    Raises:
        ValueError: If the program specification has no model.
    """
    if prog_input.model is None:
        raise ValueError("QCSchema AtomicInput conversion requires a non-None model.")

    return {
        "molecule": {
            "symbols": prog_input.structure.symbols,
            "geometry": prog_input.structure.geometry,
            "molecular_charge": prog_input.structure.charge,
            "molecular_multiplicity": prog_input.structure.multiplicity,
            "connectivity": prog_input.structure.connectivity or None,
            # Insane defaults, but they are the defaults in qcelemental
            # Setting to True instead so qcel doesn't rotate or translate the molecule
            # https://github.com/MolSSI/QCElemental/blob/8e5a8cff52a6438ff9d6c1c6bbf1aeb4f02f12e1/qcelemental/models/molecule.py#L262-L281  # noqa: E501
            "fix_com": True,
            "fix_orientation": True,
            "identifiers": prog_input.structure.identifiers.model_dump(
                exclude={
                    "name_IUPAC",
                    "name",
                    "extras",
                    "canonical_smiles_program",
                    "qcdata_version",
                }
            ),  # not on qcel model
        },
        "driver": prog_input.calctype,
        "model": prog_input.model.model_dump(exclude={"extras", "qcdata_version"}),
        "keywords": prog_input.keywords,
        "extras": prog_input.extras,
    }


def from_qcel_output_results(
    qcel_output: dict[str, Any],
) -> SinglePointData:
    """Create SinglePointData from a QCElemental v1 AtomicResult dictionary.

    Args:
        qcel_output: The QCElemental v1 output schema representation of the output.
            Must identify the producing program in provenance.creator.

    Raises:
        ValueError: If producer identity is unavailable.
    """
    provenance = qcel_output.get("provenance")
    if not isinstance(provenance, dict) or not provenance.get("creator"):
        raise ValueError("QCSchema result conversion requires provenance.creator.")

    # Collect values from keys that exist in qcdata
    qcdata_to_qcel = {
        "calcinfo_natoms": "calcinfo_natom",
        "energy": "return_energy",
        "gradient": "return_gradient",
        "hessian": "return_hessian",
    }
    results = {}
    for key in SinglePointData.model_fields:
        if key in qcdata_to_qcel:
            qcel_key = qcdata_to_qcel[key]
        else:
            qcel_key = key

        value = qcel_output["properties"].get(qcel_key)
        if value is not None:
            results[key] = value

    # Override with return_result as qcel may not have save the key value to .properties
    results[qcel_output["driver"]] = qcel_output["return_result"]

    if qcel_output.get("wavefunction"):
        results["wavefunction"] = {
            key: value
            for key, value in qcel_output["wavefunction"].items()
            if key in Wavefunction.__annotations__
        }

    results["provenance"] = Provenance(
        program=provenance["creator"],
        program_version=provenance.get("version"),
        extras=deepcopy(
            {
                key: value
                for key, value in provenance.items()
                if key not in {"creator", "version"}
            }
        ),
    )
    results["extras"] = deepcopy(qcel_output.get("extras") or {})
    return SinglePointData(**results)

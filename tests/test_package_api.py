import qcdata
from qcdata import *  # noqa: F403


def test_public_exports():
    for name in [
        "Identifiers",
        "Data",
        "DataType",
        "StructuredData",
        "StructuredDataType",
        "Inputs",
        "InputType",
        "ProgramSpec",
        "ProgramInput",
        "ProgramOutput",
    ]:
        assert globals()[name] is getattr(qcdata, name)

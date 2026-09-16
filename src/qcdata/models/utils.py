"""Utility functions for the models module."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qcdata import Structure


def to_multi_xyz(structures: Iterable["Structure"]) -> str:
    """Create a multi-structure XYZ string from a list of structures.

    Args:
        structures: An Iterable of Structure objects.

    Returns:
        The multi-structure XYZ string.
    """
    return "".join(struct.to_xyz() for struct in structures)

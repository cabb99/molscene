"""File-format parsers and the :class:`FormatRegistry`.

Parsers are Scene-agnostic: readers return ``(atoms_DataFrame, meta)`` and writers
take a scene used as a DataFrame.  Importing the format modules here registers them.
"""

from .registry import FormatRegistry
from . import pdb, cif, awsem_gro  # noqa: F401  (import triggers registration)

read = FormatRegistry.read
write = FormatRegistry.write

__all__ = ["FormatRegistry", "read", "write", "pdb", "cif", "awsem_gro"]

"""In-memory object converters and the :class:`BackendRegistry`.

Backends are Scene-agnostic: ``from_object(obj)`` returns ``(atoms_DataFrame,
meta)`` and ``to_object(scene)`` returns the foreign object.  Importing the backend
modules here registers them; the optional dependency is imported lazily only when a
conversion actually runs.
"""

from .registry import BackendRegistry
from . import prody_backend, mdtraj_backend, pdbfixer_backend  # noqa: F401

from_object = BackendRegistry.from_object
to_object = BackendRegistry.to_object

__all__ = ["BackendRegistry", "from_object", "to_object",
           "prody_backend", "mdtraj_backend", "pdbfixer_backend"]

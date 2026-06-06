"""Registry of in-memory object converters (ProDy ``AtomGroup``, mdtraj
``Trajectory``, ``PDBFixer`` ...).

Backends are **Scene-agnostic**: ``from_object(obj)`` returns ``(atoms_DataFrame,
meta)`` and ``to_object(scene)`` returns the foreign object.  Each backend also
provides ``matches(obj)`` for type dispatch (duck-typed by module/class name so it
never needs the optional dependency just to decide).  The actual conversion imports
the optional dependency lazily and raises a clear "install molscene[<extra>]" error
if it is missing.
"""


class BackendRegistry:
    backends: dict = {}     # name -> backend class (matches / from_object / to_object)

    @classmethod
    def register(cls, name):
        def decorate(backend):
            cls.backends[name] = backend
            return backend
        return decorate

    @classmethod
    def get(cls, name):
        return cls.backends.get(name)

    @classmethod
    def from_object(cls, obj):
        """Convert a recognized foreign object to ``(atoms_DataFrame, meta)``."""
        for backend in cls.backends.values():
            if backend.matches(obj):
                return backend.from_object(obj)
        t = type(obj)
        raise TypeError(
            f"No registered backend converts {t.__module__}.{t.__name__} to a Scene; "
            f"available backends: {sorted(cls.backends)}")

    @classmethod
    def to_object(cls, scene, name):
        try:
            backend = cls.backends[name]
        except KeyError:
            raise ValueError(
                f"No backend named {name!r}; available: {sorted(cls.backends)}")
        return backend.to_object(scene)

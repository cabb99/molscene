"""Registry mapping file extensions / format names to reader and writer functions.

Parsers are **Scene-agnostic**: a reader returns ``(atoms, meta)`` where ``atoms``
is a plain :class:`pandas.DataFrame` and ``meta`` is a dict (it may carry
``source_file``/``source_format`` and, for multi-frame formats,
``coordinate_frames``); a writer takes a ``scene`` (used purely as a DataFrame).
:class:`molscene.Scene` imports this package and wraps the results — nothing here
imports ``Scene``.
"""

import os


class FormatRegistry:
    """Class-level registry of file-format readers/writers keyed by extension."""

    readers: dict = {}          # ".pdb" -> reader(path, **kw) -> (DataFrame, meta)
    writers: dict = {}          # ".pdb" -> writer(scene, path, **kw)
    reader_by_name: dict = {}   # "pdb"  -> reader
    writer_by_name: dict = {}   # "pdb"  -> writer

    @classmethod
    def register_reader(cls, *extensions, name=None):
        def decorate(fn):
            for ext in extensions:
                cls.readers[ext.lower()] = fn
            if name:
                cls.reader_by_name[name] = fn
            return fn
        return decorate

    @classmethod
    def register_writer(cls, *extensions, name=None):
        def decorate(fn):
            for ext in extensions:
                cls.writers[ext.lower()] = fn
            if name:
                cls.writer_by_name[name] = fn
            return fn
        return decorate

    @staticmethod
    def _extension(path):
        return os.path.splitext(str(path))[1].lower()

    @classmethod
    def _lookup(cls, path, format, by_name, by_ext, kind):
        if format is not None:
            try:
                return by_name[format]
            except KeyError:
                raise ValueError(
                    f"No {kind} registered for format {format!r}; "
                    f"known formats: {sorted(by_name)}")
        ext = cls._extension(path)
        try:
            return by_ext[ext]
        except KeyError:
            raise ValueError(
                f"No {kind} registered for extension {ext!r}; "
                f"known extensions: {sorted(by_ext)} (or pass format=...).")

    @classmethod
    def read(cls, path, format=None, **kw):
        """Return ``(atoms_DataFrame, meta)`` for ``path``."""
        reader = cls._lookup(path, format, cls.reader_by_name, cls.readers, "reader")
        return reader(path, **kw)

    @classmethod
    def write(cls, scene, path, format=None, **kw):
        writer = cls._lookup(path, format, cls.writer_by_name, cls.writers, "writer")
        return writer(scene, path, **kw)

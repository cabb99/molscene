"""Sparse COO utilities for distance maps and contact masks.

All functions operate on plain ``numpy`` arrays ``(row, col, data, shape)`` so a
purely functional workflow never needs to instantiate a class; :class:`SparseMatrix`
is a thin wrapper that delegates to them.

Two conventions matter for correctness:

* **Absent is not zero.**  A pair that is not stored is *farther than the cutoff*,
  not at distance zero, so :func:`to_dense` fills unstored entries with ``inf`` and
  always writes ``0`` on the diagonal.  Callers wanting a boolean contact map pass
  ``fill=0``.
* **Mask mode is** ``data=None``, rather than a magic value in ``data``.  Operations
  that need distances raise on a mask instead of reading meaningless numbers.

``shape`` is the ``(n_rows, n_cols)`` pair of the map, so rectangular cross-maps
between two different selections are representable alongside square self-maps.
"""

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "SparseMatrix",
    "lookup",
    "filter_range",
    "to_dense",
]


def _as_shape(shape) -> Tuple[int, int]:
    """Normalize ``shape`` to an ``(n_rows, n_cols)`` tuple of ints.

    A scalar is read as a square map, matching the sequence-length convention used
    by callers that only ever build self-maps.
    """
    if np.isscalar(shape):
        return int(shape), int(shape)
    n_rows, n_cols = shape
    return int(n_rows), int(n_cols)


def lookup(row, col, data, shape, query_i, query_j) -> np.ndarray:
    """Read stored values at ``(query_i, query_j)`` without building the dense map.

    Equivalent to ``to_dense(...)[query_i, query_j]``.  Every queried pair must
    already be stored; querying an absent pair returns an arbitrary neighbouring
    value rather than ``inf``.
    """
    if data is None:
        raise ValueError("cannot look up distances on a mask (data is None)")
    n_rows, n_cols = _as_shape(shape)
    row = np.asarray(row, dtype=np.int64)
    col = np.asarray(col, dtype=np.int64)
    data = np.asarray(data, dtype=float)

    stored_key = row * n_cols + col
    order = np.argsort(stored_key)
    query_key = (np.asarray(query_i, dtype=np.int64) * n_cols
                 + np.asarray(query_j, dtype=np.int64))
    position = np.searchsorted(stored_key, query_key, sorter=order)
    return data[order[position]]


def filter_range(row, col, data, shape, init, fin):
    """Restrict a square map to indices ``[init, fin)`` and re-index them to zero."""
    n_rows, n_cols = _as_shape(shape)
    row = np.asarray(row, dtype=np.intp)
    col = np.asarray(col, dtype=np.intp)
    keep = (row >= init) & (row < fin) & (col >= init) & (col < fin)
    new_data = None if data is None else np.asarray(data, dtype=float)[keep]
    size = int(fin) - int(init)
    return row[keep] - init, col[keep] - init, new_data, (size, size)


def to_dense(row, col, data, shape, fill=np.inf) -> np.ndarray:
    """Rebuild the dense map, filling unstored entries with ``fill``.

    The diagonal of a square map is always ``0`` — an atom's distance to itself is
    known even though the pair is never stored.  ``data=None`` (mask mode) writes
    ``1.0`` at every stored pair.
    """
    n_rows, n_cols = _as_shape(shape)
    dense = np.full((n_rows, n_cols), fill, dtype=float)
    if n_rows == n_cols:
        np.fill_diagonal(dense, 0.0)
    row = np.asarray(row, dtype=np.intp)
    col = np.asarray(col, dtype=np.intp)
    dense[row, col] = 1.0 if data is None else np.asarray(data, dtype=float)
    return dense


class SparseMatrix:
    """COO distance map or contact mask, wrapping the functions in this module.

    Unpacks as ``row, col, data, shape`` so it is interchangeable with the plain
    tuple the kernels produce::

        row, col, data, shape = scene.distance_map(sparse=True, cutoff=8.0)

    ``data=None`` marks a mask, where only the stored positions carry meaning.
    """

    __slots__ = ("row", "col", "data", "shape")

    def __init__(self, row, col, data=None, shape=None):
        self.row = np.asarray(row, dtype=np.intp)
        self.col = np.asarray(col, dtype=np.intp)
        self.data = None if data is None else np.asarray(data, dtype=float)
        if shape is None:
            if len(self.row) == 0:
                shape = (0, 0)
            else:
                shape = (int(self.row.max()) + 1, int(self.col.max()) + 1)
        self.shape = _as_shape(shape)

    @property
    def is_mask(self) -> bool:
        return self.data is None

    def lookup(self, query_i, query_j) -> np.ndarray:
        """Read stored values at ``(query_i, query_j)``."""
        return lookup(self.row, self.col, self.data, self.shape, query_i, query_j)

    def filter_range(self, init, fin) -> "SparseMatrix":
        """Restrict to indices ``[init, fin)``, re-indexed to zero."""
        return SparseMatrix(*filter_range(self.row, self.col, self.data,
                                          self.shape, init, fin))

    def filter_cutoff(self, cutoff: float) -> "SparseMatrix":
        """Drop stored pairs farther than ``cutoff``.

        Building one map at the widest radius and narrowing it is cheaper than
        re-running the neighbour search for every cutoff of interest.
        """
        if self.data is None:
            raise ValueError("cannot filter a mask by distance (data is None)")
        keep = self.data <= cutoff
        return SparseMatrix(self.row[keep], self.col[keep], self.data[keep], self.shape)

    def to_mask(self) -> "SparseMatrix":
        """Drop the distances, keeping the stored positions."""
        return SparseMatrix(self.row, self.col, None, self.shape)

    def to_dense(self, fill=np.inf) -> np.ndarray:
        """Rebuild the dense map, filling unstored entries with ``fill``."""
        return to_dense(self.row, self.col, self.data, self.shape, fill=fill)

    def __iter__(self):
        return iter((self.row, self.col, self.data, self.shape))

    def __len__(self) -> int:
        return int(self.row.size)

    def __repr__(self) -> str:
        kind = "mask" if self.data is None else "distance"
        return f"SparseMatrix({kind}, nnz={len(self)}, shape={self.shape})"

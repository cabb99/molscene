"""Geometry kernels for distance maps and virtual Cβ reconstruction.

Pure ``numpy``/``scipy`` helpers operating on coordinate and label arrays, so this
module never imports :class:`~molscene.Scene.Scene`.  ``Scene`` wraps these kernels
the same way it wraps the ``parsers`` and ``backends`` layers: it resolves
selections, extracts coordinates, and re-wraps the result.

The sparse kernels return ``(row, col, data, shape)`` COO arrays holding only pairs
within the cutoff; :mod:`molscene.sparse` defines what an absent pair means and how
to rebuild the dense map from them.
"""

import numpy as np
import pandas
from scipy.spatial import cKDTree, distance

from . import geometry

BACKBONE_ATOMS = frozenset({"N", "CA", "C"})
"""Atoms a residue must carry to have a reconstructable backbone frame."""

CB_OFFSET = (-0.531020, -1.206181, -0.789162)
"""Cβ position in the residue backbone frame, in Ångström."""


def cb_from_backbone(N, CA, C, offset=CB_OFFSET):
    """Reconstruct Cβ coordinates from backbone N, CA, C atoms.

    Uses the standard ideal local geometry so that residues lacking an explicit
    Cβ (e.g. glycine) still receive one.  All inputs are ``(R, 3)`` arrays;
    returns an ``(R, 3)`` array of Cβ positions.

    ``offset`` is the coefficient triple in the backbone frame — either one triple
    for every residue or an ``(R, 3)`` array of per-residue coefficients, which is
    how per-amino-acid Cβ tables are expressed.  The frame is orthonormal, so the
    result is correct for coordinates in any length unit.
    """
    basis, origin = geometry.local_frame(CA, N, C)
    return geometry.place_from_frame(basis, origin, offset)


def dense_atom_map(coordsA, coordsB, self_map):
    """Dense ``(nA, nB)`` matrix of all pairwise distances."""
    if self_map:
        if len(coordsA) == 0:
            return np.zeros((0, 0))
        return distance.squareform(distance.pdist(coordsA))
    return distance.cdist(coordsA, coordsB)


def sparse_atom_map(coordsA, coordsB, cutoff, self_map):
    """COO arrays of every pair within ``cutoff``.

    Self-maps store each unordered pair in both directions and never store the
    diagonal; cross-maps store whatever the neighbour search finds, including
    zero-distance pairs when the two selections overlap.
    """
    shape = (len(coordsA), len(coordsB))
    if self_map:
        pairs = cKDTree(coordsA).query_pairs(cutoff, output_type="ndarray")
        i, j = pairs[:, 0], pairs[:, 1]
        d = np.linalg.norm(coordsA[i] - coordsA[j], axis=1)
        row = np.concatenate([i, j]).astype(np.intp)
        col = np.concatenate([j, i]).astype(np.intp)
        return row, col, np.concatenate([d, d]), shape
    coo = cKDTree(coordsA).sparse_distance_matrix(
        cKDTree(coordsB), cutoff, output_type="coo_matrix")
    return (coo.row.astype(np.intp), coo.col.astype(np.intp),
            coo.data.astype(float), shape)


def _group_labels(resA, resB, self_map):
    """Map the two label arrays onto contiguous group indices."""
    uA, invA = np.unique(resA, return_inverse=True)
    uB, invB = (uA, invA) if self_map else np.unique(resB, return_inverse=True)
    return uA, invA, uB, invB


def dense_residue_map(coordsA, coordsB, resA, resB, self_map, reduce):
    """Dense per-group map, reducing the atom-atom distances within each block.

    Grouping is done with two chained ``pandas`` reductions rather than a scatter
    over ``(n_atoms_A, n_atoms_B)`` index arrays: each group pair spans a full
    rectangle of the distance matrix, so reducing rows then columns is exact and
    avoids allocating two index buffers the size of the distance matrix itself.
    """
    D = dense_atom_map(coordsA, coordsB, self_map)
    uA, invA, uB, invB = _group_labels(resA, resB, self_map)
    if D.size == 0:
        return np.zeros((len(uA), len(uB)))
    by_row = pandas.DataFrame(D).groupby(invA).agg(reduce)
    return by_row.T.groupby(invB).agg(reduce).T.to_numpy()


def sparse_residue_map(coordsA, coordsB, resA, resB, cutoff, self_map, reduce):
    """COO per-group map built from the atom pairs within ``cutoff``.

    Only ``reduce='min'`` is meaningful here: a group-pair maximum or mean taken
    over the pairs inside the cutoff is not the maximum or mean of the group pair.
    :meth:`~molscene.Scene.Scene.distance_map` rejects the other reductions.
    """
    row, col, data, _ = sparse_atom_map(coordsA, coordsB, cutoff, self_map)
    uA, invA, uB, invB = _group_labels(resA, resB, self_map)
    shape = (len(uA), len(uB))

    ri, rj = invA[row], invB[col]
    if self_map:
        keep = ri != rj
        ri, rj, data = ri[keep], rj[keep], data[keep]

    if ri.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy(), np.empty(0, dtype=float), shape

    agg = (pandas.DataFrame({"i": ri, "j": rj, "d": data})
           .groupby(["i", "j"])["d"].agg(reduce).reset_index())
    return (agg["i"].to_numpy(np.intp), agg["j"].to_numpy(np.intp),
            agg["d"].to_numpy(float), shape)

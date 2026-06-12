"""Geometry kernels for distance maps and virtual Cβ reconstruction.

Pure ``numpy``/``scipy``/``pandas`` helpers operating on coordinate and label
arrays, so this module never imports :class:`~molscene.Scene.Scene`.  ``Scene``
wraps these kernels the same way it wraps the ``parsers`` and ``backends``
layers: it resolves selections, extracts coordinates, and re-wraps the result.
"""

import numpy as np
import pandas
from scipy.spatial import cKDTree, distance


def cb_from_backbone(N, CA, C):
    """Reconstruct Cβ coordinates from backbone N, CA, C atoms.

    Uses the standard ideal local geometry so that residues lacking an explicit
    Cβ (e.g. glycine) still receive one.  All inputs are ``(R, 3)`` arrays;
    returns an ``(R, 3)`` array of Cβ positions.
    """
    v_CA_C = C - CA
    v_CA_N = N - CA
    v_CA_C = v_CA_C / np.linalg.norm(v_CA_C, axis=1, keepdims=True)
    v_CA_N = v_CA_N / np.linalg.norm(v_CA_N, axis=1, keepdims=True)
    cross1 = np.cross(v_CA_C, v_CA_N)
    cross1 = cross1 / np.linalg.norm(cross1, axis=1, keepdims=True)
    cross2 = np.cross(cross1, v_CA_N)
    cross2 = cross2 / np.linalg.norm(cross2, axis=1, keepdims=True)
    return -0.531020 * v_CA_N - 1.206181 * cross1 + 0.789162 * cross2 + CA


def dense_atom_map(coordsA, coordsB, self_map):
    if self_map:
        return distance.squareform(distance.pdist(coordsA))
    return distance.cdist(coordsA, coordsB)


def sparse_atom_map(coordsA, coordsB, cutoff, self_map):
    nA, nB = len(coordsA), len(coordsB)
    if self_map:
        pairs = cKDTree(coordsA).query_pairs(cutoff, output_type="ndarray")
        if pairs.size == 0:
            empty = np.empty(0, dtype=np.intp)
            return empty, empty.copy(), np.empty(0, dtype=float), (nA, nB)
        d = np.linalg.norm(coordsA[pairs[:, 0]] - coordsA[pairs[:, 1]], axis=1)
        row = np.concatenate([pairs[:, 0], pairs[:, 1]]).astype(np.intp)
        col = np.concatenate([pairs[:, 1], pairs[:, 0]]).astype(np.intp)
        data = np.concatenate([d, d])
        return row, col, data, (nA, nB)
    coo = cKDTree(coordsA).sparse_distance_matrix(
        cKDTree(coordsB), cutoff, output_type="coo_matrix")
    return (coo.row.astype(np.intp), coo.col.astype(np.intp),
            coo.data.astype(float), (nA, nB))


def dense_residue_map(coordsA, coordsB, resA, resB, self_map, reduce):
    D = (distance.squareform(distance.pdist(coordsA)) if self_map
         else distance.cdist(coordsA, coordsB))
    uA, invA = np.unique(resA, return_inverse=True)
    uB, invB = (uA, invA) if self_map else np.unique(resB, return_inverse=True)
    ii = np.broadcast_to(invA[:, None], D.shape).ravel()
    jj = np.broadcast_to(invB[None, :], D.shape).ravel()
    vals = D.ravel()
    if reduce == "mean":
        out = np.zeros((len(uA), len(uB)))
        cnt = np.zeros_like(out)
        np.add.at(out, (ii, jj), vals)
        np.add.at(cnt, (ii, jj), 1.0)
        return out / cnt
    op = np.minimum if reduce == "min" else np.maximum
    out = np.full((len(uA), len(uB)), np.inf if reduce == "min" else -np.inf)
    op.at(out, (ii, jj), vals)
    return out


def sparse_residue_map(coordsA, coordsB, resA, resB, cutoff, self_map, reduce):
    uA, invA = np.unique(resA, return_inverse=True)
    uB, invB = (uA, invA) if self_map else np.unique(resB, return_inverse=True)
    if self_map:
        pairs = cKDTree(coordsA).query_pairs(cutoff, output_type="ndarray")
        if pairs.size:
            ia, ja = pairs[:, 0], pairs[:, 1]
            d = np.linalg.norm(coordsA[ia] - coordsA[ja], axis=1)
        else:
            ia = ja = np.empty(0, dtype=int)
            d = np.empty(0, dtype=float)
    else:
        coo = cKDTree(coordsA).sparse_distance_matrix(
            cKDTree(coordsB), cutoff, output_type="coo_matrix")
        ia, ja, d = coo.row, coo.col, coo.data
    ri, rj = invA[ia], invB[ja]
    if self_map and ri.size:
        keep = ri != rj
        ri, rj, d = ri[keep], rj[keep], d[keep]
        ri, rj = np.minimum(ri, rj), np.maximum(ri, rj)
    shape = (len(uA), len(uB))
    if ri.size == 0:
        empty = np.empty(0, dtype=np.intp)
        return empty, empty.copy(), np.empty(0, dtype=float), shape
    agg = (pandas.DataFrame({"i": ri, "j": rj, "d": d})
           .groupby(["i", "j"])["d"].agg(reduce).reset_index())
    row = agg["i"].to_numpy(np.intp)
    col = agg["j"].to_numpy(np.intp)
    data = agg["d"].to_numpy(float)
    if self_map:
        row, col, data = (np.concatenate([row, col]),
                          np.concatenate([col, row]),
                          np.concatenate([data, data]))
    return row, col, data, shape

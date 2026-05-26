#!/usr/bin/env python
"""
Build a CaMKII–actin holoenzyme model by transplanting the kinase–actin
binding pose from Qian's model onto the merged template.

Pipeline
--------
1.  Identify in Qian which two actins the kinase contacts (closest atoms).
    Build the KABT (kinase + 2 bound actins).
2.  In merged, cluster the 26 actin chains into 2 filaments (KMeans on
    per-chain centroids) and order each filament along its principal axis.
3.  Slide the KABT along each filament with stride-2 (F-actin's same-strand
    neighbour) — for every pair (c_i, c_{i+2}), fit KABT_actins onto the
    pair via Kabsch + SequenceMatching, apply that Transformation to
    KABT_kinase, save as a candidate kinase pose.
4.  Either auto-rank candidates by hub-distance + Hungarian assignment
    (default), or accept an explicit user-supplied mapping that says
    which source chain (A–L) goes to which candidate.
5.  Apply the per-chain kinase transform; ScLERP-morph the linker between
    the unchanged hub end and the moved kinase end.
6.  Write the final PDB (and optionally a multi-model morph movie).

Usage
-----
::

    # Default: Hungarian assignment, single-frame final structure.
    python scratch/CaMKII_example/script.py

    # Inspect candidates manually. Writes <prefix>.pdb (sources + candidates
    # in distinct chain IDs) and <prefix>.txt (labels you'll need to write
    # the assignment file).
    python scratch/CaMKII_example/script.py --save-candidates camkii_candidates

    # Use a manual assignment file. Two-column format
    # "source_chain candidate_id", e.g.
    #     A  0   # chain A → candidate 0
    #     B  3
    #     ...
    #     L  21
    python scratch/CaMKII_example/script.py --assignment myassignment.txt

    # Same, plus a 20-frame morph movie (multi-model PDB).
    python scratch/CaMKII_example/script.py --assignment myassignment.txt --animate 20
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

import molscene as ms
from molscene import SequenceMatching, Transformation


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HERE = os.path.dirname(os.path.abspath(__file__))
MERGED_PATH = os.path.join(HERE, "merged_CaMKII.pdb")
QIAN_PATH = os.path.join(HERE, "CaMKII_qian_template.pdb")
OUT_PATH = os.path.join(HERE, "camkii_on_actin.pdb")

KINASE_RANGE = (1, 290)
LINKER_RANGE = (291, 400)
HUB_RANGE = (401, 542)

CAMKII_CHAINS = list("ABCDEFGHIJKL")

# Hub topology — the dodecameric hub forms a hexagon with 6 vertex-pairs.
# Chains stack face-to-face within each vertex. Faces and vertices are
# listed in clockwise order so that adjacent vertices in the lists are
# physically adjacent on the hexagon (AB ↔ IJ adjacent, AB ↔ GL opposite).
HUB_FACE_FRONT = list("AJDLCK")
HUB_FACE_BACK = list("BIEGFH")
HUB_VERTEX_STACKS = [
    ("A", "B"),  # vertex 0 ─ AB
    ("I", "J"),  # vertex 1 ─ IJ (clockwise 60° from AB)
    ("D", "E"),  # vertex 2 ─ DE
    ("G", "L"),  # vertex 3 ─ GL (opposite to AB)
    ("C", "F"),  # vertex 4 ─ CF
    ("H", "K"),  # vertex 5 ─ HK
]

QIAN_KINASE = (1, 290)
QIAN_ACTIN_RANGES = [
    (1944, 2318), (2319, 2693), (2694, 3068), (3069, 3443), (3444, 3818),
]

KABT_CHAIN_A = "1"
KABT_CHAIN_B = "2"

# Single-character chain IDs used to label candidate poses in the candidates
# PDB so they are easy to pick out in a viewer.
CANDIDATE_CHAIN_IDS = list("0123456789") + list("mnopqrstuvwxyz")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def chain_centroid(scene: ms.Scene, chain: str) -> np.ndarray:
    return scene[scene["chain"] == chain][["x", "y", "z"]].to_numpy().mean(axis=0)


def find_kabt_actin_pair(qian: ms.Scene, kinase_range, actin_ranges):
    """Return the two ``(lo, hi)`` actin residue-ranges from Qian whose
    atoms come closest to the kinase atoms."""
    kin = qian.select(f"chain A and resid {kinase_range[0]} to {kinase_range[1]}")
    kin_xyz = kin[["x", "y", "z"]].to_numpy()

    min_dists = []
    for lo, hi in actin_ranges:
        sub = qian.select(f"chain A and resid {lo} to {hi}")
        sub_xyz = sub[["x", "y", "z"]].to_numpy()
        diffs = kin_xyz[:, None, :] - sub_xyz[None, :, :]
        d_min = float(np.sqrt((diffs ** 2).sum(axis=2)).min())
        min_dists.append(d_min)

    order = np.argsort(min_dists)
    a, b = order[0], order[1]
    print(f"  kinase contacts actin {a} ({min_dists[a]:.2f} Å) and "
          f"actin {b} ({min_dists[b]:.2f} Å)")
    return tuple(sorted([actin_ranges[a], actin_ranges[b]]))


def cluster_into_filaments(scene: ms.Scene, chains, n_filaments=2):
    centroids = np.array([chain_centroid(scene, c) for c in chains])
    labels = KMeans(n_clusters=n_filaments, n_init=10, random_state=0).fit_predict(centroids)
    return [[c for c, l in zip(chains, labels) if l == k] for k in range(n_filaments)]


def order_along_axis(scene: ms.Scene, chains):
    centroids = np.array([chain_centroid(scene, c) for c in chains])
    centered = centroids - centroids.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    axis = vt[0]
    projections = centered @ axis
    return [c for _, c in sorted(zip(projections, chains))]


def build_kabt(qian: ms.Scene):
    """Build the kinase-actin binding template from Qian's model.

    Returns ``(kabt_kinase, kabt_actins)``; the two actin monomers are
    relabelled to chains :data:`KABT_CHAIN_A` / :data:`KABT_CHAIN_B`.
    """
    (a1_lo, a1_hi), (a2_lo, a2_hi) = find_kabt_actin_pair(
        qian, QIAN_KINASE, QIAN_ACTIN_RANGES,
    )

    kabt_kinase = qian.select(f"chain A and resid {QIAN_KINASE[0]} to {QIAN_KINASE[1]}")
    a1 = qian.select(f"chain A and resid {a1_lo} to {a1_hi}").copy()
    a2 = qian.select(f"chain A and resid {a2_lo} to {a2_hi}").copy()
    a1["chain"] = KABT_CHAIN_A
    a2["chain"] = KABT_CHAIN_B
    kabt_actins = ms.Scene.concatenate([a1, a2])
    kabt_actins["chain"] = list(a1["chain"]) + list(a2["chain"])
    return kabt_kinase, kabt_actins


def fit_kabt_to_pair(kabt_actins, ci, cj, merged_pair_ca):
    """Fit KABT actins onto the actin pair (ci, cj) using the canonical filament order.

    KABT_CHAIN_A is always mapped to ci (the earlier chain along the filament) and
    KABT_CHAIN_B to cj, so each stride-2 pair produces a kinase pose anchored to its
    own unique position and adjacent overlapping pairs cannot converge to the same pose.
    """
    kabt = kabt_actins.copy()
    kabt["chain"] = kabt["chain"].map({KABT_CHAIN_A: ci, KABT_CHAIN_B: cj})
    try:
        return kabt.select("name CA").compute_transformation(
            merged_pair_ca, match=SequenceMatching(),
        )
    except (ValueError, KeyError):
        return None


# ---------------------------------------------------------------------------
# Filament-rotation optimization
# ---------------------------------------------------------------------------

def filament_helical_axis(scene: ms.Scene, fil_chains):
    """Return ``(axis_direction_unit, axis_point)`` for the helical axis of a
    filament identified by the chains in ``fil_chains``.

    The axis is computed in pure-numpy from the atom coordinates: the
    centroid of all atoms in the filament lies on the axis (rotations
    around the axis permute the monomers, so their centred sum is
    rotation-invariant), and the direction is the first principal
    component of the centred coordinates. Equivalent to building a
    z-aligned synthetic filament and Kabsch-fitting it onto the real one
    — the rotation column corresponding to z gives the direction and the
    translation gives a point on the axis.
    """
    xyz = scene[scene["chain"].isin(fil_chains)][["x", "y", "z"]].to_numpy()
    axis_point = xyz.mean(axis=0)
    centred = xyz - axis_point
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return vt[0], axis_point


def rotation_about_axis(axis_direction, axis_point, angle_rad) -> Transformation:
    """Rotation around the line through ``axis_point`` parallel to
    ``axis_direction`` as a :class:`Transformation`."""
    from scipy.spatial.transform import Rotation as _R
    axis = np.asarray(axis_direction, dtype=float)
    axis = axis / np.linalg.norm(axis)
    R = _R.from_rotvec(angle_rad * axis).as_matrix()
    t = axis_point - R @ axis_point
    return Transformation.from_matrix(R, t)


def linker_endpoints(merged: ms.Scene, chain_transforms):
    """For each chain in ``chain_transforms``, return its
    ``(src_chain, kinase_end_placed, hub_end_fixed)`` linker endpoints.

    The kinase-end (CA of the first linker residue, ``LINKER_RANGE[0]``)
    is the ``T_kin``-placed position; the hub-end (CA of the last linker
    residue, ``LINKER_RANGE[1]``) is unchanged from merged.
    """
    kin_resid = LINKER_RANGE[0]
    hub_resid = LINKER_RANGE[1]
    endpoints = []
    for src_chain, T_kin in chain_transforms:
        m_kin = ((merged["chain"] == src_chain)
                 & (merged["resid"] == kin_resid)
                 & (merged["name"] == "CA"))
        m_hub = ((merged["chain"] == src_chain)
                 & (merged["resid"] == hub_resid)
                 & (merged["name"] == "CA"))
        if not m_kin.any() or not m_hub.any():
            raise ValueError(
                f"chain {src_chain}: missing CA at residue "
                f"{kin_resid} or {hub_resid}"
            )
        p_kin = merged.loc[m_kin, ["x", "y", "z"]].to_numpy()[0]
        p_hub = merged.loc[m_hub, ["x", "y", "z"]].to_numpy()[0]
        p_kin_placed = T_kin.apply(p_kin[None, :])[0]
        endpoints.append((src_chain, p_kin_placed, p_hub))
    return endpoints


def compute_linker_spans(merged: ms.Scene, chain_transforms,
                          chain_rotations=None):
    """Distance between the first and last linker beads, per chain.

    Parameters
    ----------
    merged : Scene
    chain_transforms : list of (src_chain, T_kin)
    chain_rotations : dict {src_chain: Transformation}, optional
        Additional per-chain rotation applied to the kinase-end bead
        (e.g. a filament rotation about the helical axis). Chains not in
        the dict have no extra rotation.
    """
    chain_rotations = chain_rotations or {}
    spans = {}
    for src_chain, p_kin_placed, p_hub in linker_endpoints(merged, chain_transforms):
        T_extra = chain_rotations.get(src_chain)
        if T_extra is not None:
            p_kin_placed = T_extra.apply(p_kin_placed[None, :])[0]
        spans[src_chain] = float(np.linalg.norm(p_kin_placed - p_hub))
    return spans


def optimize_filament_rotation(merged: ms.Scene, chain_transforms_for_fil,
                               fil_chains, scan_step_deg=1.0):
    """Find the rotation about the filament's helical axis that minimises
    the sum of squared linker spans for the chains bound to it.

    Parameters
    ----------
    merged : Scene
        Original merged template (kinases at rest position).
    chain_transforms_for_fil : list of (src_chain, T_kin)
        Only the source chains that are bound to this filament.
    fil_chains : list of str
        The filament's actin chain IDs (used to compute the helical axis).
    scan_step_deg : float
        Coarse-scan step. The minimum is refined with Brent within the
        bracket of width ``2 * scan_step_deg`` around the coarse winner.

    Returns
    -------
    dict
        With keys ``angle_deg``, ``angle_rad``, ``T_rot`` (rotation about
        the filament's axis), ``axis_dir``, ``axis_point``, ``objective``
        (sum of squared spans at the optimum), ``scan_angles_deg``,
        ``scan_objective``, ``spans_at_optimum`` (src_chain -> distance).
    """
    if not chain_transforms_for_fil:
        return None

    axis_dir, axis_point = filament_helical_axis(merged, fil_chains)

    endpoints = linker_endpoints(merged, chain_transforms_for_fil)
    p_kin = np.array([ep[1] for ep in endpoints])
    p_hub = np.array([ep[2] for ep in endpoints])

    def objective(angle_rad):
        T_rot = rotation_about_axis(axis_dir, axis_point, angle_rad)
        p_kin_rot = T_rot.apply(p_kin)
        spans = np.linalg.norm(p_kin_rot - p_hub, axis=1)
        return float(np.sum(spans ** 2))

    scan_angles_deg = np.arange(0.0, 360.0, scan_step_deg)
    scan_vals = np.array([objective(np.deg2rad(a)) for a in scan_angles_deg])
    best_idx = int(np.argmin(scan_vals))
    coarse_best_deg = float(scan_angles_deg[best_idx])

    from scipy.optimize import minimize_scalar
    bracket = (
        coarse_best_deg - scan_step_deg,
        coarse_best_deg,
        coarse_best_deg + scan_step_deg,
    )
    try:
        result = minimize_scalar(
            lambda a_deg: objective(np.deg2rad(a_deg)),
            bracket=bracket,
            method="brent",
            options={"xtol": 1e-4},
        )
        if result.fun < scan_vals[best_idx]:
            best_deg = float(result.x)
            best_obj = float(result.fun)
        else:
            best_deg = coarse_best_deg
            best_obj = float(scan_vals[best_idx])
    except (ValueError, RuntimeError):
        best_deg = coarse_best_deg
        best_obj = float(scan_vals[best_idx])

    T_best = rotation_about_axis(axis_dir, axis_point, np.deg2rad(best_deg))
    p_kin_rot = T_best.apply(p_kin)
    spans = np.linalg.norm(p_kin_rot - p_hub, axis=1)
    spans_at_optimum = {ep[0]: float(s) for ep, s in zip(endpoints, spans)}

    return {
        "angle_deg": best_deg,
        "angle_rad": np.deg2rad(best_deg),
        "T_rot": T_best,
        "axis_dir": axis_dir,
        "axis_point": axis_point,
        "objective": best_obj,
        "scan_angles_deg": scan_angles_deg,
        "scan_objective": scan_vals,
        "spans_at_optimum": spans_at_optimum,
    }


# ---------------------------------------------------------------------------
# Symmetry: filament-pair geometry + symmetry transforms
# ---------------------------------------------------------------------------

def _camkii_com(merged: ms.Scene) -> np.ndarray:
    """Centre of mass of the CaMKII *hub* (chains A–L, hub residues only).

    Using the hub rather than the whole dodecamer keeps the reference stable
    across the pipeline — the hub stays put while the kinases and linkers
    move, so any frame-defining quantity anchored on the hub is consistent
    between input and output scenes.
    """
    return merged[
        merged["chain"].isin(CAMKII_CHAINS)
        & merged["resid"].between(HUB_RANGE[0], HUB_RANGE[1])
    ][["x", "y", "z"]].to_numpy().mean(axis=0)


def _polarity_signed_axis(merged: ms.Scene, fil_chains, d_raw):
    """SVD axis sign is arbitrary; flip ``d_raw`` so it points from the first
    chain in ``fil_chains`` toward the last (i.e., follows the filament's
    polarity, since :func:`order_along_axis` already sorted the chains)."""
    c_first = chain_centroid(merged, fil_chains[0])
    c_last = chain_centroid(merged, fil_chains[-1])
    return d_raw if np.dot(d_raw, c_last - c_first) >= 0 else -d_raw


def filament_pair_geometry(merged: ms.Scene, filaments):
    """Describe the geometric relationship between the two filaments.

    Axes are signed by filament polarity (centroid of first chain → centroid
    of last chain) so that ``angle_deg`` is 0° when the two filaments point
    the same way and 180° when they point opposite ways.

    Returns a dict with vectors (``d0, d1, p0, p1, n_perp, v0, v1``) and four
    scalar descriptors:

    * ``angle_deg``  — angle between the two helical-axis directions.
    * ``d_axial_A``  — axial offset of ``p1`` relative to ``p0`` along ``d̂₀``.
    * ``d_perp_A``   — perpendicular (closest-approach) distance between the
                       two helical axes.
    * ``twist_deg``  — rotation of filament 1's first-monomer radial direction
                       relative to filament 0's, measured around ``d̂₀``.
    """
    d0_raw, p0 = filament_helical_axis(merged, filaments[0])
    d1_raw, p1 = filament_helical_axis(merged, filaments[1])
    d0 = _polarity_signed_axis(merged, filaments[0], d0_raw)
    d1 = _polarity_signed_axis(merged, filaments[1], d1_raw)

    cos_ang = float(np.clip(np.dot(d0, d1), -1.0, 1.0))
    angle_deg = float(np.degrees(np.arccos(cos_ang)))

    delta = p1 - p0
    d_axial = float(np.dot(delta, d0))
    d_perp_vec = delta - d_axial * d0
    d_perp = float(np.linalg.norm(d_perp_vec))
    n_perp = d_perp_vec / d_perp if d_perp > 1e-9 else np.zeros(3)

    # Radial reference: project the first monomer's centroid perpendicular
    # to d̂₀ (filament 0's direction). Using d̂₀ on both sides means twist is
    # measured in the same rotational frame.
    c0 = chain_centroid(merged, filaments[0][0])
    c1 = chain_centroid(merged, filaments[1][0])
    v0 = (c0 - p0) - np.dot(c0 - p0, d0) * d0
    v1 = (c1 - p1) - np.dot(c1 - p1, d0) * d0
    n0, n1 = np.linalg.norm(v0), np.linalg.norm(v1)
    v0_u = v0 / n0 if n0 > 1e-9 else np.zeros(3)
    v1_u = v1 / n1 if n1 > 1e-9 else np.zeros(3)
    twist_deg = float(np.degrees(np.arctan2(
        np.dot(np.cross(v0_u, v1_u), d0),
        np.dot(v0_u, v1_u),
    )))

    return {
        "d0": d0, "d1": d1, "p0": p0, "p1": p1,
        "n_perp": n_perp, "v0": v0_u, "v1": v1_u,
        "angle_deg": angle_deg,
        "d_axial_A": d_axial,
        "d_perp_A": d_perp,
        "twist_deg": twist_deg,
    }


def print_pair_geometry(geom, label="Filament-pair geometry"):
    """Pretty-print the four descriptors from :func:`filament_pair_geometry`."""
    print(f"\n{label}:")
    print(f"  angle between axes:        {geom['angle_deg']:7.3f}°  "
          f"(0° = parallel, 180° = antiparallel)")
    print(f"  axial offset of p1 along d̂₀:  {geom['d_axial_A']:+7.2f} Å")
    print(f"  perpendicular separation:  {geom['d_perp_A']:7.2f} Å")
    print(f"  twist of filament 1 about d̂₀: {geom['twist_deg']:+7.2f}°")


def build_filament_symmetry_transform(
    merged: ms.Scene,
    filaments,
    mode: str,
    filament0_transform: Transformation = None,
) -> Transformation:
    """Return the transformation that takes filament 1 from its current pose
    to the **C2 image of filament 0** about the CaMKII COM, with the radial
    twist forced to exactly 180°.

    Parameters
    ----------
    filament0_transform : Transformation, optional
        Transform already applied (or about to be applied) to filament 0
        — typically the rotation produced by ``--optimize-filaments``. The
        target frame for filament 1 is computed against this *effective*
        filament 0 pose so the output assembly is genuinely symmetric. Pass
        identity (or ``None``) when filament 0 isn't moving.

    Notes
    -----
    Two design choices keep the result clean:

    * Position and direction follow the user-specified C2:
      ``parallel`` → C2 about ``d̂₀`` through the CaMKII COM (filament 1
      mirrors filament 0 across the COM in the d̂₀-perpendicular plane,
      keeping the same direction); ``antiparallel`` → C2 about
      ``d̂₀ × n̂_⊥`` (filament 1 flips direction and mirrors across the
      COM in the plane containing d̂₀ and n̂_⊥).
    * The radial reference target is set explicitly to ``−v̂₀_eff`` rather
      than to ``C2(v̂₀_eff)``. The C2-image of v̂₀ has 180° twist only when
      v̂₀ happens to be perpendicular to the rotation axis (true for
      ``parallel``, generically false for ``antiparallel``). Forcing the
      target radial to ``−v̂₀_eff`` adds an extra rotation about filament
      1's new axis that pins the twist at 180° in both modes.
    """
    if filament0_transform is None:
        filament0_transform = Transformation.identity()

    com_cam = _camkii_com(merged)

    # Filament 0 pose at input + effective pose post-filament0_transform.
    d0_raw, p0 = filament_helical_axis(merged, filaments[0])
    d0_in = _polarity_signed_axis(merged, filaments[0], d0_raw)
    d0_eff = filament0_transform.rotation @ d0_in
    p0_eff = filament0_transform.apply(p0[None, :])[0]

    c0 = chain_centroid(merged, filaments[0][0])
    c0_eff = filament0_transform.apply(c0[None, :])[0]
    v0_raw = (c0_eff - p0_eff) - np.dot(c0_eff - p0_eff, d0_eff) * d0_eff
    v0_eff = v0_raw / np.linalg.norm(v0_raw)

    # Filament 1 current pose (no transform yet).
    d1_raw, p1 = filament_helical_axis(merged, filaments[1])
    d1 = _polarity_signed_axis(merged, filaments[1], d1_raw)
    c1 = chain_centroid(merged, filaments[1][0])
    v1_raw = (c1 - p1) - np.dot(c1 - p1, d1) * d1
    v1 = v1_raw / np.linalg.norm(v1_raw)

    # User-specified C2 (used only for the target position).
    delta = com_cam - p0_eff
    n_perp_vec = delta - np.dot(delta, d0_eff) * d0_eff
    n_perp_mag = np.linalg.norm(n_perp_vec)
    if n_perp_mag < 1e-9:
        raise ValueError(
            "CaMKII COM lies on filament 0's helical axis; cannot define n_perp."
        )
    n_perp = n_perp_vec / n_perp_mag

    if mode == "parallel":
        c2_axis = d0_eff
        d_tgt = d0_eff
    elif mode == "antiparallel":
        c2_axis = np.cross(d0_eff, n_perp)
        c2_axis = c2_axis / np.linalg.norm(c2_axis)
        d_tgt = -d0_eff
    else:
        raise ValueError(
            f"unknown symmetry mode {mode!r} (expected 'parallel' or 'antiparallel')"
        )

    C2 = rotation_about_axis(c2_axis, com_cam, np.pi)
    p_tgt = C2.apply(p0_eff[None, :])[0]

    # Force twist to 180° by construction: v_tgt = -v0_eff (already ⊥ d_tgt
    # since v0_eff ⊥ d0_eff and d_tgt = ±d0_eff).
    v_tgt = -v0_eff

    e2_curr = np.cross(d1, v1)
    R_curr = np.column_stack([v1, e2_curr, d1])
    e2_tgt = np.cross(d_tgt, v_tgt)
    R_tgt = np.column_stack([v_tgt, e2_tgt, d_tgt])

    R = R_tgt @ R_curr.T
    t = p_tgt - R @ p1
    return Transformation.from_matrix(R, t)


# ---------------------------------------------------------------------------
# Symmetry: CaMKII orientation
# ---------------------------------------------------------------------------

def camkii_frame(merged: ms.Scene):
    """Return ``(com, stack_axis_unit, vertex_AB_dir_unit)`` for the CaMKII.

    The stack axis is the unit vector from the back-face centroid to the
    front-face centroid (the hub's two faces are flat hexagons stacked
    face-to-face). The vertex-AB direction is the perpendicular-to-stack
    component of ``(p_AB − com)``, where ``p_AB`` is the centroid of the
    two A+B chains' hub residues.
    """
    com = _camkii_com(merged)

    def _hub_centroid(chains):
        sub = merged[
            merged["chain"].isin(chains)
            & merged["resid"].between(HUB_RANGE[0], HUB_RANGE[1])
        ]
        return sub[["x", "y", "z"]].to_numpy().mean(axis=0)

    p_front = _hub_centroid(HUB_FACE_FRONT)
    p_back = _hub_centroid(HUB_FACE_BACK)
    stack = p_front - p_back
    stack = stack / np.linalg.norm(stack)

    p_ab = _hub_centroid(HUB_VERTEX_STACKS[0])
    v = p_ab - com
    v_perp = v - np.dot(v, stack) * stack
    v_perp_unit = v_perp / np.linalg.norm(v_perp)

    return com, stack, v_perp_unit


def _camkii_target_com(d0, p0, d1, p1):
    """Midpoint of the closest-approach segment between two infinite lines
    ``(p0 + t·d̂₀)`` and ``(p1 + s·d̂₁)``."""
    n = np.cross(d0, d1)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-9:
        # Parallel lines: midpoint of the projection segment.
        return 0.5 * (p0 + p1)
    # Solve for parameters that give the closest approach.
    n2 = n / n_norm
    delta = p1 - p0
    mat = np.column_stack([d0, -d1, n2])
    coeffs = np.linalg.solve(mat, delta)
    t, s, _ = coeffs
    q0 = p0 + t * d0
    q1 = p1 + s * d1
    return 0.5 * (q0 + q1)


def build_camkii_symmetry_transform(
    merged: ms.Scene,
    filaments,
    filament1_transform: Transformation,
    orientation: str,
) -> Transformation:
    """Compute the Transformation that brings the CaMKII into the symmetric pose.

    Parameters
    ----------
    merged : Scene
        Pre-symmetrisation merged template.
    filaments : list of list[str]
        ``[filament0_chains, filament1_chains]``.
    filament1_transform : Transformation
        Transform already applied to filament 1 (e.g. via ``--enforce-symmetry``
        or ``--optimize-filaments``). Identity if filament 1 hasn't moved.
        Needed so the target CaMKII COM is computed against the *post*-Block-1
        geometry.
    orientation : {'pointed', 'flat'}
        Hub orientation. See module docstring.
    """
    from scipy.spatial.transform import Rotation as _R

    com_curr, stack_curr, vAB_curr = camkii_frame(merged)

    d0, p0 = filament_helical_axis(merged, filaments[0])
    d1, p1 = filament_helical_axis(merged, filaments[1])

    # Apply filament 1's post-Block-1 transform to its axis so the target
    # COM and n_perp are measured against the actual filament-1 position
    # we'll see in the output.
    p1_eff = filament1_transform.apply(p1[None, :])[0]
    d1_eff_endpoint = filament1_transform.apply((p1 + d1)[None, :])[0]
    d1_eff = d1_eff_endpoint - p1_eff
    d1_eff = d1_eff / np.linalg.norm(d1_eff)

    com_target = _camkii_target_com(d0, p0, d1_eff, p1_eff)

    # n_perp_eff: from filament 0's axis to the target COM, ⊥ to d̂₀.
    delta = com_target - p0
    n_perp_vec = delta - np.dot(delta, d0) * d0
    n_perp_mag = np.linalg.norm(n_perp_vec)
    if n_perp_mag < 1e-9:
        raise ValueError("Target CaMKII COM lies on filament 0's axis.")
    n_perp_eff = n_perp_vec / n_perp_mag

    # Target stack axis: normal to the plane containing both filament axes
    # (= the plane spanned by d̂₀ and n̂_⊥_eff).
    e3_tgt = np.cross(d0, n_perp_eff)
    e3_tgt = e3_tgt / np.linalg.norm(e3_tgt)
    if np.dot(e3_tgt, stack_curr) < 0:
        e3_tgt = -e3_tgt

    # Target AB-vertex direction depending on orientation.
    if orientation == "pointed":
        e1_tgt = n_perp_eff
    elif orientation == "flat":
        # AB sits at +120° clockwise (i.e. +120° about +e3_tgt with the
        # clockwise-as-positive convention used in the user's vertex
        # ordering) from DE; in the "flat" pose DE points along ±d̂₀.
        R_120 = _R.from_rotvec(np.deg2rad(120) * e3_tgt).as_matrix()
        e1_tgt = R_120 @ d0
        # Project out any stack-axis component and renormalise.
        e1_tgt = e1_tgt - np.dot(e1_tgt, e3_tgt) * e3_tgt
        e1_tgt = e1_tgt / np.linalg.norm(e1_tgt)
    else:
        raise ValueError(f"unknown orientation {orientation!r} (expected 'pointed' or 'flat')")

    # Pick the sign of e1_tgt that minimises the rotation distance from
    # vAB_curr, so we don't flip the hexagon by 180° unnecessarily.
    if np.dot(e1_tgt, vAB_curr) < 0:
        e1_tgt = -e1_tgt

    # Build orthonormal current and target frames; the third axis is e2 = e3 × e1.
    e2_curr = np.cross(stack_curr, vAB_curr)
    R_curr = np.column_stack([vAB_curr, e2_curr, stack_curr])
    e2_tgt = np.cross(e3_tgt, e1_tgt)
    R_tgt = np.column_stack([e1_tgt, e2_tgt, e3_tgt])

    R = R_tgt @ R_curr.T
    t = com_target - R @ com_curr
    return Transformation.from_matrix(R, t)


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def generate_candidates(merged, kabt_kinase, kabt_actins):
    """Slide KABT along both filaments with stride-2, return all candidate poses.

    Returns
    -------
    list of dict
        Each entry: ``{'pose': Scene, 'filament': int, 'pair': (str, str),
        'rmsd': float, 'hub_distance': float}``.
    """
    print("Clustering actin chains into 2 filaments …")
    actin_chains = sorted(c for c in merged["chain"].unique() if c not in CAMKII_CHAINS)
    filaments = cluster_into_filaments(merged, actin_chains, n_filaments=2)
    filaments = [order_along_axis(merged, fil) for fil in filaments]
    for fi, fil in enumerate(filaments):
        print(f"  filament {fi} ({len(fil)} monomers): {' → '.join(fil)}")

    print("Generating kinase candidates by sliding KABT (stride-2) along each filament …")
    candidates = []
    for fi, fil in enumerate(filaments):
        for i in range(len(fil) - 2):
            ci, cj = fil[i], fil[i + 2]
            pair_ca = merged.select(f"(chain {ci} or chain {cj}) and name CA")
            T = fit_kabt_to_pair(kabt_actins, ci, cj, pair_ca)
            if T is None:
                continue
            candidates.append({
                "id": len(candidates),  # stable generation-order id, unaffected by sorting
                "pose": kabt_kinase.transform(T),
                "filament": fi,
                "pair": (ci, cj),
                "rmsd": T.rmsd,
            })

    hub_sel = " or ".join(f"chain {c}" for c in CAMKII_CHAINS)
    hub = merged.select(f"({hub_sel}) and resid {HUB_RANGE[0]} to {HUB_RANGE[1]}")
    hub_center = hub.get_center().to_numpy()
    for c in candidates:
        c["hub_distance"] = float(np.linalg.norm(c["pose"].get_center().to_numpy() - hub_center))

    candidates.sort(key=lambda c: c["hub_distance"])
    print(f"  generated {len(candidates)} candidates (sorted by distance to hub centroid)")
    return candidates, hub_center, filaments


def write_candidates_view(merged, candidates, out_prefix):
    """Write a labelled candidates PDB + a labels TXT file for manual inspection."""
    pdb_path = out_prefix + ".pdb"
    labels_path = out_prefix + ".txt"

    if len(candidates) > len(CANDIDATE_CHAIN_IDS):
        raise RuntimeError(
            f"too many candidates ({len(candidates)}) for the single-character chain "
            f"label set (max {len(CANDIDATE_CHAIN_IDS)})."
        )

    parts = []
    # Source CaMKII chains: full subunit (kinase + linker + hub) — gives the
    # user a visual reference of where each kinase currently is and where the
    # immovable hub ring sits.
    for c in CAMKII_CHAINS:
        parts.append(merged.select(f"chain {c}"))
    # Candidates: just the kinase atoms in their target pose.
    for cand in candidates:
        sub = cand["pose"].copy()
        sub["chain"] = CANDIDATE_CHAIN_IDS[cand["id"]]
        parts.append(sub)

    combined = ms.Scene(pd.concat(parts, ignore_index=True))
    combined.write_pdb(pdb_path)

    with open(labels_path, "w") as f:
        f.write("# Candidate kinase poses (sorted by ascending distance to the hub centroid).\n")
        f.write("# Open " + pdb_path + " in your viewer; each candidate is on the listed chain.\n")
        f.write("# To assign manually, fill out an assignment file with one line per\n"
                "# source kinase: \"source_chain candidate_id\". Lines starting with '#'\n"
                "# are ignored. There must be exactly 12 lines (one per source chain).\n")
        f.write("\n")
        f.write(f"# {'cand_id':>7s} {'chain':>5s} {'filament':>8s} {'pair':>9s} "
                f"{'fit_RMSD':>10s} {'dist_to_hub':>12s}\n")
        for cand in sorted(candidates, key=lambda c: c["id"]):
            ci, cj = cand["pair"]
            f.write(f"  {cand['id']:>7d} {CANDIDATE_CHAIN_IDS[cand['id']]:>5s} {cand['filament']:>8d} "
                    f"{ci + ',' + cj:>9s} {cand['rmsd']:>10.2f} {cand['hub_distance']:>12.2f}\n")
        f.write("\n# Source CaMKII chains (kinase residues 1-274 currently at):\n")
        f.write(f"# {'chain':>5s}  {'centroid_x':>10s} {'centroid_y':>10s} {'centroid_z':>10s}\n")
        for c in CAMKII_CHAINS:
            cc = merged.select(
                f"chain {c} and resid {KINASE_RANGE[0]} to {KINASE_RANGE[1]}"
            ).get_center().to_numpy()
            f.write(f"  {c:>5s}  {cc[0]:>10.2f} {cc[1]:>10.2f} {cc[2]:>10.2f}\n")
        f.write("\n# Example assignment file:\n")
        f.write("# A  0\n# B  3\n# ...\n# L  21\n")

    print(f"Wrote {pdb_path} ({len(combined)} atoms) and {labels_path}")
    print(f"\nNext step: write an assignment file (see {labels_path} for the format),")
    print( "then re-run with `--assignment <file>` (optionally with `--animate N`).")


def read_assignment(path, candidates):
    """Parse a 2-column "source_chain candidate_id" file."""
    cand_by_id = {c["id"]: c for c in candidates}
    assignments = []
    seen_sources, seen_candidates = set(), set()
    with open(path) as f:
        for ln, raw in enumerate(f, 1):
            line = raw.split("#", 1)[0].strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"{path}:{ln}: expected 'source_chain candidate_id', got {raw!r}")
            src, cid_str = parts[0], parts[1]
            try:
                cid = int(cid_str)
            except ValueError:
                raise ValueError(f"{path}:{ln}: candidate id {cid_str!r} is not an integer")
            if src not in CAMKII_CHAINS:
                raise ValueError(f"{path}:{ln}: unknown source chain {src!r} "
                                 f"(must be one of {CAMKII_CHAINS})")
            if cid not in cand_by_id:
                raise ValueError(f"{path}:{ln}: candidate id {cid} not found "
                                 f"(valid ids: {sorted(cand_by_id)})")
            if src in seen_sources:
                raise ValueError(f"{path}:{ln}: source chain {src!r} assigned twice")
            if cid in seen_candidates:
                raise ValueError(f"{path}:{ln}: candidate {cid} assigned twice")
            seen_sources.add(src)
            seen_candidates.add(cid)
            assignments.append((src, cand_by_id[cid]))
    if len(assignments) != len(CAMKII_CHAINS):
        raise ValueError(
            f"expected {len(CAMKII_CHAINS)} assignments (one per source chain), "
            f"got {len(assignments)} from {path}"
        )
    return assignments


def assign_hungarian(merged, candidates, top_n=12):
    """Take the top_n closest-to-hub candidates and Hungarian-match against source chains."""
    targets = candidates[:top_n]
    src_centers = np.array([
        merged.select(f"chain {c} and resid {KINASE_RANGE[0]} to {KINASE_RANGE[1]}")
              .get_center().to_numpy()
        for c in CAMKII_CHAINS
    ])
    tgt_centers = np.array([t["pose"].get_center().to_numpy() for t in targets])
    cost = np.linalg.norm(src_centers[:, None, :] - tgt_centers[None, :, :], axis=2)
    row_ind, col_ind = linear_sum_assignment(cost)
    print(f"  total assignment cost: {cost[row_ind, col_ind].sum():.1f} Å")
    return [(CAMKII_CHAINS[i], targets[j]) for i, j in zip(row_ind, col_ind)]


def compute_chain_transforms(merged, assignments):
    """For each (src_chain, candidate), compute the rigid Kabsch transform
    that takes the source kinase onto the candidate kinase pose."""
    chain_transforms = []
    for src_chain, candidate in assignments:
        src_kin = merged.select(
            f"chain {src_chain} and resid {KINASE_RANGE[0]} to {KINASE_RANGE[1]}"
        )
        T_kin = src_kin.compute_transformation(
            candidate["pose"],
            match=SequenceMatching(chain_pairs={src_chain: "A"}),
        )
        chain_transforms.append((src_chain, T_kin))
    return chain_transforms


def build_frame(merged, chain_transforms, alpha,
                filament_rotations=None, camkii_symmetry_transform=None):
    """Render the structure at fraction ``alpha`` of the morph (0 = rest, 1 = final).

    Parameters
    ----------
    merged, chain_transforms, alpha : see above.
    filament_rotations : list of (list[str], Transformation), optional
        Per-filament rotations to apply to the actin atoms of each
        filament's chains. Each rotation is sclerp-interpolated by
        ``alpha`` just like the kinase transforms. Pass the same list
        used to compose the chain_transforms so the actins move in lock-
        step with the kinases.
    camkii_symmetry_transform : Transformation, optional
        A single transform applied to the **hub residues** of all 12 CaMKII
        chains (sclerp-interpolated by ``alpha``). The linker morph for each
        chain uses ``T_cam_at`` as its ``t_start`` (instead of identity) so
        the hub end of the linker tracks the moved hub.
    """
    out = merged.copy()

    if filament_rotations:
        coords = out.get_coordinates().to_numpy()
        for fil_chains, T_rot in filament_rotations:
            T_at = Transformation.identity().interpolate(T_rot, alpha, method="sclerp")
            mask = out["chain"].isin(fil_chains).to_numpy()
            coords[mask] = T_at.apply(coords[mask])
        out.set_coordinates(coords)

    if camkii_symmetry_transform is not None:
        T_cam_at = Transformation.identity().interpolate(
            camkii_symmetry_transform, alpha, method="sclerp",
        )
        coords = out.get_coordinates().to_numpy()
        mask = (
            out["chain"].isin(CAMKII_CHAINS)
            & out["resid"].between(HUB_RANGE[0], HUB_RANGE[1])
        ).to_numpy()
        coords[mask] = T_cam_at.apply(coords[mask])
        out.set_coordinates(coords)
    else:
        T_cam_at = Transformation.identity()

    for src_chain, T_kin in chain_transforms:
        T_at = Transformation.identity().interpolate(T_kin, alpha, method="sclerp")
        mask = (
            (out["chain"] == src_chain)
            & (out["resid"].between(KINASE_RANGE[0], KINASE_RANGE[1]))
        ).to_numpy()
        coords = out.get_coordinates().to_numpy()
        coords[mask] = T_at.apply(coords[mask])
        out.set_coordinates(coords)
        out = out.morph_segment(
            chain=src_chain,
            resid_range=range(LINKER_RANGE[1], LINKER_RANGE[0] - 1, -1),
            t_start=T_cam_at,
            t_end=T_at,
            method="sclerp",
        )
    return out


def write_animation(merged, chain_transforms, n_frames, out_path,
                    filament_rotations=None, camkii_symmetry_transform=None):
    """Write a multi-model PDB with ``n_frames`` morph frames (rest → final)."""
    print(f"Building {n_frames}-frame morph movie …")
    # Atom layout is identical across frames, so build a per-frame coordinate
    # stack and use the multi-frame write_pdb path.
    frame_coords = []
    for i in range(n_frames):
        alpha = 0.0 if n_frames == 1 else i / (n_frames - 1)
        frame = build_frame(
            merged, chain_transforms, alpha,
            filament_rotations=filament_rotations,
            camkii_symmetry_transform=camkii_symmetry_transform,
        )
        frame_coords.append(frame.get_coordinates().to_numpy())
        print(f"  frame {i + 1:>3d}/{n_frames} (α = {alpha:.3f}) built")
    movie = merged.copy()
    movie.set_coordinate_frames(np.stack(frame_coords, axis=0))
    movie.write_pdb(out_path)
    print(f"Wrote {out_path}")


# ---------------------------------------------------------------------------
# Main / CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Transplant Qian's CaMKII–actin binding pose onto the merged template.",
    )
    p.add_argument(
        "--save-candidates", metavar="PREFIX",
        help="Save labelled candidate kinase poses (PREFIX.pdb + PREFIX.txt) "
             "for manual inspection, then exit before assignment.",
    )
    p.add_argument(
        "--assignment", metavar="FILE",
        help="Path to a 2-column 'source_chain candidate_id' file. If omitted, "
             "the script auto-assigns via Hungarian matching on hub-distance ranking.",
    )
    p.add_argument(
        "--animate", type=int, metavar="N_FRAMES", default=None,
        help="Also write an N-frame multi-model PDB showing the morph from rest "
             "to the final pose.",
    )
    p.add_argument(
        "--optimize-filaments", action="store_true",
        help="After computing the kinase transforms, rotate each actin filament "
             "around its helical axis to minimise the sum of squared linker spans "
             "(distance between the first and last linker beads). Treats the "
             "linker as a spring whose tension we want minimised and equalised.",
    )
    p.add_argument(
        "--scan-step", type=float, default=1.0, metavar="DEG",
        help="Coarse scan step in degrees for --optimize-filaments (default: %(default)s). "
             "The result is then refined with Brent's method.",
    )
    p.add_argument(
        "--enforce-symmetry", choices=["parallel", "antiparallel"], default=None,
        help="Override filament 1's pose so the pair is symmetric about the CaMKII COM. "
             "'parallel' rotates 180° around filament 0's helical-axis direction (filaments "
             "point the same way, on opposite sides of CaMKII); 'antiparallel' rotates 180° "
             "around the axis perpendicular to both d̂₀ and the COM perpendicular (filaments "
             "point opposite ways). Compatible with --optimize-filaments (which then applies "
             "only to filament 0).",
    )
    p.add_argument(
        "--enforce-camkii-symmetry", choices=["pointed", "flat"], default=None,
        help="Translate the CaMKII hub to the midpoint between the two filaments and orient "
             "the hexagonal dodecamer. 'pointed' places vertex AB perpendicular to the "
             "helical axis (a vertex points toward each filament); 'flat' aligns the HK↔DE "
             "diameter with the helical axis (an edge faces each filament). The kinases stay "
             "where the assignment placed them; the linkers re-morph to bridge the new hub.",
    )
    p.add_argument(
        "--output", default=OUT_PATH,
        help="Final structure PDB path (default: %(default)s).",
    )

    p.add_argument("--split", action="store_true", 
                   help="Split the merged template into separate CIF files for the CaMKII hub "
                   "and each actin filament, then exit. Useful for manual inspection and picking "
                   "of candidate poses in a viewer.")
    return p.parse_args()


def _spans_summary(spans):
    """Return a one-line summary of a span dict."""
    vals = np.array(list(spans.values()))
    return (f"mean {vals.mean():6.2f} Å, "
            f"std {vals.std():5.2f} Å, "
            f"min {vals.min():6.2f}, max {vals.max():6.2f}, "
            f"Σ² {float(np.sum(vals ** 2)):.1f}")


def main():
    args = parse_args()

    print("Loading inputs …")
    merged = ms.Scene.from_file(MERGED_PATH)
    qian = ms.Scene.from_pdb(QIAN_PATH)
    print(f"  merged: {len(merged)} atoms, {merged['chain'].nunique()} chains")
    print(f"  qian:   {len(qian)} atoms")

    print("Building KABT from Qian …")
    kabt_kinase, kabt_actins = build_kabt(qian)
    print(f"  KABT kinase: {len(kabt_kinase)} atoms")
    print(f"  KABT actins: {len(kabt_actins)} atoms")

    candidates, hub_center, filaments = generate_candidates(
        merged, kabt_kinase, kabt_actins
    )
    print(f"  hub ring centroid: [{hub_center[0]:.2f}, {hub_center[1]:.2f}, {hub_center[2]:.2f}]")

    # Always-on geometry printout for the filament pair as currently positioned.
    pair_geom = filament_pair_geometry(merged, filaments)
    print_pair_geometry(pair_geom, label="Filament-pair geometry (input)")

    if args.save_candidates:
        write_candidates_view(merged, candidates, args.save_candidates)
        return 0

    if args.assignment:
        print(f"Reading assignment from {args.assignment} …")
        assignments = read_assignment(args.assignment, candidates)
    else:
        print("Hungarian-assigning the top 12 candidates …")
        assignments = assign_hungarian(merged, candidates, top_n=12)

    for src_chain, candidate in assignments:
        ci, cj = candidate["pair"]
        print(f"  chain {src_chain} → candidate filament {candidate['filament']} "
              f"pair ({ci},{cj}), hub-dist {candidate['hub_distance']:.1f} Å")

    print("Computing per-chain kinase transforms …")
    chain_transforms = compute_chain_transforms(merged, assignments)

    # ------------------------------------------------------------------
    # Per-filament transform pass: --enforce-symmetry wins on filament 1,
    # --optimize-filaments otherwise. Either one (or both) populates
    # filament_rotations and chain_extra_rot.
    # ------------------------------------------------------------------
    filament_rotations = None
    if args.optimize_filaments or args.enforce_symmetry:
        spans_before = compute_linker_spans(merged, chain_transforms)
        print("\nFilament-rotation pass:")
        print("  Linker spans before:")
        for src in sorted(spans_before):
            print(f"    chain {src}: {spans_before[src]:7.2f} Å")
        print(f"    summary: {_spans_summary(spans_before)}")

        chain_to_filament = {src: cand["filament"] for src, cand in assignments}
        filament_rotations = []
        chain_extra_rot = {}
        per_filament_transforms = {}  # fi -> Transformation (used to pass T0 into T_sym for fi=1)
        for fi, fil_chains in enumerate(filaments):
            bound = [(c, T) for c, T in chain_transforms
                     if chain_to_filament[c] == fi]

            T_fi = None
            if fi == 1 and args.enforce_symmetry:
                print(f"\n  Filament {fi}: --enforce-symmetry={args.enforce_symmetry}")
                # Use filament 0's effective (post-optimisation) pose as the
                # symmetry reference so the C2 holds against the *output*
                # filament 0, not its input.
                T_fi = build_filament_symmetry_transform(
                    merged, filaments, mode=args.enforce_symmetry,
                    filament0_transform=per_filament_transforms.get(0),
                )
                rotvec = np.array([0.0, 0.0, 0.0])
                try:
                    from scipy.spatial.transform import Rotation as _R
                    rotvec = _R.from_matrix(T_fi.rotation).as_rotvec()
                except Exception:
                    pass
                print(f"    rotation axis ≈ [{rotvec[0]:+.3f}, {rotvec[1]:+.3f}, "
                      f"{rotvec[2]:+.3f}] (180°)")
            elif args.optimize_filaments and bound:
                print(f"\n  Filament {fi}: --optimize-filaments "
                      f"(chains bound: {sorted(c for c, _ in bound)})")
                result = optimize_filament_rotation(
                    merged, bound, fil_chains, scan_step_deg=args.scan_step,
                )
                T_fi = result["T_rot"]
                print(f"    helical axis dir: [{result['axis_dir'][0]:+.3f}, "
                      f"{result['axis_dir'][1]:+.3f}, {result['axis_dir'][2]:+.3f}]")
                print(f"    helical axis pt:  [{result['axis_point'][0]:+8.2f}, "
                      f"{result['axis_point'][1]:+8.2f}, {result['axis_point'][2]:+8.2f}]")
                print(f"    scan range: 0°..360° step {args.scan_step}°  →  "
                      f"min Σ² = {result['scan_objective'].min():.1f} "
                      f"@ {result['scan_angles_deg'][result['scan_objective'].argmin()]:.1f}°, "
                      f"max Σ² = {result['scan_objective'].max():.1f}")
                print(f"    refined optimum: angle = {result['angle_deg']:.3f}°, "
                      f"Σ² = {result['objective']:.1f}")

            if T_fi is None:
                continue
            per_filament_transforms[fi] = T_fi
            filament_rotations.append((fil_chains, T_fi))
            for src, _ in bound:
                chain_extra_rot[src] = T_fi

        # Compose so chain_transforms now produce the rotated-placed kinase.
        chain_transforms = [
            (src, chain_extra_rot[src].compose(T) if src in chain_extra_rot else T)
            for src, T in chain_transforms
        ]

        spans_after = compute_linker_spans(merged, chain_transforms)
        print("\n  Linker spans after:")
        for src in sorted(spans_after):
            print(f"    chain {src}: {spans_after[src]:7.2f} Å")
        print(f"    summary: {_spans_summary(spans_after)}")
        if not filament_rotations:
            filament_rotations = None

    # ------------------------------------------------------------------
    # --enforce-camkii-symmetry: rigid transform on hub residues + linker
    # re-morph from the moved hub to the (already placed) kinase.
    # ------------------------------------------------------------------
    camkii_symmetry_transform = None
    if args.enforce_camkii_symmetry:
        print(f"\nCaMKII symmetrisation: --enforce-camkii-symmetry={args.enforce_camkii_symmetry}")
        # Identify the transform that's already been applied to filament 1
        # (identity unless --enforce-symmetry or --optimize-filaments fired).
        T_fil1 = Transformation.identity()
        if filament_rotations:
            for fil_chains, T in filament_rotations:
                if fil_chains is filaments[1]:
                    T_fil1 = T
                    break
        camkii_symmetry_transform = build_camkii_symmetry_transform(
            merged, filaments, T_fil1, args.enforce_camkii_symmetry,
        )
        com_curr, _, _ = camkii_frame(merged)
        com_target = camkii_symmetry_transform.apply(com_curr[None, :])[0]
        translation = com_target - com_curr
        print(f"  CaMKII COM current: [{com_curr[0]:+.2f}, {com_curr[1]:+.2f}, {com_curr[2]:+.2f}]")
        print(f"  CaMKII COM target:  [{com_target[0]:+.2f}, {com_target[1]:+.2f}, {com_target[2]:+.2f}]")
        print(f"  translation:        {np.linalg.norm(translation):.2f} Å")

    print("\nApplying kinase transforms + morphing linkers (ScLERP) …")
    final = build_frame(
        merged, chain_transforms, alpha=1.0,
        filament_rotations=filament_rotations,
        camkii_symmetry_transform=camkii_symmetry_transform,
    )
    final.to_file(args.output)
    print(f"Wrote {args.output} ({len(final)} atoms)")

    # Always re-print the final geometry so it can be compared against the input.
    final_pair_geom = filament_pair_geometry(final, filaments)
    print_pair_geometry(final_pair_geom, label="Filament-pair geometry (output)")

    if args.animate:
        movie_path = args.output.replace(".pdb", "_movie.pdb")
        write_animation(
            merged, chain_transforms, args.animate, movie_path,
            filament_rotations=filament_rotations,
            camkii_symmetry_transform=camkii_symmetry_transform,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())

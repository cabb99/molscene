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
    return candidates, hub_center


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


def build_frame(merged, chain_transforms, alpha):
    """Render the structure at fraction ``alpha`` of the morph (0 = rest, 1 = final)."""
    out = merged.copy()
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
            t_start=Transformation.identity(),
            t_end=T_at,
            method="sclerp",
        )
    return out


def write_animation(merged, chain_transforms, n_frames, out_path):
    """Write a multi-model PDB with ``n_frames`` morph frames (rest → final)."""
    print(f"Building {n_frames}-frame morph movie …")
    # Atom layout is identical across frames, so build a per-frame coordinate
    # stack and use the multi-frame write_pdb path.
    frame_coords = []
    for i in range(n_frames):
        alpha = 0.0 if n_frames == 1 else i / (n_frames - 1)
        frame = build_frame(merged, chain_transforms, alpha)
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
        "--output", default=OUT_PATH,
        help="Final structure PDB path (default: %(default)s).",
    )
    return p.parse_args()


def main():
    args = parse_args()

    print("Loading inputs …")
    merged = ms.Scene.from_pdb(MERGED_PATH)
    qian = ms.Scene.from_pdb(QIAN_PATH)
    print(f"  merged: {len(merged)} atoms, {merged['chain'].nunique()} chains")
    print(f"  qian:   {len(qian)} atoms")

    print("Building KABT from Qian …")
    kabt_kinase, kabt_actins = build_kabt(qian)
    print(f"  KABT kinase: {len(kabt_kinase)} atoms")
    print(f"  KABT actins: {len(kabt_actins)} atoms")

    candidates, hub_center = generate_candidates(merged, kabt_kinase, kabt_actins)
    print(f"  hub ring centroid: [{hub_center[0]:.2f}, {hub_center[1]:.2f}, {hub_center[2]:.2f}]")

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

    print("Applying kinase transforms + morphing linkers (ScLERP) …")
    final = build_frame(merged, chain_transforms, alpha=1.0)
    final.to_file(args.output)
    print(f"Wrote {args.output} ({len(final)} atoms)")

    if args.animate:
        movie_path = args.output.replace(".pdb", "_movie.pdb")
        write_animation(merged, chain_transforms, args.animate, movie_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

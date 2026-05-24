"""
Tests for :meth:`molscene.Scene.morph_segment` (per-residue local-frame morph).

The linker fixture is residues 50–70 of chain A of 1JGE, which gives 21
contiguous residues to interpolate across.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from molscene import Scene, Transformation


# --- Fixtures -------------------------------------------------------------

@pytest.fixture(scope="module")
def jge():
    return Scene.from_pdb(Path("molscene/data/1jge.pdb")).select(model=[1])


LINKER_RESIDS = list(range(50, 71))   # 21 residues


def _residue_centroid(scene, chain, resid):
    sub = scene[(scene['chain'] == chain) & (scene['resid'] == resid)]
    return sub[['x', 'y', 'z']].to_numpy().mean(axis=0)


def _residue_pairwise(scene, chain, resid):
    sub = scene[(scene['chain'] == chain) & (scene['resid'] == resid)]
    xyz = sub[['x', 'y', 'z']].to_numpy()
    diffs = xyz[:, None, :] - xyz[None, :, :]
    return np.sqrt((diffs ** 2).sum(axis=-1))


# --- Test cases ----------------------------------------------------------

class TestMorphSegment:
    @pytest.mark.parametrize("method", ["sclerp", "slerp"])
    def test_identity_anchors_leave_scene_unchanged(self, jge, method):
        T = Transformation.identity()
        out = jge.morph_segment("A", LINKER_RESIDS, T, T, method=method)
        np.testing.assert_allclose(
            out.get_coordinates().to_numpy(),
            jge.get_coordinates().to_numpy(),
            atol=1e-12,
        )

    def test_empty_range_is_noop(self, jge):
        out = jge.morph_segment(
            "A", [], Transformation.identity(),
            Transformation.from_matrix(np.eye(3), [10, 0, 0]),
        )
        np.testing.assert_allclose(
            out.get_coordinates().to_numpy(),
            jge.get_coordinates().to_numpy(),
            atol=1e-12,
        )

    def test_outside_segment_atoms_untouched(self, jge):
        t_end = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [10, 0, 0],
        )
        out = jge.morph_segment("A", LINKER_RESIDS, Transformation.identity(), t_end)
        outside = jge[
            ~((jge['chain'] == 'A') & (jge['resid'].isin(LINKER_RESIDS)))
        ]
        out_outside = out.iloc[outside.index.get_indexer(outside.index)]
        # The masked rows should not move; compare directly by index.
        outside_xyz = outside[['x', 'y', 'z']].to_numpy()
        moved_xyz = out.loc[outside.index, ['x', 'y', 'z']].to_numpy()
        np.testing.assert_allclose(outside_xyz, moved_xyz, atol=1e-12)

    @pytest.mark.parametrize("method", ["sclerp", "slerp"])
    def test_pure_translation_centroids_lerp_linearly(self, jge, method):
        # Pure translation: alpha=0 anchor untouched, alpha=1 anchor shifted
        # by (10, 0, 0). Centroid of residue i must shift by alpha_i*10 on x.
        t_end = Transformation.from_matrix(np.eye(3), [10, 0, 0])
        original_centroids = np.array(
            [_residue_centroid(jge, 'A', rid) for rid in LINKER_RESIDS]
        )
        out = jge.morph_segment("A", LINKER_RESIDS, Transformation.identity(), t_end, method=method)
        new_centroids = np.array(
            [_residue_centroid(out, 'A', rid) for rid in LINKER_RESIDS]
        )
        shifts = new_centroids - original_centroids
        n = len(LINKER_RESIDS)
        for i in range(n):
            alpha = i / (n - 1)
            np.testing.assert_allclose(shifts[i], [alpha * 10.0, 0, 0], atol=1e-9)

    @pytest.mark.parametrize("method", ["sclerp", "slerp"])
    def test_intra_residue_geometry_preserved(self, jge, method):
        # Each residue must move rigidly: pairwise distances between its
        # atoms unchanged after a non-trivial anchor pair.
        t_end = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [10, 0, 5],
        )
        out = jge.morph_segment("A", LINKER_RESIDS, Transformation.identity(), t_end, method=method)
        for rid in LINKER_RESIDS:
            d_before = _residue_pairwise(jge, 'A', rid)
            d_after = _residue_pairwise(out, 'A', rid)
            np.testing.assert_allclose(d_after, d_before, atol=1e-9)

    def test_endpoints_match_anchor_transforms(self, jge):
        t_end = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [10, 0, 5],
        )
        out = jge.morph_segment("A", LINKER_RESIDS, Transformation.identity(), t_end)

        first_rid = LINKER_RESIDS[0]
        before = jge[(jge['chain'] == 'A') & (jge['resid'] == first_rid)][['x', 'y', 'z']].to_numpy()
        after = out[(out['chain'] == 'A') & (out['resid'] == first_rid)][['x', 'y', 'z']].to_numpy()
        np.testing.assert_allclose(after, before, atol=1e-12)

        last_rid = LINKER_RESIDS[-1]
        before = jge[(jge['chain'] == 'A') & (jge['resid'] == last_rid)][['x', 'y', 'z']].to_numpy()
        after = out[(out['chain'] == 'A') & (out['resid'] == last_rid)][['x', 'y', 'z']].to_numpy()
        np.testing.assert_allclose(after, t_end.apply(before), atol=1e-9)

    def test_sclerp_screw_axis_invariant(self, jge):
        # For (R_end = 90° about z, t_end = (10,0,0)), the screw axis is the
        # vertical line through (5, 5, 0). Every residue centroid before
        # morphing has a fixed (constant-α-independent) "rotation centre"
        # under sclerp: each centroid traces a circular arc on the same
        # vertical axis. Check that centroid distance to (5,5,*) stays
        # constant across the segment for the part of the motion that is
        # purely circular (z component is preserved by the rotation, the
        # centroid's relative offset from the screw axis is rigid).
        t_end = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [10, 0, 0],
        )
        n = len(LINKER_RESIDS)
        # Independently morph and check: for one specific residue (the
        # midpoint), the trajectory of its centroid under sclerp must satisfy
        # the screw-axis identity used in test_transformation.py.
        mid_idx = n // 2
        rid_mid = LINKER_RESIDS[mid_idx]
        alpha_mid = mid_idx / (n - 1)
        out = jge.morph_segment("A", LINKER_RESIDS, Transformation.identity(), t_end)
        c_before = _residue_centroid(jge, 'A', rid_mid)
        c_after = _residue_centroid(out, 'A', rid_mid)

        # Expected: applying sclerp(identity, t_end, alpha_mid) globally to the
        # original centroid must reproduce c_after, because the morph applies
        # T_i rigidly to every atom in that residue.
        T_mid = Transformation.sclerp(Transformation.identity(), t_end, alpha_mid)
        expected = T_mid.apply(c_before[None, :])[0]
        np.testing.assert_allclose(c_after, expected, atol=1e-9)

    def test_unknown_method_raises(self, jge):
        with pytest.raises(ValueError, match="unknown"):
            jge.morph_segment(
                "A", LINKER_RESIDS,
                Transformation.identity(), Transformation.identity(),
                method="nope",
            )

    def test_non_transformation_anchor_raises(self, jge):
        with pytest.raises(TypeError, match="Transformation"):
            jge.morph_segment(
                "A", LINKER_RESIDS, np.eye(3), Transformation.identity(),
            )

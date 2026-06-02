"""
Tests for the alignment-related :class:`Scene` methods: :meth:`Scene.transform`,
:meth:`Scene.match`, :meth:`Scene.compute_transformation`,
:meth:`Scene.superpose`, and :meth:`Scene.rmsd`.

The fixtures use ``1jge.pdb`` (a small NMR structure shipped with the package
under ``molscene/data``) so the tests run against a realistic atom layout.
"""

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from molscene import (
    Scene,
    Transformation,
    ColumnMatching,
    OrderMatching,
    SequenceMatching,
)


# --- Fixtures -------------------------------------------------------------

@pytest.fixture(scope="module")
def jge():
    return Scene.from_pdb(Path("molscene/data/1jge.pdb")).select(model=[1])


@pytest.fixture
def known_transform():
    R = Rotation.from_rotvec(np.deg2rad(37.0) * np.array([1, 1, 0]) / np.sqrt(2)).as_matrix()
    t = np.array([10.0, -5.0, 2.5])
    return Transformation.from_matrix(R, t)


# --- Scene.transform() ----------------------------------------------------

class TestTransform:
    def test_identity_leaves_coords_unchanged(self, jge):
        out = jge.transform(Transformation.identity())
        np.testing.assert_allclose(
            out.get_coordinates().to_numpy(),
            jge.get_coordinates().to_numpy(),
            atol=1e-12,
        )

    def test_round_trip_by_inverse(self, jge, known_transform):
        moved = jge.transform(known_transform)
        back = moved.transform(known_transform.inverse())
        np.testing.assert_allclose(
            back.get_coordinates().to_numpy(),
            jge.get_coordinates().to_numpy(),
            atol=1e-9,
        )

    def test_preserves_other_columns(self, jge, known_transform):
        out = jge.transform(known_transform)
        for col in ("resname", "chain", "resid", "name", "element"):
            assert (out[col].to_numpy() == jge[col].to_numpy()).all()
        assert len(out) == len(jge)

    def test_multiframe(self, jge, known_transform):
        coords = jge.get_coordinates().to_numpy()
        frames = np.stack([coords, coords + [1.0, 2.0, 3.0]], axis=0)
        ms = jge.copy()
        ms.set_coordinate_frames(frames)
        out = ms.transform(known_transform)
        np.testing.assert_allclose(
            out.get_coordinate_frames(),
            known_transform.apply(frames),
            atol=1e-9,
        )

    def test_rejects_raw_matrix(self, jge):
        with pytest.raises(TypeError, match="Transformation"):
            jge.transform(np.eye(3))


# --- Scene.match() --------------------------------------------------------

class TestMatch:
    def test_default_is_order_matching(self, jge):
        left, right = jge.match(jge)
        assert len(left) == len(jge)
        np.testing.assert_array_equal(
            left.get_coordinates().to_numpy(),
            right.get_coordinates().to_numpy(),
        )

    def test_tuple_dispatches_to_column_matching(self, jge):
        # Pre-filter altloc so the (chain,resid,name) key is unique.
        primary = Scene(jge[jge["altloc"].isin(["", "A"])].copy())
        left, right = primary.match(primary, match=("chain", "resid", "name"))
        assert len(left) == len(primary)

    def test_match_instance(self, jge):
        left, right = jge.match(jge, match=ColumnMatching())
        assert len(left) == len(jge)

    def test_callable_dispatched(self, jge):
        called = {}

        def matcher(a, b):
            called["yes"] = True
            return a, b

        left, right = jge.match(jge, match=matcher)
        assert called.get("yes")
        assert len(left) == len(jge)


# --- Scene.compute_transformation() / Scene.superpose() ------------------

class TestCompute:
    def test_self_compute_is_identity(self, jge):
        T = jge.compute_transformation(jge)
        np.testing.assert_allclose(T.rotation, np.eye(3), atol=1e-9)
        np.testing.assert_allclose(T.translation, 0, atol=1e-9)
        assert T.rmsd == pytest.approx(0.0, abs=1e-9)

    def test_recovers_inverse_of_applied_transform(self, jge, known_transform):
        moved = jge.transform(known_transform)
        # Going from moved → original undoes the applied transform.
        T = moved.compute_transformation(jge)
        expected = known_transform.inverse()
        np.testing.assert_allclose(T.rotation, expected.rotation, atol=1e-8)
        np.testing.assert_allclose(T.translation, expected.translation, atol=1e-8)
        assert T.rmsd == pytest.approx(0.0, abs=1e-8)

    def test_superpose_returns_aligned_scene(self, jge, known_transform):
        moved = jge.transform(known_transform)
        aligned = moved.superpose(jge)
        np.testing.assert_allclose(
            aligned.get_coordinates().to_numpy(),
            jge.get_coordinates().to_numpy(),
            atol=1e-7,
        )

    @pytest.mark.requires_molselect
    def test_subset_fit_via_select_composition(self, jge, known_transform):
        # The documented pattern: pre-select, compute, then apply to whole scene.
        moved = jge.transform(known_transform)
        T = moved.select("name CA").compute_transformation(jge.select("name CA"))
        aligned = moved.transform(T)
        np.testing.assert_allclose(
            aligned.get_coordinates().to_numpy(),
            jge.get_coordinates().to_numpy(),
            atol=1e-6,
        )

    def test_column_match_handles_shuffled_input(self, jge, known_transform):
        rng = np.random.default_rng(0)
        order = rng.permutation(len(jge))
        shuffled = Scene(jge.iloc[order].copy()).transform(known_transform)
        T = shuffled.compute_transformation(jge, match=ColumnMatching())
        expected = known_transform.inverse()
        np.testing.assert_allclose(T.rotation, expected.rotation, atol=1e-8)
        np.testing.assert_allclose(T.translation, expected.translation, atol=1e-8)

    def test_sequence_match_handles_residue_gap(self, jge, known_transform):
        chain_a = Scene(jge[jge["chain"] == "A"].copy())
        # Delete a stretch from the mobile side; SequenceMatching must still
        # find the surviving CA correspondences across the gap.
        mask = (chain_a["resid"] >= 100) & (chain_a["resid"] <= 110)
        deleted = Scene(chain_a[~mask].copy()).transform(known_transform)
        T = deleted.compute_transformation(chain_a, match=SequenceMatching())
        expected = known_transform.inverse()
        np.testing.assert_allclose(T.rotation, expected.rotation, atol=1e-7)
        np.testing.assert_allclose(T.translation, expected.translation, atol=1e-7)


# --- Scene.rmsd() ---------------------------------------------------------

class TestRMSD:
    def test_self_rmsd_is_zero(self, jge):
        assert jge.rmsd(jge) == pytest.approx(0.0, abs=1e-12)

    def test_after_translation(self, jge):
        T = Transformation.from_matrix(np.eye(3), [3, 0, 0])
        moved = jge.transform(T)
        assert jge.rmsd(moved) == pytest.approx(3.0, abs=1e-9)

    def test_align_true_undoes_rigid_motion(self, jge, known_transform):
        moved = jge.transform(known_transform)
        assert jge.rmsd(moved) > 1.0
        assert jge.rmsd(moved, align=True) == pytest.approx(0.0, abs=1e-8)

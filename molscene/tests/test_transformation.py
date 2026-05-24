"""Tests for :class:`molscene.Transformation`."""

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from molscene import Transformation
from molscene.geometry import dual_quaternion_from_pose, pose_from_dual_quaternion


# --- Helpers --------------------------------------------------------------

def _random_pose(rng):
    R = Rotation.random(random_state=rng).as_matrix()
    t = rng.normal(scale=5.0, size=3)
    return R, t


# --- Construction & properties ------------------------------------------

class TestConstruction:
    def test_identity(self):
        T = Transformation.identity()
        assert np.allclose(T.rotation, np.eye(3))
        assert np.allclose(T.translation, 0)
        assert np.allclose(T.quaternion, [0, 0, 0, 1])

    def test_from_matrix(self):
        R = Rotation.from_euler('z', 90, degrees=True).as_matrix()
        T = Transformation.from_matrix(R, [1, 2, 3])
        # scipy round-trip introduces ~1e-16 noise, so allow a tiny absolute tolerance.
        np.testing.assert_allclose(T.rotation, R, atol=1e-12)
        np.testing.assert_allclose(T.translation, [1, 2, 3])

    def test_from_quaternion(self):
        # 90° rotation about z, xyzw convention: (0, 0, sin(45°), cos(45°))
        q = np.array([0.0, 0.0, np.sin(np.pi / 4), np.cos(np.pi / 4)])
        T = Transformation.from_quaternion(q, [0, 0, 0])
        np.testing.assert_allclose(T.quaternion, q, atol=1e-12)
        np.testing.assert_allclose(
            T.rotation,
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            atol=1e-12,
        )

    def test_invalid_rotation_shape_raises(self):
        with pytest.raises(ValueError, match="shape"):
            Transformation(rotation=np.eye(4))

    def test_translation_defaults_to_zero(self):
        R = Rotation.from_euler('x', 30, degrees=True).as_matrix()
        T = Transformation(rotation=R)
        np.testing.assert_allclose(T.translation, 0)


# --- Conversions ---------------------------------------------------------

class TestConversions:
    def test_matrix_quaternion_round_trip(self):
        rng = np.random.default_rng(0)
        R, t = _random_pose(rng)
        T = Transformation.from_matrix(R, t)
        q = T.quaternion
        T2 = Transformation.from_quaternion(q, t)
        np.testing.assert_allclose(T2.rotation, R, atol=1e-12)

    def test_dual_quaternion_round_trip(self):
        rng = np.random.default_rng(1)
        for _ in range(5):
            R, t = _random_pose(rng)
            qr, qd = dual_quaternion_from_pose(R, t)
            R2, t2 = pose_from_dual_quaternion(qr, qd)
            np.testing.assert_allclose(R, R2, atol=1e-12)
            np.testing.assert_allclose(t, t2, atol=1e-12)

    def test_from_dual_quaternion(self):
        rng = np.random.default_rng(2)
        R, t = _random_pose(rng)
        qr, qd = dual_quaternion_from_pose(R, t)
        T = Transformation.from_dual_quaternion(qr, qd)
        np.testing.assert_allclose(T.rotation, R, atol=1e-12)
        np.testing.assert_allclose(T.translation, t, atol=1e-12)


# --- Algebra -------------------------------------------------------------

class TestAlgebra:
    def test_apply_single_frame(self):
        T = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [1, 2, 3],
        )
        coords = np.array([[1.0, 0, 0]])
        # rotate (1,0,0) → (0,1,0); + (1,2,3) → (1,3,3)
        np.testing.assert_allclose(T.apply(coords), [[1, 3, 3]], atol=1e-12)

    def test_apply_multi_frame(self):
        T = Transformation.from_matrix(np.eye(3), [0, 0, 5])
        frames = np.zeros((3, 4, 3))
        out = T.apply(frames)
        assert out.shape == (3, 4, 3)
        np.testing.assert_allclose(out[..., 2], 5)

    def test_inverse_undoes_apply(self):
        rng = np.random.default_rng(3)
        R, t = _random_pose(rng)
        T = Transformation.from_matrix(R, t)
        coords = rng.normal(size=(50, 3))
        back = T.inverse().apply(T.apply(coords))
        np.testing.assert_allclose(back, coords, atol=1e-9)

    def test_compose_order_convention(self):
        # (T1 @ T2)(x) == T1(T2(x))
        rng = np.random.default_rng(4)
        R1, t1 = _random_pose(rng)
        R2, t2 = _random_pose(rng)
        T1 = Transformation.from_matrix(R1, t1)
        T2 = Transformation.from_matrix(R2, t2)
        x = rng.normal(size=(20, 3))
        np.testing.assert_allclose(
            (T1 @ T2).apply(x),
            T1.apply(T2.apply(x)),
            atol=1e-12,
        )

    def test_compose_associative(self):
        rng = np.random.default_rng(5)
        T1 = Transformation.from_matrix(*_random_pose(rng))
        T2 = Transformation.from_matrix(*_random_pose(rng))
        T3 = Transformation.from_matrix(*_random_pose(rng))
        x = rng.normal(size=(10, 3))
        np.testing.assert_allclose(
            ((T1 @ T2) @ T3).apply(x),
            (T1 @ (T2 @ T3)).apply(x),
            atol=1e-12,
        )

    def test_matmul_apply_short_form(self):
        T = Transformation.from_matrix(np.eye(3), [1, 0, 0])
        x = np.array([[0., 0, 0]])
        np.testing.assert_allclose(T @ x, [[1, 0, 0]])


# --- Kabsch fit ----------------------------------------------------------

class TestFromKabsch:
    def test_recovers_known_transform(self):
        rng = np.random.default_rng(6)
        R, t = _random_pose(rng)
        P = rng.normal(size=(40, 3)) * 5
        Q = P @ R.T + t
        T = Transformation.from_kabsch(P, Q)
        np.testing.assert_allclose(T.rotation, R, atol=1e-9)
        np.testing.assert_allclose(T.translation, t, atol=1e-9)
        assert T.rmsd == pytest.approx(0.0, abs=1e-9)

    def test_rmsd_present_on_fit_absent_otherwise(self):
        rng = np.random.default_rng(7)
        T_no_fit = Transformation.identity()
        assert T_no_fit.rmsd is None
        T_fit = Transformation.from_kabsch(
            rng.normal(size=(10, 3)), rng.normal(size=(10, 3))
        )
        assert isinstance(T_fit.rmsd, float)


# --- Interpolation -------------------------------------------------------

class TestInterpolation:
    @pytest.mark.parametrize("method", ["slerp", "sclerp"])
    def test_endpoints(self, method):
        rng = np.random.default_rng(8)
        T0 = Transformation.from_matrix(*_random_pose(rng))
        T1 = Transformation.from_matrix(*_random_pose(rng))
        np.testing.assert_allclose(T0.interpolate(T1, 0.0, method=method).rotation, T0.rotation, atol=1e-9)
        np.testing.assert_allclose(T0.interpolate(T1, 0.0, method=method).translation, T0.translation, atol=1e-9)
        np.testing.assert_allclose(T0.interpolate(T1, 1.0, method=method).rotation, T1.rotation, atol=1e-9)
        np.testing.assert_allclose(T0.interpolate(T1, 1.0, method=method).translation, T1.translation, atol=1e-9)

    def test_slerp_midpoint_rotation(self):
        T0 = Transformation.identity()
        T1 = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [10, 0, 0],
        )
        mid = Transformation.slerp(T0, T1, 0.5)
        # 45° about z
        np.testing.assert_allclose(
            Rotation.from_matrix(mid.rotation).as_euler('xyz', degrees=True),
            [0, 0, 45],
            atol=1e-9,
        )
        # decoupled translation lerp: linear in alpha
        np.testing.assert_allclose(mid.translation, [5, 0, 0], atol=1e-9)

    def test_sclerp_pure_rotation_agrees_with_slerp(self):
        T0 = Transformation.identity()
        T1 = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [0, 0, 0],
        )
        for a in (0.25, 0.5, 0.75):
            sl = Transformation.slerp(T0, T1, a)
            sc = Transformation.sclerp(T0, T1, a)
            np.testing.assert_allclose(sl.rotation, sc.rotation, atol=1e-9)
            np.testing.assert_allclose(sl.translation, sc.translation, atol=1e-9)

    def test_sclerp_pure_translation_agrees_with_slerp(self):
        T0 = Transformation.identity()
        T1 = Transformation.from_matrix(np.eye(3), [10, 5, -3])
        for a in (0.25, 0.5, 0.75):
            sl = Transformation.slerp(T0, T1, a)
            sc = Transformation.sclerp(T0, T1, a)
            np.testing.assert_allclose(sl.rotation, sc.rotation, atol=1e-9)
            np.testing.assert_allclose(sl.translation, sc.translation, atol=1e-9)

    def test_sclerp_follows_screw_axis(self):
        # 90° rotation about z + translation (10, 0, 0).
        # Per Chasles, the motion is a pure rotation (no pitch) about the
        # vertical axis through (5, 5, 0). Every intermediate pose should
        # satisfy that same axis.
        T0 = Transformation.identity()
        T1 = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [10, 0, 0],
        )
        # For a fixed point on the screw axis c, (R(α) - I) @ c + t(α) == 0
        # ⇒ t(α) == (I - R(α)) c, for ALL α. Check at several α's.
        c = np.array([5.0, 5.0, 0.0])
        for a in (0.25, 0.5, 0.75):
            sc = Transformation.sclerp(T0, T1, a)
            expected_t = (np.eye(3) - sc.rotation) @ c
            np.testing.assert_allclose(sc.translation, expected_t, atol=1e-9)

    def test_sclerp_screw_with_pitch(self):
        # 90° rotation about z + translation (10, 0, 5). The z-component is
        # along the rotation axis (the pitch); the in-plane component traces
        # the screw axis. Pitch should scale linearly with alpha.
        T0 = Transformation.identity()
        T1 = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [10, 0, 5],
        )
        for a in (0.25, 0.5, 0.75):
            sc = Transformation.sclerp(T0, T1, a)
            assert sc.translation[2] == pytest.approx(5.0 * a, abs=1e-9)

    def test_unknown_method_raises(self):
        T0 = Transformation.identity()
        T1 = Transformation.identity()
        with pytest.raises(ValueError, match="unknown"):
            T0.interpolate(T1, 0.5, method="nope")


# --- repr ---------------------------------------------------------------

class TestRepr:
    def test_repr_contains_angle_and_translation(self):
        T = Transformation.from_matrix(
            Rotation.from_euler('z', 90, degrees=True).as_matrix(),
            [1, 2, 3],
        )
        txt = repr(T)
        assert "90" in txt
        assert "1.000" in txt
        assert "2.000" in txt
        assert "3.000" in txt

"""Contracts for the array-level geometry kernels in :mod:`molscene.geometry`."""
import numpy as np
import pytest

from molscene import contacts, geometry


def random_rotation(rng):
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    return q if np.linalg.det(q) > 0 else -q


def test_local_frame_is_right_handed_and_follows_its_convention():
    """Feature: local_frame puts e1 along primary, e3 in-plane toward secondary, det = +1."""
    rng = np.random.default_rng(0)
    rotation, origin = random_rotation(rng), rng.normal(size=3) * 10
    local = dict(primary=[1.5, 0, 0], secondary=[-0.4, 0, 1.3])
    world = {k: origin + rotation @ np.asarray(v, float) for k, v in local.items()}
    basis, got_origin = geometry.local_frame(origin, world["primary"], world["secondary"])
    np.testing.assert_allclose(got_origin, origin)
    np.testing.assert_allclose(basis, rotation, atol=1e-12)
    assert np.isclose(np.linalg.det(basis), 1.0)


def test_local_frame_maps_its_own_points_back_to_local_coordinates():
    """Feature: (x - origin) @ basis expresses structure points in the frame."""
    rng = np.random.default_rng(1)
    o, p, s = rng.normal(size=(3, 3))
    basis, origin = geometry.local_frame(o, p, s)
    local = (np.stack([o, p, s]) - origin) @ basis
    np.testing.assert_allclose(local[0], 0, atol=1e-12)
    np.testing.assert_allclose(local[1, 1:], 0, atol=1e-12)
    assert local[1, 0] > 0
    assert abs(local[2, 1]) < 1e-12 and local[2, 2] > 0


def test_local_frame_broadcasts_over_leading_axes():
    """Feature: stacks of residues or frames are framed in one call, like one at a time."""
    rng = np.random.default_rng(2)
    o, p, s = (rng.normal(size=(4, 5, 3)) for _ in range(3))
    basis, origin = geometry.local_frame(o, p, s)
    assert basis.shape == (4, 5, 3, 3)
    one, _ = geometry.local_frame(o[2, 3], p[2, 3], s[2, 3])
    np.testing.assert_allclose(basis[2, 3], one)


@pytest.mark.parametrize("primary,secondary", [([0, 0, 0], [1, 0, 0]),
                                               ([0, 0, 1], [0, 0, 2]),
                                               ([0, 0, np.nan], [1, 0, 0])])
def test_local_frame_rejects_degenerate_points(primary, secondary):
    """Feature: coincident, collinear or non-finite points raise instead of giving NaN axes."""
    with pytest.raises(ValueError, match="noncollinear"):
        geometry.local_frame([0, 0, 0], primary, secondary)


def test_virtual_cb_offset_is_expressed_in_the_right_handed_frame():
    """Feature: CB_OFFSET places an L-amino-acid CB in the right-handed backbone frame."""
    n, ca, c = np.array([[-0.53, 1.36, 0.0], [0.0, 0.0, 0.0], [1.52, 0.0, 0.0]])
    cb = contacts.cb_from_backbone(n[None], ca[None], c[None])[0]
    assert np.dot(np.cross(n - ca, c - ca), cb - ca) > 0
    assert np.isclose(np.linalg.norm(cb - ca), np.linalg.norm(contacts.CB_OFFSET))


@pytest.mark.parametrize("shared", [False, True])
def test_apply_transform_uses_each_frames_own_rotation(shared):
    """Feature: per-frame R and t transform the matching frame, or a shared frame per rotation."""
    rng = np.random.default_rng(3)
    coords = rng.normal(size=(3, 7, 3))
    rotations = np.stack([random_rotation(rng) for _ in range(3)])
    translations = rng.normal(size=(3, 3))
    if shared:
        coords = coords[0]
    expected = np.stack([(coords if shared else coords[i]) @ rotations[i].T + translations[i]
                         for i in range(3)])
    np.testing.assert_allclose(geometry.apply_transform(coords, rotations, translations),
                               expected)


def test_apply_transform_keeps_single_precision():
    """Feature: float32 coordinates are transformed in float32, not upcast."""
    coords = np.ones((4, 3), dtype=np.float32)
    assert geometry.apply_transform(coords, np.eye(3), np.zeros(3)).dtype == np.float32
    assert geometry.apply_transform(coords.astype(int), np.eye(3)).dtype == float


def test_apply_transform_rejects_mismatched_shapes():
    """Feature: malformed coordinates, rotations or translations raise."""
    with pytest.raises(ValueError):
        geometry.apply_transform(np.zeros((4, 2)), np.eye(3))
    with pytest.raises(ValueError):
        geometry.apply_transform(np.zeros((4, 3)), np.eye(2))
    with pytest.raises(ValueError):
        geometry.apply_transform(np.zeros((4, 3)), np.eye(3), np.zeros(2))

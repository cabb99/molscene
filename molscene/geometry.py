"""
Geometric primitives used by :class:`molscene.Scene` and
:class:`molscene.Transformation`.

This module hosts three groups of utilities:

1. ``kabsch`` / ``apply_transform`` — least-squares rigid-body alignment and
   the corresponding coordinate transform.
2. Quaternion helpers using the scipy ``xyzw`` convention (Hamilton product,
   conjugate).
3. Dual-quaternion conversion + screw-linear interpolation, used by
   :meth:`molscene.Transformation.interpolate` when ``method='sclerp'``.

The dual-quaternion representation expresses a rigid transformation as a pair
``(q_real, q_dual)`` of unit quaternions linked by ``q_dual = 0.5 * (t, 0) *
q_real``, where ``(t, 0)`` is the translation packed as a pure quaternion.
This representation makes screw-linear interpolation (ScLERP) natural and
keeps rotation and translation tightly coupled.
"""

from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Kabsch + transform application
# ---------------------------------------------------------------------------

def kabsch(mobile: np.ndarray, reference: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute the rigid-body transformation that least-squares aligns ``mobile``
    onto ``reference`` using the Kabsch algorithm.

    The returned rotation :math:`R` and translation :math:`t` satisfy

    .. math::
        \\hat{x}_i = R\\,x_i + t \\approx y_i

    for paired points :math:`x_i` (mobile) and :math:`y_i` (reference).

    Parameters
    ----------
    mobile : ndarray, shape (N, 3)
        Coordinates to be moved.
    reference : ndarray, shape (N, 3)
        Target coordinates. Must have the same shape as ``mobile`` and the
        rows must correspond atom-for-atom.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Proper rotation matrix (``det(R) = +1``).
    t : ndarray, shape (3,)
        Translation vector.
    rmsd : float
        Root-mean-square deviation between ``R @ mobile + t`` and ``reference``.

    Raises
    ------
    ValueError
        If the inputs have incompatible shapes or fewer than three points.
    """
    P = np.asarray(mobile, dtype=float)
    Q = np.asarray(reference, dtype=float)

    if P.shape != Q.shape:
        raise ValueError(
            f"mobile and reference must have the same shape, got {P.shape} and {Q.shape}"
        )
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError(f"inputs must have shape (N, 3), got {P.shape}")
    if P.shape[0] < 3:
        raise ValueError(
            f"at least 3 points are required for a well-defined rotation, got {P.shape[0]}"
        )

    centroid_P = P.mean(axis=0)
    centroid_Q = Q.mean(axis=0)
    Pc = P - centroid_P
    Qc = Q - centroid_Q

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)

    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T

    t = centroid_Q - R @ centroid_P

    diff = Pc @ R.T - Qc
    rmsd = float(np.sqrt(np.mean(np.einsum('ij,ij->i', diff, diff))))

    return R, t, rmsd


def apply_transform(coords: np.ndarray, R: np.ndarray, t: np.ndarray = None) -> np.ndarray:
    """
    Apply a rigid-body transformation ``x' = R x + t`` to a set of coordinates.

    Parameters
    ----------
    coords : ndarray
        Either a single frame of shape ``(N, 3)`` or a stack of frames of
        shape ``(F, N, 3)``.
    R : ndarray, shape (3, 3) or (F, 3, 3)
        Rotation matrix, or one rotation per frame. Per-frame rotations applied
        to a single ``(N, 3)`` frame give one transformed copy per rotation.
    t : ndarray, shape (3,) or (F, 3), optional
        Translation vector, or one per frame. Defaults to zero.

    Returns
    -------
    ndarray
        Transformed coordinates, ``(N, 3)`` or ``(F, N, 3)``. Floating-point
        input keeps its precision, so ``float32`` coordinates stay ``float32``.
    """
    coords = np.asarray(coords)
    if coords.dtype.kind != "f":
        coords = coords.astype(float)
    if coords.ndim not in (2, 3) or coords.shape[-1] != 3:
        raise ValueError(f"coords must have shape (N, 3) or (F, N, 3), got {coords.shape}")
    R = np.asarray(R, dtype=coords.dtype)
    if R.ndim not in (2, 3) or R.shape[-2:] != (3, 3):
        raise ValueError(f"R must have shape (3, 3) or (F, 3, 3), got {R.shape}")
    t = np.zeros(3, dtype=coords.dtype) if t is None else np.asarray(t, dtype=coords.dtype)
    if t.shape not in ((3,), R.shape[:-2] + (3,)):
        raise ValueError(f"t must have shape (3,) or match R's frames, got {t.shape}")
    return np.matmul(coords, np.swapaxes(R, -1, -2)) + t[..., None, :]


# ---------------------------------------------------------------------------
# Local frames and pseudo-atom placement
# ---------------------------------------------------------------------------

def local_frame(origin: np.ndarray, primary: np.ndarray,
                secondary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Build a right-handed orthonormal frame from three points.

    The basis columns are ``e1`` along ``primary - origin``, ``e2`` normal to the
    plane of the three points (along ``(secondary - origin) x e1``), and
    ``e3 = e1 x e2``, which lies in that plane on the side of ``secondary``. So
    ``origin + basis @ v`` maps local coordinates ``v`` into the structure, and
    ``(x - origin) @ basis`` maps structure coordinates ``x`` back.

    The basis is orthonormal, so coefficients expressed in it are true lengths and
    a placement built on it is independent of the coordinate units.  Frames built
    from raw (unnormalized) bond vectors do not have this property — their
    cross-product axis scales as length squared, so coefficients fitted in Ångström
    silently misplace atoms when the coordinates are in nanometres.

    Parameters
    ----------
    origin, primary, secondary : numpy.ndarray
        ``(..., 3)`` arrays of the frame origin and its two reference points;
        leading axes (residues, frames) broadcast.

    Returns
    -------
    basis : numpy.ndarray
        ``(..., 3, 3)`` array whose columns are the frame axes.
    origin : numpy.ndarray
        ``(..., 3)`` array of frame origins.

    Raises
    ------
    ValueError
        If the three points are not finite, coincide, or are collinear.
    """
    def unit(v):
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
            raise ValueError("frame points must be finite, distinct and noncollinear")
        return v / norm

    origin = np.asarray(origin, dtype=float)
    e1 = unit(np.asarray(primary, dtype=float) - origin)
    e2 = unit(np.cross(np.asarray(secondary, dtype=float) - origin, e1))
    return np.stack([e1, e2, np.cross(e1, e2)], axis=-1), origin


def place_from_frame(basis: np.ndarray, origin: np.ndarray,
                     offset) -> np.ndarray:
    """Place a point at ``origin + basis @ offset``.

    ``offset`` is either a single ``(3,)`` coefficient triple applied to every
    frame, or an ``(..., 3)`` array of per-frame coefficients.
    """
    offset = np.asarray(offset, dtype=float)
    return np.einsum('...ij,...j->...i', basis, offset) + origin


# ---------------------------------------------------------------------------
# Quaternion helpers (scipy xyzw convention)
# ---------------------------------------------------------------------------

def quat_multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` in scipy ``(x, y, z, w)`` convention."""
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array([
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
        aw * bw - ax * bx - ay * by - az * bz,
    ])


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Quaternion conjugate in ``(x, y, z, w)`` convention."""
    return np.array([-q[0], -q[1], -q[2], q[3]])


# ---------------------------------------------------------------------------
# Dual-quaternion utilities
# ---------------------------------------------------------------------------

def dual_quaternion_from_pose(R: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert a rigid pose ``(R, t)`` into a unit dual quaternion
    ``(q_real, q_dual)``.

    Parameters
    ----------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    t : ndarray, shape (3,)
        Translation vector.

    Returns
    -------
    (q_real, q_dual) : tuple of ndarray
        Each component is a length-4 array in ``(x, y, z, w)`` order.
    """
    from scipy.spatial.transform import Rotation
    R = np.asarray(R, dtype=float)
    t = np.asarray(t, dtype=float).reshape(3)
    q_real = Rotation.from_matrix(R).as_quat()
    t_quat = np.array([t[0], t[1], t[2], 0.0])
    q_dual = 0.5 * quat_multiply(t_quat, q_real)
    return q_real, q_dual


def pose_from_dual_quaternion(
    q_real: np.ndarray, q_dual: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inverse of :func:`dual_quaternion_from_pose`.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    t : ndarray, shape (3,)
        Translation vector.
    """
    from scipy.spatial.transform import Rotation
    q_real = np.asarray(q_real, dtype=float).reshape(4)
    q_dual = np.asarray(q_dual, dtype=float).reshape(4)
    t_quat = 2.0 * quat_multiply(q_dual, quat_conjugate(q_real))
    return Rotation.from_quat(q_real).as_matrix(), t_quat[:3]


# ---------------------------------------------------------------------------
# Screw-linear interpolation (ScLERP)
# ---------------------------------------------------------------------------

def screw_interpolate(
    R1: np.ndarray, t1: np.ndarray,
    R2: np.ndarray, t2: np.ndarray,
    alpha: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Screw-linear interpolation between two rigid poses.

    Treats the motion from ``(R1, t1)`` to ``(R2, t2)`` as a single screw
    motion (Chasles' theorem). The interpolated pose at fraction
    ``alpha`` ∈ [0, 1] lies on the same screw axis and advances proportionally
    along it. Equivalent to dual-quaternion ScLERP but implemented in matrix
    form to avoid numerical pitfalls at the singular limits.

    Parameters
    ----------
    R1, t1 : ndarray
        Starting pose. ``alpha == 0`` returns this pose exactly.
    R2, t2 : ndarray
        Ending pose. ``alpha == 1`` returns this pose exactly.
    alpha : float

    Returns
    -------
    R_alpha : ndarray, shape (3, 3)
    t_alpha : ndarray, shape (3,)

    Notes
    -----
    Compared to decoupled rotation slerp + translation lerp, ScLERP keeps the
    two channels in lock-step: a moving end-effector drags every intermediate
    frame along the *same* helical path, which is exactly the behaviour the
    CaMKII linker morph needs.
    """
    from scipy.spatial.transform import Rotation

    R1 = np.asarray(R1, dtype=float)
    R2 = np.asarray(R2, dtype=float)
    t1 = np.asarray(t1, dtype=float).reshape(3)
    t2 = np.asarray(t2, dtype=float).reshape(3)

    # Compute the relative pose taking (R1, t1) to (R2, t2):
    # (R_rel, t_rel) such that (R_rel @ x + t_rel) maps a point expressed in
    # frame 1 to the same point expressed in frame 2.
    r1 = Rotation.from_matrix(R1)
    r2 = Rotation.from_matrix(R2)
    r_rel = r1.inv() * r2                  # rotation in frame 1
    t_rel = r1.inv().apply(t2 - t1)        # translation in frame 1

    rotvec = r_rel.as_rotvec()
    angle = np.linalg.norm(rotvec)

    if angle < 1e-9:
        # Pure translation: screw axis is at infinity; ScLERP reduces to
        # rotation-fixed translation lerp.
        t_alpha_local = alpha * t_rel
        R_alpha_local = np.eye(3)
    else:
        axis = rotvec / angle
        # Decompose t_rel into components parallel and perpendicular to axis.
        t_par_scalar = float(np.dot(t_rel, axis))
        t_par = t_par_scalar * axis
        t_perp = t_rel - t_par

        # The screw axis offset (point on axis closest to origin in frame 1).
        # Derived from the screw motion identity
        #   (I - R) c = t_perp,  with  R = rot(axis, angle).
        cot_half = 1.0 / np.tan(angle / 2.0) if abs(np.tan(angle / 2.0)) > 1e-12 else 0.0
        c = 0.5 * t_perp + 0.5 * cot_half * np.cross(axis, t_perp)

        # Interpolated rotation and translation in frame 1.
        R_alpha_local = Rotation.from_rotvec(alpha * rotvec).as_matrix()
        t_perp_alpha = (np.eye(3) - R_alpha_local) @ c
        t_par_alpha = alpha * t_par
        t_alpha_local = t_perp_alpha + t_par_alpha

    # Convert the interpolated local pose back into the lab frame:
    # the lab-frame interpolated pose is (R1, t1) ∘ (R_alpha_local, t_alpha_local).
    R_alpha = R1 @ R_alpha_local
    t_alpha = R1 @ t_alpha_local + t1
    return R_alpha, t_alpha

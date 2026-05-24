"""
First-class rigid-body :class:`Transformation` object used by
:class:`molscene.Scene`.

A ``Transformation`` carries a rotation and a translation and behaves as a
mathematical object: it can be applied to coordinates, inverted, composed
with another transformation, and interpolated. Two interpolation modes are
available:

* ``method='slerp'`` — slerp the rotation (via :class:`scipy.spatial.transform.Slerp`)
  and lerp the translation independently. Cheap; matches the textbook
  "decoupled" interpolation.
* ``method='sclerp'`` — dual-quaternion screw-linear interpolation. Treats
  the transformation as a single screw motion (Chasles' theorem); rotation
  and translation co-evolve along the same helical path. This is the default
  because, when a moving end-effector needs to propagate a smooth twist
  through an intervening region (e.g. a flexible linker), it yields a
  visually and physically more plausible morph than decoupled slerp+lerp.

Conventions follow ``scipy.spatial.transform.Rotation``: ``(R1 * R2).apply(x)
== R1.apply(R2.apply(x))``, so ``T1 @ T2`` applies ``T2`` first and then
``T1``.
"""

from typing import Tuple, Union

import numpy as np

from .geometry import (
    apply_transform as _apply_transform,
    dual_quaternion_from_pose,
    kabsch,
    screw_interpolate,
)


_RotationLike = Union[np.ndarray, "Rotation", None]


class Transformation:
    """A rigid-body transformation ``x -> R @ x + t``."""

    __slots__ = ("_rot", "_t", "rmsd")

    def __init__(self, rotation: _RotationLike = None, translation=None):
        from scipy.spatial.transform import Rotation

        if rotation is None:
            self._rot = Rotation.identity()
        elif isinstance(rotation, Rotation):
            self._rot = rotation
        else:
            arr = np.asarray(rotation, dtype=float)
            if arr.shape == (3, 3):
                self._rot = Rotation.from_matrix(arr)
            elif arr.shape == (4,):
                self._rot = Rotation.from_quat(arr)
            else:
                raise ValueError(
                    f"rotation must be a Rotation, (3,3) matrix, or length-4 quaternion; "
                    f"got shape {arr.shape}"
                )

        if translation is None:
            self._t = np.zeros(3)
        else:
            self._t = np.asarray(translation, dtype=float).reshape(3)

        # ``rmsd`` is meaningful only when constructed via :meth:`from_kabsch`;
        # otherwise it stays ``None``.
        self.rmsd = None

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------
    @classmethod
    def identity(cls) -> "Transformation":
        return cls()

    @classmethod
    def from_matrix(cls, R, t=None) -> "Transformation":
        return cls(rotation=R, translation=t)

    @classmethod
    def from_quaternion(cls, q, t=None) -> "Transformation":
        """Construct from a quaternion in scipy ``(x, y, z, w)`` order."""
        return cls(rotation=q, translation=t)

    @classmethod
    def from_dual_quaternion(cls, q_real, q_dual) -> "Transformation":
        from .geometry import pose_from_dual_quaternion
        R, t = pose_from_dual_quaternion(q_real, q_dual)
        return cls.from_matrix(R, t)

    @classmethod
    def from_kabsch(cls, mobile, reference) -> "Transformation":
        """Least-squares fit of ``mobile`` onto ``reference``.

        The returned transformation has an ``rmsd`` attribute set to the
        residual after fitting.
        """
        R, t, rmsd = kabsch(mobile, reference)
        out = cls.from_matrix(R, t)
        out.rmsd = rmsd
        return out

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------
    @property
    def rotation(self) -> np.ndarray:
        """3×3 rotation matrix."""
        return self._rot.as_matrix()

    @property
    def translation(self) -> np.ndarray:
        """Length-3 translation vector."""
        return self._t.copy()

    @property
    def quaternion(self) -> np.ndarray:
        """Length-4 quaternion in scipy ``(x, y, z, w)`` order."""
        return self._rot.as_quat()

    @property
    def dual_quaternion(self) -> Tuple[np.ndarray, np.ndarray]:
        """``(q_real, q_dual)`` representation; see :mod:`molscene.geometry`."""
        return dual_quaternion_from_pose(self.rotation, self.translation)

    # ------------------------------------------------------------------
    # Algebra
    # ------------------------------------------------------------------
    def apply(self, coords) -> np.ndarray:
        """Apply to ``(N, 3)`` or ``(F, N, 3)`` coordinates."""
        return _apply_transform(coords, self.rotation, self._t)

    def inverse(self) -> "Transformation":
        r_inv = self._rot.inv()
        t_inv = -r_inv.apply(self._t)
        return Transformation(r_inv, t_inv)

    def compose(self, other: "Transformation") -> "Transformation":
        """``self ∘ other``: applying the result equals applying ``other``
        first and then ``self``.
        """
        new_rot = self._rot * other._rot
        new_t = self._rot.apply(other._t) + self._t
        return Transformation(new_rot, new_t)

    def __matmul__(self, other):
        if isinstance(other, Transformation):
            return self.compose(other)
        return self.apply(other)

    def __eq__(self, other) -> bool:
        if not isinstance(other, Transformation):
            return NotImplemented
        return (
            np.allclose(self.rotation, other.rotation)
            and np.allclose(self.translation, other.translation)
        )

    def __repr__(self) -> str:
        rotvec = self._rot.as_rotvec()
        angle = float(np.linalg.norm(rotvec))
        axis = rotvec / angle if angle > 1e-12 else np.zeros(3)
        return (
            f"Transformation(angle={np.degrees(angle):.3f}°, "
            f"axis=[{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}], "
            f"t=[{self._t[0]:.3f}, {self._t[1]:.3f}, {self._t[2]:.3f}])"
        )

    # ------------------------------------------------------------------
    # Interpolation
    # ------------------------------------------------------------------
    def interpolate(
        self,
        other: "Transformation",
        alpha: float,
        method: str = "sclerp",
    ) -> "Transformation":
        """Interpolate from ``self`` (alpha=0) to ``other`` (alpha=1).

        Parameters
        ----------
        other : Transformation
        alpha : float
            Interpolation fraction. ``0`` returns ``self``, ``1`` returns
            ``other``; intermediate values produce a blended pose.
        method : {'sclerp', 'slerp'}
            ``'sclerp'`` (default) — dual-quaternion screw-linear interpolation
            that keeps rotation and translation co-evolving along the same
            helical axis. ``'slerp'`` — slerp the rotation and lerp the
            translation independently.

        Returns
        -------
        Transformation
        """
        if method == "sclerp":
            R, t = screw_interpolate(
                self.rotation, self.translation,
                other.rotation, other.translation,
                alpha,
            )
            return Transformation.from_matrix(R, t)
        if method == "slerp":
            from scipy.spatial.transform import Rotation, Slerp
            rots = Rotation.concatenate([self._rot, other._rot])
            slerp = Slerp([0.0, 1.0], rots)
            r_alpha = slerp(alpha)
            t_alpha = (1.0 - alpha) * self._t + alpha * other._t
            return Transformation(r_alpha, t_alpha)
        raise ValueError(f"unknown interpolation method: {method!r}")

    # Convenience class-method aliases.
    @classmethod
    def slerp(cls, a: "Transformation", b: "Transformation", alpha: float) -> "Transformation":
        return a.interpolate(b, alpha, method="slerp")

    @classmethod
    def sclerp(cls, a: "Transformation", b: "Transformation", alpha: float) -> "Transformation":
        return a.interpolate(b, alpha, method="sclerp")

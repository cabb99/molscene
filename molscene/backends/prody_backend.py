"""ProDy ``AtomGroup`` <-> Scene converter (optional dependency ``prody``).

Module is named ``prody_backend`` (not ``prody``) so ``import prody`` inside it
resolves to the third-party package, never to this module.
"""

import numpy as np
import pandas as pd

from .registry import BackendRegistry


def _require_prody():
    try:
        import prody
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The ProDy backend requires the optional 'prody' package. "
            "Install it with:  pip install molscene[prody]") from exc
    return prody


# ProDy getter -> canonical molscene column (coordinates handled separately).
_FROM_PRODY = {
    "getSerials": "serial", "getNames": "name", "getAltlocs": "altloc",
    "getResnames": "resname", "getChids": "chain", "getResnums": "resid",
    "getIcodes": "icode", "getOccupancies": "occupancy", "getBetas": "beta",
    "getElements": "element", "getCharges": "charge",
}


@BackendRegistry.register("prody")
class ProdyBackend:

    @staticmethod
    def matches(obj):
        return (type(obj).__module__.split(".")[0] == "prody"
                and hasattr(obj, "getCoords"))

    @staticmethod
    def from_object(atomgroup):
        """ProDy ``AtomGroup``/``Selection`` -> ``(atoms_DataFrame, meta)``."""
        _require_prody()
        coords = atomgroup.getCoords()
        if coords is None:
            raise ValueError("ProDy object has no coordinates")
        coords = np.asarray(coords, dtype=float)
        data = {"x": coords[:, 0], "y": coords[:, 1], "z": coords[:, 2]}
        for getter, col in _FROM_PRODY.items():
            fn = getattr(atomgroup, getter, None)
            if fn is None:
                continue
            vals = fn()
            if vals is not None:
                data[col] = np.asarray(vals)
        return pd.DataFrame(data), {}

    @staticmethod
    def to_object(scene):
        """Scene -> ProDy ``AtomGroup``."""
        prody = _require_prody()
        ag = prody.AtomGroup("molscene")
        ag.setCoords(scene[["x", "y", "z"]].to_numpy(dtype=float))

        def _set(method, column, cast):
            if column not in scene.columns:
                return
            try:
                getattr(ag, method)(cast(scene[column]))
            except Exception:  # pragma: no cover - tolerate odd dtypes
                pass

        as_str = lambda s: s.astype(str).to_numpy()
        as_int = lambda s: pd.to_numeric(s, errors="coerce").fillna(0).astype(int).to_numpy()
        as_float = lambda s: pd.to_numeric(s, errors="coerce").fillna(0.0).astype(float).to_numpy()

        _set("setNames", "name", as_str)
        _set("setResnames", "resname", as_str)
        _set("setResnums", "resid", as_int)
        _set("setChids", "chain", as_str)
        _set("setIcodes", "icode", as_str)
        _set("setAltlocs", "altloc", as_str)
        _set("setElements", "element", as_str)
        _set("setBetas", "beta", as_float)
        _set("setOccupancies", "occupancy", as_float)
        _set("setSerials", "serial", as_int)
        _set("setCharges", "charge", as_float)
        return ag

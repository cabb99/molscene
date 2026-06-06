"""mdtraj ``Trajectory`` <-> Scene converter (optional dependency ``mdtraj``).

Multi-frame trajectories map to/from ``Scene``'s ``coordinate_frames``.  mdtraj
works in nanometres; molscene in Angstrom, so coordinates are scaled by 10.
Module is named ``mdtraj_backend`` so ``import mdtraj`` resolves to the package.
"""

import string

import numpy as np
import pandas as pd

from .registry import BackendRegistry


def _require_mdtraj():
    try:
        import mdtraj
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The mdtraj backend requires the optional 'mdtraj' package. "
            "Install it with:  pip install molscene[mdtraj]") from exc
    return mdtraj


def _chain_letter(index):
    return string.ascii_uppercase[index % 26]


def _md_element(mdtraj, symbol):
    sym = str(symbol).strip()
    if not sym:
        return mdtraj.element.virtual
    sym = sym[0].upper() + sym[1:].lower()
    try:
        return mdtraj.element.get_by_symbol(sym)
    except KeyError:
        return mdtraj.element.virtual


@BackendRegistry.register("mdtraj")
class MDTrajBackend:

    @staticmethod
    def matches(obj):
        return (type(obj).__module__.split(".")[0] == "mdtraj"
                and type(obj).__name__ == "Trajectory")

    @staticmethod
    def from_object(traj):
        """mdtraj ``Trajectory`` -> ``(atoms_DataFrame, meta)`` (nm -> Angstrom)."""
        _require_mdtraj()
        top = traj.topology
        rows = []
        for atom in top.atoms:
            res = atom.residue
            chain = res.chain
            chain_id = getattr(chain, "chain_id", None) or _chain_letter(chain.index)
            rows.append(dict(
                recname="ATOM",
                serial=atom.serial if atom.serial is not None else atom.index + 1,
                name=atom.name, resname=res.name, chain=str(chain_id),
                resid=int(res.resSeq), icode="",
                element=(atom.element.symbol if atom.element is not None else "")))
        atoms = pd.DataFrame(rows)

        xyz = np.asarray(traj.xyz, dtype=float) * 10.0   # nm -> Angstrom
        atoms["x"], atoms["y"], atoms["z"] = xyz[0, :, 0], xyz[0, :, 1], xyz[0, :, 2]
        meta = {"source_format": "mdtraj"}
        if xyz.shape[0] > 1:
            meta["coordinate_frames"] = xyz
        return atoms, meta

    @staticmethod
    def to_object(scene):
        """Scene -> mdtraj ``Trajectory`` (Angstrom -> nm)."""
        mdtraj = _require_mdtraj()
        top = mdtraj.Topology()
        chains, residues, df = {}, {}, pd.DataFrame(scene)
        for row in df.itertuples(index=False):
            ch_id = str(getattr(row, "chain", "A"))
            if ch_id not in chains:
                chains[ch_id] = top.add_chain()
            resid = int(getattr(row, "resid", 1))
            key = (ch_id, resid, str(getattr(row, "icode", "")))
            if key not in residues:
                residues[key] = top.add_residue(
                    str(getattr(row, "resname", "")), chains[ch_id], resSeq=resid)
            top.add_atom(str(row.name), _md_element(mdtraj, getattr(row, "element", "")),
                         residues[key])

        frames = (getattr(scene, "_meta", {}) or {}).get("coordinate_frames")
        if frames is not None:
            xyz = np.asarray(frames, dtype=float) / 10.0
        else:
            xyz = (df[["x", "y", "z"]].to_numpy(dtype=float) / 10.0)[None, :, :]
        return mdtraj.Trajectory(xyz=xyz, topology=top)

"""PDBFixer <-> Scene converter + isolated structure repair (optional deps
``pdbfixer`` and ``openmm``).

Provides:
- ``from_object(fixer)``: a cleaned ``PDBFixer`` object -> ``(atoms_DataFrame, meta)``.
- ``clean(...)``: build + clean a ``PDBFixer`` inline (mirrors the old from_fixPDB).
- ``repair_pdb`` / ``repair_pdbs``: run PDBFixer in a **child process** so OpenMM's
  C-level memory is reclaimed on exit (important for batch cleaning).
"""

import multiprocessing
import tempfile
from pathlib import Path

import pandas

from .registry import BackendRegistry


def _require_fixer():
    try:
        import pdbfixer  # noqa: F401
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "PDB repair requires the optional 'pdbfixer' (and 'openmm') packages. "
            "Install with:  pip install molscene[fixer]   or   "
            "conda install -c conda-forge pdbfixer openmm") from exc


_COLS = ['recname', 'serial', 'name', 'altloc', 'resname', 'chain', 'resid',
         'icode', 'x', 'y', 'z', 'occupancy', 'beta', 'element', 'charge']


@BackendRegistry.register("pdbfixer")
class PDBFixerBackend:

    @staticmethod
    def matches(obj):
        return type(obj).__name__ == "PDBFixer" or \
            type(obj).__module__.split(".")[0] == "pdbfixer"

    @staticmethod
    def from_object(fixer):
        """A ``PDBFixer`` (openmm topology + positions) -> ``(atoms_DataFrame, meta)``."""
        import pdbfixer
        data = []
        for atom, pos in zip(fixer.topology.atoms(), fixer.positions):
            residue = atom.residue
            pos = pos.value_in_unit(pdbfixer.pdbfixer.unit.angstrom)
            data.append(dict(zip(_COLS, [
                'ATOM', int(atom.id), atom.name, '', residue.name, residue.chain.id,
                int(residue.id), '', pos[0], pos[1], pos[2], 0, 0,
                atom.element.symbol, ''])))
        atoms = pandas.DataFrame(data)[_COLS]
        atoms.index = atoms['serial']
        return atoms, {}

    # no to_object: PDBFixer is an input/cleanup format, not an export target.


def clean(filename=None, pdbfile=None, pdbxfile=None, url=None, pdbid=None):
    """Build and clean a ``PDBFixer`` inline (replace nonstandard residues, remove
    heterogens, add missing atoms/hydrogens). Returns the ``PDBFixer`` object."""
    import pdbfixer
    fixer = pdbfixer.PDBFixer(
        filename=str(filename) if filename is not None else None,
        pdbfile=str(pdbfile) if pdbfile is not None else None,
        pdbxfile=str(pdbxfile) if pdbxfile is not None else None,
        url=str(url) if url is not None else None,
        pdbid=str(pdbid) if pdbid is not None else None)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues):
        chain_tmp = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain_tmp.residues())):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    return fixer


def _repair_worker(pdb_file, chain, cleaned_path):
    """Run PDBFixer in a child process and write a cleaned PDB (module-level so it
    is picklable by :mod:`multiprocessing`)."""
    import pdbfixer
    PDBFile = pdbfixer.pdbfixer.app.PDBFile
    fixer = pdbfixer.PDBFixer(filename=str(pdb_file))
    if chain is not None:
        chains = list(fixer.topology.chains())
        to_remove = [i for i, c in enumerate(chains) if c.id not in chain]
        fixer.removeChains(to_remove)
    fixer.findMissingResidues()
    chains = list(fixer.topology.chains())
    for key in list(fixer.missingResidues):
        chain_obj = chains[key[0]]
        if key[1] == 0 or key[1] == len(list(chain_obj.residues())):
            del fixer.missingResidues[key]
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    try:
        fixer.addMissingAtoms()
    except Exception:
        pass
    fixer.addMissingHydrogens(7.0)
    with open(cleaned_path, "w") as fh:
        PDBFile.writeFile(fixer.topology, fixer.positions, fh)


def repair_pdb(pdb_file, chain=None, output=None) -> Path:
    """Repair a PDB/CIF with PDBFixer in an isolated child process; returns the
    cleaned-PDB path (``<tempdir>/<stem>_cleaned.pdb`` by default)."""
    _require_fixer()
    pdb_file = Path(pdb_file)
    if output is None:
        output = Path(tempfile.gettempdir()) / f"{pdb_file.stem}_cleaned.pdb"
    output = Path(output)
    proc = multiprocessing.Process(
        target=_repair_worker, args=(str(pdb_file), chain, str(output)))
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise RuntimeError(f"PDB repair failed for {pdb_file.name} (exit code {proc.exitcode})")
    return output


def repair_pdbs(jobs, output_dir=None) -> list:
    """Repair many ``(pdb_file, chain)`` jobs in parallel, each in its own process."""
    _require_fixer()
    out_dir = Path(output_dir) if output_dir is not None else Path(tempfile.gettempdir())
    procs, outputs = [], []
    for pdb_file, chain in jobs:
        pdb_file = Path(pdb_file)
        out = out_dir / f"{pdb_file.stem}_cleaned.pdb"
        outputs.append(out)
        procs.append((multiprocessing.Process(
            target=_repair_worker, args=(str(pdb_file), chain, str(out))), pdb_file))
    for proc, _ in procs:
        proc.start()
    for proc, pdb_file in procs:
        proc.join()
        if proc.exitcode != 0:
            raise RuntimeError(
                f"PDB repair failed for {Path(pdb_file).name} (exit code {proc.exitcode})")
    return outputs

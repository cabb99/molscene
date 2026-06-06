"""PDB format reader and writer (Scene-agnostic).

``read_pdb`` returns ``(atoms_DataFrame, meta)``; ``write_pdb`` takes a scene (used
as a DataFrame) and supports multi-frame output via ``_meta['coordinate_frames']``.
"""

import logging
from pathlib import Path

import numpy as np
import pandas

from .registry import FormatRegistry

logger = logging.getLogger(__name__)


# --- field helpers --------------------------------------------------------------

def _pdb_atom_name_field(name, element) -> str:
    """Justify an atom name into the 4-character PDB name field (cols 13-16)."""
    name = str(name)
    if len(name) >= 4:
        return name[:4]
    if len(str(element)) == 1 and not name[:1].isdigit():
        return f" {name:<3}"
    return f"{name:<4}"


def _format_pdb_charge(charge) -> str:
    """Format a formal charge into the 2-character PDB field (e.g. ``"2+"``)."""
    try:
        c = int(round(float(charge)))
    except (ValueError, TypeError):
        return "  "
    if c == 0:
        return "  "
    return f"{min(abs(c), 9)}{'+' if c > 0 else '-'}"


def _parse_pdb_charge(field) -> float:
    """Parse a PDB charge field (``"2+"``, ``"1-"``, ``"-1"`` or blank)."""
    s = str(field).strip()
    if not s:
        return 0.0
    if s[-1] in "+-":
        mag, sign = s[:-1].strip(), s[-1]
        if mag.isdigit():
            return float(mag) * (1.0 if sign == "+" else -1.0)
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


# --- writer ---------------------------------------------------------------------

def format_atoms(scene) -> str:
    """Return spec-compliant 80-column PDB ATOM/HETATM lines for ``scene``."""
    t = pandas.DataFrame(scene).copy()
    defaults = {
        'recname': 'ATOM', 'name': 'X', 'altloc': '', 'resname': '',
        'chain': 'A', 'resid': 1, 'icode': '', 'occupancy': 0.0,
        'beta': 0.0, 'element': '', 'charge': 0,
    }
    for col, default in defaults.items():
        if col not in t:
            t[col] = default
    if 'serial' not in t:
        t['serial'] = np.arange(1, len(t) + 1)
    for axis in ('x', 'y', 'z'):
        assert axis in t.columns, f'Coordinate {axis} not in particle definition'

    lines = []
    for atom in t.itertuples(index=False):
        recname = str(getattr(atom, 'recname', 'ATOM')) or 'ATOM'
        altloc = (str(atom.altloc)[:1] if atom.altloc != '' else ' ')
        icode = (str(atom.icode)[:1] if atom.icode != '' else ' ')
        line = (
            f"{recname:<6}"
            f"{int(atom.serial) % 100000:>5}"
            " "
            f"{_pdb_atom_name_field(atom.name, atom.element):<4}"
            f"{altloc:1}"
            f"{str(atom.resname):>3.3}"
            " "
            f"{str(atom.chain)[:1] or ' ':1}"
            f"{int(atom.resid) % 10000:>4}"
            f"{icode:1}"
            "   "
            f"{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}"
            f"{atom.occupancy:>6.2f}{atom.beta:>6.2f}"
            "          "
            f"{str(atom.element):>2.2}"
            f"{_format_pdb_charge(atom.charge):>2}"
        )
        assert len(line) == 80, f'PDB line is not 80 columns ({len(line)}):\n{line}'
        lines.append(line)
    return '\n'.join(lines) + '\n' if lines else ''


@FormatRegistry.register_writer('.pdb', '.ent', name='pdb')
def write_pdb(scene, file_name=None, verbose=False):
    """Serialize ``scene`` to PDB; multi-frame ``_meta['coordinate_frames']`` becomes
    MODEL blocks.  Returns the text when ``file_name is None``."""
    if verbose:
        logger.info("Writing pdb file (%d atoms): %s", len(scene), file_name)

    meta = getattr(scene, '_meta', {}) or {}
    frames = meta.get('coordinate_frames')
    if frames is not None:
        df = pandas.DataFrame(scene).copy()
        chunks = []
        for i, frame in enumerate(np.asarray(frames)):
            df[['x', 'y', 'z']] = frame
            chunks.append(f'MODEL     {i + 1:>4d}\n')
            chunks.append(format_atoms(df))
            chunks.append('ENDMDL\n')
        chunks.append('END\n')
        text = ''.join(chunks)
    else:
        text = format_atoms(scene)

    if file_name is None:
        return text
    with open(file_name, 'w+') as out:
        out.write(text)


# --- reader ---------------------------------------------------------------------

def _pdb_line(line):
    return dict(
        recname=line[0:6].strip(), serial=line[6:11], name=line[12:16].strip(),
        altloc=line[16:17].strip(), resname=line[17:20].strip(),
        chain=line[21:22].strip(), resid=line[22:26], icode=line[26:27].strip(),
        x=line[30:38], y=line[38:46], z=line[46:54],
        occupancy=line[54:60].strip(), beta=line[60:66].strip(),
        element=line[76:78].strip(), charge=line[78:80].strip(),
        segment=line[72:76].strip() if len(line) > 72 else '')


@FormatRegistry.register_reader('.pdb', '.ent', name='pdb')
def read_pdb(file, **kwargs):
    """Parse a PDB file into ``(atoms_DataFrame, meta)``."""
    with open(file, 'r') as pdb:
        lines, mod_lines, model_numbers, model_number = [], [], [], 1
        for i, line in enumerate(pdb):
            if len(line) <= 6:
                continue
            header = line[:6]
            if header in ('ATOM  ', 'HETATM'):
                try:
                    lines.append(_pdb_line(line))
                except ValueError as e:
                    logger.error("Malformed PDB atom record at line %d: %r", i, line.rstrip())
                    raise ValueError(
                        f"Could not parse PDB atom record at line {i}: {line.rstrip()!r}") from e
                model_numbers.append(model_number)
            elif header == "MODRES":
                mod_lines.append(dict(
                    recname=line[0:6].strip(), idCode=line[7:11].strip(),
                    resname=line[12:15].strip(), chain=line[16:17].strip(),
                    resid=int(line[18:22]), icode=line[22:23].strip(),
                    stdRes=line[24:27].strip(), comment=line[29:70].strip()))
            elif header == "MODEL ":
                model_number = int(line[10:14])

    atoms = pandas.DataFrame(lines)[[
        'recname', 'serial', 'name', 'altloc', 'resname', 'chain', 'resid',
        'icode', 'x', 'y', 'z', 'occupancy', 'beta', 'element', 'charge', 'segment']]
    atoms['serial'] = pandas.to_numeric(atoms['serial'], errors='coerce').fillna(0).astype(int)
    atoms['resid'] = pandas.to_numeric(atoms['resid'], errors='coerce').fillna(0).astype(int)
    for axis in ('x', 'y', 'z'):
        atoms[axis] = pandas.to_numeric(atoms[axis], errors='coerce').fillna(0.0)
    atoms['occupancy'] = pandas.to_numeric(atoms['occupancy'], errors='coerce').fillna(1.0)
    atoms['beta'] = pandas.to_numeric(atoms['beta'], errors='coerce').fillna(1.0)
    atoms['charge'] = atoms['charge'].map(_parse_pdb_charge)
    atoms['model'] = model_numbers
    atoms['molecule'] = 0

    meta = dict(kwargs)
    meta['source_file'] = str(Path(file).resolve())
    meta['source_format'] = 'pdb'
    if mod_lines:
        meta['modified_residues'] = pandas.DataFrame(mod_lines)
    return atoms, meta

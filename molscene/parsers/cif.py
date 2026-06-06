"""mmCIF (PDBx) reader and writer (Scene-agnostic).

Also exposes :func:`read_cif_category`, the low-level loop reader reused by
``Scene.compute_anisou`` and the DSSP secondary-structure code.
"""

import re
from pathlib import Path

import numpy as np
import pandas

from .registry import FormatRegistry

_cif_tokenizer = re.compile(r"""'([^']*)'    |  # single-quoted → group 1
                                "([^"]*)"    |  # double-quoted → group 2
                                \#[^\n]*     |  # comment (no group)
                                ([^\s'"#]+)     # unquoted → group 3
                            """, re.VERBOSE)


def read_cif_category(file_path, category):
    """Read a single ``loop_`` category from a CIF file into a DataFrame.

    ``category`` is the prefix, e.g. ``'_atom_site'`` or ``'_dssp_struct_summary'``.
    """
    data, header, in_section = [], [], False
    prefix = category + '.'
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            if line.startswith('loop_'):
                in_section = False
            elif line.startswith(prefix):
                header.append(line.split('.')[-1])
                in_section = True
            elif in_section:
                tokens = []
                for m in _cif_tokenizer.finditer(line):
                    sq, dq, word = m.group(1), m.group(2), m.group(3)
                    val = sq if sq is not None else (dq if dq is not None else word)
                    if val is not None:
                        tokens.append(val)
                if tokens:
                    data.append(tokens)
    return pandas.DataFrame(data, columns=header)


# resid uses auth_seq_id (author numbering), following the ProDy convention.
_CIF_PDB_RENAME = {
    'id': 'serial', 'label_atom_id': 'name', 'label_alt_id': 'altloc',
    'label_comp_id': 'resname', 'label_asym_id': 'chain', 'auth_seq_id': 'resid',
    'pdbx_PDB_ins_code': 'icode', 'Cartn_x': 'x', 'Cartn_y': 'y', 'Cartn_z': 'z',
    'occupancy': 'occupancy', 'B_iso_or_equiv': 'beta', 'type_symbol': 'element',
    'pdbx_formal_charge': 'charge', 'pdbx_PDB_model_num': 'model'}


@FormatRegistry.register_reader('.cif', '.mmcif', '.pdbx', name='cif')
def read_cif(file_path, **kwargs):
    """Parse an mmCIF ``_atom_site`` loop into ``(atoms_DataFrame, meta)``."""
    cif_atoms = read_cif_category(file_path, '_atom_site').rename(_CIF_PDB_RENAME, axis=1)
    cif_atoms = cif_atoms.replace({'.': pandas.NA, '?': pandas.NA})

    _int_cols = {'serial': 0, 'resid': 0, 'charge': 0, 'model': 1}
    _float_cols = {'x': 0.0, 'y': 0.0, 'z': 0.0, 'occupancy': 1.0, 'beta': 0.0}
    _nullable_int_cols = ['label_seq_id']
    _str_defaults = {'altloc': '', 'icode': '', 'group_PDB': '', 'label_entity_id': '',
                     'auth_comp_id': '', 'auth_asym_id': '', 'auth_atom_id': ''}

    for col, default in _int_cols.items():
        if col in cif_atoms.columns:
            cif_atoms[col] = pandas.to_numeric(cif_atoms[col], errors='coerce').fillna(default).astype(int)
        else:
            cif_atoms[col] = default
    for col, default in _float_cols.items():
        if col in cif_atoms.columns:
            cif_atoms[col] = pandas.to_numeric(cif_atoms[col], errors='coerce').fillna(default)
    for col in _nullable_int_cols:
        if col in cif_atoms.columns:
            cif_atoms[col] = pandas.to_numeric(cif_atoms[col], errors='coerce').astype('Int64')
    for col, default in _str_defaults.items():
        if col in cif_atoms.columns:
            cif_atoms[col] = cif_atoms[col].fillna(default)

    # CIF has no PDB-style segment; auth_asym_id is the natural segment (ProDy convention).
    cif_atoms['segment'] = cif_atoms['auth_asym_id'] if 'auth_asym_id' in cif_atoms.columns else ''

    meta = dict(kwargs)
    meta['source_file'] = str(Path(file_path).resolve())
    meta['source_format'] = 'cif'
    return cif_atoms, meta


def _cif_quote(val):
    if val is np.nan:
        return '.'
    if not isinstance(val, str):
        val = str(val)
    if "'" in val and '"' in val:
        return '"' + val.replace('"', "'") + '"'
    if "'" in val:
        return '"' + val + '"'
    if '"' in val:
        return "'" + val + "'"
    if any(c.isspace() for c in val) or val == '' or val.startswith('#') or val.startswith(';'):
        return '"' + val + '"'
    return val


@FormatRegistry.register_writer('.cif', '.mmcif', '.pdbx', name='cif')
def write_cif(scene, file_name=None, verbose=False):
    """Serialize ``scene`` to a PDBx/mmCIF ``_atom_site`` loop. Returns the text when
    ``file_name is None``."""
    t = pandas.DataFrame(scene).copy()
    t['serial'] = np.arange(1, len(t) + 1) if 'serial' not in t else t['serial']
    for col, default in {'name': 'A', 'altloc': '?', 'resname': 'R', 'chain': 'C',
                         'resid': 1, 'resIC': 1, 'icode': '', 'occupancy': 0,
                         'beta': 0, 'element': 'C', 'model': 0, 'charge': 0}.items():
        if col not in t:
            t[col] = default
    for axis in ('x', 'y', 'z'):
        assert axis in t.columns, f'Coordinate {axis} not in particle definition'

    for col in ['serial', 'resid', 'resIC', 'model', 'charge']:
        t[col] = pandas.to_numeric(t[col], errors='coerce').fillna(0).astype(int)
    for col in ['x', 'y', 'z', 'occupancy', 'beta']:
        t[col] = pandas.to_numeric(t[col], errors='coerce').fillna(0.0)
    for col in ['name', 'altloc', 'resname', 'chain', 'icode', 'element']:
        t[col] = t[col].astype(str).str.strip().replace('', '.')

    header = ("data_pdbx\n#\nloop_\n"
              "_atom_site.group_PDB\n_atom_site.id\n_atom_site.label_atom_id\n"
              "_atom_site.label_comp_id\n_atom_site.label_asym_id\n_atom_site.label_seq_id\n"
              "_atom_site.label_alt_id\n_atom_site.auth_atom_id\n_atom_site.auth_comp_id\n"
              "_atom_site.auth_asym_id\n_atom_site.auth_seq_id\n_atom_site.pdbx_PDB_ins_code\n"
              "_atom_site.Cartn_x\n_atom_site.Cartn_y\n_atom_site.Cartn_z\n"
              "_atom_site.occupancy\n_atom_site.B_iso_or_equiv\n_atom_site.type_symbol\n"
              "_atom_site.pdbx_formal_charge\n_atom_site.pdbx_PDB_model_num\n")

    t['line'] = 'ATOM'
    for col in ['serial',
                'name', 'resname', 'chain', 'resid', 'altloc',
                'name', 'resname', 'chain', 'resid', 'icode',   # label_* then auth_*
                'x', 'y', 'z', 'occupancy', 'beta', 'element', 'charge', 'model']:
        t['line'] += " "
        t['line'] += t[col].apply(_cif_quote)
    t['line'] += '\n'
    text = header + ''.join(t['line']) + '#\n'

    if file_name is None:
        return text
    with open(file_name, 'w+') as out:
        out.write(text)

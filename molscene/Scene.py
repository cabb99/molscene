"""
Python library to allow easy handling of coordinate files for molecular dynamics using pandas DataFrames.
"""


import pandas
import numpy as np
import io
import tempfile
from typing import Union, Tuple, Sequence, List
from pathlib import Path
import re
from scipy.spatial import cKDTree, distance
import logging
from . import utils
from . import contacts
from .bonds import compute_bonds as _compute_bonds
from .data.element_info import element_info
from .transformation import Transformation
from .matching import as_matching as _as_matching


logger = logging.getLogger(__name__)

_MOLSELECT_EVALUATOR = None


def _molselect_evaluator():
    """Return a lazily-built molselect evaluator (parser + builder + backend).

    Constructed once per process because parser construction is non-trivial.
    """
    global _MOLSELECT_EVALUATOR
    if _MOLSELECT_EVALUATOR is None:
        try:
            from molselect.python.evaluator import Evaluator
            from molselect.python.backends.pandas import PandasStructure
            from molselect.python.parser import SelectionParser
            from molselect.python.builder import ASTBuilder
        except ImportError as exc:  # pragma: no cover - exercised without the optional dep
            raise ImportError(
                "String atom selection requires the optional 'molselect' package. "
                "Install it with:  pip install molscene[selection]  "
                "(kwargs-based Scene.select(col=values) works without it)."
            ) from exc
        parser = SelectionParser()
        builder = ASTBuilder(parser)
        _MOLSELECT_EVALUATOR = Evaluator(PandasStructure, parser=parser, builder=builder)
    return _MOLSELECT_EVALUATOR


# Sentinel distinguishing "threshold not passed" from "threshold=None" in the
# back-compatible distance_map signature.
_DM_UNSET = object()


__author__ = 'Carlos Bueno'

_protein_residues = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                     'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                     'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                     'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                     'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

_DNA_residues = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T'}

_RNA_residues = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'}

def _canonicalize_element_symbol(symbol):
    if isinstance(symbol, str) and symbol.isalpha() and symbol.isupper():
        return symbol[0] + symbol[1:].lower()
    return symbol


def _read_cif_category(file_path, category):
    """Read a single ``loop_`` category (delegates to the cif parser)."""
    from molscene.parsers.cif import read_cif_category
    return read_cif_category(file_path, category)


def _dihedral(p0, p1, p2, p3):
    """Return the dihedral angle in degrees defined by four points."""
    b1 = p1 - p0
    b2 = p2 - p1
    b3 = p3 - p2
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m = np.cross(n1, b2 / np.linalg.norm(b2))
    x = np.dot(n1, n2)
    y = np.dot(m, n2)
    return -np.degrees(np.arctan2(y, x))


class _FrameAccessor:
    def __init__(self, scene: "Scene"):
        self._scene = scene

    def __getitem__(self, index):
        # Retrieve the multi-frame array from the parent Scene.
        frames = self._scene.get_coordinate_frames()
        # Support integer indexing (or slicing that returns one or more frames)
        new_coords = frames[index]
        # If a single frame is selected, new_coords has shape (n_atoms, 3).
        # In that case, create a new Scene that has the same metadata but with
        # the coordinates replaced by this frame. Importantly, we do NOT copy
        # the entire multi-frame data.
        if new_coords.ndim == 2:
            new_scene = self._scene.copy(deep=True)
            # Remove the heavy multi-frame data from the new scene.
            new_scene._meta.pop('coordinate_frames', None)
            new_scene.set_coordinates(new_coords)
            return new_scene
        # If the index returns multiple frames (e.g. a slice), return a list
        # of Scene objects, one per frame.
        elif new_coords.ndim == 3:
            scenes = []
            for coords in new_coords:
                new_scene = self._scene.copy(deep=True)
                new_scene._meta.pop('coordinate_frames', None)
                new_scene.set_coordinates(coords)
                scenes.append(new_scene)
            return scenes
        else:
            raise ValueError("Invalid frame dimensions")

    def __iter__(self):
        frames = self._scene.get_coordinate_frames()
        for i in range(frames.shape[0]):
            yield self[i]


class Scene(pandas.DataFrame):

    _metadata = ["_meta"]

    _columns = {'recname': 'Record name',
                'serial': 'Atom serial number',
                'name': 'Atom name',
                'altloc': 'Alternate location indicator',
                'resname': 'Residue name',
                'chain': 'Chain identifier',
                'resid': 'Residue sequence number',
                'icode': 'Code for insertion of residues',
                'x': 'Orthogonal coordinates for X in Angstroms',
                'y': 'Orthogonal coordinates for Y in Angstroms',
                'z': 'Orthogonal coordinates for Z in Angstroms',
                'occupancy': 'Occupancy',
                'beta': 'Temperature factor',
                'element': 'Element symbol',
                'charge': 'Charge on the atom',
                'model': 'Model number',
                # 'residue': 'Residue index (0-based)',
                # 'fragment': 'Chain index (0-based)',
                # 'index': 'Atom index (0-based)',
                'molecule': 'Molecule name'}
    
    

    # Initialization
    def __init__(self, particles, **kwargs):
        """Create a scene from particles.

        ``particles`` may be anything :class:`pandas.DataFrame` accepts (an
        ``(N, 3)`` array of coordinates, a DataFrame with ``x``/``y``/``z``
        columns, etc.). Any extra keyword arguments are stored as scene-level
        metadata (see ``_meta``). The ``Scene`` is a DataFrame wrapper with the
        canonical structural columns inferred and a metadata dict attached.
        """
        super().__init__(particles)
        # Add metadata dictionary
        self.__dict__['_meta'] = {}

        if all([col in self.columns for col in ['x', 'y', 'z']]):
            pass
        elif any([col in self.columns for col in ['x', 'y', 'z']]):
            raise ValueError(f"Incomplete coordinates, missing columns: {set(['x', 'y', 'z']) - set(self.columns)}")
        elif len(self.columns) == 3:
            self.columns=['x', 'y', 'z']
        else:
            raise ValueError("Incorrect particle format")
        
        if 'chain' not in self.columns:
            self['chain'] = ['A'] * len(self)
        if 'resid' not in self.columns:
            self['resid'] = [1] * len(self)
        if 'icode' not in self.columns:
            self['icode'] = [''] * len(self)
        if 'altloc' not in self.columns:
            self['altloc'] = [''] * len(self)
        if 'model' not in self.columns:
            self['model'] = [1] * len(self)
        if 'name' not in self.columns:
            self['name'] = [f'P{i:03}' for i in range(len(self))]
        if 'element' not in self.columns:
            self['element'] = ['C'] * len(self)
        if 'occupancy' not in self.columns:
            self['occupancy'] = [1.0] * len(self)
        if 'beta' not in self.columns:
            self['beta'] = [1.0] * len(self)
        if 'resname' not in self.columns:
            self['resname'] = [''] * len(self)

        # Element-derived columns. Mapping every element through
        # canonicalization is the costliest part of __init__, and __init__ runs
        # on every internally-constructed frame (slices, arithmetic), so only
        # pay for it when one of the derived columns is actually missing.
        if not {'mass', 'atomicnumber', 'radius'} <= set(self.columns):
            element_symbols = self['element'].map(_canonicalize_element_symbol)
            if 'mass' not in self.columns:
                self['mass'] = element_symbols.map(element_info.mass).fillna(0.0)
            if 'atomicnumber' not in self.columns:
                self['atomicnumber'] = element_symbols.map(element_info.atomicnumber).fillna(0).astype(int)
            if 'radius' not in self.columns:
                self['radius'] = element_symbols.map(element_info.radius).fillna(0.0)
        if 'type' not in self.columns:
            self['type'] = self['element']
        
        # Create an integer index for the chains
        if 'fragment' not in self.columns:
            chain_map = {b: a for a, b in enumerate(self['chain'].unique())}
            self['fragment'] = self['chain'].map(chain_map).astype(int)

        # Create an integer index for the residues
        if 'residue' not in self.columns:
            # Construct a global unique residue key
            residue_keys = (
                self['fragment'].astype(str) + '_' +
                self['resid'].astype(str) + '_' +
                self['icode'].astype(str)
            )

            # Get unique residue keys and map to integers
            unique_keys = pandas.Series(residue_keys.unique())
            key_to_index = dict(zip(unique_keys, range(len(unique_keys))))

            # Map each residue key to its index
            self['residue'] = residue_keys.map(key_to_index).astype(int)

        # Create an integer index for the atoms
        if 'index' not in self.columns:
            self['index'] = range(len(self))


        # Add metadata
        for attr, value in kwargs.items():
            self._meta[attr] = value

    def compute_mass(self):
        """Return a copy with a ``mass`` column (per-atom mass in daltons).

        ``mass`` is normally populated already at construction time; this is a
        no-op copy when the column is present.
        """
        out = self.copy()
        if 'mass' not in out.columns:
            out['mass'] = out['element'].map(_canonicalize_element_symbol).map(element_info.mass).fillna(0)
        return out

    def compute_phi_psi(self):
        """Compute backbone phi and psi dihedral angles (degrees) per atom.

        Phi (φ) = dihedral(C_prev, N_i, CA_i, C_i)
        Psi (ψ) = dihedral(N_i, CA_i, C_i, N_next)

        Terminal residues and non-protein atoms get ``NaN``.

        Returns
        -------
        Scene
            Copy of ``self`` with ``phi`` and ``psi`` columns added.
        """
        out = self.copy()
        phi = np.full(len(out), np.nan)
        psi = np.full(len(out), np.nan)

        # Work per fragment (chain) so residue numbering from different
        # chains never mixes.
        for _, frag in out.groupby('fragment'):
            # Extract backbone atoms: N, CA, C
            backbone = frag[frag['name'].isin(['N', 'CA', 'C'])]

            # Group by residue and collect ordered coords (N, CA, C)
            residue_atoms: dict = {}  # residue_idx -> {'N': xyz, 'CA': xyz, 'C': xyz}
            for row in backbone.itertuples():
                res = row.residue
                if res not in residue_atoms:
                    residue_atoms[res] = {}
                residue_atoms[res][row.name] = np.array([row.x, row.y, row.z])

            # Keep only residues that have all three backbone atoms
            complete = {
                r: atoms for r, atoms in residue_atoms.items()
                if 'N' in atoms and 'CA' in atoms and 'C' in atoms
            }
            sorted_res = sorted(complete)

            # Walk consecutive pairs
            for i in range(len(sorted_res)):
                res = sorted_res[i]
                atoms_i = complete[res]
                mask = frag['residue'] == res

                # Phi: need C from previous residue
                if i > 0:
                    prev = sorted_res[i - 1]
                    atoms_prev = complete[prev]
                    angle = _dihedral(
                        atoms_prev['C'], atoms_i['N'],
                        atoms_i['CA'], atoms_i['C'],
                    )
                    phi[mask.values] = angle

                # Psi: need N from next residue
                if i < len(sorted_res) - 1:
                    nxt = sorted_res[i + 1]
                    atoms_next = complete[nxt]
                    angle = _dihedral(
                        atoms_i['N'], atoms_i['CA'],
                        atoms_i['C'], atoms_next['N'],
                    )
                    psi[mask.values] = angle

        out['phi'] = phi
        out['psi'] = psi
        return out

    def compute_bonds(self):
        """Build bond graph from residue topology.

        Adds the following columns:

        - ``numbonds`` — number of covalent bonds per atom.
        - ``pfrag`` — connected-component index among protein residues
          (``-1`` for non-protein atoms).
        - ``nfrag`` — connected-component index among nucleic-acid residues
          (``-1`` for non-nucleic atoms).

        The bond adjacency list is stored in ``_meta['bonds']``.

        Returns
        -------
        Scene
            Copy with the new columns.
        """
        return _compute_bonds(self)

    def compute_anisou(self):
        """Read anisotropic displacement parameters from the source file.

        Adds columns ``ufx``, ``ufy``, ``ufz`` (diagonal elements of the
        anisotropic U tensor: U11, U22, U33 in Å²).  Atoms without ANISOU
        records get ``0.0``.

        The Scene must have been loaded via :meth:`from_pdb` or
        :meth:`from_cif` so that ``_meta['source_file']`` is set.

        Returns
        -------
        Scene
            Copy with the new columns.
        """
        source = self._meta.get('source_file')
        fmt = self._meta.get('source_format')
        if source is None:
            raise ValueError(
                "Cannot compute ANISOU: no source file recorded. "
                "Load via Scene.from_pdb() or Scene.from_cif()."
            )

        out = self.copy()
        if fmt == 'pdb':
            out = self._read_anisou_pdb(out, source)
        elif fmt == 'cif':
            out = self._read_anisou_cif(out, source)
        else:
            raise ValueError(f"Unknown source format: {fmt!r}")
        return out

    @staticmethod
    def _read_anisou_pdb(out, file_path):
        """Parse ANISOU records from a PDB file and merge into *out*."""
        anisou_lines = []
        model_number = 1
        with open(file_path, 'r') as f:
            for line in f:
                if len(line) > 6:
                    header = line[:6]
                    if header == 'ANISOU':
                        try:
                            serial = int(line[6:11])
                            u11 = int(line[28:35]) / 10000.0
                            u22 = int(line[35:42]) / 10000.0
                            u33 = int(line[42:49]) / 10000.0
                            anisou_lines.append({
                                'serial': serial, 'model': model_number,
                                'ufx': u11, 'ufy': u22, 'ufz': u33,
                            })
                        except (ValueError, IndexError):
                            pass
                    elif header == 'MODEL ':
                        model_number = int(line[10:14])
        if anisou_lines:
            anisou_df = pandas.DataFrame(anisou_lines)
            out = out.merge(
                anisou_df, on=['serial', 'model'], how='left',
            )
        for col in ('ufx', 'ufy', 'ufz'):
            if col not in out.columns:
                out[col] = 0.0
            else:
                out[col] = out[col].fillna(0.0)
        return out

    @staticmethod
    def _read_anisou_cif(out, file_path):
        """Parse anisotropic data from a CIF file and merge into *out*."""
        aniso = _read_cif_category(file_path, '_atom_site_anisotrop')
        if len(aniso) > 0 and 'id' in aniso.columns:
            _aniso_rename = {}
            for col in aniso.columns:
                if col == 'id':
                    _aniso_rename[col] = 'serial'
                elif 'U[1][1]' in col:
                    _aniso_rename[col] = 'ufx'
                elif 'U[2][2]' in col:
                    _aniso_rename[col] = 'ufy'
                elif 'U[3][3]' in col:
                    _aniso_rename[col] = 'ufz'
            aniso = aniso.rename(_aniso_rename, axis=1)
            if 'serial' in aniso.columns:
                aniso['serial'] = pandas.to_numeric(
                    aniso['serial'], errors='coerce'
                ).fillna(0).astype(int)
                for col in ('ufx', 'ufy', 'ufz'):
                    if col in aniso.columns:
                        aniso[col] = pandas.to_numeric(
                            aniso[col], errors='coerce'
                        ).fillna(0.0)
                merge_cols = ['serial'] + [
                    c for c in ('ufx', 'ufy', 'ufz') if c in aniso.columns
                ]
                out = out.merge(
                    aniso[merge_cols], on='serial', how='left',
                )
        for col in ('ufx', 'ufy', 'ufz'):
            if col not in out.columns:
                out[col] = 0.0
            else:
                out[col] = out[col].fillna(0.0)
        return out

    def compute_secondary_structure(self, dssp_file=None, **kwargs):
        """
        Annotate each residue with its DSSP secondary structure.

        Parameters
        ----------
        dssp_file : str or path-like, optional
            Path to a precomputed DSSP mmCIF (the output of ``mkdssp``,
            containing the ``_dssp_struct_summary`` category). When given, the
            file is parsed directly and no external tool is run — useful for
            reproducibility and for environments without DSSP installed. When
            omitted, ``mkdssp`` is run on this scene, which requires the
            ``mkdssp`` executable on ``PATH``.

        Returns
        -------
        Scene
            A copy of the scene with the DSSP columns (notably
            ``secondary_structure`` and ``accessibility``) merged in per
            residue.

        Notes
        -----
        A precomputed file must be generated from this scene's coordinates
        (DSSP reads the ``label_seq_id`` numbering written by
        :meth:`write_cif`), so it round-trips through ``write_cif`` → ``mkdssp``.
        """
        if dssp_file is None:
            import subprocess
            try:
                subprocess.call(['mkdssp', '--version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except FileNotFoundError:
                raise FileNotFoundError(
                    "DSSP is not installed or not found in PATH. Install `mkdssp` "
                    "or pass a precomputed DSSP file via `dssp_file=`."
                )

            # Use DSSP to calculate secondary structure into a temporary mmCIF.
            with tempfile.NamedTemporaryFile('w+', suffix='.cif', delete=False) as dsspout:
                with tempfile.NamedTemporaryFile('w+', suffix='.cif', delete=False) as tmp:
                    #Run dssp
                    self.write_cif(tmp.name)
                    subprocess.call(['mkdssp', '--calculate-accessibility', tmp.name, dsspout.name])
            dssp_file = dsspout.name

        dssp_df = _read_cif_category(dssp_file, '_dssp_struct_summary')

        # Rename columns to pdb convention
        _dssp_rename = {'label_comp_id': 'resname',
                        'label_asym_id': 'chain',
                        'label_seq_id': 'resid'}

        dssp_df = dssp_df.rename(_dssp_rename, axis=1)
        dssp_df['resid'] = dssp_df['resid'].astype(int)

        # Merge dssp_df to self using the 'resid', 'chain' and 'resname' columns
        return self.merge(dssp_df, on=['resid', 'chain', 'resname'], how='left', suffixes=('', '_dssp'))

    def set_coordinate_frames(self, frames: np.ndarray):
        """
        Set the coordinate frames from a NumPy array.

        Parameters
        ----------
        frames : np.ndarray
            A NumPy array of shape (n_frames, n_atoms, 3).

        Raises
        ------
        TypeError
            If frames is not a NumPy array.
        ValueError
            If the array does not have three dimensions or the last dimension is not 3,
            or if the number of atoms (second dimension) does not match the number of rows.
        """
        if not isinstance(frames, np.ndarray):
            raise TypeError("frames must be a numpy array")
        if frames.ndim != 3 or frames.shape[2] != 3:
            raise ValueError("frames must be a 3D numpy array with shape (n_frames, n_atoms, 3)")
        if frames.shape[1] != len(self):
            raise ValueError("The number of atoms in frames must match the number of rows in the Scene")
        self._meta['coordinate_frames'] = frames
        # Update the current coordinates to the first frame.
        self.set_coordinates(frames[0])

    def get_coordinate_frames(self) -> np.ndarray:
        """
        Retrieve the multi-frame coordinates.

        Returns
        -------
        np.ndarray
            A NumPy array of shape (n_frames, n_atoms, 3). If no frames have been set,
            the current single-frame coordinates are returned with shape (1, n_atoms, 3).
        """
        if 'coordinate_frames' in self._meta:
            return self._meta['coordinate_frames']
        else:
            return self.get_coordinates().to_numpy().reshape(1, -1, 3)

    @property
    def n_frames(self) -> int:
        """
        Number of frames stored in the coordinate frames.

        Returns
        -------
        int
            The number of frames.
        """
        return self.get_coordinate_frames().shape[0]

    @property
    def frames(self) -> _FrameAccessor:
        """
        Accessor to select individual frames.

        Example
        -------
        >>> frame10 = scene.frames[10]
        """
        return _FrameAccessor(self)
    
    def iterframes(self):
        """
        Iterate over frames.

        Yields
        ------
        Scene
            A new Scene for each frame (with the coordinates replaced).
        """
        return iter(self.frames)

    def get_frame_coordinates(self, frame_index: int) -> np.ndarray:
        """
        Get the coordinates for a particular frame.

        Parameters
        ----------
        frame_index : int
            The index of the desired frame.

        Returns
        -------
        np.ndarray
            An array of shape (n_atoms, 3) for that frame.
        """
        frames = self.get_coordinate_frames()
        return frames[frame_index]

    def set_frame_coordinates(self, frame_index: int):
        """
        Set the Scene’s current coordinates to those of a specific frame.

        Parameters
        ----------
        frame_index : int
            The index of the frame to set as current.
        """
        frames = self.get_coordinate_frames()
        self.set_coordinates(frames[frame_index])

    def select(self, selstr: str = "", **kwargs) -> "Scene":
        """
        Atom selection.

        Two ways to call:

        * **Selection string** (preferred). Any non-empty ``selstr`` is
          evaluated by `molselect <https://github.com/cabb99/molselect>`_,
          giving access to the full VMD-style selection grammar
          (booleans, ``within``, ``same residue as``, regex, etc.).
          Example: ``scene.select("chain A and resid 1 to 100 and name CA")``.
        * **Column-equality kwargs** (back-compat shortcut). When ``selstr``
          is empty, each kwarg ``col=values`` keeps rows where ``self[col]``
          is one of ``values``. ``altloc`` additionally allows the empty/`.`
          placeholder.

        Combine selstr and kwargs to and-merge their masks.

        Returns
        -------
        Scene
            A new Scene containing only the selected atoms; metadata is
            preserved.
        """
        sel = pandas.Series(True, index=self.index)

        if selstr:
            ev = _molselect_evaluator()
            result = ev.parse(self, selstr)
            # molselect returns a backend-wrapped sub-structure whose .df
            # carries the original DataFrame index.
            kept = result.df.index if hasattr(result, "df") else result
            sel &= self.index.isin(kept)

        for key, values in kwargs.items():
            if key == "altloc":
                sel &= self["altloc"].isin(list(values) + ["", "."])
            else:
                sel &= self[key].isin(values)

        return self._subset(sel)

    def _subset(self, sel) -> "Scene":
        """Build a sub-Scene from boolean mask ``sel``, keeping index-aligned
        metadata consistent with the removed atoms.

        Atom indices stay stable (so ``scene - subset`` and round-trips keep
        working); ``index_``-prefixed columns on the atoms and any index-aligned
        ``_meta`` (bond/angle/dihedral tables, ``coordinate_frames``, the
        ``bonds`` adjacency) drop their references to dropped atoms, while
        surviving references are left untouched.
        """
        subset = self.loc[sel].copy()
        if 'index' in subset.columns:
            kept = subset['index'].to_numpy()
        else:
            kept = np.flatnonzero(np.asarray(sel))
        kept_set = set(int(k) for k in kept)

        # Per-atom index_ metadata: blank out references to dropped atoms.
        for col in subset.columns:
            if isinstance(col, str) and col.startswith('index_'):
                subset[col] = subset[col].where(subset[col].isin(kept_set))

        new_meta = {key: self._filter_index_metadata(key, val, kept, kept_set, len(self))
                    for key, val in self._meta.items()}
        return Scene(subset, **new_meta)

    @staticmethod
    def _filter_index_metadata(key, value, kept, kept_set, n_old):
        """Keep one ``_meta`` value consistent with a subset of atoms.

        Atom indices are *stable* (preserved through selection), so surviving
        references are left untouched and only references to dropped atoms are
        removed:

        * DataFrame with ``index_*`` columns (bond/angle/dihedral tables): drop
          rows that reference a dropped atom.
        * ``coordinate_frames`` (``(n_frames, n_atoms, 3)`` array): keep the
          selected atoms.
        * ``bonds`` adjacency list (length ``n_atoms``): keep selected atoms and
          their surviving neighbours.

        Any other metadata is returned unchanged.
        """
        if isinstance(value, pandas.DataFrame):
            idx_cols = [c for c in value.columns
                        if isinstance(c, str) and c.startswith('index_')]
            if not idx_cols:
                return value
            keep = value[idx_cols].isin(kept_set).all(axis=1)
            return value[keep].reset_index(drop=True)
        if (key == 'coordinate_frames' and isinstance(value, np.ndarray)
                and value.ndim == 3 and value.shape[1] == n_old):
            return value[:, kept, :]
        if key == 'bonds' and isinstance(value, (list, tuple)) and len(value) == n_old:
            return [[n for n in value[i] if n in kept_set] for i in kept]
        return value

    @classmethod
    def _from_parsed(cls, atoms, meta):
        """Wrap ``(atoms_DataFrame, meta)`` from a parser/backend into a Scene,
        applying any ``coordinate_frames`` carried in ``meta``."""
        meta = dict(meta)
        frames = meta.pop('coordinate_frames', None)
        scene = cls(atoms, **meta)
        if frames is not None:
            scene.set_coordinate_frames(np.asarray(frames))
        return scene

    @classmethod
    def from_pdb(cls, file, **kwargs):
        from molscene.parsers import pdb as _pdb
        return cls._from_parsed(*_pdb.read_pdb(file, **kwargs))

    @classmethod
    def from_cif(cls, file_path, **kwargs):
        from molscene.parsers import cif as _cif
        return cls._from_parsed(*_cif.read_cif(file_path, **kwargs))

    @classmethod
    def from_awsem_gro(cls, path, **kwargs):
        """Read an ``awsem_gro`` (AWSEM fragment-memory) file into a Scene."""
        from molscene.parsers import awsem_gro as _ag
        return cls._from_parsed(*_ag.read_awsem_gro(path, **kwargs))

    @classmethod
    def from_object(cls, obj):
        """Build a Scene from a recognized foreign object (ProDy ``AtomGroup``,
        mdtraj ``Trajectory``, ``PDBFixer`` ...)."""
        from molscene import backends
        return cls._from_parsed(*backends.from_object(obj))

    @classmethod
    def from_prody(cls, atomgroup) -> "Scene":
        """Build a Scene from a ProDy ``AtomGroup``/``Selection`` (extra ``prody``)."""
        return cls.from_object(atomgroup)

    @classmethod
    def from_mdtraj(cls, trajectory) -> "Scene":
        """Build a Scene from an mdtraj ``Trajectory`` (extra ``mdtraj``)."""
        return cls.from_object(trajectory)

    def to_prody(self):
        """Convert this Scene to a ProDy ``AtomGroup`` (extra ``prody``)."""
        from molscene import backends
        return backends.to_object(self, 'prody')

    def to_mdtraj(self):
        """Convert this Scene to an mdtraj ``Trajectory`` (extra ``mdtraj``)."""
        from molscene import backends
        return backends.to_object(self, 'mdtraj')

    @classmethod
    def from_fixer(cls, fixer, **kwargs):
        """Parse a PDBFixer object (openmm topology + positions) into a Scene."""
        from molscene.backends.pdbfixer_backend import PDBFixerBackend
        atoms, meta = PDBFixerBackend.from_object(fixer)
        meta.update(kwargs)
        return cls._from_parsed(atoms, meta)

    @classmethod
    def from_fixPDB(cls, filename=None, pdbfile=None, pdbxfile=None, url=None, pdbid=None,
                    isolated=False, chain=None, **kwargs):
        """Clean a structure with PDBFixer (replace nonstandard residues, remove
        heterogens, add missing atoms/hydrogens) and parse it into a Scene.

        ``isolated=True`` runs the cleanup in a child process so OpenMM memory is
        reclaimed on exit (useful for batch cleaning); it needs a ``filename`` and
        may filter to ``chain``.
        """
        from molscene.backends import pdbfixer_backend as _fix
        if isolated:
            if filename is None:
                raise ValueError("from_fixPDB(isolated=True) requires a filename")
            cleaned = _fix.repair_pdb(filename, chain=chain)
            return cls.from_pdb(str(cleaned), **kwargs)
        fixer = _fix.clean(filename=filename, pdbfile=pdbfile, pdbxfile=pdbxfile,
                           url=url, pdbid=pdbid)
        return cls.from_fixer(fixer, **kwargs)

    @classmethod
    def from_file(cls, filename, format=None):
        from molscene import parsers
        return cls._from_parsed(*parsers.read(filename, format=format))

    def to_file(self, filename, format=None):
        from molscene import parsers
        parsers.write(self, filename, format=format)

    @classmethod
    def concatenate(cls, scene_list):
        #Set chain names
        chainID = []
        name_generator = utils.chain_name_generator()
        for scene in scene_list:
            if 'chain' not in scene:
                chainID += [next(name_generator)]*len(scene)
            else:
                chains = list(scene['chain'].unique())
                chains.sort()
                chain_replace = {chain: next(name_generator) for chain in chains}
                chainID += list(scene['chain'].replace(chain_replace))
        name_generator.close()
        model = pandas.concat(scene_list)
        model['chain'] = chainID
        model.index = range(len(model))
        return cls(model)

    # Writing
    def write_pdb(self, file_name=None, verbose=False):
        """Serialize to PDB; multi-frame coordinates become ``MODEL`` blocks.

        Returns an :class:`io.StringIO` when ``file_name`` is ``None``.
        """
        from molscene.parsers import pdb as _pdb
        text = _pdb.write_pdb(self, file_name, verbose=verbose)
        return io.StringIO(text) if file_name is None else None

    def write_cif(self, file_name=None, verbose=False):
        """Serialize to a PDBx/mmCIF ``_atom_site`` loop (round-trips via
        :meth:`from_cif`). Returns an :class:`io.StringIO` when ``file_name`` is ``None``."""
        from molscene.parsers import cif as _cif
        text = _cif.write_cif(self, file_name, verbose=verbose)
        return io.StringIO(text) if file_name is None else None

    def write_gro(self, file_name, box_size=None, verbose=False):
        """
        Write the Scene to a GRO file.

        Parameters:
        -----------
        file_name : str
            Name of the output GRO file.

        box_size : float or tuple of floats, optional
            The box dimensions in nanometers (x, y, z). If None, it will be set based on the coordinates.

        verbose : bool, optional
            If True, prints additional information.

        Raises:
        -------
        ValueError
            If required columns are missing.
        """
        if verbose:
            logger.info("Writing GRO file (%d atoms): %s", len(self), file_name)

        # Prepare data
        gro_atoms = self.copy()

        # Ensure required columns are present
        required_columns = ['resid', 'resname', 'name', 'x', 'y', 'z']
        for col in required_columns:
            if col not in gro_atoms.columns:
                raise ValueError(f"Column '{col}' is required for writing GRO file.")

        # Handle 'serial' column
        if 'serial' not in gro_atoms.columns:
            gro_atoms['serial'] = np.arange(1, len(gro_atoms) + 1)

        # Convert types and handle formatting
        gro_atoms['resid'] = gro_atoms['resid'].astype(int) % 100000  # Limit to 5 digits
        gro_atoms['serial'] = gro_atoms['serial'].astype(int) % 100000  # Limit to 5 digits
        gro_atoms['resname'] = gro_atoms['resname'].astype(str).str[:5]
        gro_atoms['name'] = gro_atoms['name'].astype(str).str[:5]

        # Divide coordinates by 10 to convert from Angstroms to nanometers
        gro_atoms['x'] = gro_atoms['x'] / 10.0
        gro_atoms['y'] = gro_atoms['y'] / 10.0
        gro_atoms['z'] = gro_atoms['z'] / 10.0

        # If box_size is not specified, set it based on the coordinates
        if box_size is None:
            x_max = gro_atoms['x'].max()
            y_max = gro_atoms['y'].max()
            z_max = gro_atoms['z'].max()
            x_min = gro_atoms['x'].min()
            y_min = gro_atoms['y'].min()
            z_min = gro_atoms['z'].min()
            # Add a buffer of 1.0 nm to each dimension
            box_size = (x_max - x_min + 1.0, y_max - y_min + 1.0, z_max - z_min + 1.0)
        elif isinstance(box_size, (float, int)):
            box_size = (box_size, box_size, box_size)

        # Start writing the GRO file
        with open(file_name, 'w') as f:
            f.write('Generated by Scene.write_gro\n')
            f.write(f'{len(gro_atoms):5d}\n')
            for _, atom in gro_atoms.iterrows():
                line = f"{atom['resid']:5d}{atom['resname']:<5}{atom['name']:>5}{atom['serial']:5d}" \
                    f"{atom['x']:8.3f}{atom['y']:8.3f}{atom['z']:8.3f}\n"
                f.write(line)
            # Write box dimensions
            f.write(f"{box_size[0]:10.5f}{box_size[1]:10.5f}{box_size[2]:10.5f}\n")

    def write_gro_per_chain(self, base_filename, box_size=None, verbose=False):
        """
        Write each chain in the Scene to a separate GRO file.

        Parameters:
        -----------
        base_filename : str
            Base filename to use for output GRO files. The chain ID will be appended to the base filename.

        box_size : float or tuple of floats, optional
            The box dimensions in nanometers. If None, it will be set based on the coordinates.

        verbose : bool, optional
            If True, prints additional information.

        Raises:
        -------
        ValueError
            If 'chain' column is missing.
        """
        if 'chain' not in self.columns:
            raise ValueError("Column 'chain' is required to write GRO files per chain.")

        unique_chains = self['chain'].unique()
        for chain_id in unique_chains:
            chain_data = self[self['chain'] == chain_id]
            chain_scene = Scene(chain_data, **self._meta)
            output_filename = f"{base_filename}_{chain_id}.gro"
            if verbose:
                logger.info("Writing chain '%s' to %s", chain_id, output_filename)
            chain_scene.write_gro(output_filename, box_size=box_size, verbose=verbose)

    def awsem_memory_chain_lengths(self, chain=None):
        """Ordered ``[(chain, n_residues), ...]`` of regular protein residues."""
        from molscene.parsers import awsem_gro as _ag
        return _ag.chain_lengths(self, chain)

    def write_awsem_gro(self, file_name=None, chain=None):
        """Write an AWSEM fragment-memory (``awsem_gro``) file.

        The bespoke "Structure-Based gro" consumed by OpenAWSEM's
        ``fragment_memory_term`` -- *not* a standard GROMACS gro.  Returns an
        :class:`io.StringIO` when ``file_name`` is ``None``.
        """
        from molscene.parsers import awsem_gro as _ag
        text = _ag.write_awsem_gro(self, file_name, chain=chain)
        return io.StringIO(text) if file_name is None else None

    def write_awsem_memory_gro(self, file_name=None, chain=None):
        """Deprecated alias for :meth:`write_awsem_gro`."""
        return self.write_awsem_gro(file_name, chain=chain)

    # get methods
    def get_coordinates(self):
        return self[['x', 'y', 'z']]

    def get_sequence(self):
        """
        Return the sequence for each chain in the Scene as a dictionary.
        Uses _protein_residues, _DNA_residues, and _RNA_residues for mapping.
        Unknown residues are mapped to 'X'.
        Returns
        -------
        dict
            Mapping from chain to sequence string.
        """
        seq_dict = {}
        # Group by chain and resSeq
        grouped = self.sort_values(['chain', 'resid']).drop_duplicates(
            subset=['chain', 'resid']
        )
        for chain_id, group in grouped.groupby('chain'):
            seq = ''
            for _, row in group.iterrows():
                res = str(row['resname']).strip()
                # Try protein, then DNA, then RNA
                if res in _protein_residues:
                    seq += _protein_residues[res]
                elif res in _DNA_residues:
                    seq += _DNA_residues[res]
                elif res in _RNA_residues:
                    seq += _RNA_residues[res]
                else:
                    seq += 'X'
            seq_dict[chain_id] = seq
        return seq_dict

    def set_coordinates(self, coordinates):
        self[['x', 'y', 'z']] = coordinates

    def copy(self, deep=True):
        return Scene(super().copy(deep), **self._meta)

    def correct_modified_aminoacids(self):
        out = self.copy()
        if 'modified_residues' in self._meta:
            for i, row in out.modified_residues.iterrows():
                sel = ((out['resname'] == row['resname']) &
                       (out['chain'] == row['chain']) &
                       (out['resid'] == row['resid']))
                out.loc[sel, 'resname'] = row['stdRes']
        return out

    def rotate(self, rotation_matrix):
        return self.dot(rotation_matrix)

    def translate(self, other):
        new = self.copy()
        new.set_coordinates(self.get_coordinates() + other)
        return new

    def dot(self, other):
        new = self.copy()
        new.set_coordinates(self.get_coordinates().dot(other))
        return new

    def transform(self, transformation: "Transformation") -> "Scene":
        """
        Apply a :class:`Transformation` to the coordinates.

        Multi-frame coordinate stacks (see :meth:`set_coordinate_frames`) are
        transformed frame-wise.

        Parameters
        ----------
        transformation : Transformation

        Returns
        -------
        Scene
            New scene with the transformation applied; metadata preserved.
        """
        if not isinstance(transformation, Transformation):
            raise TypeError(
                f"transform expects a Transformation, got {type(transformation).__name__}; "
                "wrap raw matrices with Transformation.from_matrix(R, t)."
            )

        out = self.copy(deep=True)
        if 'coordinate_frames' in self._meta:
            frames = self.get_coordinate_frames()
            new_frames = transformation.apply(frames)
            out._meta['coordinate_frames'] = new_frames
            out.set_coordinates(new_frames[0])
        else:
            coords = self.get_coordinates().to_numpy()
            out.set_coordinates(transformation.apply(coords))
        return out

    def morph_segment(
        self,
        chain: str,
        resid_range,
        t_start: "Transformation",
        t_end: "Transformation",
        method: str = "sclerp",
    ) -> "Scene":
        """
        Per-residue rigid blend between two anchor transformations.

        Each residue in ``chain`` whose ``resid`` is in ``resid_range`` is
        moved rigidly by an interpolated transformation
        ``T_i = t_start.interpolate(t_end, α_i, method=method)``, where
        ``α_i = i / (N − 1)`` runs from 0 at the first listed residue to 1 at
        the last. Atoms outside ``chain`` / ``resid_range`` are untouched.

        This is the per-residue local-frame morph used by the CaMKII linker
        builder: ``method='sclerp'`` (default) sweeps every residue along the
        common screw axis between the two anchors, propagating the motion
        smoothly across an intervening flexible region. With ``method='slerp'``
        the rotation and translation channels evolve independently.

        Intra-residue geometry is preserved exactly (each residue moves as a
        rigid body); inter-residue backbone bonds may stretch by a small
        amount across the segment — acceptable for an initial model that
        downstream MD/minimisation will relax.

        Parameters
        ----------
        chain : str
            Chain identifier; only atoms with ``chain == chain`` and ``resid``
            in ``resid_range`` participate.
        resid_range : iterable of int
            Residues to morph, listed in the order they appear along the
            segment (e.g. ``range(275, 341)``). Order determines ``α``.
        t_start, t_end : Transformation
            Anchor transformations at ``α = 0`` and ``α = 1`` respectively.
        method : {'sclerp', 'slerp'}, optional
            Interpolation mode forwarded to :meth:`Transformation.interpolate`.

        Returns
        -------
        Scene
            New scene with the segment's atoms repositioned; metadata
            preserved.
        """
        if not isinstance(t_start, Transformation) or not isinstance(t_end, Transformation):
            raise TypeError("t_start and t_end must be Transformation instances")

        resids = list(resid_range)
        if not resids:
            return self.copy(deep=True)

        out = self.copy(deep=True)
        coords = out.get_coordinates().to_numpy().copy()

        n = len(resids)
        for i, resid in enumerate(resids):
            alpha = 0.0 if n == 1 else i / (n - 1)
            T_i = t_start.interpolate(t_end, alpha, method=method)
            mask = ((out['chain'] == chain) & (out['resid'] == resid)).to_numpy()
            if not mask.any():
                continue
            coords[mask] = T_i.apply(coords[mask])

        out.set_coordinates(coords)
        return out

    def match(self, other: "Scene", match=None) -> Tuple["Scene", "Scene"]:
        """
        Pair atoms between two scenes using a :class:`Matching` strategy.

        Parameters
        ----------
        other : Scene
            The second scene to pair against.
        match : Matching | str | tuple | callable | None, optional
            Matching strategy. See :func:`molscene.matching.as_matching` for
            accepted forms. ``None`` (default) uses :class:`OrderMatching`
            which requires the two scenes to already have the same length and
            row order.

        Returns
        -------
        (Scene, Scene)
            Two scenes of equal length whose rows correspond atom-for-atom.
        """
        return _as_matching(match).pair(self, other)

    def compute_transformation(
        self,
        reference: "Scene",
        match=None,
    ) -> "Transformation":
        """
        Compute the rigid-body :class:`Transformation` that least-squares
        superposes ``self`` onto ``reference``.

        Subset fitting is done by composing :meth:`select` with this method::

            T = scene.select("name CA").compute_transformation(
                    ref.select("name CA"))
            moved = scene.transform(T)

        The returned :class:`Transformation` carries an ``rmsd`` attribute
        holding the residual over the atoms used for the fit.

        Parameters
        ----------
        reference : Scene
        match : Matching | str | tuple | callable | None, optional
            Strategy for pairing atoms between ``self`` and ``reference``.

        Returns
        -------
        Transformation
        """
        mobile, target = self.match(reference, match=match)
        P = mobile.get_coordinates().to_numpy()
        Q = target.get_coordinates().to_numpy()
        return Transformation.from_kabsch(P, Q)

    def superpose(self, reference: "Scene", match=None) -> "Scene":
        """
        Return a copy of ``self`` rigid-body-aligned onto ``reference``.

        Equivalent to ``self.transform(self.compute_transformation(reference,
        match=match))``.
        """
        return self.transform(self.compute_transformation(reference, match=match))

    def rmsd(
        self,
        other: "Scene",
        match=None,
        align: bool = False,
    ) -> float:
        """
        Root-mean-square deviation between two scenes.

        Parameters
        ----------
        other : Scene
        match : Matching | str | tuple | callable | None, optional
            Strategy for pairing atoms. ``None`` uses :class:`OrderMatching`.
        align : bool, optional
            If ``True``, return the RMSD after optimal Kabsch superposition;
            otherwise return the positional RMSD as-is.

        Returns
        -------
        float
        """
        mobile, target = self.match(other, match=match)
        P = mobile.get_coordinates().to_numpy()
        Q = target.get_coordinates().to_numpy()

        if align:
            return Transformation.from_kabsch(P, Q).rmsd

        diff = P - Q
        return float(np.sqrt(np.mean(np.einsum('ij,ij->i', diff, diff))))

    def _resolve_selection(self, selection) -> "Scene":
        """Resolve a selection argument to a sub-:class:`Scene`.

        Accepts ``None`` (the whole Scene), a molselect string, an existing
        ``Scene``, or a boolean mask / index array understood by ``DataFrame.loc``.
        """
        if selection is None:
            return self
        if isinstance(selection, Scene):
            return selection
        if isinstance(selection, str):
            return self.select(selection)
        return Scene(self.loc[selection].copy(), **self._meta)

    def distance_map(self, selection=None, complementary_selection=None, *,
                     sparse=False, cutoff=None, by=None, reduce="min",
                     threshold=_DM_UNSET) -> Union[np.ndarray, tuple]:
        """Distance map between two atom selections.

        Parameters
        ----------
        selection, complementary_selection : None | str | Scene | mask, optional
            Atom selections (a molselect string, a sub-``Scene``, a boolean mask,
            or an index array; ``None`` = all atoms).  When
            ``complementary_selection`` is ``None`` the result is the **symmetric
            self-map** of ``selection``; otherwise it is the **rectangular
            cross-map** between ``selection`` (rows) and ``complementary_selection``
            (cols).
        sparse : bool, optional
            If ``False`` (default) return a dense ``ndarray``.  If ``True`` return a
            sparse ``(row, col, data, shape)`` tuple holding only pairs whose
            distance is ``<= cutoff``.
        cutoff : float, optional
            Required when ``sparse=True``.
        by : {None, 'residue'}, optional
            If ``'residue'``, aggregate atom-atom distances to a per-residue map
            (grouping on the ``residue`` column) using ``reduce`` — i.e. the
            classic minimum-distance contact map.
        reduce : {'min', 'max', 'mean'}, optional
            Reduction used when ``by='residue'`` (default ``'min'``).
        threshold : optional
            Deprecated back-compat shim for the old ``distance_map(threshold=...)``
            signature: ``None`` → dense all-atom matrix; ``float`` → legacy sparse
            ``(pairs, dists)`` from :meth:`distance_map_sparse`.

        Returns
        -------
        numpy.ndarray or tuple
            Dense ``(L, M)`` distance matrix, or a sparse
            ``(row, col, data, shape)`` tuple.

        Examples
        --------
        >>> s.distance_map("name CA")                      # CA self contact map
        >>> s.distance_map("chain A", "chain B")           # interface cross map
        >>> s.virtual_cb().distance_map()                  # CB_force map
        >>> s.distance_map(by="residue", reduce="min",     # min-distance contact map
        ...                sparse=True, cutoff=12.0)
        """
        # Backwards-compatible threshold API.
        if threshold is not _DM_UNSET:
            return (self.distance_map_dense() if threshold is None
                    else self.distance_map_sparse(threshold))

        if sparse and cutoff is None:
            raise ValueError("distance_map(sparse=True) requires a cutoff distance")
        if reduce not in ("min", "max", "mean"):
            raise ValueError(f"reduce must be 'min', 'max' or 'mean', got {reduce!r}")

        self_map = complementary_selection is None
        A = self._resolve_selection(selection)
        B = A if self_map else self._resolve_selection(complementary_selection)

        coordsA = A.get_coordinates().to_numpy()
        coordsB = coordsA if self_map else B.get_coordinates().to_numpy()

        if by is None:
            if not sparse:
                return contacts.dense_atom_map(coordsA, coordsB, self_map)
            return contacts.sparse_atom_map(coordsA, coordsB, cutoff, self_map)

        if by != "residue":
            raise ValueError(f"by must be None or 'residue', got {by!r}")
        resA = A["residue"].to_numpy()
        resB = resA if self_map else B["residue"].to_numpy()
        if not sparse:
            return contacts.dense_residue_map(coordsA, coordsB, resA, resB, self_map, reduce)
        return contacts.sparse_residue_map(coordsA, coordsB, resA, resB, cutoff, self_map, reduce)

    def virtual_cb(self) -> "Scene":
        """Return a Scene of one reconstructed Cβ pseudo-atom per residue.

        The Cβ position is built from the backbone ``N``, ``CA`` and ``C`` atoms
        (ideal geometry), so glycines and residues lacking an explicit Cβ still
        get one.  Only residues having all three backbone atoms are included; the
        ``CA`` row is used as the template for every other column.
        """
        frame = pandas.DataFrame(self)
        bb = frame[frame["name"].isin(["N", "CA", "C"])]
        piv = bb.pivot_table(index="residue", columns="name",
                             values=["x", "y", "z"], aggfunc="first").dropna()
        if len(piv) == 0:
            return Scene(frame.iloc[0:0].copy(), **self._meta)
        N = piv[[("x", "N"), ("y", "N"), ("z", "N")]].to_numpy()
        CA = piv[[("x", "CA"), ("y", "CA"), ("z", "CA")]].to_numpy()
        C = piv[[("x", "C"), ("y", "C"), ("z", "C")]].to_numpy()
        cb = contacts.cb_from_backbone(N, CA, C)

        template = (frame[(frame["name"] == "CA") & (frame["residue"].isin(piv.index))]
                    .drop_duplicates("residue").set_index("residue")
                    .loc[piv.index].reset_index())
        template["x"], template["y"], template["z"] = cb[:, 0], cb[:, 1], cb[:, 2]
        template["name"] = "CB"
        return Scene(template, **self._meta)
    
    def distance_map_dense(self) -> np.ndarray:
        """
        Dense n×n distance matrix.
        Equivalent to your original, but via pdist/squareform for speed.
        """
        coords = self.get_coordinates().to_numpy()
        return distance.squareform(distance.pdist(coords))


    def distance_map_sparse(self, threshold: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fast, memory-light "sparse" distances ≤ threshold.

        Returns
        -------
        pairs : numpy.ndarray
            ``(M, 2)`` array of index pairs ``[i, j]``.
        dists : numpy.ndarray
            ``(M,)`` array of corresponding distances.
        """
        if threshold is None:
            raise ValueError("Must supply a threshold for sparse distance_map")

        coords = self.get_coordinates().to_numpy()
        tree = cKDTree(coords)
        pairs = tree.query_pairs(threshold, output_type='ndarray')  # shape (N, 2)

        diffs = coords[pairs[:, 0]] - coords[pairs[:, 1]]
        dists = np.linalg.norm(diffs, axis=1)

        # symmetric pairs: stack (i,j) and (j,i) as rows
        pairs_sym = np.vstack([pairs, pairs[:, ::-1]])  # shape (2N, 2)
        dists_sym = np.tile(dists, 2)

        return pairs_sym, dists_sym

    def get_center(self) -> pandas.Series:
        """
        Compute the centroid (geometric center) of the atomic coordinates.

        Returns
        -------
        pandas.Series
            A Series with index ['x','y','z'] giving the mean of each coordinate.
        """
        # select the three coord columns and take their column‐wise mean
        return self[['x','y','z']].mean()

    def center(self) -> "Scene":
        """
        Return a new Scene with coordinates shifted so the centroid is at the origin.

        Returns
        -------
        Scene
            A new Scene object with centered coordinates.
        """
        ctr = self.get_center()
        # make a shallow copy of metadata and DataFrame
        out = self.copy(deep=True)
        # subtract the centroid Series from each row (axis=1 => align on column names)
        out[['x','y','z']] = out[['x','y','z']].sub(ctr, axis=1)
        return out


    def __repr__(self):
        try:
            return f'<Scene ({len(self)})>\n{super().__repr__()}'
        except Exception:
            return '<Scene (Uninitialized)>'

    def __add__(self, other: Union["Scene", float, Sequence, pandas.Series]) -> "Scene":
        if isinstance(other, Scene):
            logging.debug("Scene + Scene: concatenation")
            df = pandas.concat([self, other], ignore_index=True)
            return Scene(df, **self._meta)
        
        logging.debug(f"Scene + {type(other)}: translation")
        delta = _as_delta(other).to_numpy()  # shape (3,)
        out = self.copy(deep=True)
        
        if 'coordinate_frames' in self._meta:
            logging.debug("Scene + vector: multi-frame translation")
            frames = self.get_coordinate_frames()
            new_frames = frames + delta[None, None, :]
            out._meta['coordinate_frames'] = new_frames
            out.set_coordinates(new_frames[0])
        
        else:
            logging.debug("Scene + vector: single-frame translation")
            out[['x','y','z']] = out[['x','y','z']] + delta
        return out

    def __radd__(self, other):
        logging.debug(f"{type(other)} + Scene: __radd__ called")
        return self.__add__(other) 
    
    def __sub__(self, other: Union["Scene", float, Sequence, pandas.Series]) -> "Scene":
        if isinstance(other, Scene):
            logging.debug("Scene - Scene: remove atoms with matching index")
            mask = ~self['index'].isin(other['index'])
            df = self.loc[mask].reset_index(drop=True)
            return Scene(df, **self._meta)
        
        logging.debug(f"Scene - {type(other)}: translation by -delta")
        delta = _as_delta(other).to_numpy()
        out = self.copy(deep=True)
        
        if 'coordinate_frames' in self._meta:
            logging.debug("Scene - vector: multi-frame translation")
            frames = self.get_coordinate_frames()
            new_frames = frames - delta[None, None, :]
            out._meta['coordinate_frames'] = new_frames
            out.set_coordinates(new_frames[0])
        
        else:
            logging.debug("Scene - vector: single-frame translation")
            out[['x','y','z']] = out[['x','y','z']].to_numpy() - delta
        return out

    def __rsub__(self, other: Union[float, Sequence, pandas.Series]):
        logging.debug(f"{type(other)} - Scene: elementwise subtraction")
        delta = _as_delta(other).to_numpy()
        out = self.copy(deep=True)
        
        if 'coordinate_frames' in self._meta:
            logging.debug("vector - Scene: multi-frame")
            frames = self.get_coordinate_frames()
            new_frames = delta[None, None, :] - frames
            out._meta['coordinate_frames'] = new_frames
            out.set_coordinates(new_frames[0])
        
        else:
            logging.debug("vector - Scene: single-frame")
            out[['x','y','z']] = delta - out[['x','y','z']].to_numpy()
        return out

    def __mul__(self, other: Union[float, Sequence, pandas.Series]) -> "Scene":
        logging.debug(f"Scene * {type(other)}: scaling")
        factor = _as_delta(other).to_numpy()
        out = self.copy(deep=True)
        
        if 'coordinate_frames' in self._meta:
            logging.debug("Scene * vector: multi-frame scaling")
            frames = self.get_coordinate_frames()
            new_frames = frames * factor[None, None, :]
            out._meta['coordinate_frames'] = new_frames
            out.set_coordinates(new_frames[0])
        
        else:
            logging.debug("Scene * vector: single-frame scaling")
            out[['x','y','z']] = out[['x','y','z']].to_numpy() * factor
        return out

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other: Union[float, Sequence, pandas.Series]) -> "Scene":
        logging.debug(f"Scene / {type(other)}: division")
        divisor = _as_delta(other).to_numpy()
        out = self.copy(deep=True)
        
        if 'coordinate_frames' in self._meta:
            logging.debug("Scene / vector: multi-frame division")
            frames = self.get_coordinate_frames()
            new_frames = frames / divisor[None, None, :]
            out._meta['coordinate_frames'] = new_frames
            out.set_coordinates(new_frames[0])
        
        else:
            logging.debug("Scene / vector: single-frame division")
            out[['x','y','z']] = out[['x','y','z']].to_numpy() / divisor
        
        return out

    def __neg__(self) -> "Scene":
        logging.debug("Scene: negation/reflection")
        out = self.copy(deep=True)
        
        if 'coordinate_frames' in self._meta:
            logging.debug("Scene: multi-frame negation")
            frames = self.get_coordinate_frames()
            new_frames = -frames
            out._meta['coordinate_frames'] = new_frames
            out.set_coordinates(new_frames[0])
        else:
        
            logging.debug("Scene: single-frame negation")
            out[['x','y','z']] = -out[['x','y','z']].to_numpy()
        return out

    @property
    def _constructor(self):
        def _create_scene_if_complete(particles, *args, **kwargs):
            cols = getattr(particles, 'columns', None)
            if cols is not None and all(col in cols for col in self._columns.keys()):
                return Scene(particles, **kwargs)
            else:
                return pandas.DataFrame(particles, *args, **kwargs)
        return _create_scene_if_complete

    def __finalize__(self, other, method=None, **kwargs):
        """Give each derived (or concatenated) scene its own copy of ``_meta``."""
        if isinstance(other, Scene):
            self.__dict__['_meta'] = dict(other.__dict__.get('_meta', {}))
        elif method == "concat" and hasattr(other, "objs"):
            merged: dict = {}
            for obj in other.objs:
                meta = getattr(obj, '_meta', None)
                if isinstance(meta, dict):
                    merged.update(meta)
            self.__dict__['_meta'] = merged
        elif '_meta' not in self.__dict__:
            self.__dict__['_meta'] = {}
        return self

    def __getattribute__(self, name):
        """
        Override attribute lookup only to provide access to items stored in _meta.
        All normal attributes (including methods, and DataFrame properties like 'columns')
        are obtained via the standard mechanism.
        """
        try:
            return super().__getattribute__(name)
        except AttributeError:
            # If not found normally, check if it is stored in _meta.
            _meta = object.__getattribute__(self, '_meta') if '_meta' in self.__dict__ else {}
            if name in _meta:
                return _meta[name]
            raise AttributeError(f"{self.__class__.__name__} has no attribute {name}")

    def __setattr__(self, attr, value):
        if (attr == '_meta'
                or attr in type(self)._internal_names_set
                or attr in type(self)._metadata):
            object.__setattr__(self, attr, value)
            return

        # If the attribute name is one of the DataFrame's columns, assign to that column.
        try:
            columns = super().__getattribute__('columns')
        except AttributeError:
            columns = None

        if columns is not None and attr in columns:
            self[attr] = value
            return

        # If it's a built-in DataFrame attribute, set it normally.
        if hasattr(pandas.DataFrame, attr):
            super().__setattr__(attr, value)
        else:
            # Otherwise, store it in _meta.
            self._meta[attr] = value

    __array_priority__ = 1000  # Ensure Scene takes precedence in numpy ops

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Handle numpy ufuncs like np.add, np.subtract, np.multiply, etc.
        Route to corresponding dunder methods.
        """
       
        if method != "__call__":
            return NotImplemented

        # Unpack inputs
        logging.debug(f"Scene.__array_ufunc__({ufunc}, {method}, {inputs})")
        if ufunc == np.add:
            a, b = inputs
            if isinstance(a, Scene):
                return a.__add__(b)
            elif isinstance(b, Scene):
                return b.__radd__(a)
        elif ufunc == np.subtract:
            a, b = inputs
            if isinstance(a, Scene):
                return a.__sub__(b)
            elif isinstance(b, Scene):
                return b.__rsub__(a)
        elif ufunc == np.multiply:
            a, b = inputs
            if isinstance(a, Scene):
                return a.__mul__(b)
            elif isinstance(b, Scene):
                return b.__rmul__(a)
        elif ufunc == np.true_divide:
            a, b = inputs
            if isinstance(a, Scene):
                return a.__truediv__(b)
        elif ufunc == np.negative:
            (a,) = inputs
            if isinstance(a, Scene):
                return a.__neg__()

        return NotImplemented

# helpers outside the class

def _as_delta(other) -> pandas.Series:
    """
    Normalize a scalar, sequence of length-3, or Series
    into a pandas.Series indexed ['x','y','z'].
    """
    if isinstance(other, pandas.Series):
        # Check that the series has 'x', 'y', 'z' as index, and reorder if necessary
        if set(other.index) != {'x', 'y', 'z'}:
            raise ValueError(f"Series index must be ['x','y','z'], not {other.index}")
        # Reorder the series to match ['x','y','z']
        delta = other.reindex(['x','y','z']).astype(float)
    elif isinstance(other, (int, float)):
        delta = pandas.Series([other]*3, index=['x','y','z'], dtype=float)
    else:
        arr = np.asarray(other, float)
        if arr.shape == (3,):
            delta = pandas.Series(arr, index=['x','y','z'])
        else:
            raise ValueError(f"Cannot interpret {other!r} as a 3-vector")
    return delta


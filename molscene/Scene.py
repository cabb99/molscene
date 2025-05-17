import pandas
import numpy as np
import io

"""
Python library to allow easy handling of coordinate files for molecular dynamics using pandas DataFrames.
"""
from . import utils

__author__ = 'Carlos Bueno'

_protein_residues = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                     'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                     'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                     'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
                     'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}

_DNA_residues = {'DA': 'A', 'DC': 'C', 'DG': 'G', 'DT': 'T'}

_RNA_residues = {'A': 'A', 'C': 'C', 'G': 'G', 'U': 'U'}

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
    
    _columns = {'recname': 'Record name',
                'serial': 'Atom serial number',
                'name': 'Atom name',
                'altLoc': 'Alternate location indicator',
                'resName': 'Residue name',
                'chainID': 'Chain identifier',
                'resSeq': 'Residue sequence number',
                'iCode': 'Code for insertion of residues',
                'x': 'Orthogonal coordinates for X in Angstroms',
                'y': 'Orthogonal coordinates for Y in Angstroms',
                'z': 'Orthogonal coordinates for Z in Angstroms',
                'occupancy': 'Occupancy',
                'tempFactor': 'Temperature factor',
                'element': 'Element symbol',
                'charge': 'Charge on the atom',
                'model': 'Model number',
                # 'res_index': 'Residue index',
                # 'chain_index': 'Chain index',
                'molecule': 'Molecule name',
                'resname': 'Residue name'}
    
    
    # Initialization
    def __init__(self, particles, altLoc='A', model=1, **kwargs):
        """Create an empty scene from particles.
        The Scene object is a wraper of a pandas DataFrame with extra information"""
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
        
        if 'chainID' not in self.columns:
            self['chainID'] = ['A'] * len(self)
        if 'resSeq' not in self.columns:
            self['resSeq'] = [1] * len(self)
        if 'iCode' not in self.columns:
            self['iCode'] = [''] * len(self)
        if 'altLoc' not in self.columns:
            self['altLoc'] = [''] * len(self)
        if 'model' not in self.columns:
            self['model'] = [1] * len(self)
        if 'name' not in self.columns:
            self['name'] = [f'P{i:03}' for i in range(len(self))]
        if 'element' not in self.columns:
            self['element'] = ['C'] * len(self)
        if 'occupancy' not in self.columns:
            self['occupancy'] = [1.0] * len(self)
        if 'tempFactor' not in self.columns:
            self['tempFactor'] = [1.0] * len(self)
        if 'resName' not in self.columns:
            self['resName'] = [''] * len(self)
        
        # Create an integer index for the chains
        if 'chain_index' not in self.columns:
            chain_map = {b: a for a, b in enumerate(self['chainID'].unique())}
            self['chain_index'] = self['chainID'].replace(chain_map).astype(int)

        # Create an integer index for the residues
        if 'res_index' not in self.columns:
            # Construct a global unique residue key
            residue_keys = (
                self['chain_index'].astype(str) +
                self['resSeq'].astype(str) +
                self['iCode'].astype(str)
            )

            # Get unique residue keys and map to integers
            unique_keys = pandas.Series(residue_keys.unique())
            key_to_index = dict(zip(unique_keys, range(len(unique_keys))))

            # Map each residue key to its index
            self['res_index'] = residue_keys.replace(key_to_index).astype(int)

        # Create an integer index for the atoms
        if 'atom_index' not in self.columns:
            self['atom_index'] = range(len(self))

        # Add metadata
        for attr, value in kwargs.items():
            self._meta[attr] = value

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

    def select(self, **kwargs):
        index = self.index
        sel = pandas.Series([True] * len(index), index=index)
        for key in kwargs:
            if key == 'altLoc':
                sel &= (self['altLoc'].isin(['', '.'] + kwargs['altLoc']))
            elif key == 'model':
                sel &= (self['model'].isin(kwargs['model']))
            else:
                sel &= (self[key].isin(kwargs[key]))

        # Assert there are not repeated atoms
        index = self[sel][['chain_index', 'res_index', 'name']]
        if len(index.duplicated()) == 0:
            print("Duplicated atoms found")
            print(index[index.duplicated()])
            self._meta['duplicated'] = True

        return Scene(self[sel], **self._meta)

    def split_models(self):
        # TODO: Implement splitting based on model and altLoc.
        # altLoc can be present in multiple regions (1zir)
        pass

    #        for m in self['model'].unique():
    #            for a in sel:
    #                pass

    @classmethod
    def from_pdb(cls, file, **kwargs):
        def pdb_line(line):
            l = dict(recname=line[0:6].strip(),
                     serial=line[6:11],
                     name=line[12:16].strip(),
                     altLoc=line[16:17].strip(),
                     resName=line[17:20].strip(),
                     chainID=line[21:22].strip(),
                     resSeq=line[22:26],
                     iCode=line[26:27].strip(),
                     x=line[30:38],
                     y=line[38:46],
                     z=line[46:54],
                     occupancy=line[54:60].strip(),
                     tempFactor=line[60:66].strip(),
                     element=line[76:78].strip(),
                     charge=line[78:80].strip())
            return l

        with open(file, 'r') as pdb:
            lines = []
            mod_lines = []
            model_numbers = []
            model_number = 1
            for i, line in enumerate(pdb):
                if len(line) > 6:
                    header = line[:6]
                    if header == 'ATOM  ' or header == 'HETATM':
                        try:
                            lines += [pdb_line(line)]
                        except ValueError as e:
                            print(e)
                            print(f"Error in line {i}")
                            print(line)
                            raise ValueError
                        model_numbers += [model_number]
                    elif header == "MODRES":
                        m = dict(recname=str(line[0:6]).strip(),
                                 idCode=str(line[7:11]).strip(),
                                 resName=str(line[12:15]).strip(),
                                 chainID=str(line[16:17]).strip(),
                                 resSeq=int(line[18:22]),
                                 iCode=str(line[22:23]).strip(),
                                 stdRes=str(line[24:27]).strip(),
                                 comment=str(line[29:70]).strip())
                        mod_lines += [m]
                    elif header == "MODEL ":
                        model_number = int(line[10:14])
        pdb_atoms = pandas.DataFrame(lines)
        pdb_atoms = pdb_atoms[['recname', 'serial', 'name', 'altLoc',
                               'resName', 'chainID', 'resSeq', 'iCode',
                               'x', 'y', 'z', 'occupancy', 'tempFactor',
                               'element', 'charge']]
        
        # Apply type conversions and set default values
        pdb_atoms['serial'] = pandas.to_numeric(pdb_atoms['serial'], errors='coerce').fillna(0).astype(int)
        pdb_atoms['resSeq'] = pandas.to_numeric(pdb_atoms['resSeq'], errors='coerce').fillna(0).astype(int)
        pdb_atoms['x'] = pandas.to_numeric(pdb_atoms['x'], errors='coerce').fillna(0.0)
        pdb_atoms['y'] = pandas.to_numeric(pdb_atoms['y'], errors='coerce').fillna(0.0)
        pdb_atoms['z'] = pandas.to_numeric(pdb_atoms['z'], errors='coerce').fillna(0.0)
        pdb_atoms['occupancy'] = pandas.to_numeric(pdb_atoms['occupancy'], errors='coerce').fillna(1.0)
        pdb_atoms['tempFactor'] = pandas.to_numeric(pdb_atoms['tempFactor'], errors='coerce').fillna(1.0)
        pdb_atoms['charge'] = pandas.to_numeric(pdb_atoms['tempFactor'], errors='coerce').fillna(0.0)
        pdb_atoms['model'] = model_numbers
        pdb_atoms['molecule'] = 0

        if len(mod_lines) > 0:
            kwargs.update(dict(modified_residues=pandas.DataFrame(mod_lines)))

        return cls(pdb_atoms, **kwargs)

    # @classmethod
    # def from_cif(cls, file, **kwargs):
    #     _cif_pdb_rename = {'id': 'serial',
    #                        'label_atom_id': 'name',
    #                        'label_alt_id': 'altLoc',
    #                        'label_comp_id': 'resName',
    #                        'label_asym_id': 'chainID',
    #                        'label_seq_id': 'resSeq',
    #                        'pdbx_PDB_ins_code': 'iCode',
    #                        'Cartn_x': 'x',
    #                        'Cartn_y': 'y',
    #                        'Cartn_z': 'z',
    #                        'occupancy': 'occupancy',
    #                        'B_iso_or_equiv': 'tempFactor',
    #                        'type_symbol': 'element',
    #                        'pdbx_formal_charge': 'charge',
    #                        'pdbx_PDB_model_num': 'model'}

    #     data = []
    #     with open(file) as f:
    #         reader = utils.PdbxReader(f)
    #         reader.read(data)
    #     block = data[0]
    #     atom_data = block.getObj('atom_site')
    #     cif_atoms = pandas.DataFrame([atom_data.getFullRow(i) for i in range(atom_data.getRowCount())],
    #                                  columns=atom_data.getAttributeList(),
    #                                  index=range(atom_data.getRowCount()))
    #     # Rename columns to pdb convention
    #     cif_atoms = cif_atoms.rename(_cif_pdb_rename, axis=1)
    #     for col in cif_atoms.columns:
    #         try:
    #             cif_atoms[col] = cif_atoms[col].astype(float)
    #             if ((cif_atoms[col].astype(int) - cif_atoms[col]) ** 2).sum() == 0:
    #                 cif_atoms[col] = cif_atoms[col].astype(int)
    #             continue
    #         except ValueError:
    #             pass

    #     cif_atoms['serial'] = pandas.to_numeric(cif_atoms['serial'], errors='coerce').fillna(0).astype(int)
    #     cif_atoms['resSeq'] = pandas.to_numeric(cif_atoms['resSeq'], errors='coerce').fillna(0).astype(int)
    #     cif_atoms['x'] = pandas.to_numeric(cif_atoms['x'], errors='coerce').fillna(0.0)
    #     cif_atoms['y'] = pandas.to_numeric(cif_atoms['y'], errors='coerce').fillna(0.0)
    #     cif_atoms['z'] = pandas.to_numeric(cif_atoms['z'], errors='coerce').fillna(0.0)
    #     cif_atoms['occupancy'] = pandas.to_numeric(cif_atoms['occupancy'], errors='coerce').fillna(1.0)
    #     cif_atoms['tempFactor'] = pandas.to_numeric(cif_atoms['tempFactor'], errors='coerce').fillna(1.0)
    #     cif_atoms['charge'] = pandas.to_numeric(cif_atoms['tempFactor'], errors='coerce').fillna(0.0)


    #     return cls(cif_atoms, **kwargs)
    
    @classmethod
    def from_cif(cls, file_path, **kwargs):
        """
        Extracts only the _atom section from an mmCIF file.

        Args:
            file_path (str): Path to the CIF file.

        Returns:
            list: List of parsed atom data rows.
        """
       
        import re
        atom_data = []
        atom_header = []
        in_atom_section = False
        tokenizer = re.compile(r"""'[^']*'      |  # single-quoted
                                    "[^"]*"     |  # double-quoted
                                    \#[^\n]*    |  # comment
                                    [^\s'"#]+      # unquoted
                                """, re.VERBOSE)

        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith("#"):
                    continue
                
                # Detect the start of the _atom section
                if line.startswith("loop_"):
                    in_atom_section = False  # Reset section flag
                
                elif line.startswith("_atom_site."):
                    atom_header.append(line.split('.')[-1])
                    in_atom_section = True  # Found relevant section, start collecting headers
                
                elif in_atom_section:
                        atom_data.append([
                                            token.strip("'\"")        # strip any surrounding quotes
                                            for token in tokenizer.findall(line)
                                            if not token.startswith('#')  # drop the comment token (and everything after)
                                        ])

        cif_atoms = pandas.DataFrame(atom_data,columns=atom_header)
        
        # Rename columns to pdb convention
        _cif_pdb_rename = {'id': 'serial',
                           'label_atom_id': 'name',
                           'label_alt_id': 'altLoc',
                           'label_comp_id': 'resName',
                           'label_asym_id': 'chainID',
                           'label_seq_id': 'resSeq',
                           'pdbx_PDB_ins_code': 'iCode',
                           'Cartn_x': 'x',
                           'Cartn_y': 'y',
                           'Cartn_z': 'z',
                           'occupancy': 'occupancy',
                           'B_iso_or_equiv': 'tempFactor',
                           'type_symbol': 'element',
                           'pdbx_formal_charge': 'charge',
                           'pdbx_PDB_model_num': 'model'}

        cif_atoms = cif_atoms.rename(_cif_pdb_rename, axis=1)
        for col in cif_atoms.columns:
            try:
                cif_atoms[col] = cif_atoms[col].astype(float)
                if ((cif_atoms[col].astype(int) - cif_atoms[col]) ** 2).sum() == 0:
                    cif_atoms[col] = cif_atoms[col].astype(int)
                continue
            except ValueError:
                pass

        cif_atoms['serial'] = pandas.to_numeric(cif_atoms['serial'], errors='coerce').fillna(0).astype(int)
        cif_atoms['resSeq'] = pandas.to_numeric(cif_atoms['resSeq'], errors='coerce').fillna(0).astype(int)
        cif_atoms['x'] = pandas.to_numeric(cif_atoms['x'], errors='coerce').fillna(0.0)
        cif_atoms['y'] = pandas.to_numeric(cif_atoms['y'], errors='coerce').fillna(0.0)
        cif_atoms['z'] = pandas.to_numeric(cif_atoms['z'], errors='coerce').fillna(0.0)
        cif_atoms['occupancy'] = pandas.to_numeric(cif_atoms['occupancy'], errors='coerce').fillna(1.0)
        cif_atoms['tempFactor'] = pandas.to_numeric(cif_atoms['tempFactor'], errors='coerce').fillna(1.0)
        cif_atoms['charge'] = pandas.to_numeric(cif_atoms['tempFactor'], errors='coerce').fillna(0.0)
                
        return cls(cif_atoms, **kwargs)

    @classmethod
    def from_gro(cls, gro, **kwargs):
        raise NotImplementedError

    @classmethod
    def from_fixPDB(cls, filename=None, pdbfile=None, pdbxfile=None, url=None, pdbid=None,
                    **kwargs):
        """Uses the pdbfixer library to fix a pdb file, replacing non standard residues, removing
        hetero-atoms and adding missing hydrogens. The input is a pdb file location,
        the output is a fixer object, which is a pdb in the openawsem format."""
        import pdbfixer

        filename=str(filename) if filename is not None else None
        pdbfile=str(pdbfile) if pdbfile is not None else None
        pdbxfile=str(pdbxfile) if pdbxfile is not None else None
        url=str(url) if url is not None else None
        pdbid=str(pdbid) if pdbid is not None else None

        fixer = pdbfixer.PDBFixer(filename=filename, pdbfile=pdbfile, pdbxfile=pdbxfile, url=url, pdbid=pdbid)
        fixer.findMissingResidues()
        chains = list(fixer.topology.chains())
        keys = fixer.missingResidues.keys()
        for key in list(keys):
            chain_tmp = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain_tmp.residues())):
                del fixer.missingResidues[key]

        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()  # Warning: importing 'simtk.openmm' is deprecated.  Import 'openmm' instead.
        fixer.addMissingHydrogens(7.0)

        pdb = fixer
        """ Parses a pdb in the openmm format and outputs a table that contains all the information
        on a pdb file """
        cols = ['recname', 'serial', 'name', 'altLoc',
                'resName', 'chainID', 'resSeq', 'iCode',
                'x', 'y', 'z', 'occupancy', 'tempFactor',
                'element', 'charge']
        data = []

        for atom, pos in zip(pdb.topology.atoms(), pdb.positions):
            residue = atom.residue
            chain = residue.chain
            pos = pos.value_in_unit(pdbfixer.pdbfixer.unit.angstrom)
            data += [dict(zip(cols, ['ATOM', int(atom.id), atom.name, '',
                                     residue.name, chain.id, int(residue.id), '',
                                     pos[0], pos[1], pos[2], 0, 0,
                                     atom.element.symbol, '']))]
        atom_list = pandas.DataFrame(data)
        atom_list = atom_list[cols]
        atom_list.index = atom_list['serial']
        return cls(atom_list, **kwargs)
    
    @classmethod
    def from_fixer(cls, fixer, **kwargs):
        import pdbfixer
        pdb = fixer
        """ Parses a pdb in the openmm format and outputs a table that contains all the information
        on a pdb file """
        cols = ['recname', 'serial', 'name', 'altLoc',
                'resName', 'chainID', 'resSeq', 'iCode',
                'x', 'y', 'z', 'occupancy', 'tempFactor',
                'element', 'charge']
        data = []

        for atom, pos in zip(pdb.topology.atoms(), pdb.positions):
            residue = atom.residue
            chain = residue.chain
            pos = pos.value_in_unit(pdbfixer.pdbfixer.unit.angstrom)
            data += [dict(zip(cols, ['ATOM', int(atom.id), atom.name, '',
                                     residue.name, chain.id, int(residue.id), '',
                                     pos[0], pos[1], pos[2], 0, 0,
                                     atom.element.symbol, '']))]
        atom_list = pandas.DataFrame(data)
        atom_list = atom_list[cols]
        atom_list.index = atom_list['serial']
        return cls(atom_list, **kwargs)
    
    @classmethod
    def from_file(cls, filename):
        if filename.endswith('.pdb'):
            return cls.from_pdb(filename)
        elif filename.endswith('.cif'):
            return cls.from_cif(filename)
        elif filename.endswith('.gro'):
            return cls.from_gro(filename)
        else:
            raise ValueError('Unknown file format')
        
    def to_file(self, filename):
        if filename.endswith('.pdb'):
            self.write_pdb(filename)
        elif filename.endswith('.cif'):
            self.write_cif(filename)
        elif filename.endswith('.gro'):
            self.write_gro(filename)
        else:
            raise ValueError('Unknown file format')

    @classmethod
    def concatenate(cls, scene_list):
        #Set chain names
        chainID = []
        name_generator = utils.chain_name_generator()
        for scene in scene_list:
            if 'chainID' not in scene:
                chainID += [next(name_generator)]*len(scene)
            else:
                chains = list(scene['chainID'].unique())
                chains.sort()
                chain_replace = {chain: next(name_generator) for chain in chains}
                chainID += list(scene['chainID'].replace(chain_replace))
        name_generator.close()
        model = pandas.concat(scene_list)
        model['chainID'] = chainID
        model.index = range(len(model))
        return cls(model)

    # Writing
    def write_pdb(self, file_name=None, verbose=False):

        # TODO Add connectivity output
        # Fill empty columns
        if verbose:
            print(f"Writing pdb file ({len(self)} atoms): {file_name}")

        pdb_table = self.copy()
        pdb_table['serial'] = np.arange(1, len(self) + 1) if 'serial' not in pdb_table else pdb_table['serial']
        pdb_table['name'] = 'A' if 'name' not in pdb_table else pdb_table['name']
        pdb_table['altLoc'] = '' if 'altLoc' not in pdb_table else pdb_table['altLoc']
        pdb_table['resName'] = 'R' if 'resName' not in pdb_table else pdb_table['resName']
        pdb_table['chainID'] = 'C' if 'chainID' not in pdb_table else pdb_table['chainID']
        pdb_table['resSeq'] = 1 if 'resSeq' not in pdb_table else pdb_table['resSeq']
        pdb_table['iCode'] = '' if 'iCode' not in pdb_table else pdb_table['iCode']
        assert 'x' in pdb_table.columns, 'Coordinate x not in particle definition'
        assert 'y' in pdb_table.columns, 'Coordinate x not in particle definition'
        assert 'z' in pdb_table.columns, 'Coordinate x not in particle definition'
        pdb_table['occupancy'] = 0 if 'occupancy' not in pdb_table else pdb_table['occupancy']
        pdb_table['tempFactor'] = 0 if 'tempFactor' not in pdb_table else pdb_table['tempFactor']
        pdb_table['element'] = '' if 'element' not in pdb_table else pdb_table['element']
        pdb_table['charge'] = 0 if 'charge' not in pdb_table else pdb_table['charge']

        # Override chain names if molecule is present
        if 'molecule' in pdb_table:
            import string
            cc = utils.chain_name_generator(format='pdb')
            molecules = self['molecule'].unique()
            cc_d = dict(zip(molecules, cc))
            # cc_d = dict(zip(range(1, len(cc) + 1), cc))
            pdb_table['chainID'] = self['molecule'].replace(cc_d)

        # Write pdb file
        lines = ''
        for i, atom in pdb_table.iterrows():
            line = f'ATOM  {i%100000:>5} {atom["name"]:^4} {atom["resName"]:<3} {atom["chainID"]}{atom["resSeq"]:>4}' + \
                   '    ' + \
                   f'{atom.x:>8.3f}{atom.y:>8.3f}{atom.z:>8.3f}' + ' ' * 22 + f'{atom.element:2}' + ' ' * 2
            assert len(line) == 80, f'An item in the atom table is longer than expected\n{line}'
            lines += line + '\n'

        if file_name is None:
            return io.StringIO(lines)
        else:
            with open(file_name, 'w+') as out:
                out.write(lines)

    def write_cif(self, file_name=None, verbose=False):
        """Write a PDBx/mmCIF file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the molecular system being written
        file : file=stdout
            A file to write the file to
        entry : str=None
            The entry ID to assign to the CIF file
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDBx/mmCIF format.  Otherwise, the output file will be invalid.
        """
        """Write out a model to a PDBx/mmCIF file.

        Parameters
        ----------
        topology : Topology
            The Topology defining the model to write
        positions : list
            The list of atomic positions to write
        file : file=stdout
            A file to write the model to
        modelIndex : int=1
            The model number of this frame
        keepIds : bool=False
            If True, keep the residue and chain IDs specified in the Topology
            rather than generating new ones.  Warning: It is up to the caller to
            make sure these are valid IDs that satisfy the requirements of the
            PDBx/mmCIF format.  Otherwise, the output file will be invalid.
        """
        # TODO Add connectivity output
        if verbose:
            print(f"Writing cif file ({len(self)} atoms): {file_name}")

        # Fill empty columns
        pdbx_table = self.copy()
        pdbx_table['serial'] = np.arange(1, len(self) + 1) if 'serial' not in pdbx_table else pdbx_table['serial']
        pdbx_table['name'] = 'A' if 'name' not in pdbx_table else pdbx_table['name']
        pdbx_table['altLoc'] = '?' if 'altLoc' not in pdbx_table else pdbx_table['altLoc']
        pdbx_table['resName'] = 'R' if 'resName' not in pdbx_table else pdbx_table['resName']
        pdbx_table['chainID'] = 'C' if 'chainID' not in pdbx_table else pdbx_table['chainID']
        pdbx_table['resSeq'] = 1 if 'resSeq' not in pdbx_table else pdbx_table['resSeq']
        pdbx_table['resIC'] = 1 if 'resIC' not in pdbx_table else pdbx_table['resIC']
        pdbx_table['iCode'] = '' if 'iCode' not in pdbx_table else pdbx_table['iCode']
        assert 'x' in pdbx_table.columns, 'Coordinate x not in particle definition'
        assert 'y' in pdbx_table.columns, 'Coordinate x not in particle definition'
        assert 'z' in pdbx_table.columns, 'Coordinate x not in particle definition'
        pdbx_table['occupancy'] = 0 if 'occupancy' not in pdbx_table else pdbx_table['occupancy']
        pdbx_table['tempFactor'] = 0 if 'tempFactor' not in pdbx_table else pdbx_table['tempFactor']
        pdbx_table['element'] = 'C' if 'element' not in pdbx_table else pdbx_table['element']
        pdbx_table['model'] = 0 if 'model' not in pdbx_table else pdbx_table['model']

        # If the column is a string convert it to a float
        for col in ['serial', 'resSeq', 'resIC', 'model','charge']:
            pdbx_table[col] = pandas.to_numeric(pdbx_table[col], errors='coerce').fillna(0).astype(int)
        for col in ['x', 'y', 'z', 'occupancy', 'tempFactor']:
            pdbx_table[col] = pandas.to_numeric(pdbx_table[col], errors='coerce').fillna(0.0)
            
        #If the column is a string convert and empty string to a dot
        for col in ['name', 'altLoc', 'resName', 'chainID', 'iCode', 'element']:
            pdbx_table[col] = pdbx_table[col].str.strip().replace('', '.')

        # print(pdbx_table)
        # pdbx_table.fillna('.', inplace=True)
        # pdbx_table.replace(' ', '.', inplace=True)

        lines = ""
        lines += "data_pdbx\n"
        lines += "#\n"
        lines += "loop_\n"
        lines += "_atom_site.group_PDB\n"
        lines += "_atom_site.id\n"
        lines += "_atom_site.label_atom_id\n"
        lines += "_atom_site.label_comp_id\n"
        lines += "_atom_site.label_asym_id\n"
        lines += "_atom_site.label_seq_id\n"
        lines += "_atom_site.label_alt_id\n"
        lines += "_atom_site.auth_atom_id\n"
        lines += "_atom_site.auth_comp_id\n"
        lines += "_atom_site.auth_asym_id\n"
        lines += "_atom_site.auth_seq_id\n"
        lines += "_atom_site.pdbx_PDB_ins_code\n"
        lines += "_atom_site.Cartn_x\n"
        lines += "_atom_site.Cartn_y\n"
        lines += "_atom_site.Cartn_z\n"
        lines += "_atom_site.occupancy\n"
        lines += "_atom_site.B_iso_or_equiv\n"
        lines += "_atom_site.type_symbol\n"
        lines += "_atom_site.pdbx_formal_chrge\n"
        lines += "_atom_site.pdbx_PDB_model_num\n"

        pdbx_table['line'] = 'ATOM'

        def cif_quote(val):
            if val is np.nan:
                return '.'
            if not isinstance(val, str):
                val = str(val)
            if "'" in val and '"' in val:
                # If both quotes are present (unusual), use double quotes and replace double quotes with single quotes
                return '"' + val.replace('"', "'") + '"'
            elif "'" in val:
                return '"' + val + '"'
            elif '"' in val:
                return "'" + val + "'"
            elif any(c.isspace() for c in val) or val == '' or val.startswith('#') or val.startswith(';'):
                #quote the string if it contains spaces or is empty
                return '"' + val + '"'
            else:
                return val

        for col in ['serial',
                    'name', 'resName', 'chainID', 'resSeq', 'iCode',
                    'name', 'resName', 'chainID', 'resSeq','iCode',
                    'x', 'y', 'z',
                    'occupancy', 'tempFactor',
                    'element', 'charge', 'model']:
            pdbx_table['line'] += " "
            pdbx_table['line'] += pdbx_table[col].apply(cif_quote)
        pdbx_table['line'] += '\n'
        lines += ''.join(pdbx_table['line'])
        lines += '#\n'

        if file_name is None:
            return io.StringIO(lines)
        else:
            with open(file_name, 'w+') as out:
                out.write(lines)

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
            print(f"Writing GRO file ({len(self)} atoms): {file_name}")

        # Prepare data
        gro_atoms = self.copy()

        # Ensure required columns are present
        required_columns = ['resSeq', 'resName', 'name', 'x', 'y', 'z']
        for col in required_columns:
            if col not in gro_atoms.columns:
                raise ValueError(f"Column '{col}' is required for writing GRO file.")

        # Handle 'serial' column
        if 'serial' not in gro_atoms.columns:
            gro_atoms['serial'] = np.arange(1, len(gro_atoms) + 1)

        # Convert types and handle formatting
        gro_atoms['resSeq'] = gro_atoms['resSeq'].astype(int) % 100000  # Limit to 5 digits
        gro_atoms['serial'] = gro_atoms['serial'].astype(int) % 100000  # Limit to 5 digits
        gro_atoms['resName'] = gro_atoms['resName'].astype(str).str[:5]
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
                line = f"{atom['resSeq']:5d}{atom['resName']:<5}{atom['name']:>5}{atom['serial']:5d}" \
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
            If 'chainID' column is missing.
        """
        if 'chainID' not in self.columns:
            raise ValueError("Column 'chainID' is required to write GRO files per chain.")

        unique_chains = self['chainID'].unique()
        for chain_id in unique_chains:
            chain_data = self[self['chainID'] == chain_id]
            chain_scene = Scene(chain_data, **self._meta)
            output_filename = f"{base_filename}_{chain_id}.gro"
            if verbose:
                print(f"Writing chain '{chain_id}' to {output_filename}")
            chain_scene.write_gro(output_filename, box_size=box_size, verbose=verbose)

    # get methods
    def get_coordinates(self):
        return self[['x', 'y', 'z']]

    def get_sequence(self):
        pass

    def set_coordinates(self, coordinates):
        self[['x', 'y', 'z']] = coordinates

    def copy(self, deep=True):
        return Scene(super().copy(deep), **self._meta)

    def correct_modified_aminoacids(self):
        out = self.copy()
        if 'modified_residues' in self._meta:
            for i, row in out.modified_residues.iterrows():
                sel = ((out['resName'] == row['resName']) &
                       (out['chainID'] == row['chainID']) &
                       (out['resSeq'] == row['resSeq']))
                out.loc[sel, 'resName'] = row['stdRes']
        return out

    def rotate(self, rotation_matrix):
        return self.dot(rotation_matrix)

    def translate(self, other):
        new = self.copy()
        new.at[:, ['x', 'y', 'z']] = self.get_coordinates() + other
        return new

    def dot(self, other):
        new = self.copy()
        new.at[:, ['x', 'y', 'z']] = self.get_coordinates().dot(other)
        return new

    def __repr__(self):
        try:
            return f'<Scene ({len(self)})>\n{super().__repr__()}'
        except AttributeError:
            return '<Scene (Uninitialized)>'


    def __add__(self, other):
        if isinstance(other, self.__class__):
            concatenated = pandas.concat([self, other], axis=0)
            return Scene(concatenated)
        else:
            coord = self.get_coordinates()
            temp = self.copy()
            temp.set_coordinates(coord + other)
            return temp

    def __radd__(self, other):
        if isinstance(other, self.__class__):
            return other + s
        else:
            return Scene(super().__add__(other))

    def __mul__(self, other):
        coord = self.get_coordinates()
        temp = self.copy()
        temp.set_coordinates(coord * other)
        return temp

    def __rmul__(self, other):
        coord = self.get_coordinates()
        temp = self.copy()
        temp.set_coordinates(other * coord)
        return temp

    @property
    def _constructor(self):
        # Check if the DataFrame contains all the required columns
        if all(col in self.columns for col in self._columns.keys()):
            return Scene
        else:
            print("Warning: Missing required columns. Returning a standard DataFrame.")
            print([col for col in self._columns.keys() if col not in self.columns])
            return pandas.DataFrame

    # def __getattr__(self, attr):
    #     if '_meta' in self.__dict__ and attr in self._meta:
    #         return self._meta[attr]
    #     elif attr in self.columns:
    #         return self[attr]
    #     else:
    #         raise AttributeError(f"type object {str(self.__class__)} has no attribute {str(attr)}")
        
    # def __getattr__(self, attr):
    #     # Safely retrieve _meta without triggering __getattr__ again.
    #     meta = object.__getattribute__(self, '_meta') if '_meta' in self.__dict__ else {}

    #     if attr in meta:
    #         return meta[attr]

    #     # Use object.__getattribute__ to get columns without recursion.
    #     cols = object.__getattribute__(self, 'columns')
    #     if attr in cols:
    #         return self[attr]

    #     raise AttributeError(f"{self.__class__.__name__} has no attribute {attr}")

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
        # Always set _meta normally.
        if attr == '_meta':
            super().__setattr__(attr, value)
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

if __name__ == '__main__':
    particles = pandas.DataFrame([[0, 0, 0],
                                  [0, 1, 0],
                                  [0, 0, 1]],
                                 columns=['x', 'y', 'z'])
    s = Scene(particles)
    s.write_pdb('test.pdb')

    s = Scene.from_pdb('test.pdb')

    s.write_cif('test.cif')

    s = Scene.from_cif('test.cif')

    s = Scene.from_fixPDB(pdbid='1JGE')

    s1 = Scene(particles)
    s1.write_pdb('test.pdb')
    s2 = Scene.from_pdb('test.pdb')
    s2.write_cif('test.cif')
    s3 = Scene.from_cif('test.cif')
    s3.write_pdb('test2.pdb')
    s4 = Scene.from_pdb('test2.pdb')

    s1.to_csv('particles_1.csv')
    s2.to_csv('particles_2.csv')
    s3.to_csv('particles_3.csv')
    s4.to_csv('particles_4.csv')

"""
import numpy as np
import pandas as pd

def h5store(filename, df, **kwargs):
    store = pd.HDFStore(filename)
    store.put('mydata', df)
    store.get_storer('mydata').attrs.metadata = kwargs
    store.close()

def h5load(store):
    data = store['mydata']
    metadata = store.get_storer('mydata').attrs.metadata
    return data, metadata

a = pd.DataFrame(
    data=pd.np.random.randint(0, 100, (10, 5)), columns=list('ABCED'))

filename = '/tmp/data.h5'
metadata = dict(local_tz='US/Eastern')
h5store(filename, a, **metadata)
with pd.HDFStore(filename) as store:
    data, metadata = h5load(store)

print(data)
#     A   B   C   E   D
# 0   9  20  92  43  25
# 1   2  64  54   0  63
# 2  22  42   3  83  81
# 3   3  71  17  64  53
# 4  52  10  41  22  43
# 5  48  85  96  72  88
# 6  10  47   2  10  78
# 7  30  80   3  59  16
# 8  13  52  98  79  65
# 9   6  93  55  40   3

$DATE$ $TIME$
"""
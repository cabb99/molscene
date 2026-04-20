import pytest
from molscene import Scene
from pathlib import Path
import pandas as pd
import numpy as np
import os
import warnings
import pytest


if os.environ.get('MOLSCENE_FAIL_ON_WARNINGS', '').lower() in {'1', 'true', 'yes'}:
    pytestmark = pytest.mark.filterwarnings('error')
    warnings.simplefilter('error')

# Utility to get a test file path (pytest tmp_path by default, scratch if env set)
def get_test_file_path(tmp_path, suffix):
    if os.environ.get('MOLSCENE_TEST_SCRATCH', '').lower() in {'1', 'true', 'yes'}:
        scratch_dir = Path('molscene/tests/scratch')
        scratch_dir.mkdir(parents=True, exist_ok=True)
        return scratch_dir / suffix
    else:
        return tmp_path / suffix

@pytest.fixture
def pdbfile():
    return Path('molscene/data/1zir.pdb')

@pytest.fixture
def ciffile():
    return Path('molscene/data/1zir.cif')

def test_Scene_exists():
    Scene


def test_from_matrix():
    s = Scene([[0, 0, 0],[0, 0, 1]])
    assert len(s) == 2


def test_from_numpy():
    import numpy as np
    a = np.random.random([100, 3]) * 100
    s = Scene(a)
    assert len(s) == 100


def test_from_dataframe():
    import numpy as np
    import pandas
    a = np.random.random([100, 3]) * 100
    atoms = pandas.DataFrame(a, columns=['z', 'y', 'x'])
    s = Scene(atoms)
    assert s['x'][20] == atoms['x'][20]

def test_from_pdb(pdbfile):
    s = Scene.from_pdb(pdbfile)
    assert len(s) == 1771
    atom = s.loc[1576]
    #print(atom)
    assert atom['serial'] == 1577
    assert atom['resid'] == 1170
    assert atom['name'] == 'SD'
    assert atom['resname'] == 'MET'
    assert atom['chain'] == 'A'
    assert atom['altloc'] == 'G'

def test_from_cif(ciffile):
    s = Scene.from_cif(ciffile)
    assert len(s) == 1771
    atom = s.loc[1576]
    #print(atom)
    assert atom['serial'] == 1577
    assert atom['resid'] == 1170  # auth_seq_id (author numbering)
    assert atom['name'] == 'SD'
    assert atom['resname'] == 'MET'
    assert atom['chain'] == 'A'
    assert atom['altloc'] == 'G'


def test_cif_quoted_atom_names_preserved():
    """Atom names like O5' must survive CIF double-quoting (e.g. "O5'")."""
    s = Scene.from_cif(Path('molscene/data/1zbl.cif'))
    assert "O5'" in s['name'].values, "O5' was mangled during CIF parsing"


def test_cif_pdb_resid_agreement(pdbfile, ciffile):
    # CIF resid (now auth_seq_id) should match PDB resid for the same atom
    pdb = Scene.from_pdb(pdbfile)
    cif = Scene.from_cif(ciffile)
    for idx in [0, 500, 1576]:
        assert pdb.loc[idx, 'resid'] == cif.loc[idx, 'resid'], (
            f"resid mismatch at index {idx}: PDB={pdb.loc[idx, 'resid']} CIF={cif.loc[idx, 'resid']}")


def test_pdb_columns(pdbfile):

    s = Scene.from_pdb(pdbfile)
    # Atom 0: GLY N — occupancy=1.0, beta=13.32, charge=0
    a = s.loc[0]
    assert a['x'] == pytest.approx(28.412)
    assert a['y'] == pytest.approx(-8.856)
    assert a['z'] == pytest.approx(19.490)
    assert a['occupancy'] == pytest.approx(1.0)
    assert a['beta'] == pytest.approx(13.32)
    assert a['charge'] == pytest.approx(0.0)
    assert a['element'] == 'N'

    # Atom 500: ARG CB — occupancy=1.0, beta=17.41, charge=0
    a = s.loc[500]
    assert a['occupancy'] == pytest.approx(1.0)
    assert a['beta'] == pytest.approx(17.41)
    assert a['charge'] == pytest.approx(0.0)

    # Atom 1576: MET SD — occupancy=0.59, beta=25.42, charge=0
    a = s.loc[1576]
    assert a['occupancy'] == pytest.approx(0.59)
    assert a['beta'] == pytest.approx(25.42)
    assert a['charge'] == pytest.approx(0.0)


def test_cif_columns(ciffile):
    
    s = Scene.from_cif(ciffile)
    # Atom 0: GLY N — same coordinates as PDB
    a = s.loc[0]
    assert a['x'] == pytest.approx(28.412)
    assert a['y'] == pytest.approx(-8.856)
    assert a['z'] == pytest.approx(19.490)
    assert a['occupancy'] == pytest.approx(1.0)
    assert a['beta'] == pytest.approx(13.32)
    assert a['charge'] == pytest.approx(0.0)
    assert a['element'] == 'N'

    # Atom 500: ARG CB
    a = s.loc[500]
    assert a['occupancy'] == pytest.approx(1.0)
    assert a['beta'] == pytest.approx(17.41)
    assert a['charge'] == pytest.approx(0.0)

    # Atom 1576: MET SD
    a = s.loc[1576]
    assert a['occupancy'] == pytest.approx(0.59)
    assert a['beta'] == pytest.approx(25.42)
    assert a['charge'] == pytest.approx(0.0)


def test_residue_key_no_collision():
    # Without a separator, fragment=1 + resid=11 concatenates to "111",
    # which collides with fragment=11 + resid=1.
    # Using a separator ('_') avoids this: "1_11_" vs "11_1_".
    data = pd.DataFrame({
        'x': [0, 0], 'y': [0, 0], 'z': [0, 0],
        'chain': ['A', 'B'],
        'resid': [11, 1],
        'fragment': [1, 11],
        'icode': ['', ''],
    })
    s = Scene(data)
    # fragment=1 resid=11 and fragment=11 resid=1 must get different residue indices
    assert s.loc[0, 'residue'] != s.loc[1, 'residue']


def test_compute_mass(pdbfile):
    s = Scene.from_pdb(pdbfile)
    # mass is now auto-populated during construction
    assert 'mass' in s.columns
    # Check known element masses for a few atoms
    # Atom 0: element N -> 14.007
    assert s.loc[0, 'mass'] == pytest.approx(14.007)
    # Atom 500: element C -> 12.011
    assert s.loc[500, 'mass'] == pytest.approx(12.011)
    # Atom 1576: element S -> 32.07
    assert s.loc[1576, 'mass'] == pytest.approx(32.07)
    # All masses should be > 0 for real atoms
    assert (s['mass'] > 0).all()
    # compute_mass still works (returns copy, no-op if already present)
    s_mass = s.compute_mass()
    assert 'mass' in s_mass.columns


# --- Tests for icode rename ---

def test_icode_column_pdb(pdbfile):
    """iCode was renamed to icode (lowercase)."""
    s = Scene.from_pdb(pdbfile)
    assert 'icode' in s.columns
    assert 'iCode' not in s.columns
    # All icode values for 1zir should be empty strings
    assert (s['icode'] == '').all()


def test_icode_column_cif(ciffile):
    s = Scene.from_cif(ciffile)
    assert 'icode' in s.columns
    assert 'iCode' not in s.columns


def test_icode_default_on_bare_scene():
    """Bare Scene (from coords only) should get icode, not iCode."""
    s = Scene([[0, 0, 0]])
    assert 'icode' in s.columns
    assert 'iCode' not in s.columns


# --- Tests for segment column ---

def test_segment_column_pdb(pdbfile):
    """PDB segment comes from columns 73-76 (usually blank for most PDBs)."""
    s = Scene.from_pdb(pdbfile)
    assert 'segment' in s.columns
    # 1zir.pdb typically has blank segment columns
    assert s['segment'].dtype == object


def test_segment_column_cif(ciffile):
    """CIF segment should be derived from label_entity_id."""
    s = Scene.from_cif(ciffile)
    assert 'segment' in s.columns
    # 1zir.cif has label_entity_id values (typically "1", "2", etc.)
    assert len(s['segment'].unique()) > 0
    # Should not contain empty strings since label_entity_id is present
    assert not (s['segment'] == '').all()


def test_segment_column_cif_matches_entity_id(ciffile):
    """CIF segment values should match label_entity_id when present."""
    s = Scene.from_cif(ciffile)
    if 'label_entity_id' in s.columns:
        assert (s['segment'] == s['label_entity_id']).all()


# --- Tests for atomicnumber column ---

def test_atomicnumber_column_pdb(pdbfile):
    s = Scene.from_pdb(pdbfile)
    assert 'atomicnumber' in s.columns
    # N -> 7, C -> 6, O -> 8, S -> 16
    assert s.loc[0, 'atomicnumber'] == 7   # N
    assert s.loc[1, 'atomicnumber'] == 6   # C (CA)
    assert s.loc[3, 'atomicnumber'] == 8   # O
    assert s.loc[1576, 'atomicnumber'] == 16  # S (SD in MET)
    assert s['atomicnumber'].dtype == int


def test_atomicnumber_column_cif(ciffile):
    s = Scene.from_cif(ciffile)
    assert 'atomicnumber' in s.columns
    assert s.loc[0, 'atomicnumber'] == 7   # N


# --- Tests for radius column ---

def test_radius_column_pdb(pdbfile):
    s = Scene.from_pdb(pdbfile)
    assert 'radius' in s.columns
    # VdW radii: N -> 1.55, C -> 1.70, O -> 1.52, S -> 1.80
    assert s.loc[0, 'radius'] == pytest.approx(1.55)   # N
    assert s.loc[1, 'radius'] == pytest.approx(1.70)   # C
    assert s.loc[3, 'radius'] == pytest.approx(1.52)   # O
    assert s.loc[1576, 'radius'] == pytest.approx(1.80) # S
    assert (s['radius'] > 0).all()


def test_radius_column_cif(ciffile):
    s = Scene.from_cif(ciffile)
    assert 'radius' in s.columns
    assert s.loc[0, 'radius'] == pytest.approx(1.55)   # N


# --- Tests for type column ---

def test_type_column_pdb(pdbfile):
    s = Scene.from_pdb(pdbfile)
    assert 'type' in s.columns
    # type should equal element for standard atoms
    assert s.loc[0, 'type'] == 'N'
    assert s.loc[1, 'type'] == 'C'
    assert (s['type'] == s['element']).all()


def test_type_column_cif(ciffile):
    s = Scene.from_cif(ciffile)
    assert 'type' in s.columns
    assert (s['type'] == s['element']).all()


# --- Tests for element-derived columns on all structures ---

@pytest.mark.parametrize("structure", ['1r70', '1zbl', '1zir'])
@pytest.mark.parametrize("fmt", ['pdb', 'cif'])
def test_element_derived_columns_present(structure, fmt):
    """All structures should have mass, atomicnumber, radius, type columns."""
    path = Path(f'molscene/data/{structure}.{fmt}')
    if not path.exists():
        pytest.skip(f'{path} not found')
    if fmt == 'pdb':
        s = Scene.from_pdb(path)
    else:
        s = Scene.from_cif(path)
    for col in ['mass', 'atomicnumber', 'radius', 'type']:
        assert col in s.columns, f"Missing column '{col}' in {path}"
    # mass and radius should be > 0 for all atoms with known elements
    known = s['element'].isin(['C', 'N', 'O', 'S', 'H', 'P', 'Fe', 'Zn', 'Ca', 'Mg', 'Na', 'Cl'])
    assert (s.loc[known, 'mass'] > 0).all()
    assert (s.loc[known, 'radius'] > 0).all()
    assert (s.loc[known, 'atomicnumber'] > 0).all()


def test_compute_secondary_structure(ciffile):
    s = Scene.from_cif(ciffile)
    result = s.compute_secondary_structure()
    # Row count should be preserved (one SS per residue, broadcast to all atoms)
    assert len(result) == len(s)
    # secondary_structure column should be present
    assert 'secondary_structure' in result.columns
    # Check specific residue assignments
    # GLY A 1001 -> coil (.)
    r1001 = result[(result['chain'] == 'A') & (result['resid'] == 1001)].iloc[0]
    assert r1001['secondary_structure'] == '.'
    # PHE A 1005 -> strand (E)
    r1005 = result[(result['chain'] == 'A') & (result['resid'] == 1005)].iloc[0]
    assert r1005['secondary_structure'] == 'E'
    # LEU A 1111 -> helix (H)
    r1111 = result[(result['chain'] == 'A') & (result['resid'] == 1111)].iloc[0]
    assert r1111['secondary_structure'] == 'H'
    # accessibility should be numeric
    assert r1005['accessibility'] == '0.0'


def test_select(pdbfile):
    s = Scene.from_pdb(pdbfile)
    # print(s['altLoc'].unique())
    assert len(s.select(altloc=['A','C','E','G'])) == 1613



def test_split_models():
    # TODO: define how to split complex models
    pass

def test_wrong_init():
    with pytest.raises(ValueError):
        s = Scene(1)
    with pytest.raises(ValueError):
        s = Scene([0,1,2,3])
    with pytest.raises(ValueError):
        s = Scene([[0,1,2,3],[4,5,6,7]])
    
    temp = pd.DataFrame([[0,1,2,3],[4,5,6,7]], columns=['x','y','z','w'])
    s = Scene(temp)

    temp = pd.DataFrame([[0,1],[4,5]], columns=['x','y'])
    with pytest.raises(ValueError):
        s = Scene(temp)    

def test_metadata():
    s= Scene([[0,1,2],[4,5,6]])
    assert 'test' not in s._meta.keys()
    s.test = 'test'
    assert ['test'] == list(s._meta.keys())
    s.x=[3,4]
    assert 'x' not in s._meta.keys()
    assert s['x'][0] == 3

def test_from_fixPDB(pdbfile):
    s = Scene.from_fixPDB(pdbfile)
    assert len(s) == 2849
    sel = s[(s['name'] == 'SD') & (s['resid'] == 1170)]
    assert len(sel) == 1
    atom = sel.iloc[0]
    #print(atom)
    assert atom['serial'] != 1577
    assert atom['resid'] == 1170
    assert atom['name'] == 'SD'
    assert atom['resname'] == 'MET'
    assert atom['chain'] == 'A'

def test_from_fixer(pdbfile):
    import pdbfixer
    fixer = pdbfixer.PDBFixer(filename=str(pdbfile))
    fixer.findMissingResidues()
    fixer.removeHeterogens(keepWater=False)
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()  # Warning: importing 'simtk.openmm' is deprecated.  Import 'openmm' instead.
    fixer.addMissingHydrogens(7.0)
    s = Scene.from_fixer(fixer)
    assert len(s) == 2849
    sel = s[(s['name'] == 'SD') & (s['resid'] == 1170)]
    assert len(sel) == 1
    atom = sel.iloc[0]
    #print(atom)
    assert atom['serial'] != 1577
    assert atom['resid'] == 1170
    assert atom['name'] == 'SD'
    assert atom['resname'] == 'MET'
    assert atom['chain'] == 'A'

def test_get_coordinates(pdbfile):
    s = Scene.from_pdb(pdbfile)
    assert s.get_coordinates().shape == (1771, 3)

def test_set_coordinates(pdbfile):
    import numpy as np
    s = Scene.from_pdb(pdbfile)
    temp = s[['x','y','z']].values
    temp*=0
    temp+=1
    s.set_coordinates(temp)
    assert s.get_coordinates().shape == (1771, 3)
    assert s.get_coordinates().iloc[0,0] == 1




class Test_Read_Write():
    def _convert(self, reader, writer, mol, tmp_path):
        if reader == 'pdb':
            s1 = Scene.from_pdb(f'molscene/data/{mol}.pdb')
        elif reader == 'cif':
            s1 = Scene.from_cif(f'molscene/data/{mol}.cif')
        elif reader == 'gro':
            s1 = Scene.from_gro(f'molscene/data/{mol}.gro')
        elif reader == 'fixPDB_pdb':
            s1 = Scene.from_fixPDB(pdbfile=f'molscene/data/{mol}.pdb')
        elif reader == 'fixPDB_cif':
            s1 = Scene.from_fixPDB(pdbxfile=f'molscene/data/{mol}.cif')
        elif reader == 'fixPDB_pdbid':
            s1 = Scene.from_fixPDB(pdbid=f'{mol}')

        if writer == 'pdb':
            fname = get_test_file_path(tmp_path, f'{reader}_{writer}_{mol}.pdb')
            s1.write_pdb(fname)
            s2 = Scene.from_pdb(fname)
        elif writer == 'cif':
            fname = get_test_file_path(tmp_path, f'{reader}_{writer}_{mol}.cif')
            s1.write_cif(fname)
            s2 = Scene.from_cif(fname)
        elif writer == 'gro':
            fname = get_test_file_path(tmp_path, f'{reader}_{writer}_{mol}.gro')
            s1.write_gro(fname)
            s2 = Scene.from_gro(fname)

        s1.to_csv(get_test_file_path(tmp_path, 's1.csv'))
        s2.to_csv(get_test_file_path(tmp_path, 's2.csv'))
        print(len(s1))
        assert (len(s1) == len(s2)), f"The number of particles before reading ({len(s1)}) and after writing ({len(s2)})" \
                                     f" are different.\nCheck the file: {fname}"

    @pytest.mark.parametrize('writer', ['pdb', 'cif'])
    @pytest.mark.parametrize('reader', ['pdb', 'cif'])
    @pytest.mark.parametrize('mol', ['1r70', '1zbl', '1zir'])
    def test_convert(self, reader, writer, mol, tmp_path):
        self._convert(reader, writer, mol, tmp_path)


@pytest.fixture
def simple_scene():
    """Fixture to create a simple Scene with 3 atoms."""
    particles = pd.DataFrame([[0, 0, 0],
                              [0, 1, 0],
                              [0, 0, 1]],
                             columns=['x', 'y', 'z'])
    return Scene(particles)

@pytest.fixture
def scene_with_trajectory(simple_scene):
    """Fixture to create a Scene with multi-frame coordinate data."""
    n_frames = 5
    n_atoms = len(simple_scene)
    frames = np.random.rand(n_frames, n_atoms, 3) * 10  # coordinates in Angstroms
    simple_scene.set_coordinate_frames(frames)
    return simple_scene, frames

def test_n_frames(scene_with_trajectory):
    """Test that the Scene correctly reports the number of frames."""
    scene, frames = scene_with_trajectory
    assert scene.n_frames == frames.shape[0], "n_frames should match the stored number of frames."

def test_get_frame_coordinates(scene_with_trajectory):
    """Test that get_frame_coordinates retrieves the correct frame."""
    scene, frames = scene_with_trajectory
    frame_index = 2
    np.testing.assert_array_equal(scene.get_frame_coordinates(frame_index), frames[frame_index])

def test_set_frame_coordinates(scene_with_trajectory):
    """Test that set_frame_coordinates correctly updates the Scene's coordinates."""
    scene, frames = scene_with_trajectory
    frame_index = 3
    scene.set_frame_coordinates(frame_index)
    np.testing.assert_array_equal(scene.get_coordinates().to_numpy(), frames[frame_index])

def test_frames_accessor(scene_with_trajectory):
    """Test that accessing a specific frame via Scene.frames[index] returns the correct Scene."""
    scene, frames = scene_with_trajectory
    frame_index = 1
    frame_scene = scene.frames[frame_index]

    np.testing.assert_array_equal(frame_scene.get_coordinates().to_numpy(), frames[frame_index])
    
    # Ensure the returned Scene does not retain multi-frame metadata
    assert 'coordinate_frames' not in frame_scene._meta, "Returned frame scene should not have coordinate_frames in _meta."

def test_iterframes(scene_with_trajectory):
    """Test that iterframes() correctly iterates over all frames."""
    scene, frames = scene_with_trajectory
    count = 0
    for frame_scene in scene.iterframes():
        np.testing.assert_array_equal(frame_scene.get_coordinates().to_numpy(), frames[count])
        assert len(list(frame_scene.columns))>3
        count += 1
    assert count == scene.n_frames, "iterframes() should yield exactly n_frames scenes."

def test_distance_map():
    import numpy as np
    from molscene import Scene

    # Simple 3-point triangle: (0,0,0), (1,0,0), (0,1,0)
    coords = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    s = Scene(coords)

    # Dense distance map
    dense = s.distance_map(threshold=None)
    assert dense.shape == (3, 3)
    np.testing.assert_allclose(np.diag(dense), 0)
    np.testing.assert_allclose(dense[0, 1], 1)
    np.testing.assert_allclose(dense[0, 2], 1)
    np.testing.assert_allclose(dense[1, 2], np.sqrt(2))

    # Sparse distance map with threshold=1.01
    pairs, dists = s.distance_map_sparse(threshold=1.01)
    assert pairs.shape[1] == 2
    # Confirm all distances are ≤ threshold
    assert np.all(dists <= 1.01)
    # Confirm exact expected pairs present
    expected_pairs = {(0, 1), (1, 0), (0, 2), (2, 0)}
    actual_pairs = {tuple(p) for p in pairs}
    assert expected_pairs <= actual_pairs  # all expected pairs must be found
    np.testing.assert_allclose(dists, 1.0)

    # Sparse distance map with threshold=2.0
    pairs, dists = s.distance_map_sparse(threshold=2.0)
    assert pairs.shape[1] == 2
    assert np.all(dists <= 2.0)
    # Should include (1,2) and (2,1) now
    expected_pairs = {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)}
    actual_pairs = {tuple(p) for p in pairs}
    assert expected_pairs <= actual_pairs
    # Confirm that the only distances present are the ones we expect
    expected_dists = [1.0, 1.0, 1.0, 1.0, np.sqrt(2), np.sqrt(2)]
    np.testing.assert_allclose(sorted(dists), sorted(expected_dists))

def test_get_sequence_1zbl():
    s = Scene.from_cif('molscene/data/1zbl.cif')
    seqs = s.get_sequence()
    # Reference sequences from user prompt
    ref = {
        'A': 'GACACCUGAUUC',
        'B': 'GAATCAGGTGTC',
        'D': 'EEIIWESLSVDVGSQGNPGIVEYKGVDTKTGEVLFEREPIPIGTNNMGEFLAIVHGLRYLKERNSRKPIYSDSQTAIKWVKDKKAKSTLVRNEETALIWKLVDEAEEWLNTHTYETPILKWQTDKWGEIKANY'
    }
    # Allow for possible chainID mapping (e.g. auth labels)
    for k, v in ref.items():
        assert k in seqs, f"Chain {k} not found in parsed sequence. Found: {list(seqs.keys())}"
        assert seqs[k].startswith(v[:6]), f"Chain {k} sequence does not match reference. Got: {seqs[k][:6]}, expected: {v[:6]}"
        assert seqs[k] == v, f"Chain {k} sequence mismatch.\nExpected: {v}\nGot: {seqs[k]}"


if __name__ == '__main__':
    pass


# --- Tests for phi/psi dihedral calculation ---

def test_compute_phi_psi_columns(pdbfile):
    """compute_phi_psi should add phi and psi columns."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_phi_psi()
    assert 'phi' in result.columns
    assert 'psi' in result.columns
    assert len(result) == len(s)


def test_compute_phi_psi_first_residue_no_phi(pdbfile):
    """The first residue in a chain has no phi (needs C from prev residue)."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_phi_psi()
    # Find the first residue index in fragment 0
    frag0 = result[result['fragment'] == 0]
    first_res = frag0['residue'].min()
    first_atoms = frag0[frag0['residue'] == first_res]
    assert first_atoms['phi'].isna().all()


def test_compute_phi_psi_last_residue_no_psi(pdbfile):
    """The last residue in a chain has no psi (needs N from next residue)."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_phi_psi()
    frag0 = result[result['fragment'] == 0]
    last_res = frag0['residue'].max()
    last_atoms = frag0[frag0['residue'] == last_res]
    assert last_atoms['psi'].isna().all()


def test_compute_phi_psi_range(pdbfile):
    """Interior phi/psi should be in [-180, 180]."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_phi_psi()
    valid_phi = result['phi'].dropna()
    valid_psi = result['psi'].dropna()
    assert (valid_phi >= -180).all() and (valid_phi <= 180).all()
    assert (valid_psi >= -180).all() and (valid_psi <= 180).all()


def test_compute_phi_psi_known_values():
    """Test phi/psi with a minimal synthetic backbone (known geometry)."""
    # Build 3 residues of backbone atoms in an extended (beta) conformation.
    # Extended sheet: phi ≈ -120°, psi ≈ +120° (roughly).
    # We'll use ideal coordinates for an anti-parallel beta strand.
    coords = {
        # Residue 0: N, CA, C
        'N0':  np.array([0.000, 0.000, 0.000]),
        'CA0': np.array([1.458, 0.000, 0.000]),
        'C0':  np.array([2.009, 1.420, 0.000]),
        # Residue 1: N, CA, C
        'N1':  np.array([3.327, 1.556, 0.000]),
        'CA1': np.array([3.983, 2.850, 0.000]),
        'C1':  np.array([5.483, 2.711, 0.000]),
        # Residue 2: N, CA, C
        'N2':  np.array([6.071, 1.517, 0.000]),
        'CA2': np.array([7.519, 1.358, 0.000]),
        'C2':  np.array([8.087, 2.760, 0.000]),
    }
    rows = []
    for i in range(3):
        for aname in ['N', 'CA', 'C']:
            c = coords[f'{aname}{i}']
            rows.append({
                'x': c[0], 'y': c[1], 'z': c[2],
                'name': aname, 'element': aname[0],
                'chain': 'A', 'resid': i + 1,
                'resname': 'ALA',
            })

    s = Scene(pd.DataFrame(rows))
    result = s.compute_phi_psi()

    # Residue 0: no phi, has psi
    res0 = result[result['resid'] == 1]
    assert res0['phi'].isna().all()
    assert res0['psi'].notna().all()

    # Residue 1 (middle): has both phi and psi
    res1 = result[result['resid'] == 2]
    assert res1['phi'].notna().all()
    assert res1['psi'].notna().all()

    # Residue 2: has phi, no psi
    res2 = result[result['resid'] == 3]
    assert res2['phi'].notna().all()
    assert res2['psi'].isna().all()

    # Check angles are finite and in valid range (coplanar coords give 0 or ±180)
    phi_1 = res1['phi'].iloc[0]
    psi_1 = res1['psi'].iloc[0]
    assert -180 <= phi_1 <= 180, f"phi out of range: {phi_1}"
    assert -180 <= psi_1 <= 180, f"psi out of range: {psi_1}"
    # For the all-z=0 planar layout, dihedrals are exactly 0 or ±180
    assert abs(phi_1) == pytest.approx(180, abs=1) or phi_1 == pytest.approx(0, abs=1)
    assert abs(psi_1) == pytest.approx(180, abs=1) or psi_1 == pytest.approx(0, abs=1)


def test_compute_phi_psi_multichain(pdbfile):
    """Phi/psi should not bleed across chains."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_phi_psi()
    for _, frag in result.groupby('fragment'):
        first_res = frag['residue'].min()
        last_res = frag['residue'].max()
        # First residue of each chain: no phi
        assert frag[frag['residue'] == first_res]['phi'].isna().all()
        # Last residue of each chain: no psi
        assert frag[frag['residue'] == last_res]['psi'].isna().all()


# ---------- compute_bonds / numbonds / pfrag / nfrag ----------

def test_compute_bonds_columns(pdbfile):
    """compute_bonds adds numbonds, pfrag, nfrag columns."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_bonds()
    assert 'numbonds' in result.columns
    assert 'pfrag' in result.columns
    assert 'nfrag' in result.columns


def test_compute_bonds_ca_numbonds(pdbfile):
    """CA atoms in standard residues should have 2-3 bonds."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_bonds()
    ca = result[result['name'] == 'CA']
    # Filter to first occurrence per residue (altloc duplicates get 0 bonds)
    ca = ca.drop_duplicates(subset=['chain', 'resid', 'icode', 'name'], keep='first')
    assert (ca['numbonds'] >= 2).all()
    assert (ca['numbonds'] <= 3).all()


def test_compute_bonds_pfrag_isolation():
    """Protein fragments from separate chains get distinct pfrag ids."""
    s = Scene.from_pdb(Path('molscene/data/1zbl.pdb'))
    result = s.compute_bonds()
    m1 = result[result['model'] == 1]
    protein = m1[m1['resname'].isin(['ALA', 'ARG', 'ASN', 'ASP', 'CYS',
        'GLN', 'GLU', 'GLY', 'HIS', 'ILE', 'LEU', 'LYS', 'MET', 'PHE',
        'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'])]
    chains = protein.groupby('chain')['pfrag'].nunique()
    # Each protein chain should have exactly 1 pfrag
    assert (chains == 1).all()


def test_compute_bonds_multi_model():
    """Both models in a multi-model file should get the same bond counts."""
    s = Scene.from_pdb(Path('molscene/data/1zbl.pdb'))
    result = s.compute_bonds()
    m1 = result[result['model'] == 1]
    m2 = result[result['model'] == 2]
    assert len(m1) == len(m2)
    assert m1['numbonds'].mean() == pytest.approx(m2['numbonds'].mean())
    assert (m1['numbonds'] > 0).sum() == (m2['numbonds'] > 0).sum()


def test_compute_bonds_nfrag():
    """Nucleic fragments should be detected in structures with DNA/RNA."""
    s = Scene.from_pdb(Path('molscene/data/1zbl.pdb'))
    result = s.compute_bonds()
    m1 = result[result['model'] == 1]
    nuc = m1[m1['nfrag'] > 0]
    assert len(nuc) > 0, "should detect nucleic fragments in 1zbl"


def test_compute_bonds_water_zero():
    """Water molecules should have 0 bonds (no connectivity in CCD)."""
    s = Scene.from_pdb(Path('molscene/data/1zbl.pdb'))
    result = s.compute_bonds()
    water = result[result['resname'] == 'HOH']
    if len(water) > 0:
        assert (water['numbonds'] == 0).all()


def test_compute_bonds_stores_meta(pdbfile):
    """Bond adjacency list should be stored in _meta['bonds']."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_bonds()
    assert hasattr(result, '_meta') and 'bonds' in result._meta
    adj = result._meta['bonds']
    assert len(adj) == len(result)


# ---------- compute_anisou ----------

_ANISOU_PDB = """\
ATOM      1  N   ALA A   1       1.000   2.000   3.000  1.00 10.00           N
ANISOU    1  N   ALA A   1      200    300    400     50     60     70       N
ATOM      2  CA  ALA A   1       2.000   3.000   4.000  1.00 12.00           C
ANISOU    2  CA  ALA A   1      500    600    700     80     90    100       C
ATOM      3  C   ALA A   1       3.000   4.000   5.000  1.00 14.00           C
END
"""


def test_compute_anisou_pdb(tmp_path):
    """compute_anisou reads ANISOU records from a PDB file."""
    pdb = tmp_path / "anisou.pdb"
    pdb.write_text(_ANISOU_PDB)
    s = Scene.from_pdb(pdb)
    assert 'ufx' not in s.columns  # not loaded eagerly
    result = s.compute_anisou()
    assert 'ufx' in result.columns
    assert 'ufy' in result.columns
    assert 'ufz' in result.columns
    # Atom 1: U11=200 → 0.02, U22=300 → 0.03, U33=400 → 0.04
    row0 = result.iloc[0]
    assert row0['ufx'] == pytest.approx(0.02)
    assert row0['ufy'] == pytest.approx(0.03)
    assert row0['ufz'] == pytest.approx(0.04)
    # Atom 2: U11=500 → 0.05, U22=600 → 0.06, U33=700 → 0.07
    row1 = result.iloc[1]
    assert row1['ufx'] == pytest.approx(0.05)
    assert row1['ufy'] == pytest.approx(0.06)
    assert row1['ufz'] == pytest.approx(0.07)


def test_compute_anisou_missing_records(tmp_path):
    """Atoms without ANISOU records get 0.0."""
    pdb = tmp_path / "anisou.pdb"
    pdb.write_text(_ANISOU_PDB)
    s = Scene.from_pdb(pdb)
    result = s.compute_anisou()
    # Atom 3 (C) has no ANISOU record
    row2 = result.iloc[2]
    assert row2['ufx'] == 0.0
    assert row2['ufy'] == 0.0
    assert row2['ufz'] == 0.0


def test_compute_anisou_no_source_raises():
    """compute_anisou raises if no source file is recorded."""
    s = Scene([[0, 0, 0]])
    with pytest.raises(ValueError, match="no source file"):
        s.compute_anisou()


def test_compute_anisou_no_anisou_in_file(pdbfile):
    """Files without ANISOU records get all-zero columns."""
    s = Scene.from_pdb(pdbfile)
    result = s.compute_anisou()
    assert (result['ufx'] == 0.0).all()
    assert (result['ufy'] == 0.0).all()
    assert (result['ufz'] == 0.0).all()

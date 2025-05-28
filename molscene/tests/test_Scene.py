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
    assert atom['resid'] == 170
    assert atom['name'] == 'SD'
    assert atom['resname'] == 'MET'
    assert atom['chain'] == 'A'
    assert atom['altloc'] == 'G'


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

import pytest
from molscene import Scene
from pathlib import Path
import pandas as pd


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
    assert atom['resSeq'] == 1170
    assert atom['name'] == 'SD'
    assert atom['resName'] == 'MET'
    assert atom['chainID'] == 'A'
    assert atom['altLoc'] == 'G'

def test_from_cif(ciffile):
    s = Scene.from_cif(ciffile)
    assert len(s) == 1771
    atom = s.loc[1576]
    #print(atom)
    assert atom['serial'] == 1577
    assert atom['resSeq'] == 170
    assert atom['name'] == 'SD'
    assert atom['resName'] == 'MET'
    assert atom['chainID'] == 'A'
    assert atom['altLoc'] == 'G'


def test_select(pdbfile):
    s = Scene.from_pdb(pdbfile)
    # print(s['altLoc'].unique())
    assert len(s.select(altLoc=['A','C','E','G'])) == 1613



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
    sel = s[(s['name'] == 'SD') & (s['resSeq'] == 1170)]
    assert len(sel) == 1
    atom = sel.iloc[0]
    #print(atom)
    assert atom['serial'] != 1577
    assert atom['resSeq'] == 1170
    assert atom['name'] == 'SD'
    assert atom['resName'] == 'MET'
    assert atom['chainID'] == 'A'

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
    sel = s[(s['name'] == 'SD') & (s['resSeq'] == 1170)]
    assert len(sel) == 1
    atom = sel.iloc[0]
    #print(atom)
    assert atom['serial'] != 1577
    assert atom['resSeq'] == 1170
    assert atom['name'] == 'SD'
    assert atom['resName'] == 'MET'
    assert atom['chainID'] == 'A'

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
    def _convert(self, reader, writer, mol):
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
            fname = f'molscene/tests/scratch/{reader}_{writer}_{mol}.pdb'
            s1.write_pdb(fname)
            s2 = Scene.from_pdb(fname)
        elif writer == 'cif':
            fname = f'molscene/tests/scratch/{reader}_{writer}_{mol}.cif'
            s1.write_cif(fname)
            s2 = Scene.from_cif(fname)
        elif writer == 'gro':
            fname = f'molscene/tests/scratch/{reader}_{writer}_{mol}.gro'
            s1.write_gro(fname)
            s2 = Scene.from_gro(fname)

        s1.to_csv('molscene/tests/scratch/s1.csv')
        s2.to_csv('molscene/tests/scratch/s2.csv')
        print(len(s1))
        assert (len(s1) == len(s2)), f"The number of particles before reading ({len(s1)}) and after writing ({len(s2)})" \
                                     f" are different.\nCheck the file: {fname}"

    def test_convert(self):
        for reader in ['pdb', 'cif']:  # ,'fixPDB_pdb','fixPDB_cif','fixPDB_pdbid']:
            for writer in ['pdb', 'cif']:
                for mol in ['1r70', '1zbl', '1zir']:
                    self._convert(reader, writer, mol)

    def test_pdb2pdb(self):
        for mol in ['1r70', '1zbl', '1zir', '3wu2']:
            yield self._convert, 'pdb', 'pdb', mol

    # def _cif_to_cif(self, mol):
    #    s1 = Scene.from_cif(f'data/{mol}')
    #    s1.write_cif(f'scratch/{mol}')
    #    s2 = Scene.from_cif(f'data/{mol}')
    #    assert len(s1) == len(s2)

    # def test_cif(self):
    #    for mol in ['1r70', '1zbl', '1zir', '3wu2']:
    #        yield self._cif_to_cif, f'{mol}.cif'




if __name__ == '__main__':
    pass

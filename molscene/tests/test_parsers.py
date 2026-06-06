"""Tests for the parsers/ (file formats) and backends/ (object converters) layers."""

import numpy as np
import pytest

from molscene import Scene
from molscene.parsers import FormatRegistry

PDB = 'molscene/data/1jge.pdb'


def test_registry_is_populated():
    assert {'.pdb', '.cif', '.gro'} <= set(FormatRegistry.readers)
    assert {'pdb', 'cif', 'awsem_gro'} <= set(FormatRegistry.reader_by_name)


def test_unknown_format_raises(tmp_path):
    s = Scene.from_pdb(PDB)
    with pytest.raises(ValueError):
        FormatRegistry.read('mystery.xyz')
    with pytest.raises(ValueError):
        FormatRegistry.write(s, str(tmp_path / 'mystery.xyz'))
    with pytest.raises(ValueError):
        FormatRegistry.read('mystery.pdb', format='nope')


def test_explicit_format_overrides_extension(tmp_path):
    s = Scene.from_pdb(PDB)
    p = tmp_path / 'structure.dat'           # extension carries no info
    s.to_file(str(p), format='pdb')
    back = Scene.from_file(str(p), format='pdb')
    assert len(back) == len(s)


@pytest.mark.parametrize('ext', ['.pdb', '.cif'])
def test_format_roundtrip(tmp_path, ext):
    s = Scene.from_pdb(PDB)
    p = tmp_path / f'rt{ext}'
    s.to_file(str(p))
    back = Scene.from_file(str(p))
    assert len(back) == len(s)
    ca_in = s[s['name'] == 'CA'][['x', 'y', 'z']].to_numpy()
    ca_out = back[back['name'] == 'CA'][['x', 'y', 'z']].to_numpy()
    np.testing.assert_allclose(ca_out, ca_in, atol=1e-3)


def test_awsem_gro_read_write_roundtrip(tmp_path):
    # single chain so resid uniquely identifies a CA (awsem_gro carries no chain)
    s = Scene.from_pdb(PDB).select(chain=['A'])
    p = tmp_path / 'mem.gro'
    s.write_awsem_gro(str(p))
    back = Scene.from_awsem_gro(str(p))
    assert back._meta['source_format'] == 'awsem_gro'
    ca_in = (s[s['name'] == 'CA'].sort_values('occupancy', ascending=False, kind='stable')
             .drop_duplicates('resid'))
    src = {int(r.resid): np.array([r.x, r.y, r.z]) for r in ca_in.itertuples()}
    ca_back = back[back['name'] == 'CA']
    assert len(ca_back) == len(src)
    # awsem_gro stores nm at 3 decimals -> ~0.01 Angstrom resolution
    for r in ca_back.itertuples():
        np.testing.assert_allclose([r.x, r.y, r.z], src[int(r.resid)], atol=1e-2)


def test_backend_unknown_object_raises():
    from molscene.backends import BackendRegistry
    with pytest.raises(TypeError):
        BackendRegistry.from_object(object())


def test_backend_roundtrip_prody():
    prody = pytest.importorskip("prody")
    prody.confProDy(verbosity="none")
    s = Scene.from_pdb(PDB)
    # pdb -> prody -> Scene -> prody must equal pdb -> prody (lossless through Scene)
    ag1 = s.to_prody()
    ag2 = Scene.from_prody(ag1).to_prody()
    np.testing.assert_allclose(ag2.getCoords(), ag1.getCoords(), atol=1e-4)
    assert list(ag2.getNames()) == list(ag1.getNames())
    assert list(ag2.getResnums()) == list(ag1.getResnums())


def test_backend_roundtrip_mdtraj():
    mdtraj = pytest.importorskip("mdtraj")          # skipped where mdtraj is absent
    s = Scene.from_pdb(PDB)
    t1 = s.to_mdtraj()
    t2 = Scene.from_mdtraj(t1).to_mdtraj()
    assert t1.n_atoms == t2.n_atoms
    np.testing.assert_allclose(t2.xyz, t1.xyz, atol=1e-4)

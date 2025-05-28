"""This module contains unit tests for :mod:`~prody.select`."""

import os
import pytest
import pandas as pd
from lark import Lark
from molscene.Scene import Scene
from molscene.selection.transformer import PandasTransformer

# Read the same PDB as before using Scene
PDB_PATH = os.path.join(os.path.dirname(__file__), '../data/pdb3mht.pdb')
scene = Scene.from_pdb(PDB_PATH)

# Build the parser
grammar_path = os.path.join(os.path.dirname(__file__), '../selection/selection_syntax.lark')
with open(grammar_path) as f:
    grammar_text = f.read()
parser = Lark(grammar_text, parser='lalr', start=['start', 'expr'])

# Minimal selection tests: only 'none', 'all', 'acidic'
SELECTION_TESTS = [
    ('none', 0),
    ('all', len(scene)),
    ('acidic', int((scene['resname'] == 'GLU').sum() + (scene['resname'] == 'ASP').sum())),
]

@pytest.fixture(scope="module")
def transformer():
    return PandasTransformer(scene, parser=parser)

@pytest.mark.parametrize("selstr, natoms", SELECTION_TESTS)
def test_flag_selection(transformer, selstr, natoms):
    tree = parser.parse(selstr, start='start')
    result = transformer.transform(tree)
    assert len(result) == natoms, f"Selection '{selstr}' expected {natoms} atoms, got {len(result)}"
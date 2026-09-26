"""Shared pytest configuration for the molscene test suite.

Provides the ``requires_molselect`` marker so tests that exercise string atom
selection (which needs the optional ``molselect`` dependency) are skipped
automatically when it is not installed. Mark such a test with::

    @pytest.mark.requires_molselect
    def test_string_selection(...):
        ...
"""

import importlib.util

import pandas as pd
import pytest

from molscene import Scene

HAS_MOLSELECT = importlib.util.find_spec("molselect") is not None


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_molselect: test needs the optional 'molselect' package "
        "(install with `pip install molscene[selection]`)",
    )


def pytest_collection_modifyitems(config, items):
    if HAS_MOLSELECT:
        return
    skip = pytest.mark.skip(
        reason="optional dependency 'molselect' not installed "
        "(pip install molscene[selection])"
    )
    for item in items:
        if "requires_molselect" in item.keywords:
            item.add_marker(skip)


def _toy_protein_frame():
    """Three residues (chain A: 1, 2; chain B: 1) with a non-collinear backbone.

    Residue A2 is a glycine, so it carries no explicit CB and exercises the
    reconstruction path that has to invent one.
    """
    rows = []
    for chain, resid, base in [('A', 1, (0, 0, 0)), ('A', 2, (8, 0, 0)),
                               ('B', 1, (0, 0, 15))]:
        bx, by, bz = base
        offs = {'N': (0.0, 1.0, 0.0), 'CA': (1.0, 0.0, 0.0),
                'C': (2.0, 1.0, 0.2), 'CB': (1.0, -1.0, 0.5)}
        resname = 'GLY' if (chain == 'A' and resid == 2) else 'ALA'
        for name, (ox, oy, oz) in offs.items():
            if name == 'CB' and resname == 'GLY':
                continue
            rows.append(dict(name=name, resname=resname, chain=chain, resid=resid,
                             x=bx + ox, y=by + oy, z=bz + oz))
    return pd.DataFrame(rows)


@pytest.fixture
def toy_protein():
    """A small three-residue protein Scene shared across geometry tests."""
    return Scene(_toy_protein_frame())

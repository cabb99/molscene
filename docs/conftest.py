"""Pytest fixtures for the documentation doctests.

The ``.rst`` files under ``docs/`` are executed as doctests
(``pytest --doctest-glob="*.rst"``) so that every published snippet is verified
and protected from regression. This fixture injects a small shared namespace so
the snippets stay readable instead of repeating import/setup boilerplate:

* ``np`` / ``pd`` — numpy and pandas
* ``Scene`` and the matching/transformation classes
* ``example_structure`` — path to a small PDB shipped in the source tree, used
  by the file-reading examples
* ``workdir`` — a throwaway directory (pytest ``tmp_path``) for files written
  by the examples, so doctests never pollute the working tree

The executed snippets deliberately avoid string atom selection so the doc
doctests pass with or without the optional ``molselect`` dependency; string
selection is exercised separately in ``molscene/tests/test_select.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from molscene import (
    Scene,
    Transformation,
    OrderMatching,
    ColumnMatching,
    SequenceMatching,
    as_matching,
)

# Data lives in the source tree (not shipped in the wheel); resolve it relative
# to the repository root so the path is valid in a checkout / CI run.
_DATA = Path(__file__).resolve().parents[1] / "molscene" / "data"


@pytest.fixture(autouse=True)
def _doctest_namespace(doctest_namespace, tmp_path):
    doctest_namespace.update(
        np=np,
        pd=pd,
        Scene=Scene,
        Transformation=Transformation,
        OrderMatching=OrderMatching,
        ColumnMatching=ColumnMatching,
        SequenceMatching=SequenceMatching,
        as_matching=as_matching,
        example_structure=str(_DATA / "1jge.pdb"),
        workdir=tmp_path,
    )

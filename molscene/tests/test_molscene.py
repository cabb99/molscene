"""
Unit and regression test for the molscene package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import molscene


def test_molscene_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "molscene" in sys.modules

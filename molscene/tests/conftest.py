"""Shared pytest configuration for the molscene test suite.

Provides the ``requires_molselect`` marker so tests that exercise string atom
selection (which needs the optional ``molselect`` dependency) are skipped
automatically when it is not installed. Mark such a test with::

    @pytest.mark.requires_molselect
    def test_string_selection(...):
        ...
"""

import importlib.util

import pytest

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

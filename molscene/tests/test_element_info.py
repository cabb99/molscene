"""Tests for molscene.data.element_info."""

import warnings
import pytest
from molscene.data.element_info import ElementInfo, element_info


class TestElementInfo:
    """Tests for the ElementInfo singleton and its data."""

    def test_has_expected_attributes(self):
        assert hasattr(element_info, "mass")
        assert hasattr(element_info, "atomicnumber")
        assert hasattr(element_info, "radius")

    def test_carbon(self):
        assert element_info.atomicnumber["C"] == 6
        assert abs(element_info.mass["C"] - 12.011) < 0.01
        assert abs(element_info.radius["C"] - 1.70) < 0.05

    def test_hydrogen(self):
        assert element_info.atomicnumber["H"] == 1
        assert abs(element_info.mass["H"] - 1.008) < 0.01
        assert abs(element_info.radius["H"] - 1.20) < 0.05

    def test_all_dicts_same_keys(self):
        assert element_info.mass.keys() == element_info.atomicnumber.keys()
        assert element_info.mass.keys() == element_info.radius.keys()

    def test_at_least_92_elements(self):
        assert len(element_info.mass) >= 92

    def test_atomic_numbers_sequential(self):
        numbers = sorted(element_info.atomicnumber.values())
        assert numbers[0] == 1
        assert numbers == list(range(1, len(numbers) + 1))

    def test_masses_positive(self):
        for sym, m in element_info.mass.items():
            assert m > 0, f"{sym} has non-positive mass {m}"

    def test_unknown_element_raises(self):
        with pytest.raises(KeyError):
            _ = element_info.mass["Xx"]


class TestDeprecatedImport:
    """The old element_masses module should still work but warn."""

    def test_deprecation_warning(self):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Re-import triggers the warning only on first import;
            # force it by reloading.
            import importlib
            import molscene.data.element_masses as mod
            importlib.reload(mod)
            dep = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep) >= 1
            assert "deprecated" in str(dep[0].message).lower()

    def test_shim_values_match(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            import importlib
            import molscene.data.element_masses as mod
            importlib.reload(mod)
            assert mod._element_masses["C"] == element_info.mass["C"]
            assert mod._element_atomicnumbers["O"] == element_info.atomicnumber["O"]
            assert mod._element_radii["N"] == element_info.radius["N"]

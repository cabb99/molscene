"""
Tests for :meth:`Scene.select` and its molselect-backed string selection.
"""

from pathlib import Path

import pytest

from molscene import Scene


@pytest.fixture(scope="module")
def jge():
    return Scene.from_pdb(Path("molscene/data/1jge.pdb")).select(model=[1])


class TestSelect:
    def test_empty_selstr_returns_full_scene(self, jge):
        out = jge.select()
        assert len(out) == len(jge)

    def test_string_selection_round_trip(self, jge):
        out = jge.select("chain A and resid 50 to 55 and name CA")
        # Independently filter via kwargs; molselect's range is inclusive on both ends.
        expected = jge[
            (jge["chain"] == "A")
            & (jge["resid"].between(50, 55))
            & (jge["name"] == "CA")
        ]
        assert len(out) == len(expected) > 0
        assert (out["resid"].to_numpy() == expected["resid"].to_numpy()).all()
        assert (out["name"] == "CA").all()
        assert (out["chain"] == "A").all()

    def test_boolean_combinators(self, jge):
        out = jge.select("(name CA or name CB) and resid 50 to 60")
        expected = jge[
            jge["name"].isin(["CA", "CB"])
            & jge["resid"].between(50, 60)
        ]
        assert len(out) == len(expected) > 0

    def test_within_distance_selection(self, jge):
        # Atoms within 3 Å of any water oxygen.
        out = jge.select("within 3 of resname HOH")
        assert len(out) > 0
        # Water atoms themselves satisfy "within 0 of HOH" so they appear
        # in the result.
        assert (out["resname"] == "HOH").any()

    def test_combine_selstr_and_kwargs(self, jge):
        # Use selstr for atom-name + kwargs for chain — the and-merge keeps
        # both filters applied.
        out = jge.select("name CA", chain=["A"])
        assert len(out) > 0
        assert (out["chain"] == "A").all()
        assert (out["name"] == "CA").all()

    def test_kwargs_alone_still_work(self, jge):
        out = jge.select(chain=["A"], name=["CA"])
        expected = jge[(jge["chain"] == "A") & (jge["name"] == "CA")]
        assert len(out) == len(expected) > 0

    def test_metadata_preserved(self, jge):
        s = jge.copy()
        s.author = "alice"
        out = s.select("chain A and name CA")
        assert out.author == "alice"

    def test_invalid_selection_raises(self, jge):
        with pytest.raises(Exception):
            jge.select("this is not a valid molselect expression")

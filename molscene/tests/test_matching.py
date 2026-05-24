"""Tests for :mod:`molscene.matching`."""

from pathlib import Path

import numpy as np
import pytest

from molscene import (
    Scene,
    OrderMatching,
    ColumnMatching,
    SequenceMatching,
    as_matching,
)
from molscene.matching import _CallableMatching, _needleman_wunsch


# --- Fixtures -------------------------------------------------------------

@pytest.fixture(scope="module")
def jge():
    return Scene.from_pdb(Path("molscene/data/1jge.pdb")).select(model=[1])


# --- as_matching() coercion ----------------------------------------------

class TestAsMatching:
    def test_none_yields_order(self):
        assert isinstance(as_matching(None), OrderMatching)

    def test_string_dispatches(self):
        assert isinstance(as_matching("order"), OrderMatching)
        assert isinstance(as_matching("columns"), ColumnMatching)
        assert isinstance(as_matching("sequence"), SequenceMatching)

    def test_unknown_string_raises(self):
        with pytest.raises(ValueError, match="unknown matching"):
            as_matching("not-a-strategy")

    def test_tuple_yields_column_matching(self):
        m = as_matching(("chain", "resid"))
        assert isinstance(m, ColumnMatching)
        assert m.on == ("chain", "resid")

    def test_callable_wrapped(self):
        m = as_matching(lambda a, b: (a, b))
        assert isinstance(m, _CallableMatching)

    def test_instance_passthrough(self):
        inst = OrderMatching()
        assert as_matching(inst) is inst

    def test_bad_type_raises(self):
        with pytest.raises(TypeError, match="cannot interpret"):
            as_matching(42)


# --- OrderMatching -------------------------------------------------------

class TestOrderMatching:
    def test_self_pair(self, jge):
        left, right = OrderMatching().pair(jge, jge)
        assert len(left) == len(jge)
        np.testing.assert_array_equal(
            left.get_coordinates().to_numpy(),
            right.get_coordinates().to_numpy(),
        )

    def test_length_mismatch_raises(self, jge):
        smaller = Scene(jge.iloc[:100].copy())
        with pytest.raises(ValueError, match="equal lengths"):
            OrderMatching().pair(jge, smaller)


# --- ColumnMatching ------------------------------------------------------

class TestColumnMatching:
    def test_match_after_shuffle(self, jge):
        rng = np.random.default_rng(0)
        order = rng.permutation(len(jge))
        shuffled = Scene(jge.iloc[order].copy())
        left, right = ColumnMatching().pair(jge, shuffled)
        for col in ("chain", "resid", "name", "altloc"):
            assert (left[col].values == right[col].values).all()
        np.testing.assert_allclose(
            left.get_coordinates().to_numpy(),
            right.get_coordinates().to_numpy(),
        )

    def test_default_key_handles_altlocs(self, jge):
        # 1JGE has altlocs A/B at some residues; matching self with the
        # default key should preserve both (no spurious collapse).
        left, right = ColumnMatching().pair(jge, jge)
        # number of altloc=='B' atoms on both sides must match.
        assert (left["altloc"] == "B").sum() == (jge["altloc"] == "B").sum()
        assert (right["altloc"] == "B").sum() == (jge["altloc"] == "B").sum()

    def test_partial_overlap(self, jge):
        first = Scene(jge.iloc[: len(jge) // 2].copy())
        second = Scene(jge.iloc[len(jge) // 2 - 100:].copy())  # ~100-atom overlap
        left, right = ColumnMatching().pair(first, second)
        assert len(left) == len(right)
        assert 0 < len(left) <= 100
        for col in ("chain", "resid", "name", "altloc"):
            assert (left[col].values == right[col].values).all()

    def test_missing_key_raises(self, jge):
        with pytest.raises(KeyError, match="Missing match keys"):
            ColumnMatching(on=("chain", "not_a_column")).pair(jge, jge)


# --- SequenceMatching ----------------------------------------------------

class TestNeedlemanWunsch:
    def test_self_align(self):
        ai, aj = _needleman_wunsch("ACGT", "ACGT")
        assert list(zip(ai, aj)) == [(0, 0), (1, 1), (2, 2), (3, 3)]

    def test_internal_gap(self):
        ai, aj = _needleman_wunsch("ACGT", "AGT")
        # The 'C' (s1[1]) is dropped; 'A','G','T' align.
        assert list(zip(ai, aj)) == [(0, 0), (2, 1), (3, 2)]

    def test_block_gap(self):
        ai, aj = _needleman_wunsch("AAACCCGGG", "AAAGGG")
        assert list(zip(ai, aj)) == [(0, 0), (1, 1), (2, 2), (6, 3), (7, 4), (8, 5)]


class TestSequenceMatching:
    def test_self_pair_on_protein_chain(self, jge):
        chain_a = Scene(jge[jge["chain"] == "A"].copy())
        left, right = SequenceMatching().pair(chain_a, chain_a)
        # Every CA residue in chain A should pair to itself.
        n_ca = ((chain_a["name"] == "CA")).sum()
        assert len(left) == n_ca
        np.testing.assert_allclose(
            left.get_coordinates().to_numpy(),
            right.get_coordinates().to_numpy(),
        )

    def test_pair_with_internal_deletion(self, jge):
        chain_a = Scene(jge[jge["chain"] == "A"].copy())
        # Drop residues 100–110 from the "mobile" copy; the aligned pairs
        # must skip those positions but cover the rest.
        mask = (chain_a["resid"] >= 100) & (chain_a["resid"] <= 110)
        deleted = Scene(chain_a[~mask].copy())
        left, right = SequenceMatching().pair(deleted, chain_a)
        # No row may carry a deleted-region resid.
        assert not ((left["resid"] >= 100) & (left["resid"] <= 110)).any()
        assert not ((right["resid"] >= 100) & (right["resid"] <= 110)).any()
        # Resids must agree on both sides (sequence-identical residues align).
        np.testing.assert_array_equal(left["resid"].values, right["resid"].values)

    def test_no_overlap_raises(self, jge):
        chain_a = Scene(jge[jge["chain"] == "A"].copy())
        # rename the mobile chain so the auto-pairing finds nothing in common.
        relabeled = chain_a.copy()
        relabeled["chain"] = "Z"
        with pytest.raises(ValueError, match="no chains to pair"):
            SequenceMatching().pair(relabeled, chain_a)

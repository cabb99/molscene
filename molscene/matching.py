"""
Strategies for pairing atoms between two :class:`molscene.Scene` instances.

A ``Matching`` takes two scenes and returns two same-length sub-scenes whose
rows correspond atom-for-atom. The result feeds the Kabsch fit inside
:meth:`molscene.Scene.compute_transformation` (and indirectly
:meth:`molscene.Scene.superpose` and :meth:`molscene.Scene.rmsd`).

Three concrete strategies ship with MolScene:

* :class:`OrderMatching` — row ``i`` ↔ row ``i`` (default). Requires equal
  lengths.
* :class:`ColumnMatching` — inner-join on a tuple of columns. Default keys
  are the canonical PDB unique-atom identifiers
  ``(chain, resid, iCode, name, altloc)``.
* :class:`SequenceMatching` — per-chain Needleman–Wunsch on the one-letter
  protein/nucleic sequence (one atom per residue, ``CA`` by default).

Users can plug in arbitrary matchers either by subclassing :class:`Matching`
or by passing any callable ``(mobile, reference) -> (Scene, Scene)`` —
:func:`as_matching` will wrap it transparently.
"""

from abc import ABC, abstractmethod
from typing import Callable, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas


_DEFAULT_COLUMN_KEYS = ("chain", "resid", "iCode", "name", "altloc")


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

class Matching(ABC):
    """Strategy for pairing atoms between two scenes."""

    @abstractmethod
    def pair(self, mobile, reference) -> Tuple["Scene", "Scene"]:
        """Return two same-length sub-scenes whose rows correspond."""


# ---------------------------------------------------------------------------
# Order matching: trivial, row-by-row
# ---------------------------------------------------------------------------

class OrderMatching(Matching):
    """Pair atoms by row order; both scenes must have the same length."""

    def pair(self, mobile, reference):
        if len(mobile) != len(reference):
            raise ValueError(
                f"OrderMatching requires equal lengths, got {len(mobile)} and {len(reference)}"
            )
        return mobile, reference

    def __repr__(self) -> str:
        return "OrderMatching()"


# ---------------------------------------------------------------------------
# Column matching: inner-join on a tuple of columns
# ---------------------------------------------------------------------------

class ColumnMatching(Matching):
    """Pair atoms whose values in the ``on`` columns are identical.

    Duplicate keys on either side are resolved by keeping the first occurrence,
    so the result is always 1-to-1. The default key
    ``(chain, resid, iCode, name, altloc)`` is the strict PDB unique-atom
    identifier; drop ``altloc`` (or pre-filter with ``select(altloc=['A'])``)
    when working with scenes that only carry a single alternate location.
    """

    def __init__(self, on: Sequence[str] = _DEFAULT_COLUMN_KEYS):
        self.on = tuple(on)

    def pair(self, mobile, reference):
        from .Scene import Scene  # local import to avoid circularity

        on = list(self.on)
        missing_left = [c for c in on if c not in mobile.columns]
        missing_right = [c for c in on if c not in reference.columns]
        if missing_left or missing_right:
            raise KeyError(
                f"Missing match keys: mobile={missing_left}, reference={missing_right}"
            )

        left = pandas.DataFrame(mobile).drop_duplicates(subset=on, keep="first").reset_index(drop=True)
        right = pandas.DataFrame(reference).drop_duplicates(subset=on, keep="first").reset_index(drop=True)

        keys = pandas.merge(
            left[on].assign(_l=np.arange(len(left))),
            right[on].assign(_r=np.arange(len(right))),
            on=on,
            how="inner",
        )

        left_matched = left.iloc[keys["_l"].to_numpy()].reset_index(drop=True)
        right_matched = right.iloc[keys["_r"].to_numpy()].reset_index(drop=True)

        return (
            Scene(left_matched, **mobile._meta),
            Scene(right_matched, **reference._meta),
        )

    def __repr__(self) -> str:
        return f"ColumnMatching(on={self.on!r})"


# ---------------------------------------------------------------------------
# Sequence matching: per-chain Needleman–Wunsch
# ---------------------------------------------------------------------------

class SequenceMatching(Matching):
    """Pair atoms via per-chain pairwise sequence alignment.

    For every ``(chain_mobile, chain_reference)`` pair (either user-supplied
    via ``chain_pairs`` or defaulted to chains that share a name on both
    sides), the one-letter sequences are extracted via
    :meth:`molscene.Scene.get_sequence` and globally aligned with the
    Needleman–Wunsch algorithm. Only residues that align without gaps
    contribute pairs; the matched ``atom`` (default ``"CA"``) from each
    aligned residue is emitted into the output.

    Parameters
    ----------
    atom : str
        The atom name to take from each aligned residue (``"CA"`` for
        proteins, ``"P"`` for nucleic acids, etc.).
    chain_pairs : mapping or iterable of pairs, optional
        Explicit ``mobile_chain -> reference_chain`` mapping. If ``None``,
        chains with identical names on both sides are paired up.
    match_score, mismatch_score, gap_open, gap_extend : float
        Needleman–Wunsch scoring parameters.
    """

    def __init__(
        self,
        atom: str = "CA",
        chain_pairs: Optional[Union[Mapping[str, str], Iterable[Tuple[str, str]]]] = None,
        match_score: float = 1.0,
        mismatch_score: float = -1.0,
        gap_open: float = -10.0,
        gap_extend: float = -0.5,
    ):
        self.atom = atom
        self.match_score = float(match_score)
        self.mismatch_score = float(mismatch_score)
        self.gap_open = float(gap_open)
        self.gap_extend = float(gap_extend)
        if chain_pairs is None:
            self.chain_pairs = None
        elif isinstance(chain_pairs, Mapping):
            self.chain_pairs = list(chain_pairs.items())
        else:
            self.chain_pairs = [tuple(p) for p in chain_pairs]

    def pair(self, mobile, reference):
        from .Scene import Scene

        if self.chain_pairs is None:
            common = sorted(set(mobile["chain"].unique()) & set(reference["chain"].unique()))
            pairs = [(c, c) for c in common]
        else:
            pairs = self.chain_pairs

        if not pairs:
            raise ValueError(
                "SequenceMatching found no chains to pair; provide chain_pairs="
                " explicitly or ensure the chain names overlap."
            )

        seq_mob = mobile.get_sequence()
        seq_ref = reference.get_sequence()

        left_indices, right_indices = [], []
        for cm, cr in pairs:
            if cm not in seq_mob or cr not in seq_ref:
                continue
            s1, s2 = seq_mob[cm], seq_ref[cr]
            if not s1 or not s2:
                continue

            mob_resids = (
                mobile[(mobile["chain"] == cm) & (mobile["name"] == self.atom)]
                .drop_duplicates(subset=["chain", "resid"])["resid"]
                .to_numpy()
            )
            ref_resids = (
                reference[(reference["chain"] == cr) & (reference["name"] == self.atom)]
                .drop_duplicates(subset=["chain", "resid"])["resid"]
                .to_numpy()
            )
            # The sequence returned by ``get_sequence`` already enumerates
            # residues in resid-sorted order; align lengths defensively.
            n = min(len(s1), len(mob_resids))
            m = min(len(s2), len(ref_resids))
            s1, s2 = s1[:n], s2[:m]
            mob_resids, ref_resids = mob_resids[:n], ref_resids[:m]

            aligned_i, aligned_j = _needleman_wunsch(
                s1, s2,
                match=self.match_score,
                mismatch=self.mismatch_score,
                gap_open=self.gap_open,
                gap_extend=self.gap_extend,
            )

            for i, j in zip(aligned_i, aligned_j):
                rid_m = mob_resids[i]
                rid_r = ref_resids[j]
                left_rows = mobile.index[
                    (mobile["chain"] == cm)
                    & (mobile["resid"] == rid_m)
                    & (mobile["name"] == self.atom)
                ]
                right_rows = reference.index[
                    (reference["chain"] == cr)
                    & (reference["resid"] == rid_r)
                    & (reference["name"] == self.atom)
                ]
                if len(left_rows) == 0 or len(right_rows) == 0:
                    continue
                left_indices.append(left_rows[0])
                right_indices.append(right_rows[0])

        if not left_indices:
            raise ValueError("SequenceMatching produced no aligned atoms.")

        left = pandas.DataFrame(mobile).loc[left_indices].reset_index(drop=True)
        right = pandas.DataFrame(reference).loc[right_indices].reset_index(drop=True)
        return Scene(left, **mobile._meta), Scene(right, **reference._meta)

    def __repr__(self) -> str:
        return (
            f"SequenceMatching(atom={self.atom!r}, "
            f"match={self.match_score}, mismatch={self.mismatch_score}, "
            f"gap_open={self.gap_open}, gap_extend={self.gap_extend})"
        )


# ---------------------------------------------------------------------------
# Coercion helper
# ---------------------------------------------------------------------------

_BUILTIN_NAMES = {
    "order": OrderMatching,
    "columns": ColumnMatching,
    "sequence": SequenceMatching,
}


def as_matching(spec) -> Matching:
    """Coerce ``spec`` into a :class:`Matching` instance.

    Accepted forms:

    * ``None``                                  → :class:`OrderMatching`
    * :class:`Matching` instance                → returned as-is
    * string ``'order' | 'columns' | 'sequence'`` → corresponding default-constructed class
    * tuple or list of column names              → :class:`ColumnMatching` on those columns
    * callable ``(mobile, reference) -> (Scene, Scene)`` → wrapped as a one-off matcher
    """
    if spec is None:
        return OrderMatching()
    if isinstance(spec, Matching):
        return spec
    if isinstance(spec, str):
        try:
            return _BUILTIN_NAMES[spec]()
        except KeyError:
            raise ValueError(
                f"unknown matching strategy {spec!r}; "
                f"expected one of {sorted(_BUILTIN_NAMES)}"
            )
    if isinstance(spec, (tuple, list)):
        return ColumnMatching(on=spec)
    if callable(spec):
        return _CallableMatching(spec)
    raise TypeError(f"cannot interpret {spec!r} as a Matching")


class _CallableMatching(Matching):
    """Adapter that turns a bare ``(mobile, reference) -> (Scene, Scene)`` callable
    into a :class:`Matching`."""

    def __init__(self, fn: Callable):
        self.fn = fn

    def pair(self, mobile, reference):
        return self.fn(mobile, reference)

    def __repr__(self) -> str:
        return f"_CallableMatching({self.fn!r})"


# ---------------------------------------------------------------------------
# Needleman–Wunsch (pure numpy)
# ---------------------------------------------------------------------------

def _needleman_wunsch(
    s1: str, s2: str,
    *,
    match: float = 1.0,
    mismatch: float = -1.0,
    gap_open: float = -10.0,
    gap_extend: float = -0.5,
):
    """Global pairwise alignment with affine gap penalties.

    Returns two equal-length arrays of indices ``(i_in_s1, j_in_s2)`` for the
    positions that align without gaps.
    """
    n, m = len(s1), len(s2)
    NEG_INF = -1e18

    # M[i,j] = best score ending with s1[i-1] aligned to s2[j-1]
    # X[i,j] = best score ending with s1[i-1] aligned to a gap (deletion from s2)
    # Y[i,j] = best score ending with s2[j-1] aligned to a gap (insertion into s2)
    M = np.full((n + 1, m + 1), NEG_INF)
    X = np.full((n + 1, m + 1), NEG_INF)
    Y = np.full((n + 1, m + 1), NEG_INF)

    M[0, 0] = 0.0
    for i in range(1, n + 1):
        X[i, 0] = gap_open + (i - 1) * gap_extend
    for j in range(1, m + 1):
        Y[0, j] = gap_open + (j - 1) * gap_extend

    # Trace pointers: 0 = M, 1 = X, 2 = Y, -1 = origin
    trace_M = np.full((n + 1, m + 1), -1, dtype=np.int8)
    trace_X = np.full((n + 1, m + 1), -1, dtype=np.int8)
    trace_Y = np.full((n + 1, m + 1), -1, dtype=np.int8)

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            score = match if s1[i - 1] == s2[j - 1] else mismatch
            cands = (M[i - 1, j - 1], X[i - 1, j - 1], Y[i - 1, j - 1])
            best = max(cands)
            trace_M[i, j] = cands.index(best)
            M[i, j] = best + score

            cands_x = (M[i - 1, j] + gap_open, X[i - 1, j] + gap_extend)
            X[i, j] = max(cands_x)
            trace_X[i, j] = 0 if cands_x[0] >= cands_x[1] else 1

            cands_y = (M[i, j - 1] + gap_open, Y[i, j - 1] + gap_extend)
            Y[i, j] = max(cands_y)
            trace_Y[i, j] = 0 if cands_y[0] >= cands_y[1] else 2

    # Backtrace from the best end cell among the three matrices at (n, m).
    end_scores = (M[n, m], X[n, m], Y[n, m])
    cur_state = end_scores.index(max(end_scores))
    i, j = n, m
    aligned_i, aligned_j = [], []
    while i > 0 or j > 0:
        if cur_state == 0:
            # diagonal step
            if i == 0 or j == 0:
                break
            aligned_i.append(i - 1)
            aligned_j.append(j - 1)
            cur_state = int(trace_M[i, j])
            i -= 1
            j -= 1
        elif cur_state == 1:
            if i == 0:
                break
            cur_state = int(trace_X[i, j])
            i -= 1
        else:  # cur_state == 2
            if j == 0:
                break
            cur_state = int(trace_Y[i, j])
            j -= 1
        if cur_state == -1:
            break

    aligned_i.reverse()
    aligned_j.reverse()
    return np.array(aligned_i, dtype=int), np.array(aligned_j, dtype=int)

#!/usr/bin/env python3
"""Download residue connectivity from the PDB Chemical Component Dictionary.

Usage::

    python devtools/scripts/update_connectivity.py

Fetches individual component CIF files from the RCSB for every standard
protein, DNA, and RNA residue, parses ``_chem_comp_bond`` to extract
heavy-atom connectivity, and writes ``molscene/data/connectivity.json``.

Source
------
PDB Chemical Component Dictionary (CCD)
https://www.wwpdb.org/data/ccd
Component files: https://files.rcsb.org/ligands/view/{COMP_ID}.cif
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

CCD_URL = "https://files.rcsb.org/ligands/view/{comp_id}.cif"
OUTPUT = Path(__file__).resolve().parents[2] / "molscene" / "data" / "connectivity.json"

# Standard residue IDs to fetch
PROTEIN_RESIDUES = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY",
    "HIS", "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER",
    "THR", "TRP", "TYR", "VAL",
]
DNA_RESIDUES = ["DA", "DC", "DG", "DT"]
RNA_RESIDUES = ["A", "C", "G", "U"]
OTHER_RESIDUES = ["HOH"]

ALL_RESIDUES = PROTEIN_RESIDUES + DNA_RESIDUES + RNA_RESIDUES + OTHER_RESIDUES

# Inter-residue link atoms (connecting consecutive residues in a polymer)
PROTEIN_LINK = ["C", "N"]      # peptide bond: C(i) → N(i+1)
NUCLEIC_LINK = ["O3'", "P"]   # phosphodiester: O3'(i) → P(i+1)

_CIF_TOKENIZER = re.compile(
    r"""\"[^\"]*\"   |  # double-quoted
        '[^']*'      |  # single-quoted
        \#[^\n]*     |  # comment
        [^\s'"#]+       # unquoted token
    """,
    re.VERBOSE,
)


def _fetch_component(comp_id: str) -> str:
    """Download a CCD component CIF file."""
    url = CCD_URL.format(comp_id=comp_id)
    req = urllib.request.Request(url, headers={"User-Agent": "molscene-updater/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def _parse_bonds(cif_text: str) -> list:
    """Extract ``_chem_comp_bond`` pairs, heavy atoms only.

    Returns a list of ``[atom1, atom2]`` pairs with quote characters
    stripped but primes preserved (e.g. ``"O3'"``).
    """
    lines = cif_text.splitlines()
    header: list = []
    data: list = []
    in_section = False
    prefix = "_chem_comp_bond."

    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("loop_"):
            in_section = False
        elif stripped.startswith(prefix):
            header.append(stripped.split(".")[-1])
            in_section = True
        elif in_section:
            tokens = [
                t.strip("'\"")
                for t in _CIF_TOKENIZER.findall(stripped)
                if not t.startswith("#")
            ]
            if len(tokens) == len(header):
                data.append(dict(zip(header, tokens)))
            else:
                # End of loop data
                in_section = False

    # Filter to heavy-atom bonds
    bonds: list = []
    for row in data:
        a1 = row.get("atom_id_1", "")
        a2 = row.get("atom_id_2", "")
        # Skip hydrogen atoms (names starting with H, or 1H, 2H, 3H)
        if _is_hydrogen(a1) or _is_hydrogen(a2):
            continue
        bonds.append([a1, a2])
    return bonds


def _is_hydrogen(name: str) -> bool:
    """Return True if *name* is a hydrogen atom name."""
    if not name:
        return False
    if name[0] == "H":
        return True
    # Numbered hydrogens like 1HB, 2HG
    if len(name) >= 2 and name[0].isdigit() and name[1] == "H":
        return True
    return False


def _strip_primes(name: str) -> str:
    """Strip prime characters from atom names (C1' → C1)."""
    return name.replace("'", "")


def build_connectivity() -> dict:
    """Download and assemble the connectivity table."""
    result: dict = {
        "_source": (
            "PDB Chemical Component Dictionary (CCD) — "
            "https://www.wwpdb.org/data/ccd"
        ),
        "protein_link": [_strip_primes(a) for a in PROTEIN_LINK],
        "nucleic_link": [_strip_primes(a) for a in NUCLEIC_LINK],
        "residues": {},
    }

    for comp_id in ALL_RESIDUES:
        print(f"  Fetching {comp_id} ...", end=" ", flush=True)
        try:
            cif_text = _fetch_component(comp_id)
            bonds = _parse_bonds(cif_text)
            # Strip primes to match CIF parser output (label_atom_id without quotes)
            bonds = [[_strip_primes(a), _strip_primes(b)] for a, b in bonds]
            result["residues"][comp_id] = bonds
            print(f"{len(bonds)} bonds")
        except Exception as e:
            print(f"FAILED: {e}")
            result["residues"][comp_id] = []

    return result


def write_json(data: dict, path: Path) -> None:
    """Write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    n = len(data["residues"])
    print(f"Wrote {n} residues to {path}")


def main() -> None:
    print(f"Downloading connectivity from CCD ...")
    data = build_connectivity()
    write_json(data, OUTPUT)

    # Sanity checks
    ala = data["residues"]["ALA"]
    ala_pairs = {tuple(sorted(p)) for p in ala}
    assert ("CA", "N") in ala_pairs, "ALA should have N-CA bond"
    assert ("C", "CA") in ala_pairs, "ALA should have CA-C bond"
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()

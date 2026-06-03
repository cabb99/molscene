#!/usr/bin/env python3
"""Download element data from PubChem and write molscene/data/elements.json.

Usage::

    python devtools/scripts/update_elements.py

Fetches the PubChem periodic-table CSV via their public REST API and converts
it to the compact JSON format consumed by ``molscene.data.element_info``.

Source
------
PubChem Periodic Table — https://pubchem.ncbi.nlm.nih.gov/periodic-table/
REST endpoint: https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV
"""

import csv
import io
import json
import sys
import urllib.request
from pathlib import Path

PUBCHEM_URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/periodictable/CSV"
OUTPUT = Path(__file__).resolve().parents[2] / "molscene" / "data" / "elements.json"


def fetch_csv(url: str) -> str:
    """Download the PubChem periodic-table CSV."""
    req = urllib.request.Request(url, headers={"User-Agent": "molscene-updater/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read().decode("utf-8")


def parse_elements(text: str) -> dict:
    """Parse CSV text into the elements dict used by molscene."""
    reader = csv.DictReader(io.StringIO(text))
    elements: dict = {
        "_source": "PubChem Periodic Table — https://pubchem.ncbi.nlm.nih.gov/periodic-table/",
    }
    for row in reader:
        symbol = row["Symbol"]
        atomic_number = int(row["AtomicNumber"])
        mass_str = row["AtomicMass"]
        radius_str = row["AtomicRadius"]  # picometers

        mass = float(mass_str) if mass_str else 0.0
        # Convert pm → Å (divide by 100)
        radius = round(float(radius_str) / 100, 2) if radius_str else 0.0

        elements[symbol] = {
            "atomicnumber": atomic_number,
            "mass": mass,
            "radius": radius,
        }
    return elements


def write_json(elements: dict, path: Path) -> None:
    """Write the elements dict as formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(elements, f, indent=2)
        f.write("\n")
    print(f"Wrote {len(elements) - 1} elements to {path}")


def main() -> None:
    print(f"Fetching element data from {PUBCHEM_URL} ...")
    text = fetch_csv(PUBCHEM_URL)
    elements = parse_elements(text)
    write_json(elements, OUTPUT)

    # Quick sanity check
    c = elements.get("C", {})
    assert c["atomicnumber"] == 6, "Carbon should be Z=6"
    assert abs(c["mass"] - 12.011) < 0.01, "Carbon mass looks wrong"
    assert abs(c["radius"] - 1.70) < 0.05, "Carbon radius looks wrong"
    print("Sanity checks passed.")


if __name__ == "__main__":
    main()

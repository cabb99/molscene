"""Element property look-ups backed by ``elements.json``.

The module exposes a single :class:`ElementInfo` instance called
``element_info`` with dict-valued attributes for each property::

    >>> from molscene.data.element_info import element_info
    >>> element_info.mass["C"]
    12.011
    >>> element_info.atomicnumber["O"]
    8
    >>> element_info.radius["N"]   # Van der Waals radius in Å
    1.55

Data is sourced from PubChem and can be refreshed by running
``devtools/scripts/update_elements.py``.
"""

import json
from pathlib import Path
from typing import Dict

_DATA_FILE = Path(__file__).parent / "elements.json"


class ElementInfo:
    """Read-only container for per-element properties."""

    def __init__(self, path: Path = _DATA_FILE) -> None:
        with open(path) as f:
            raw = json.load(f)

        self.mass: Dict[str, float] = {}
        self.atomicnumber: Dict[str, int] = {}
        self.radius: Dict[str, float] = {}

        for key, val in raw.items():
            if key.startswith("_"):
                continue
            self.mass[key] = val["mass"]
            self.atomicnumber[key] = val["atomicnumber"]
            self.radius[key] = val["radius"]


element_info = ElementInfo()

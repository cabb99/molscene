# MolScene

[![GitHub Actions Build Status](https://github.com/cabb99/molscene/workflows/CI/badge.svg)](https://github.com/cabb99/molscene/actions?query=workflow%3ACI)
[![codecov](https://codecov.io/gh/cabb99/molscene/branch/main/graph/badge.svg)](https://codecov.io/gh/cabb99/molscene/branch/main)
[![Documentation Status](https://readthedocs.org/projects/molscene/badge/?version=latest)](https://molscene.readthedocs.io/en/latest/?badge=latest)

**Molecular structures as pandas DataFrames.**

MolScene reads molecular structures (PDB, mmCIF) into [pandas](https://pandas.pydata.org/) DataFrames and writes them back out (PDB, mmCIF, GRO). Every `Scene` is a DataFrame so the full pandas toolbox (filtering, grouping, plotting, `to_csv`, etc.) works out of the box. MolScene adds a layer on top for molecular structural data operations: atom selection, rigid-body transformations and alignment, residue morphing, multi-frame trajectories, and distance maps.

```python
from molscene import Scene

s = Scene.from_pdb("1jge.pdb")          # load a structure into a DataFrame
ca = s.select("chain A and name CA")    # VMD-style atom selection
ca = ca - ca.get_center()               # center it on the origin (operator math)
ca.write_pdb("ca_centered.pdb")         # write it back out
```

## Why use MolScene?

- **Just a DataFrame.** `Scene` subclasses `pandas.DataFrame`, so columns like
  `chain`, `resid`, `resname`, `name`, `x`, `y`, `z` are all there to slice, group, and analyze with common data analysis tools.
- **Atom selection.** Use a familiar VMD-style string (`"protein and within 5 of resname HEM"`) via the optional [`molselect`](https://github.com/cabb99/molselect) engine, or plain keyword filters (`s.select(chain=["A"], name=["CA"])`) with no extra dependencies.
- **Geometry reads like math.** `+`, `-`, `*`, `/` translate, scale, and reflect coordinates; `scene1 + scene2` concatenates two structures.
- **Built-in structural alignment** Pair atoms by order, by columns, or by sequence (Needleman–Wunsch), then `superpose`, compute `rmsd`, or get a `Transformation` (rotation + translation).
- **Smooth morphing and trajectories.** Interpolate between conformations with
  screw-linear (dual-quaternion) or slerp motion, and carry multiple coordinate
  frames per scene.
- **Metadata travels with the data.** Attach arbitrary metadata to a scene, like bonds, angles or dihedrals, that depend on atom indices; it is preserved through selections and transformations.

## Installation

```bash
pip install molscene
```

String atom selection (`Scene.select("chain A and resid 1 to 100")`) is powered by the optional [`molselect`](https://github.com/cabb99/molselect) package. Install it with the extra:

```bash
pip install "molscene[selection]"
```

The keyword form `Scene.select(chain=["A"])` works without it.

> **Conda:** a conda-forge recipe lives in [`recipe/`](recipe/); `conda install`
> support will follow once the package is published to conda-forge.

## Quickstart

```python
import numpy as np
from molscene import Scene

# Load from a file (format auto-detected by from_file, or use from_pdb/from_cif/from_gro)
s = Scene.from_pdb("1jge.pdb")

# It's a DataFrame — use pandas directly
print(s["resname"].value_counts())
print(len(s), "atoms")

# Select atoms (string needs molscene[selection]; kwargs always work)
backbone = s.select("name CA C N O")
chain_a  = s.select(chain=["A"])

# Geometry with operators
centered = s - s.get_center()
shifted  = s + np.array([10, 0, 0])
scaled   = s * 2.0

# Align one structure onto another and measure RMSD
mobile = Scene.from_pdb("model.pdb")
ref    = Scene.from_pdb("native.pdb")
aligned = mobile.superpose(ref, match="sequence")
print(aligned.rmsd(ref, match="sequence"))

# Write back out (format from extension)
centered.write_pdb("centered.pdb")
```

See the [full documentation](https://molscene.readthedocs.io/) for the User Guide
(with API reference) and Developer Guide.

## Documentation

- **Getting Started**: installation and first steps
- **User Guide**: full API reference
- **Examples**
- **Developer Guide**

Build the docs locally with:

```bash
pip install "molscene[docs]"
sphinx-build -b html docs docs/_build/html
```

## Contributing

Contributions are welcome — see [`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md)
and the [Code of Conduct](CODE_OF_CONDUCT.md). Run the test suite with:

```bash
pip install "molscene[test,selection]"
pytest molscene/tests/
```

## License

MIT — see [LICENSE](LICENSE).

### Copyright

Copyright (c) 2025-2026, Carlos Bueno

#### Acknowledgements

Project skeleton based on the
[Computational Molecular Science Python Cookiecutter](https://github.com/molssi/cookiecutter-cms)
version 1.10.

MolScene Roadmap
===============================================

Overview
--------

`MolScene` is a lightweight, pandas-based scene manager for molecular systems. It focuses on:

* **DataFrame-like API**: seamless integration with pandas for filtering, grouping, and I/O.
* **Minimal assumptions**: users define atom types, bonds, and topology as they wish.
* **Extensibility**: modular subpackages, plugin hooks, and clear extension points.
* **Frame support**: handle time series (movies) with atoms appearing/disappearing.
* **Scene composition**: relative placement, rotation, morphing of sub-scenes.
* **Export adapters**: easy export to NGLView, VMD scripts, Blender, and standard coordinate files.

Modular Architecture & File Layout
----------------------------------

.. code-block:: text

    molscene/
    ├── __init__.py           # top‑level API: Scene, read_scene, write_scene
    ├── core.py               # Scene class, MetadataStore, facade methods
    ├── selection.py          # SelectionEngine: VMD‑style DSL
    ├── sequence.py           # SequenceView: FASTA export, mutation
    ├── geometry.py           # rotate, translate, distance, morph routines
    ├── io/                   # parsers & writers
    │   ├── __init__.py
    │   ├── factory.py        # read_scene / write_scene factories
    │   ├── pdb.py            # parse/write PDB
    │   ├── cif.py            # parse/write mmCIF
    │   └── gro.py            # parse/write GRO
    ├── export/               # export adapters
    │   ├── __init__.py
    │   ├── nglview.py        # to_nglview()
    │   └── vmd.py            # to_vmd_script()
    ├── utils.py              # shared helpers (chain naming, residue maps)
    └── plugins.py            # plugin registry & discovery

Each submodule is small and self‑contained, making it easy to extend and test.

Core Design Patterns
--------------------

* **Facade (core.py)**: `Scene` exposes a unified API while delegating to helper modules.
* **Factory (io/factory.py)**: `read_scene(filename)` and `write_scene(scene, filename)` choose the appropriate parser/writer.
* **Strategy (selection.py)**: enable multiple selection backends (pandas, MDAnalysis, mdtraj).
* **Adapter (export/)**: wrap external tools (NGLView, VMD) behind consistent methods.
* **Observer / Hooks (plugins.py)**: lightweight pre/post hooks on load, select, transform, and write operations.

Feature Roadmap
---------------

Core Data & Metadata
~~~~~~~~~~~~~~~~~~~~

* **MetadataStore**: attach per‑atom and per‑frame arrays (charges, angles, dihedrals, custom labels).
* **Dynamic frames**: presence masks for atoms, enabling movies with birth/death events.
* **Hybrid layers**: link atomistic + coarse‑grained representations with bidirectional mapping.

Analysis Utilities
~~~~~~~~~~~~~~~~~~

* **PCA & clustering**: built‑in wrappers (scikit‑learn) for dimensionality reduction and selection of conformations.
* **Morphing**: interpolate between two scenes into a sequence of frames.

Scene Composition
~~~~~~~~~~~~~~~~~

* **Relative placement**: `scene.place(subscene, rotation, translation)` and docking shortcuts.
* **Box / solvation stubs**: minimal methods to generate coordinate-only solvent boxes.

Export & Interoperability
~~~~~~~~~~~~~~~~~~~~~~~~~

* **Coordinate exports**: PDB, mmCIF, GRO, XYZ via core I/O.
* **Graphics adapters**: NGLView widget, VMD script, Blender exporter, surface/mesh OBJ.

Plugin & Hook System
~~~~~~~~~~~~~~~~~~~~

* **I/O plugins**: allow third parties to register new file formats without modifying core.
* **Export plugins**: extend export targets easily.

Step‑by‑Step Implementation Plan
--------------------------------

Implementation Checklist
~~~~~~~~~~~~~~~~~~~~~~~~

- [ ] **Scaffold `core.py`**
    - [ ] Implement `Scene` class with minimal DataFrame wrapping.
    - [ ] Build `MetadataStore` for per‑atom / per‑frame metadata.
    - [ ] Expose `.sel` and `.sequence` properties (placeholder implementations).

- [ ] **Factory & I/O**
    - [ ] Create `io/factory.py` with `read_scene()` / `write_scene()`.
    - [ ] Stub out `io/pdb.py`, `io/cif.py`, `io/gro.py` parsers and writers.
    - [ ] Wire into `read_scene` by file extension.

- [ ] **Selection Engine**
    - [ ] In `selection.py`, implement `SelectionEngine.select(expr: str)`.
    - [ ] Start with basic parsing for `name`, `resname`, `chainID`, numeric ranges, logical ops.

- [ ] **Sequence View**
    - [ ] In `sequence.py`, build `SequenceView` for `.to_fasta()` and `.mutate()`.
    - [ ] Hook onto `Scene.sequence` property.

- [ ] **Geometry Utilities**
    - [ ] Move existing `rotate`, `translate`, `distance_map_*` into `geometry.py`.
    - [ ] Implement `morph` interpolation function.

- [ ] **Export Adapters**
    - [ ] In `export/nglview.py`, write `to_nglview(scene)` using NGLView’s Python API.
    - [ ] In `export/vmd.py`, start simple `to_vmd_script(scene)` generator.

- [ ] **Plugin System**
    - [ ] Define `plugins.py` registry API: `register_reader(ext, func)`, `register_exporter(name, func)`.
    - [ ] Enable discovery via entry points.

- [ ] **Testing & CI**
    - [ ] Create `tests/` directory mirroring modules: `test_core.py`, `test_io_pdb.py`, `test_selection.py`, `test_sequence.py`, `test_export.py`.
    - [ ] Add continuous integration for linting, type‑checking (mypy), and pytest.

- [ ] **Documentation & Tutorials**
    - [ ] Write docstrings for all public APIs.
    - [ ] Develop a tutorial notebook demonstrating: reading a PDB, selecting CA atoms, exporting FASTA, docking ligand, and visualizing with NGLView.

- [ ] **Release & Versioning**
    - [ ] Bump to `v2.0.0-alpha`.
    - [ ] Publish on PyPI, announce on GitHub with migration guide.

.. note::
   To render checkboxes in HTML, use the MyST or recommonmark Markdown parser, or a Sphinx extension that supports GitHub-style checklists. In plain reStructuredText, checkboxes will appear as literal text.

.. note::
   This roadmap keeps core functionality narrow and focused on coordinate + DataFrame management while providing clear extension points for sequence, selection, I/O, and graphics. It avoids chemistry assumptions, leaving topology and free‑energy analysis to specialized libraries.

   *Next up: kick off the `core.py` scaffold with the MetadataStore and basic facade methods.*

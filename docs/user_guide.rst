User Guide
==========

This guide covers MolScene feature by feature and ends with the complete
:ref:`API reference <api-reference>`. For short, runnable recipes — every one of
which is executed as part of the test suite — see :doc:`examples`.

.. contents:: On this page
   :local:
   :depth: 2

The Scene object
----------------

:class:`~molscene.Scene` subclasses :class:`pandas.DataFrame`. Each row is an
atom; columns hold per-atom fields. Reading a structure populates the canonical
columns ``recname``, ``serial``, ``name``, ``altloc``, ``resname``, ``chain``,
``resid``, ``iCode``, ``x``, ``y``, ``z``, ``occupancy``, ``beta``, ``element``
(plus derived helpers such as ``fragment``, ``residue index``).

You can build a ``Scene`` directly from coordinates or a DataFrame:

.. code-block:: python

    import numpy as np
    import pandas as pd
    from molscene import Scene

    Scene(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]]))   # (N, 3) array
    Scene(pd.DataFrame({"x": [0.0], "y": [0.0], "z": [0.0]}))

Because it is a DataFrame, everything you already do with pandas works:
``scene[scene["element"] == "C"]``, ``scene.groupby("resname")``,
``scene.to_csv(...)``, plotting, and so on. The structure-aware methods below
layer on top of that.

Reading and writing files
--------------------------

MolScene reads PDB and mmCIF files:

.. code-block:: python

    s_pdb  = Scene.from_pdb("structure.pdb")
    s_cif  = Scene.from_cif("structure.cif")
    s_auto = Scene.from_file("structure.cif")   # format from the extension

.. note::

   Reading GRO files (:meth:`~molscene.Scene.from_gro`) is not implemented yet
   and raises :class:`NotImplementedError`; GRO *writing* is supported (see
   below).

If `PDBFixer <https://github.com/openmm/pdbfixer>`_ is installed, you can clean
and protonate structures on the way in with
:meth:`~molscene.Scene.from_fixPDB` / :meth:`~molscene.Scene.from_fixer`.

Write with the matching methods (or :meth:`~molscene.Scene.to_file`, which
picks the writer from the extension):

.. code-block:: python

    scene.write_pdb("out.pdb")
    scene.write_cif("out.cif")
    scene.write_gro("out.gro")
    scene.to_file("out.pdb")
    scene.to_csv("atoms.csv", index=False)   # inherited from pandas

Atom selection
--------------

:meth:`~molscene.Scene.select` has two complementary modes that can be combined
(their masks are AND-merged):

**Selection strings** (preferred) are evaluated by
`molselect <https://github.com/cabb99/molselect>`_, giving the full VMD-style
grammar — booleans, ranges, ``within``, ``same residue as``, regular
expressions, and more:

.. code-block:: python

    scene.select("chain A and resid 1 to 100 and name CA")
    scene.select("protein and not hydrogen")
    scene.select("within 5 of resname HEM")

.. note::

   String selection requires the optional ``molselect`` package
   (``pip install "molscene[selection]"``). The complete selection-language
   reference lives in the
   `molselect documentation <https://github.com/cabb99/molselect>`_. If
   ``molselect`` is not installed, a string selection raises a clear
   ``ImportError`` pointing you to the extra.

**Keyword filters** keep rows where a column matches any of the given values
and need no extra dependency:

.. code-block:: python

    scene.select(chain=["A"])
    scene.select(resname=["ALA", "GLY"], name=["CA"])

Selection always returns a new ``Scene`` and preserves metadata.

Sequence, mass, and secondary structure
----------------------------------------

.. code-block:: python

    seq    = scene.get_sequence()       # one-letter sequence per chain
    scene  = scene.compute_mass()       # returns a copy with a 'mass' column
    scene["mass"].sum()                 # e.g. total mass

:meth:`~molscene.Scene.compute_secondary_structure` shells out to DSSP and
returns a copy of the scene with the DSSP assignment merged in per residue:

.. code-block:: python

    scene = scene.compute_secondary_structure()

.. note::

   Secondary-structure assignment requires the external ``mkdssp`` (DSSP)
   executable on your ``PATH``; install it separately (e.g. via conda-forge's
   ``dssp`` package). Without it the call raises :class:`FileNotFoundError`.

Coordinate operators
---------------------

Arithmetic operators act atom-wise on coordinates while preserving metadata and
every coordinate frame:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Expression
     - Meaning
   * - ``scene + v`` / ``scene - v``
     - translate by a scalar or 3-vector ``v``
   * - ``scene * f`` / ``scene / f``
     - scale per axis by a scalar or 3-vector ``f``
   * - ``-scene``
     - reflect through the origin
   * - ``scene1 + scene2``
     - concatenate the two atom lists
   * - ``scene1 - scene2``
     - remove atoms of ``scene2`` from ``scene1``

Equivalent method forms exist: :meth:`~molscene.Scene.translate`,
:meth:`~molscene.Scene.rotate` / :meth:`~molscene.Scene.dot` (apply a 3×3
matrix), :meth:`~molscene.Scene.get_center` and :meth:`~molscene.Scene.center`.
The precise rules and errors are documented in
:ref:`scene-arithmetic-operators`.

.. code-block:: python

    import numpy as np

    centered = scene - scene.get_center()
    rotated  = scene.rotate(np.eye(3))
    combined = protein + ligand

Transformations and structural alignment
-----------------------------------------

A :class:`~molscene.Transformation` is a first-class rigid-body motion
(rotation + translation). Apply it to a scene with
:meth:`~molscene.Scene.transform`, invert it, compose two of them, or
interpolate between them (``slerp`` or dual-quaternion ``sclerp``).

To align two structures you first decide how their atoms correspond. MolScene
ships three :class:`~molscene.Matching` strategies:

* :class:`~molscene.OrderMatching` — row *i* ↔ row *i* (default; equal lengths).
* :class:`~molscene.ColumnMatching` — inner-join on key columns
  (default ``chain, resid, iCode, name, altloc``).
* :class:`~molscene.SequenceMatching` — per-chain Needleman–Wunsch alignment of
  the one-letter sequence (one atom per residue, ``CA`` by default).

Any callable ``(mobile, reference) -> (Scene, Scene)`` also works via
:func:`~molscene.as_matching`.

.. code-block:: python

    # Superpose `mobile` onto `reference` and measure fit quality
    aligned = mobile.superpose(reference, match="sequence")
    error   = aligned.rmsd(reference, match="sequence")

    # Or get the Transformation explicitly
    T = mobile.compute_transformation(reference, match="columns")
    moved = mobile.transform(T)

Morphing
--------

:meth:`~molscene.Scene.morph_segment` rigidly repositions each residue in a
chain's ``resid_range`` by a :class:`~molscene.Transformation` interpolated
between two anchors — ``t_start`` at the first residue and ``t_end`` at the
last — propagating a smooth twist across the segment (screw-linear by default).
Each residue moves as a rigid body; atoms outside the range are untouched.

.. code-block:: python

    morphed = scene.morph_segment("A", range(275, 341), t_start, t_end)

Multi-frame coordinates
------------------------

A single ``Scene`` can carry a stack of coordinate frames (a trajectory or a
morph movie):

.. code-block:: python

    scene.set_coordinate_frames(frames)   # (n_frames, n_atoms, 3)
    scene.n_frames
    for frame in scene.iterframes():
        ...                               # `frame` is a Scene for that frame

Distance maps
-------------

.. code-block:: python

    D = scene.distance_map()                       # dense (N, N) matrix
    pairs, dists = scene.distance_map_sparse(5.0)  # only pairs within 5 Å

Metadata
--------

Attach arbitrary metadata to a scene as attributes; it is stored in a private
``_meta`` dict and inherited by sub-scenes produced through selection,
filtering, or transformation:

.. code-block:: python

    scene.author = "CB"
    scene.pH = 7.4
    sub = scene.select(chain=["A"])
    sub.author        # -> "CB"

Columns whose names start with ``index_`` are treated as index-aligned metadata
and are carried along consistently during selection — handy for per-bond or
per-pair properties.

.. _api-reference:

API reference
-------------

.. toctree::
   :maxdepth: 2

   api

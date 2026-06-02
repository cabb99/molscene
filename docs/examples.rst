MolScene Usage Examples
============================

.. contents:: Table of Contents
   :local:
   :depth: 2

Every ``>>>`` snippet on this page is executed by the test suite
(``pytest --doctest-glob="*.rst"``), so the examples are guaranteed to work
against the current code.

.. note::

   Two names appear in the file-based examples below: ``example_structure`` is
   the path to a small PDB shipped with the MolScene source, and ``workdir`` is
   a throwaway directory for any files written. In your own code replace them
   with your structure's path and wherever you want output to go.


1. Creating and inspecting a Scene
----------------------------------

A ``Scene`` can be created from an ``(N, 3)`` array of coordinates or from a
DataFrame with ``x``, ``y``, ``z`` columns. It *is* a DataFrame, so it has a
row per atom and the usual structural columns.

.. code-block:: python

    >>> import numpy as np
    >>> from molscene import Scene
    >>> coords = np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0]])
    >>> scene = Scene(coords)
    >>> len(scene)
    3
    >>> scene.get_coordinates().shape
    (3, 3)
    >>> {"chain", "resid", "name", "x", "y", "z"} <= set(scene.columns)
    True


2. Reading and writing files
-----------------------------

MolScene reads PDB and mmCIF files and writes PDB, mmCIF, and GRO.
:meth:`~molscene.Scene.from_file` / :meth:`~molscene.Scene.to_file` pick the
format from the extension.

.. code-block:: python

    >>> from molscene import Scene
    >>> protein = Scene.from_pdb(example_structure).select(model=[1])
    >>> len(protein) > 0
    True

    >>> # write a few formats into the scratch directory
    >>> pdb_out = workdir / "out.pdb"
    >>> protein.write_pdb(pdb_out)
    >>> pdb_out.exists()
    True
    >>> protein.to_file(str(workdir / "out.cif"))   # format from the extension
    >>> protein.write_gro(workdir / "out.gro")
    >>> (workdir / "out.gro").exists()
    True

    >>> # round-trip preserves the atom count
    >>> reloaded = Scene.from_pdb(pdb_out)
    >>> len(reloaded) == len(protein)
    True

You can also dump the table with the inherited pandas ``to_csv``:

.. code-block:: python

    >>> protein.to_csv(workdir / "atoms.csv", index=False)
    >>> (workdir / "atoms.csv").exists()
    True


3. Atom selection
-----------------

Keyword filters keep rows whose column matches any of the given values and need
no extra dependency:

.. code-block:: python

    >>> chain_a = protein.select(chain=["A"])
    >>> bool((chain_a["chain"] == "A").all())
    True
    >>> ca = protein.select(name=["CA"])
    >>> bool((ca["name"] == "CA").all())
    True

A non-empty selection *string* uses the full VMD-style grammar via the optional
`molselect <https://github.com/cabb99/molselect>`_ package
(``pip install "molscene[selection]"``):

.. code-block:: python

    pocket = protein.select("protein and within 5 of resname HOH")
    backbone = protein.select("name CA C N O and not hydrogen")


4. Geometry with operators
---------------------------

Arithmetic operators act atom-wise on coordinates and preserve metadata:

.. code-block:: python

    >>> import numpy as np
    >>> s = Scene(np.array([[0., 0, 0], [2, 0, 0], [0, 2, 0]]))
    >>> s.get_center().to_numpy().round(3).tolist()
    [0.667, 0.667, 0.0]

    >>> centered = s - s.get_center()                 # move centroid to origin
    >>> bool(np.allclose(centered.get_center().to_numpy(), 0))
    True
    >>> (s + np.array([1, 2, 3])).get_coordinates().iloc[0].to_numpy().tolist()
    [1.0, 2.0, 3.0]
    >>> (s * 2.0).get_coordinates().iloc[1].to_numpy().tolist()
    [4.0, 0.0, 0.0]

    >>> len(s + s)        # Scene + Scene concatenates atoms
    6
    >>> len(s - s)        # Scene - Scene removes shared atoms
    0

Rotations apply a 3×3 matrix (``rotate`` and ``dot`` are equivalent):

.. code-block:: python

    >>> Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float)
    >>> bool(np.allclose(s.rotate(Rz).get_coordinates().to_numpy(),
    ...                   s.dot(Rz).get_coordinates().to_numpy()))
    True


5. Transformations
------------------

A :class:`~molscene.Transformation` is a rigid-body motion you can apply,
invert, and compose:

.. code-block:: python

    >>> from molscene import Transformation
    >>> T = Transformation.from_matrix(np.eye(3), [5, 0, 0])
    >>> T.translation.tolist()
    [5.0, 0.0, 0.0]
    >>> moved = s.transform(T)
    >>> bool(np.allclose(moved.get_coordinates().to_numpy(),
    ...                  s.get_coordinates().to_numpy() + [5, 0, 0]))
    True
    >>> back = moved.transform(T.inverse())            # round-trip
    >>> bool(np.allclose(back.get_coordinates().to_numpy(),
    ...                  s.get_coordinates().to_numpy()))
    True


6. Structural alignment
-----------------------

Pair atoms with a :class:`~molscene.Matching` strategy, then superpose or
measure RMSD. Here a displaced copy is aligned back onto the original:

.. code-block:: python

    >>> mobile = s.transform(Transformation.from_matrix(np.eye(3), [10, 0, 0]))
    >>> aligned = mobile.superpose(s)                  # default: OrderMatching
    >>> round(float(aligned.rmsd(s)), 6)
    0.0
    >>> T = mobile.compute_transformation(s)           # the fitted motion
    >>> round(float(T.rmsd), 6)
    0.0

    >>> from molscene import OrderMatching
    >>> a, b = OrderMatching().pair(mobile, s)
    >>> len(a) == len(b) == len(s)
    True


7. Morphing a segment
---------------------

:meth:`~molscene.Scene.morph_segment` rigidly repositions each residue in a
range by a transformation interpolated between two anchors:

.. code-block:: python

    >>> seg = Scene(np.array([[0., 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0]]))
    >>> seg["resid"] = [1, 1, 2, 2]
    >>> t0 = Transformation.identity()
    >>> t1 = Transformation.from_matrix(np.eye(3), [0, 0, 5])
    >>> morphed = seg.morph_segment("A", [1, 2], t0, t1)
    >>> len(morphed) == len(seg)
    True
    >>> # residue 1 sits at alpha=0 (identity anchor) and is unchanged
    >>> bool(np.allclose(morphed.select(resid=[1]).get_coordinates().to_numpy(),
    ...                  seg.select(resid=[1]).get_coordinates().to_numpy()))
    True


8. Multi-frame coordinates
--------------------------

A single scene can carry a stack of coordinate frames:

.. code-block:: python

    >>> frames = np.stack([s.get_coordinates().to_numpy() + d for d in range(3)])
    >>> frames.shape
    (3, 3, 3)
    >>> s.set_coordinate_frames(frames)
    >>> s.n_frames
    3
    >>> [len(frame) for frame in s.iterframes()]
    [3, 3, 3]


9. Distance maps
----------------

.. code-block:: python

    >>> pts = Scene(np.array([[0., 0, 0], [3, 0, 0], [0, 4, 0]]))
    >>> D = pts.distance_map()
    >>> D.shape
    (3, 3)
    >>> bool(np.allclose(np.diag(D), 0))
    True
    >>> round(float(D[0, 1]), 1), round(float(D[0, 2]), 1)
    (3.0, 4.0)
    >>> pairs, dists = pts.distance_map_sparse(3.5)
    >>> pairs.shape[1] == 2 and len(pairs) == len(dists)
    True


10. Metadata travels with the data
-----------------------------------

Scene-level metadata is stored on the scene and inherited by sub-scenes:

.. code-block:: python

    >>> m = Scene(np.array([[0., 0, 0], [1, 0, 0]]))
    >>> m.author = "CB"
    >>> m.note = {"pH": 7.4}
    >>> m.author
    'CB'
    >>> m.select(chain=["A"]).author        # preserved through selection
    'CB'


Tips & tricks
-------------

* **Per-atom metadata** is just a new column:

  .. code-block:: python

      >>> m["charge"] = [0.1, -0.1]
      >>> m["charge"].tolist()
      [0.1, -0.1]

* **Sequence and mass** helpers:

  .. code-block:: python

      >>> seq = protein.get_sequence()        # dict of one-letter sequences per chain
      >>> "A" in seq
      True
      >>> with_mass = protein.compute_mass()  # copy with a 'mass' column
      >>> float(with_mass["mass"].sum()) > 0
      True

* **Secondary structure** (:meth:`~molscene.Scene.compute_secondary_structure`)
  requires the external ``mkdssp`` executable on your ``PATH``.

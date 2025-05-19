.. _scene-arithmetic-operators:

Scene Arithmetic & Operators
============================

The ``Scene`` class overloads standard Python arithmetic and NumPy ufuncs to provide intuitive, atom-wise transformations while preserving metadata and (optionally) multi-frame trajectories. Below is a summary of how each operator works, what inputs are allowed, and what errors are raised.

1. Addition (``+`` / ``__add__`` / ``__radd__``)
-------------------------------------------------

.. code-block:: python

   # Translate by a 3-vector (scalar, sequence of length 3, or pandas.Series index ['x','y','z']):
   translated = scene + 1.5              # adds (1.5,1.5,1.5) to every atom
   translated = scene + [1,2,3]          # adds (1,2,3)
   translated = pd.Series([1,2,3],index=['x','y','z'])
   translated = scene + that_series

   # Reflected addition via right-add:
   translated = [1,2,3] + scene

* **Multi-frame** scenes: All frames are translated identically.
* **Metadata** is always preserved (same ``.author``, ``.note``, etc.).
* **Error** if you attempt ``scene + scene`` through the “vector” path (caught by ``_as_delta``); instead see “concatenation” below.

Concatenation: Scene + Scene
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   combined = scene1 + scene2

* **Semantics**: concatenates the two atom lists end-to-end (chain IDs remapped if needed).
* **Coordinates**: stacked along the atom axis.
* **Metadata**: inherited from ``scene1`` (with no automatic merge of differing keys).
* **Multi-frame**: concatenates every frame in lockstep.

2. Subtraction (``-`` / ``__sub__`` / ``__rsub__``)
---------------------------------------------------

.. code-block:: python

   # Translate by –delta:
   shifted  = scene - 2.0
   reverse  = 2.0 - scene

   # Remove shared atoms:
   pruned   = scene1 - scene2

* **Vector subtraction**: like addition, but subtracting a 3-vector.
* **Scene–Scene subtraction**: removes any atom in ``scene1`` whose ``atom_index`` appears in ``scene2``.

  * If they share exactly the same atoms, ``scene – scene`` yields an **empty** Scene of length 0.
* **Multi-frame**: all frames are shifted or pruned identically.
* **Metadata**: always preserved.

3. Multiplication (``*`` / ``__mul__`` / ``__rmul__``)
------------------------------------------------------

.. code-block:: python

   scaled = scene * 2.5
   scaled = 2.5 * scene
   scaled = scene * (1,2,3)           # per-axis scaling
   scaled = scene * pd.Series(...)    # indexed ['x','y','z']

* **Semantics**: scales each coordinate by the given factor per axis.
* **Error** if you attempt ``scene * scene`` (or any non-vector input larger than length 3)—raises ``ValueError``.
* **Multi-frame**: scales every frame.
* **Metadata**: preserved.

4. True Division (``/`` / ``__truediv__``)
------------------------------------------

.. code-block:: python

   shrunk = scene / 2.0

* **Semantics**: divides each coordinate by the given per-axis divisor.
* **Error** if you attempt ``scene / scene``—raises ``ValueError``.
* **Multi-frame**: divides every frame.
* **Metadata**: preserved.

5. Negation (``-scene`` / ``__neg__``)
--------------------------------------

.. code-block:: python

   inverted = -scene

* **Semantics**: reflects all coordinates through the origin.
* **Multi-frame**: negates every frame.
* **Metadata**: preserved.

6. Combined with NumPy Ufuncs
-----------------------------

All of the above operators are also routed through ``Scene.__array_ufunc__``, so that expressions like:

.. code-block:: python

   np.add(scene, [1,2,3])
   np.subtract([1,2,3], scene)
   np.multiply(scene, 2.0)
   np.negative(scene)

behave identically to their Python-operator counterparts.

7. Error Handling
-----------------

* **Vector ops** (``+v``, ``-v``, ``*v``, ``/v``) expect ``v`` to be:

  * A scalar (``int`` or ``float``),
  * A length-3 sequence (``list``, ``tuple``, 1D ``ndarray``), or
  * A ``pandas.Series`` with **exactly** the index ``['x','y','z']``.
* **Invalid vector** inputs (wrong length or wrong Series index) raise:

  .. code-block:: python

     ValueError: Cannot interpret {other!r} as a 3-vector

* **Scene-vs-Scene**:

  * **Supported**: ``scene + scene`` (concatenate), ``scene - scene`` (prune).
  * **Unsupported**: all other arithmetic combinations (``scene * scene``, ``scene / scene``, etc.) raise:

    .. code-block:: python

       ValueError: Cannot interpret {scene!r} as a 3-vector

* **Coordinate-frame mismatch**: if you set multi-frame data of the wrong shape, you’ll see:

  .. code-block:: python

     ValueError: frames must be a 3D numpy array with shape (n_frames, n_atoms, 3)

8. Examples
-----------

.. code-block:: python

   import numpy as np
   from molscene import Scene

   coords = np.array([[0,0,0],[1,1,1]])
   s1 = Scene(coords)
   s2 = Scene(coords + 5)

   # Concatenate atoms
   s3 = s1 + s2
   len(s3)              # 4 atoms

   # Remove overlaps
   empty = s1 - s1
   len(empty)           # 0

   # Translate & scale
   t = s1 + [1,2,3]
   u = 2 * s1
   v = s1 / np.array([2,2,2])

   # Reflect
   w = -s1

This design gives you a powerful, DataFrame-based “molecular scene” that you can shift, stack, prune, and scale using familiar arithmetic, all while carrying along your metadata and trajectory frames seamlessly.

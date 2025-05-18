MolScene Usage Examples
============================

.. contents:: Table of Contents
   :local:
   :depth: 2


1. Creating and Inspecting a Scene
----------------------------------

A 'Scene' can be created from a numpy array or a pandas DataFrame. 
The array should be of shape (N,3) for N atoms, and the DataFrame should have columns 'x', 'y', and 'z' for coordinates.

.. code-block:: python

    >>> import numpy as np
    >>> from molscene import Scene
    >>> coords = np.array([[0,0,0],[1,0,0],[0,1,0]])
    >>> scene = Scene(coords)
    >>> len(scene)
    3
    >>> scene.get_coordinates().shape
    (3, 3)
    >>> scene
    <Scene (3)>
       x  y  z chainID  ...  resName chain_index res_index  atom_index
    0  0  0  0       A  ...                    0         0           0
    1  1  0  0       A  ...                    0         0           1
    2  0  1  0       A  ...                    0         0           2
    <BLANKLINE>
    [3 rows x 16 columns]

.. code-block:: python

    >>> import pandas as pd
    >>> from molscene import Scene
    >>> df = pd.DataFrame([[0,0,0],[1,0,0],[0,1,0]], columns=['x','y','z'])
    >>> scene2 = Scene(df)
    >>> len(scene)
    3
    >>> scene.get_coordinates().shape
    (3, 3)
    >>> scene
        <Scene (3)>
       x  y  z chainID  ...  resName chain_index res_index  atom_index
    0  0  0  0       A  ...                    0         0           0
    1  1  0  0       A  ...                    0         0           1
    2  0  1  0       A  ...                    0         0           2
    <BLANKLINE>
    [3 rows x 16 columns]

2. File I/O
-----------

Reading
~~~~~~~
You can read PDB, mmCIF, and GRO files. The ``Scene.from_file()`` method will automatically detect the file type.

.. code-block:: python

    >>> from molscene import Scene
    >>> scene_pdb  = Scene.from_pdb('1abc.pdb')  # doctest: +SKIP
    >>> scene_cif  = Scene.from_cif('1abc.cif')  # doctest: +SKIP
    >>> scene_fix  = Scene.from_fixPDB(pdbfile='1abc.pdb')  # doctest: +SKIP
    >>> scene_auto = Scene.from_file('1abc.pdb')  # doctest: +SKIP

Writing
~~~~~~~
You can write to PDB, mmCIF, and GRO files. The ``Scene.to_file()`` method will automatically detect the file type.

.. code-block:: python

    >>> scene_pdb.to_file('out.pdb')   # doctest: +SKIP
    >>> scene_cif.to_file('out.cif')   # doctest: +SKIP
    # .gro once available

Also, you can use the ``.to_csv()`` method to write to a CSV file. This is useful for exporting data in a tabular format that can be easily read by other programs or libraries.

.. code-block:: python

    >>> scene.to_csv('atoms.csv', index=False)  # doctest: +SKIP


3. Metadata Handling
--------------------

Every ``Scene`` has a private ``_meta`` dict you can read or write via attributes:

.. code-block:: python

    >>> from molscene import Scene
    >>> scene = Scene(np.array([[0,0,0],[1,0,0],[0,1,0]]))
    >>> scene._meta
    {}
    >>> scene.author = "CB"
    >>> scene.description = "Test peptide"
    >>> scene.author
    'CB'
    >>> scene._meta
    {'author': 'CB', 'description': 'Test peptide'}
    >>> sub = scene.select(chainID=['A'])
    >>> sub.author
    'CB'

The metadata in MolScene is intrinsically linked to each Scene object. 
When you create a sub-scene through selection or filtering, the associated metadata is automatically inherited by the resulting sub-scene.
The metadata dictionary can store a wide range of objects, including other DataFrames, dictionaries, or any custom Python objects you wish to associate with the scene.
If your DataFrame includes columns that are indexed to the original DataFrame, this metadata will be preserved during selection or filtering operations.
To ensure that a column is recognized as metadata, its name should begin with the prefix index_. 
This convention is particularly useful for storing information such as bonds, angles, or other properties that may be associated with multiple atoms within the scene.

.. code-block:: python

    ## TO BE IMPLEMENTED


4. Merging and Splitting Scenes
-------------------------------

Concatenation
~~~~~~~~~~~~~

Use the ``+`` operator (or ``.concatenate``) to stitch two scenes end-to-end:

.. code-block:: python

    >>> import numpy as np
    >>> from molscene import Scene
    >>> scene1 = Scene(np.random.rand(5,3))
    >>> scene2 = Scene(np.random.rand(3,3)) * 2.0
    >>> merged = scene1 + scene2
    >>> len(merged)
    8

Under the hood, ``scene1 + scene2`` does a pandas-concat of the rows.

Splitting by Chain (or any column)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

There’s no built-in ``.split_chains()``, but it’s easy:

.. code-block:: python

    >>> chains = merged['chainID'].unique()  # doctest: +SKIP
    >>> by_chain = {c: merged.select(chainID=[c]) for c in chains}  # doctest: +SKIP
    >>> scenes = [merged.select(chainID=[c]) for c in chains]  # doctest: +SKIP

Each sub-scene is a full ``Scene`` you can write out or transform independently.


5. Geometric Transforms & Operators
-----------------------------------

Translation
~~~~~~~~~~~

The ``+`` and ``-`` operators also work as vector translations:

.. code-block:: python

    >>> moved = scene + np.array([1,2,3])
    >>> moved2 = scene.translate([1,2,3])

Centering
~~~~~~~~~

There are multiple ways to center a Scene so that its centroid is at the origin:

.. doctest::

    >>> center = scene.get_center()
    >>> centered = scene - center
    >>> # or, equivalently
    >>> centered = scene.center()

All of these approaches will shift the coordinates so that the centroid is at (0, 0, 0).

Scaling
~~~~~~~

Multiply by a scalar (or 3-vector) to scale:

.. code-block:: python

    >>> big = scene * 10.0
    >>> squished = scene * np.array([1,1,0.5])

Rotation
~~~~~~~~

Use ``.rotate()`` or ``.dot()`` with a 3×3 rotation matrix:

.. code-block:: python

    >>> import numpy as np
    >>> theta = np.pi/2
    >>> Rz = np.array([[np.cos(theta), -np.sin(theta), 0],
    ...                [np.sin(theta),  np.cos(theta), 0],
    ...                [0,              0,             1]])
    >>> rotated = scene.rotate(Rz)
    >>> rot2 = scene.dot(Rz)

Center & Align Two Scenes
~~~~~~~~~~~~~~~~~~~~~~~~~

Put ``scene2`` on top of ``scene1`` by matching centroids:

.. code-block:: python

    >>> c1 = scene1.get_coordinates().mean().to_numpy()
    >>> c2 = scene2.get_coordinates().mean().to_numpy()
    >>> aligned = scene2 - c2 + c1


6. Distance Maps
----------------

.. code-block:: python

    >>> D = scene.distance_map()
    >>> pairs, dists = scene.distance_map_sparse(threshold=5.0)


7. Putting It All Together: Example Workflow
--------------------------------------------

.. code-block:: python

    >>> from molscene import Scene
    >>> ligand = Scene(np.array([[0,0,0],[1,0,0],[0,1,0]]))
    >>> protein = Scene(np.array([[0,0,0],[1,0,0],[0,1,0]]))
    >>> ligand_c = ligand - ligand.get_coordinates().mean().to_numpy()
    >>> protein_c = protein - protein.get_coordinates().mean().to_numpy()
    >>> offset = np.array([0,0,10])
    >>> ligand_pos = ligand_c + (protein_c.get_coordinates().mean().to_numpy() + offset)
    >>> system = protein_c + ligand_pos
    >>> system.author = "Carol B"
    >>> system.pH = 7.4


Tips & Tricks
-------------

* **Per-atom metadata** (e.g. custom charges or flags) can simply be new columns:

  .. code-block:: python

      >>> scene['charge'] = np.array([0.1, 0.2, 0.3])

* **Frame movies**: call ``scene.set_coordinate_frames(frames)``, then iterate:

  .. code-block:: python

      >>> for frame in scene.iterframes():  # doctest: +SKIP
      ...     do_something(frame)

* **Splitting by selection**: any keyword to ``.select()``—e.g. ``scene.select(resSeq=[10,20,30])``.

* **Combining transforms**:

  .. code-block:: python

      >>> new = (scene - center).rotate(Rz) * 2.0 + np.array([1,1,1])

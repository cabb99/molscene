Getting Started
===============

MolScene represents molecular structures as :class:`pandas.DataFrame` objects.
A :class:`~molscene.Scene` *is* a DataFrame, with one row per atom and columns
such as ``chain``, ``resid``, ``resname``, ``name`` and the Cartesian
coordinates ``x``, ``y``, ``z`` — plus a thin layer of structure-aware methods
on top.

Installation
------------

Install the latest release from PyPI:

.. code-block:: bash

    pip install molscene

String atom selection (``Scene.select("chain A and resid 1 to 100")``) is
powered by the optional `molselect <https://github.com/cabb99/molselect>`_
package. Enable it with the ``selection`` extra:

.. code-block:: bash

    pip install "molscene[selection]"

The keyword form ``Scene.select(chain=["A"])`` works without that extra.

Other optional extras:

* ``molscene[test]`` — the test suite (pytest, pytest-cov)
* ``molscene[docs]`` — building this documentation

First steps
-----------

Load a structure. The format is taken from the file extension by
:meth:`~molscene.Scene.from_file`, or you can call the explicit readers
:meth:`~molscene.Scene.from_pdb`, :meth:`~molscene.Scene.from_cif`, or
:meth:`~molscene.Scene.from_gro`:

.. code-block:: python

    from molscene import Scene

    scene = Scene.from_pdb("1jge.pdb")

Because a ``Scene`` is a DataFrame, you can inspect it with the pandas API:

.. code-block:: python

    len(scene)                      # number of atoms
    scene.columns                   # available per-atom columns
    scene["resname"].value_counts() # residue composition
    scene.head()

Select atoms. A non-empty selection string is evaluated by ``molselect``;
keyword arguments filter by column equality and need no extra dependency:

.. code-block:: python

    backbone = scene.select("name CA C N O")     # VMD-style string
    chain_a  = scene.select(chain=["A"])         # kwargs (always available)
    pocket   = scene.select("within 5 of resname HEM")

Transform coordinates with operators or methods:

.. code-block:: python

    import numpy as np

    centered = scene - scene.get_center()   # move centroid to the origin
    shifted  = scene + np.array([10, 0, 0]) # translate
    scaled   = scene * 2.0                  # scale about the origin

Write the result back out. The format again follows the extension:

.. code-block:: python

    centered.to_file("centered.pdb")   # or .write_pdb / .write_cif / .write_gro

Where to next
-------------

* The :doc:`user_guide` covers every feature in depth and contains the full
  API reference.
* :doc:`examples` is a collection of short, copy-paste recipes.
* The :doc:`developer_guide` explains the design decisions behind MolScene.

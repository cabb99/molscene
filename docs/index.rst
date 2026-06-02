.. molscene documentation master file, created by
   sphinx-quickstart on Thu Mar 15 13:55:56 2018.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to MolScene's documentation!
=========================================================

**Molecular structures as pandas DataFrames.** MolScene loads, manipulates, and
writes molecular structure files (PDB, mmCIF, GRO) as
:class:`pandas.DataFrame` objects, adding atom selection, rigid-body
transformations and alignment, residue morphing, multi-frame trajectories, and
distance maps on top of a familiar tabular interface.

.. grid:: 1 1 3 3

    .. grid-item-card:: Getting Started
      :margin: 0 3 0 0

      Install MolScene and learn the basics.

      .. button-link:: ./getting_started.html
         :color: primary
         :outline:
         :expand:

         To the Getting Started Guide



    .. grid-item-card::  User Guide
      :margin: 0 3 0 0

      Every feature in depth, with the full API reference.

      .. button-link:: ./user_guide.html
         :color: primary
         :outline:
         :expand:

         To the User Guide



    .. grid-item-card::  Developer Guide
      :margin: 0 3 0 0

      Design decisions and internals.

      .. button-link:: ./developer_guide.html
         :color: primary
         :outline:
         :expand:

         To the Developer Guide


.. toctree::
   :maxdepth: 2
   :hidden:
   :titlesonly:

   getting_started
   user_guide
   examples
   developer_guide


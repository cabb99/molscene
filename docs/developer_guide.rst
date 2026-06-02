Developer Guide
===============

This guide records the design decisions behind MolScene — the *why* behind the
code. For per-operator semantics see :ref:`scene-arithmetic-operators`; for the
forward-looking feature plan see :doc:`developer_roadmap`.

.. contents:: On this page
   :local:
   :depth: 2

A Scene *is* a DataFrame
------------------------

:class:`~molscene.Scene` subclasses :class:`pandas.DataFrame` rather than
wrapping one. The goal is that structural data feels like tabular data: users
already know how to filter, group, join, and export DataFrames, and MolScene
should not make them relearn any of that. Subclassing keeps the entire pandas
surface available and means a selection or slice of a ``Scene`` is still a
``Scene``.

Two consequences drive much of the implementation:

* **Custom constructor.** pandas frequently builds new frames internally (during
  slicing, arithmetic, concatenation). ``Scene`` overrides the constructor hooks
  so those internally-produced frames come back as ``Scene`` objects with their
  metadata intact, instead of plain ``DataFrame``.
* **Metadata must ride along.** Because operations constantly create new frames,
  scene-level metadata is stored in a private ``_meta`` dict and re-attached to
  derived scenes. Attribute access (``scene.author = ...``) reads and writes
  this dict.

Metadata model
--------------

Scene-level metadata (``author``, ``pH``, a force field, even another
DataFrame) lives in ``_meta`` and is inherited by any sub-scene produced through
selection, filtering, or a transformation.

Per-atom metadata is *just columns* — add ``scene["charge"] = ...`` and it
filters and slices like everything else. For properties indexed to atoms but not
one-per-row (bonds, angles, pairs), columns whose names begin with ``index_``
are recognized as index-aligned metadata and kept consistent through selection.

Coordinate frames
------------------

A ``Scene`` stores a single set of ``x, y, z`` columns but can additionally hold
a stack of coordinate frames — shape ``(n_frames, n_atoms, 3)`` — for
trajectories and morph movies. Frames are validated on assignment and every
coordinate operator (translate, scale, reflect, transform) is applied to all
frames in lockstep so a multi-frame scene stays internally consistent.

Geometry and transformations
-----------------------------

Low-level math lives in :mod:`molscene.geometry` (Kabsch superposition,
quaternion and dual-quaternion conversions, screw interpolation) and is kept
free of any ``Scene`` knowledge so it is easy to test and reuse.

:class:`~molscene.Transformation` is a first-class rigid-body motion built on
top of those helpers, following :class:`scipy.spatial.transform.Rotation`
conventions (so ``T1 @ T2`` applies ``T2`` first). It can be created from a
matrix, quaternion, dual quaternion, or a Kabsch fit, and supports two
interpolation modes:

* ``slerp`` — slerp the rotation and lerp the translation independently
  (the textbook decoupled interpolation).
* ``sclerp`` — dual-quaternion screw-linear interpolation, treating the motion
  as a single screw (Chasles' theorem) so rotation and translation co-evolve
  along one helical path. This is the default because it produces more
  physically plausible morphs when a twist must propagate through a flexible
  region.

Matching is a strategy
----------------------

Aligning two structures requires deciding which atoms correspond. That decision
is deliberately separated from the alignment math via the
:class:`~molscene.Matching` strategy interface, so
:meth:`~molscene.Scene.superpose`, :meth:`~molscene.Scene.rmsd`, and
:meth:`~molscene.Scene.compute_transformation` all share one pairing mechanism.
The shipped strategies (:class:`~molscene.OrderMatching`,
:class:`~molscene.ColumnMatching`, :class:`~molscene.SequenceMatching`) cover
the common cases, and any callable ``(mobile, reference) -> (Scene, Scene)`` can
be plugged in through :func:`~molscene.as_matching`.

File parsing
------------

PDB parsing uses fixed-column slicing per the format spec. mmCIF parsing uses a
small regular-expression tokenizer that reads a single ``loop_`` category
(e.g. ``_atom_site``) straight into a DataFrame, which keeps the dependency
surface minimal and the hot path simple.

.. note::

   An experimental Cython mmCIF tokenizer (``molscene/pdbxparser``) was
   prototyped to speed up very large files. It was removed from the shipped
   package because it was never wired into ``Scene`` and added build
   complexity (a C compiler and Cython at build time). The pure-Python
   tokenizer is the supported path; the prototype is preserved only in version
   control history.

The optional selection dependency
----------------------------------

String atom selection is delegated to the separate
`molselect <https://github.com/cabb99/molselect>`_ package, which owns the
selection grammar and parser. Keeping it a separate, *optional* dependency
(``molscene[selection]``) means MolScene installs and imports with only
pandas/numpy/scipy, and the keyword form of
:meth:`~molscene.Scene.select` always works. The import is lazy and raises a
clear, actionable error if a user asks for a string selection without
``molselect`` installed.

Testing
-------

Tests live in ``molscene/tests`` and run with ``pytest``. They write only to
pytest's ``tmp_path`` by default (set ``MOLSCENE_TEST_SCRATCH=1`` to redirect
output to ``molscene/tests/scratch`` for inspection). Tests that exercise string
selection are skipped automatically when ``molselect`` is not installed, so the
suite is green with or without the optional dependency.

.. toctree::
   :hidden:

   developer_notes_operators
   developer_roadmap

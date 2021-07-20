
.. currentmodule:: bcn

Changelog
=========

Version related info
--------------------

Given that this is my thesis project and I expect no one to work alongside it with me, the versioning scheme is pretty casual.

There are two main ways to query version information about this project, `bcn.__version__` and `bcn.version_info`.

.. data:: version_info

   A `namedtuple` of the form ``major.minor.build``, where:

   - ``major`` is a major release, representing either many new features or a significant refactor to the underlying code.

   - ``minor`` is a minor release, representing some new features on the given major release and hopefully minimal breaking changes.

   - ``build`` is a counter that loosely represents each build, revision, commit, docs build, or even file save for the given minor release.

.. data:: __version__

   A string representation of the version. e.g. ``"1.1.23"``.

.. _whats_new:

v0.4
----

New features
~~~~~~~~~~~~

- Add :ref:`bcn.branches.informed <optics_informed_branches>` module
- Add `Fault` class
- Add `BCN.evaluate`
- Add `bcn.branches.Branches.name` property
- Add `bcn.branches.Branches.latex` property

v0.3.38 (Jun 29, 2021)
----------------------

New features
~~~~~~~~~~~~

- Add this changelog.
- Add `bcn.BCN.default_weights_filename` and `bcn.BCN.default_results_filename`.
- Add a ``from_path`` parameter to `bcn.BCN.train` that allows the user to specify a results directory path, from which the model should look for and load its weights and results, instead of explicitly specifying the two file paths via ``from_weights`` and ``from_results``.
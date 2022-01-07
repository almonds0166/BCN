
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

   - ``build`` is a counter that loosely corresponds to the amount of times I saved any file in the project for the given minor release.

.. data:: __version__

   A string representation of the version. e.g. ``"1.2.345"``.

.. _whats_new:

v1.2.1180 (Jan 07, 2022)
------------------------

- Submitted thesis!
- Update docs to reflect thesis
- GUI works without crashing

v1.1.166 (Sep 30, 2021)
-----------------------

New features
~~~~~~~~~~~~

- Add ``limit`` parameter to `BCN.evaluate`. `BCN.run_wp` limits the training set evaluation to 10,000 samples instead of 60,000, in the interest of training times.
- Add ``webhook`` parameter to `BCN.run_wp`.
- Add `Results.wp_layers` to keep track of which layers were perturbed.
- Add ``padding``, ``width``, and ``depth`` kwargs to the `~bcn.Fault` class.
- Add ``fault`` argument to `BCN.run_epoch` and `BCN.run_epochs`.
- Add ``activation`` argument to `BCN` and `BCNLayer`.
- Add `BCN.confusion` for computing interclass confusion matrices.

Bug fixes
~~~~~~~~~

- `BCN.train` now appropriately adds version & device info to `BCN.results`.

v1.0.257 (Aug 21, 2021)
-----------------------

Breaking changes
~~~~~~~~~~~~~~~~

- Change `Results` class
   - When loading from a file, *update* ``Results.__dict__`` instead of entirely *replacing* it, in the interest of backward- and forward-compatibility.
   - Remove ``Results.best_epoch``.
   - Remove ``Results.best_valid_loss``.
   - Remove ``Results.version``.
   - Add `~Results.best`, representing the index of the *maximum F-1 score* (instead of minimum validation loss).
   - Add `~Results.versions`, a set containing all the versions that the model was trained under.
   - Add `~Results.devices`, a set containing all the devices the model was trained on, to help put training times into perspective.
   - Add `~Results.step`, the number of weight perturbation steps
- Change BCN network matrices
   - Harness the power of 4D batch tensors and vectorization for (marginally?) improved training times!
   - Encode the used device in the local filename to prevent device-related errors.

Bug fixes
~~~~~~~~~

- Fix local networks filenames weren't specific (e.g. it now writes ``uniform.NextToNN`` instead of just ``NextToNN``).

New features
~~~~~~~~~~~~

- Add `BCN.run_wp` method to run weight perturbation!
- Add `BCN.clone` method to duplicate a model.
- Add `BCN.evaluate` to evaluate models without training them.
- Add `WPApproach` enum.
- Add `Connections.ONE_TO_1`.
- Add missing `BCN.dropout`.

Migrating
~~~~~~~~~

See the :ref:`migration page <migration_1>` for more detailed info.

- Use `bcn.v0.migrate_results` to convert a v0 Results file to a v1 Results file.
- Use `bcn.v0.migrate_weights` to convert a v0 weights file to a v1 weights file.
- Delete the contents of the local ./networks/ folder

v0.4.98 (Jul 20, 2021)
----------------------

New features
~~~~~~~~~~~~

- Add :ref:`bcn.branches.informed <optics_informed_branches>` module.
- Add `Fault` class.
- Add `BCN.evaluate`.
- Add `bcn.branches.Branches.name` property.
- Add `bcn.branches.Branches.latex` property.

v0.3.38 (Jun 29, 2021)
----------------------

New features
~~~~~~~~~~~~

- Add this changelog.
- Add `bcn.BCN.default_weights_filename` and `bcn.BCN.default_results_filename`.
- Add a ``from_path`` parameter to `bcn.BCN.train` that allows the user to specify a results directory path, from which the model should look for and load its weights and results, instead of explicitly specifying the two file paths via ``from_weights`` and ``from_results``.
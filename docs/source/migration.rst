
.. currentmodule:: bcn

Migrating to newer versions
===========================

.. _migration_1:

Migrating from v0 to v1
-----------------------

Networks are modeled differently in v1, so the contents of the local ``./networks/`` folder should be scrapped.

To convert a v0 Results file to a v1 Results file, use `bcn.v0.migrate_results`. Likewise, use `bcn.v0.migrate_weights` to convert a v0 weights file to a v1 weights file.

v1 comes with a compatbility submodule accessible via ::

   >>> import bcn.v0

Below are the members of this submodule.

.. attribute:: bcn.v0.Results

   A v0 copy of the `bcn.Results` class.

.. autofunction:: bcn.v0.migrate_results

.. autofunction:: bcn.v0.migrate_weights
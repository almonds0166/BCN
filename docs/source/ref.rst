
.. currentmodule:: bcn

BCN reference
=============

Models
------

BCN
~~~

.. autoclass:: BCN
   :members:

BCNLayer
~~~~~~~~

.. autoclass:: BCNLayer
   :members:

Classes related to training
---------------------------

Dataset
~~~~~~~

.. autoclass:: Dataset

   .. attribute:: Dataset.MNIST

      Represents the MNIST dataset of handwritten digits.

   .. attribute:: Dataset.FASHION

      Represents the `Fashion-MNIST dataset`_ of fashion articles.

      .. _`Fashion-MNIST dataset`: https://github.com/zalandoresearch/fashion-mnist

Fault
~~~~~

.. autoclass:: Fault
   :members:

Results
~~~~~~~

.. autoclass:: Results
   :members:

TrainingScheme
~~~~~~~~~~~~~~

.. autoclass:: TrainingScheme
   :members:

Connections
-----------

.. autoclass:: Connections

   .. attribute:: Connections.ONE_TO_9

      Represents a connection scheme in which each neuron in the input plane connects to at most the nine nearest neighbors in the output plane.

   .. attribute:: Connections.ONE_TO_25

      Represents a connection scheme in which each neuron in the input plane connects to at most the 25 nearest neighbors in the output plane.

   .. attribute:: Connections.ONE_TO_49

      Represents a connection scheme in which each neuron in the input plane connects to at most the 49 nearest neighbors in the output plane.

   .. attribute:: Connections.ONE_TO_81

      Represents a connection scheme in which each neuron in the input plane connects to at most the 81 nearest neighbors in the output plane.

   .. attribute:: Connections.FULLY_CONNECTED

      Represents a connection scheme in which each neuron in the input plane connects to *all* neurons in the output plane.

      .. warning::

         Use with caution.

         One of the tradeoffs I've made with how I've designed the BCN connections is intuitiveness & convenience for file sizes. Specifically, the more connected a model, the larger the connections take up in memory.

         More specifically, if a 30x30 1-to-9 connection scheme is 29 MB, then a 30x30 1-to-81 connection scheme would be 261 MB, and a fully connected 30x30 network would be ... 2.9 GB???

Branches
--------

Base class
~~~~~~~~~~

.. autoclass:: bcn.branches.Branches
   :members:

Direct connections only
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bcn.branches.DirectOnly

Empirically based branches
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bcn.branches.Vital

"Simple" branches
~~~~~~~~~~~~~~~~~

These branches have equal power in the center, first ring, and second ring, where applicable.

.. autoclass:: bcn.branches.simple.NearestNeighbor
.. autoclass:: bcn.branches.simple.NextToNN
.. autoclass:: bcn.branches.simple.NearestNeighborOnly
.. autoclass:: bcn.branches.simple.NextToNNOnly
.. autoclass:: bcn.branches.simple.IndirectOnly

Uniform branches
~~~~~~~~~~~~~~~~

These branches have equal non-zero values.

.. autoclass:: bcn.branches.uniform.NearestNeighbor
.. autoclass:: bcn.branches.uniform.NextToNN
.. autoclass:: bcn.branches.uniform.NearestNeighborOnly
.. autoclass:: bcn.branches.uniform.NextToNNOnly
.. autoclass:: bcn.branches.uniform.IndirectOnly

.. _optics_informed_branches:

Optics-informed branches
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bcn.branches.informed.Kappa
.. autoclass:: bcn.branches.informed.IndirectOnly

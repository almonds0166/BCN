
.. currentmodule:: bcn

BCN reference
=============

.. _models:

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

.. _dataset:

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

.. _connections:

WPApproach
~~~~~~~~~~

.. autoclass:: WPApproach

   .. attribute:: WPApproach.RASTER

      Represents perturbing the layers in order from input to output, then repeating, suggestive of raster data structures.

   .. attribute:: WPApproach.COCKTAIL

      Represents perturbing the layers  first from input to output, then from output to input, then repeating, suggestive of cocktail shakers.

   .. attribute:: WPApproach.RANDOM

      Represents choosing the next layer to perturb randomly.

Connections
-----------

.. autoclass:: Connections

   .. attribute:: Connections.ONE_TO_1

      Represents a connection scheme in which each neuron in the input plane directly connects only to the nearest neuron in the next layer.

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

.. _branches:

Branches
--------

Base class
~~~~~~~~~~

.. autoclass:: bcn.branches.Branches
   :members:

Direct connections only
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bcn.branches.DirectOnly

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & 1.00 & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

Empirically based branches
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bcn.branches.Vital

.. _uniform:

Uniform branches
~~~~~~~~~~~~~~~~

These branches have equal non-zero values.

.. autoclass:: bcn.branches.uniform.NearestNeighbor

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & 0.11 & 0.11 & 0.11 & \> & \> & \>\\
      \> & \> & \> & 0.11 & 0.11 & 0.11 & \> & \> & \>\\
      \> & \> & \> & 0.11 & 0.11 & 0.11 & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.uniform.NextToNN

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.uniform.NearestNeighborOnly

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & 0.12 & 0.12 & 0.12 & \> & \> & \>\\
      \> & \> & \> & 0.12 & \> & 0.12 & \> & \> & \>\\
      \> & \> & \> & 0.12 & 0.12 & 0.12 & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.uniform.NextToNNOnly

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.06 & 0.06 & 0.06 & 0.06 & 0.06 & \> & \>\\
      \> & \> & 0.06 & \> & \> & \> & 0.06 & \> & \>\\
      \> & \> & 0.06 & \> & \> & \> & 0.06 & \> & \>\\
      \> & \> & 0.06 & \> & \> & \> & 0.06 & \> & \>\\
      \> & \> & 0.06 & 0.06 & 0.06 & 0.06 & 0.06 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.uniform.IndirectOnly

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & 0.04 & 0.04 & \> & 0.04 & 0.04 & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & 0.04 & 0.04 & 0.04 & 0.04 & 0.04 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

"Simple" branches
~~~~~~~~~~~~~~~~~

These branches have equal power in the center, first ring, and second ring, where applicable.

.. autoclass:: bcn.branches.simple.NearestNeighbor

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & 0.06 & 0.06 & 0.06 & \> & \> & \>\\
      \> & \> & \> & 0.06 & 0.50 & 0.06 & \> & \> & \>\\
      \> & \> & \> & 0.06 & 0.06 & 0.06 & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.simple.NextToNN

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.02 & 0.02 & 0.02 & 0.02 & 0.02 & \> & \>\\
      \> & \> & 0.02 & 0.04 & 0.04 & 0.04 & 0.02 & \> & \>\\
      \> & \> & 0.02 & 0.04 & 0.33 & 0.04 & 0.02 & \> & \>\\
      \> & \> & 0.02 & 0.04 & 0.04 & 0.04 & 0.02 & \> & \>\\
      \> & \> & 0.02 & 0.02 & 0.02 & 0.02 & 0.02 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.simple.NearestNeighborOnly

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & 0.12 & 0.12 & 0.12 & \> & \> & \>\\
      \> & \> & \> & 0.12 & \> & 0.12 & \> & \> & \>\\
      \> & \> & \> & 0.12 & 0.12 & 0.12 & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.simple.NextToNNOnly

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.06 & 0.06 & 0.06 & 0.06 & 0.06 & \> & \>\\
      \> & \> & 0.06 & \> & \> & \> & 0.06 & \> & \>\\
      \> & \> & 0.06 & \> & \> & \> & 0.06 & \> & \>\\
      \> & \> & 0.06 & \> & \> & \> & 0.06 & \> & \>\\
      \> & \> & 0.06 & 0.06 & 0.06 & 0.06 & 0.06 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.simple.IndirectOnly

   The center connection has the following profile.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.03 & 0.03 & 0.03 & 0.03 & 0.03 & \> & \>\\
      \> & \> & 0.03 & 0.06 & 0.06 & 0.06 & 0.03 & \> & \>\\
      \> & \> & 0.03 & 0.06 & \> & 0.06 & 0.03 & \> & \>\\
      \> & \> & 0.03 & 0.06 & 0.06 & 0.06 & 0.03 & \> & \>\\
      \> & \> & 0.03 & 0.03 & 0.03 & 0.03 & 0.03 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. _optics_informed_branches:

Optics-informed branches
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: bcn.branches.informed.Kappa

   ``Kappa(0.0)`` is ordinary `~bcn.branches.DirectOnly` branches.

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & 1.00 & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

   ``Kappa(1.0)`` looks like:

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & \> & \>\\
      \> & \> & 0.00 & 0.05 & 0.05 & 0.05 & 0.00 & \> & \>\\
      \> & \> & 0.00 & 0.05 & 0.59 & 0.05 & 0.00 & \> & \>\\
      \> & \> & 0.00 & 0.05 & 0.05 & 0.05 & 0.00 & \> & \>\\
      \> & \> & 0.00 & 0.00 & 0.00 & 0.00 & 0.00 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

   ``Kappa(1.5)`` looks like:

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.01 & 0.01 & 0.01 & 0.01 & 0.01 & \> & \>\\
      \> & \> & 0.01 & 0.08 & 0.08 & 0.08 & 0.01 & \> & \>\\
      \> & \> & 0.01 & 0.08 & 0.26 & 0.08 & 0.01 & \> & \>\\
      \> & \> & 0.01 & 0.08 & 0.08 & 0.08 & 0.01 & \> & \>\\
      \> & \> & 0.01 & 0.01 & 0.01 & 0.01 & 0.01 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. autoclass:: bcn.branches.informed.IndirectOnly

   .. math::

      \begin{matrix}
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & 0.03 & 0.03 & 0.03 & 0.03 & 0.03 & \> & \>\\
      \> & \> & 0.03 & 0.07 & 0.07 & 0.07 & 0.03 & \> & \>\\
      \> & \> & 0.03 & 0.07 & \> & 0.07 & 0.03 & \> & \>\\
      \> & \> & 0.03 & 0.07 & 0.07 & 0.07 & 0.03 & \> & \>\\
      \> & \> & 0.03 & 0.03 & 0.03 & 0.03 & 0.03 & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>\\
      \> & \> & \> & \> & \> & \> & \> & \> & \>
      \end{matrix}

.. currentmodule:: bcn

Code matchup
============

This page describes how the concepts discussed in the thesis are represented in this repository project.

+----------------------------------+---------------------------+-------------------------+
|              Concept             |          Example          |         In code         |
+==================================+===========================+=========================+
|    :ref:`Model <models>` size    |           16x16           |        ``16x16``        |
|                                  +---------------------------+-------------------------+
|                                  |          30x30x3          |       ``30x30x3``       |
|                                  +---------------------------+-------------------------+
|                                  |                         etc.                        |
+----------------------------------+---------------------------+-------------------------+
| :ref:`Connections <connections>` |           1-to-9          |          ``@9``         |
|                                  +---------------------------+-------------------------+
|                                  |          1-to-25          |         ``@25``         |
|                                  +---------------------------+-------------------------+
|                                  |          1-to-49          |         ``@49``         |
|                                  +---------------------------+-------------------------+
|                                  |      Fully connected      |         ``@Inf``        |
+----------------------------------+---------------------------+-------------------------+
|    :ref:`Branches <branches>`    |  Direct connections only  |      ``DirectOnly``     |
|                                  +---------------------------+-------------------------+
|                                  |     Nearest neighbors     |   ``NearestNeighbor``   |
|                                  +---------------------------+-------------------------+
|                                  | Next-to-nearest neighbors |       ``NextToNN``      |
|                                  +---------------------------+-------------------------+
|                                  |         First ring        | ``NearestNeighborOnly`` |
|                                  +---------------------------+-------------------------+
|                                  |        Second ring        |     ``NextToNNOnly``    |
|                                  +---------------------------+-------------------------+
|                                  | Indirect connections only |     ``IndirectOnly``    |
|                                  +---------------------------+-------------------------+
|                                  |    Grating strength 1.0   |      ``Kappa(1.0)``     |
+----------------------------------+---------------------------+-------------------------+
|     :ref:`Dataset <dataset>`     |           MNIST           |        ``MNIST``        |
|                                  +---------------------------+-------------------------+
|                                  |       Fashion-MNIST       |       ``FASHION``       |
+----------------------------------+---------------------------+-------------------------+
|            Batch size            |      Batch size of 64     |         ``b64``         |
|                                  +---------------------------+-------------------------+
|                                  |     Batch size of 256     |         ``b256``        |
|                                  +---------------------------+-------------------------+
|                                  |                         etc.                        |
+----------------------------------+---------------------------+-------------------------+
|          Training trial          |          Trial 1          |          ``t1``         |
|                                  +---------------------------+-------------------------+
|                                  |         Trial Name        |        ``tName``        |
|                                  +---------------------------+-------------------------+
|                                  |                         etc.                        |
+----------------------------------+-----------------------------------------------------+

.. I originally generated the above with https://www.tablesgenerator.com/text_tables

For example, a weights file with the name ``weights_16x16x6@9-uniform.NextToNNOnly.FASHION.b64.t2.pt`` corresponds to the second trial of a model of height 16, width 16, and depth 6, with 1-to-9 connections and second ring branches (specifically from the :ref:`uniform <uniform>` module), trained on the Fashion-MNIST dataset with a batch size of 64.

Training BCN models
===================

Models are represented by `bcn.BCN` instances. The number of connections and the type of branches are specified with `bcn.Connections` and `bcn.branches.Branches` objects, respectively.

The dataset, batch size, and similar parameters are defined using the `BCN.train` method with a `bcn.TrainingScheme` object that represents the training scheme.

Setting up a model
------------------

Every model has a width and depth, and these are specified with the two positional arguments of `bcn.BCN`. For instance, the following code creates a 28x28x3 model with default (1-to-9) connections and default (direct-only) branches. ::

   >>> from bcn import BCN
   >>> model = BCN(28, 3)

To specify a different type of connections, such as 1-to-25 (`Connections.ONE_TO_25`), specify so by supplying the ``connections`` parameter with a `Connections` option. ::

   >>> from bcn import Connections
   >>> model = BCN(28, 3, connections=Connections.ONE_TO_25)

And to specify a different type of branches, such as uniform second ring branches (`branches.uniform.NextToNNOnly`), specify so by supplying the ``branches`` parameter with a `branches.Branches` object. ::

   >>> from bcn.branches.uniform import NextToNNOnly
   >>> nnn = NextToNNOnly()
   >>> model = BCN(28, 3, branches=nnn)

Check out the `bcn.BCN` reference for other customizable parameters.

Training the model
------------------

Next we need to set up the training scheme. Like, what dataset will we train on? What's the batch size? This information is specified with a `bcn.TrainingScheme` object. It takes only keyworded arguments. Use the width argument in case the dataset should be scaled to a size other than 28 (the original size of MNIST and Fashion-MNIST). ::

   >>> from bcn import TrainingScheme, Dataset
   >>> scheme = TrainingScheme(
   ...    dataset=Dataset.MNIST,
   ...    batch_size=64,
   ... )

Then set the scheme with the `BCN.train` method. Results can be saved using the ``save_path`` parameter. This method further accepts some extra information that are encoded in the results. See the documentation to learn what it accepts. ::

   >>> model.train(
   ...    scheme=scheme,
   ...    save_path="./results/",
   ...    tag="Special test model",
   ... )

Finally, to train for, say, 100 epochs, use `BCN.run_epochs`: ::

   >>> model.run_epochs(100)


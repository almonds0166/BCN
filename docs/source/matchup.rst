
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
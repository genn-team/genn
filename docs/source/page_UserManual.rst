.. index:: pair: page; User Manual
.. _doxid-dc/d05/UserManual:

User Manual
===========

\vspace{0cm}\mbox{}\vspace{0cm}



.. _doxid-dc/d05/UserManual_1Contents:

Contents
~~~~~~~~

* :ref:`Introduction <doxid-dc/d05/UserManual_1sIntro>`

* :ref:`Defining a network model <doxid-df/dc3/sectDefiningNetwork>`

* :ref:`Neuron models <doxid-de/ded/sectNeuronModels>`

* :ref:`Weight update models <doxid-d5/d24/sectSynapseModels>`

* :ref:`Postsynaptic integration methods <doxid-dd/de4/sect_postsyn>`

* :ref:`Current source models <doxid-d0/d1e/sectCurrentSourceModels>`

* :ref:`Synaptic matrix types <doxid-d5/d39/subsect34>`

* :ref:`Variable initialisation <doxid-d4/dc6/sectVariableInitialisation>`

* :ref:`Sparse connectivity initialisation <doxid-d5/dd4/sectSparseConnectivityInitialisation>`





.. _doxid-dc/d05/UserManual_1sIntro:

Introduction
~~~~~~~~~~~~

GeNN is a software library for facilitating the simulation of neuronal network models on NVIDIA CUDA enabled GPU hardware. It was designed with computational neuroscience models in mind rather than artificial neural networks. The main philosophy of GeNN is two-fold:

#. GeNN relies heavily on code generation to make it very flexible and to allow adjusting simulation code to the model of interest and the GPU hardware that is detected at compile time.

#. GeNN is lightweight in that it provides code for running models of neuronal networks on GPU hardware but it leaves it to the user to write a final simulation engine. It so allows maximal flexibility to the user who can use any of the provided code but can fully choose, inspect, extend or otherwise modify the generated code. They can also introduce their own optimisations and in particular control the data flow from and to the GPU in any desired granularity.

This manual gives an overview of how to use GeNN for a novice user and tries to lead the user to more expert use later on. With that we jump right in.

:ref:`Previous <doxid-df/ddb/ReleaseNotes>` \| :ref:`Top <doxid-dc/d05/UserManual>` \| :ref:`Next <doxid-df/dc3/sectDefiningNetwork>`

.. toctree::
	:hidden:

	page_sectCurrentSourceModels.rst
	page_sectDefiningNetwork.rst
	page_sectNeuronModels.rst
	page_sect_postsyn.rst
	page_sectSparseConnectivityInitialisation.rst
	page_subsect34.rst
	page_sectVariableInitialisation.rst
	page_sectSynapseModels.rst

.. rubric:: Related Pages:

|	:doc:`page_sectCurrentSourceModels`
|	:doc:`page_sectDefiningNetwork`
|	:doc:`page_sectNeuronModels`
|	:doc:`page_sect_postsyn`
|	:doc:`page_sectSparseConnectivityInitialisation`
|	:doc:`page_subsect34`
|	:doc:`page_sectVariableInitialisation`
|	:doc:`page_sectSynapseModels`



.. index:: pair: class; pygenn::genn_groups::NeuronGroup
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup:

class pygenn::genn_groups::NeuronGroup
======================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Class representing a group of neurons. :ref:`More...<details-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	class NeuronGroup: public :ref:`pygenn::genn_groups::Group<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group>`
	{
	public:
		// fields
	
		 :target:`neuron<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1ad69938b27e2ae30629c960edd5e1bafa>`;
		 :target:`spikes<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1aa1683ce8d6cfd66179ce489f7389f774>`;
		 :target:`spike_count<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a1d2af6b44512827de67598b630f1448e>`;
		 :target:`spike_que_ptr<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1af22df1573e71abd14c7e36cf9d5d071e>`;
		 :target:`is_spike_source_array<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1abc7253978f963067a6c518b98653b4ba>`;
		 :target:`type<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1ad0d8239f83e7e11402c4b10df8455f96>`;
		 :target:`pop<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a4a569b27b4c1517027d02a588079091b>`;

		// methods
	
		def :ref:`__init__<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a2eb4fa60b1914bafa1bcd1cb5f986234>`();
		def :ref:`current_spikes<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a77fd60d55b344be646dc11adb5316e7e>`();
		def :ref:`delay_slots<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a389da0244755995123f2a1913287b97a>`();
		def :target:`size<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1ae01b45db7a3199b124be2330d22ec01d>`();
		def :ref:`set_neuron<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a67bb528752a537c966cdad2fa785fb48>`();
		def :ref:`add_to<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1acee03b53c099c4263fd65ff8a8814aac>`();
		def :ref:`add_extra_global_param<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1af646754621dc897e2ce83365660e5b36>`();
		def :ref:`load<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a9291a4c6fddaba4f50537d06941211ef>`();
		def :ref:`reinitialise<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1ad94bce8552a3efa969fef0d088a2795c>`();
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// fields
	
		 :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>`;
		 :ref:`vars<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1ad8e33584773714170465d5166c9c5e3e>`;
		 :ref:`extra_global_params<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a646ab45304e93c1cec854df96e8fb197>`;

		// methods
	
		def :ref:`__init__<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1ac4f6b8f8fb67862785fd508f23d50140>`();
		def :ref:`set_var<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a9a44ddf9d465e2f272d8e5d16aa82c54>`();

.. _details-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Class representing a group of neurons.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a2eb4fa60b1914bafa1bcd1cb5f986234:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__()

Init :ref:`NeuronGroup <doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- name

		- string name of the group

.. index:: pair: function; current_spikes
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a77fd60d55b344be646dc11adb5316e7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def current_spikes()

Current spikes from GeNN.

.. index:: pair: function; delay_slots
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a389da0244755995123f2a1913287b97a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def delay_slots()

Maximum delay steps needed for this group.

.. index:: pair: function; set_neuron
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a67bb528752a537c966cdad2fa785fb48:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_neuron()

Set neuron, its parameters and initial variables.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model

		- type as string of intance of the model

	*
		- param_space

		- dict with model parameters

	*
		- var_space

		- dict with model variables

.. index:: pair: function; add_to
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1acee03b53c099c4263fd65ff8a8814aac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_to()

Add this :ref:`NeuronGroup <doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>` to a model.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- model_spec

		- ``:ref:`pygenn.genn_model.GeNNModel <doxid-db/d57/classpygenn_1_1genn__model_1_1GeNNModel>``` to add to

	*
		- num_neurons

		- int number of neurons

.. index:: pair: function; add_extra_global_param
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1af646754621dc897e2ce83365660e5b36:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_extra_global_param()

Add extra global parameter.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- param_name

		- string with the name of the extra global parameter

	*
		- param_values

		- iterable or a single value

.. index:: pair: function; load
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a9291a4c6fddaba4f50537d06941211ef:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def load()

Loads neuron group.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- slm

		- SharedLibraryModel instance for acccessing variables

	*
		- scalar

		- String specifying "scalar" type

.. index:: pair: function; reinitialise
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1ad94bce8552a3efa969fef0d088a2795c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def reinitialise()

Reinitialise neuron group.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- slm

		- SharedLibraryModel instance for acccessing variables

	*
		- scalar

		- String specifying "scalar" type


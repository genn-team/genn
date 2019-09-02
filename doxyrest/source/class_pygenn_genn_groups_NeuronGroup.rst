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
	
		def :ref:`__init__<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a9804b52565c959b8f7220227a02b1b21>`(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name);
		def :ref:`current_spikes<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1af331b05c1525ca3cafc9f006d88d1746>`(self self);
		def :ref:`delay_slots<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a0e0da2c47058822597a9a802380e762f>`(self self);
		def :target:`size<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a3cd4c9a8da6c73efc7c403c44435cde4>`(self self);
		def :ref:`set_neuron<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1aee609924fc55381b0761ae1b587d4b7e>`(self self, model model, param_space param_space, var_space var_space);
		def :ref:`add_to<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a597150844c84e62e5e98bbb18774b1cc>`(self self, model_spec model_spec, num_neurons num_neurons);
		def :ref:`add_extra_global_param<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a870207702ea6ffc852401643a2dd9960>`(self self, param_name param_name, param_values param_values);
		def :ref:`load<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a4d0e3c87d99f1c348a3a8849882d2606>`(self self, slm slm, scalar scalar);
		def :ref:`reinitialise<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1aad5a7be34e8bf7ee1ef233b094e8cbd9>`(self self, slm slm, scalar scalar);
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
	
		def :ref:`__init__<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a44c2cfa0fba7209cc89c0e70b3882f71>`(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name);
		def :ref:`set_var<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a5b79e0c32a3f3c3e03c2a03baa6e13d3>`(self self, var_name var_name, values values);

.. _details-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Class representing a group of neurons.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a9804b52565c959b8f7220227a02b1b21:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name)

Init :ref:`NeuronGroup <doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- name

		- string name of the group

.. index:: pair: function; current_spikes
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1af331b05c1525ca3cafc9f006d88d1746:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def current_spikes(self self)

Current spikes from GeNN.

.. index:: pair: function; delay_slots
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a0e0da2c47058822597a9a802380e762f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def delay_slots(self self)

Maximum delay steps needed for this group.

.. index:: pair: function; set_neuron
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1aee609924fc55381b0761ae1b587d4b7e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_neuron(
		self self,
		model model,
		param_space param_space,
		var_space var_space
		)

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
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a597150844c84e62e5e98bbb18774b1cc:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_to(self self, model_spec model_spec, num_neurons num_neurons)

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
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a870207702ea6ffc852401643a2dd9960:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_extra_global_param(
		self self,
		param_name param_name,
		param_values param_values
		)

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
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1a4d0e3c87d99f1c348a3a8849882d2606:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def load(self self, slm slm, scalar scalar)

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
.. _doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup_1aad5a7be34e8bf7ee1ef233b094e8cbd9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def reinitialise(self self, slm slm, scalar scalar)

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


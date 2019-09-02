.. index:: pair: class; pygenn::genn_groups::CurrentSource
.. _doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource:

class pygenn::genn_groups::CurrentSource
========================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Class representing a current injection into a group of neurons. :ref:`More...<details-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	class CurrentSource: public :ref:`pygenn::genn_groups::Group<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group>`
	{
	public:
		// fields
	
		 :target:`current_source_model<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1aa2e59cb8e23e22392bc2c15dd571fe47>`;
		 :target:`target_pop<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a9af0c16aa535ae2cfd25ec4444deb537>`;
		 :target:`pop<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a8559a76c85d0c217d97c7797b38e8411>`;

		// methods
	
		def :ref:`__init__<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a67f5e692501129aa592af2bb1cbd7dc4>`(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name);
		def :ref:`size<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1ad720bb8e5a3825e186756b21ab02efa1>`(self self);
	
		def :target:`size<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a22d0a70cf9e9129a0696065fb0c3f823>`(
			self self,
			_ _
			);
	
		def :ref:`set_current_source_model<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a60fec602182442bd50577602ee30f57e>`(self self, model model, param_space param_space, var_space var_space);
		def :ref:`add_to<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1ac48d2951f6388fd03df83f80bb8e1b97>`(self self, nn_model nn_model, :ref:`pop<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a8559a76c85d0c217d97c7797b38e8411>` pop);
		def :ref:`add_extra_global_param<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1ae6e20b0ea58f3e85341b835b07f5126c>`(self self, param_name param_name, param_values param_values);
	
		def :target:`load<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a87b63ce2402a65933e817c9a136c3314>`(
			self self,
			slm slm,
			scalar scalar
			);
	
		def :ref:`reinitialise<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a0593259780b1ea7857b9a37a5456f391>`(self self, slm slm, scalar scalar);
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

.. _details-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Class representing a current injection into a group of neurons.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a67f5e692501129aa592af2bb1cbd7dc4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name)

Init :ref:`CurrentSource <doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- name

		- string name of the current source

.. index:: pair: function; size
.. _doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1ad720bb8e5a3825e186756b21ab02efa1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def size(self self)

Number of neuron in the injected population.

.. index:: pair: function; set_current_source_model
.. _doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a60fec602182442bd50577602ee30f57e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_current_source_model(
		self self,
		model model,
		param_space param_space,
		var_space var_space
		)

Set curront source model, its parameters and initial variables.



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
.. _doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1ac48d2951f6388fd03df83f80bb8e1b97:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def add_to(self self, nn_model nn_model, :ref:`pop<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a8559a76c85d0c217d97c7797b38e8411>` pop)

Inject this :ref:`CurrentSource <doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource>` into population and add it to the GeNN NNmodel.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- pop

		- instance of :ref:`NeuronGroup <doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>` into which this :ref:`CurrentSource <doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource>` should be injected

	*
		- nn_model

		- GeNN NNmodel

.. index:: pair: function; add_extra_global_param
.. _doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1ae6e20b0ea58f3e85341b835b07f5126c:

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

.. index:: pair: function; reinitialise
.. _doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource_1a0593259780b1ea7857b9a37a5456f391:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def reinitialise(self self, slm slm, scalar scalar)

Reinitialise current source.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- slm

		- SharedLibraryModel instance for acccessing variables

	*
		- scalar

		- String specifying "scalar" type


.. index:: pair: class; pygenn::genn_groups::Group
.. _doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group:

class pygenn::genn_groups::Group
================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Parent class of :ref:`NeuronGroup <doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>`, :ref:`SynapseGroup <doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>` and :ref:`CurrentSource <doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource>`. :ref:`More...<details-d1/db3/classpygenn_1_1genn__groups_1_1Group>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	class Group: public object
	{
	public:
		// fields
	
		 :target:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>`;
		 :target:`vars<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1ad8e33584773714170465d5166c9c5e3e>`;
		 :target:`extra_global_params<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a646ab45304e93c1cec854df96e8fb197>`;

		// methods
	
		def :ref:`__init__<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a44c2cfa0fba7209cc89c0e70b3882f71>`(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name);
		def :ref:`set_var<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a5b79e0c32a3f3c3e03c2a03baa6e13d3>`(self self, var_name var_name, values values);
	};

	// direct descendants

	class :ref:`CurrentSource<doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource>`;
	class :ref:`NeuronGroup<doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>`;
	class :ref:`SynapseGroup<doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>`;
.. _details-d1/db3/classpygenn_1_1genn__groups_1_1Group:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Parent class of :ref:`NeuronGroup <doxid-dc/dc9/classpygenn_1_1genn__groups_1_1NeuronGroup>`, :ref:`SynapseGroup <doxid-d5/d49/classpygenn_1_1genn__groups_1_1SynapseGroup>` and :ref:`CurrentSource <doxid-da/d83/classpygenn_1_1genn__groups_1_1CurrentSource>`.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a44c2cfa0fba7209cc89c0e70b3882f71:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__(self self, :ref:`name<doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a2c166d9ace64b65eca3d4d0d91e0bf0d>` name)

Init :ref:`Group <doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- name

		- string name of the :ref:`Group <doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group>`

.. index:: pair: function; set_var
.. _doxid-d1/db3/classpygenn_1_1genn__groups_1_1Group_1a5b79e0c32a3f3c3e03c2a03baa6e13d3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_var(self self, var_name var_name, values values)

Set values for a Variable.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- var_name

		- string with the name of the variable

	*
		- values

		- iterable or a single value


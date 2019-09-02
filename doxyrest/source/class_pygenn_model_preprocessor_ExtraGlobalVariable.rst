.. index:: pair: class; pygenn::model_preprocessor::ExtraGlobalVariable
.. _doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable:

class pygenn::model_preprocessor::ExtraGlobalVariable
=====================================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Class holding information about GeNN extra global pointer variable. :ref:`More...<details-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	class ExtraGlobalVariable: public object
	{
	public:
		// fields
	
		 :target:`name<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1ae430888ed2c89d03a737951c8ed51486>`;
		 :target:`type<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a032fcb1bb1cca6edc589a3ff22b0c406>`;
		 :target:`view<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a0fc912e4388cacc0188f65ecabb1466c>`;
		 :target:`values<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a7955b71fac4270579250247bbdaef199>`;

		// methods
	
		def :ref:`__init__<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a788a8a4c49c71a1ca8ecea63043c1c7b>`(
			self self,
			variable_name variable_name,
			variable_type variable_type,
			:ref:`values<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a7955b71fac4270579250247bbdaef199>` values = None
			);
	
		def :ref:`set_values<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a09e4e8058d9f945990944f7676fed73b>`(self self, :ref:`values<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a7955b71fac4270579250247bbdaef199>` values);
	};
.. _details-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Class holding information about GeNN extra global pointer variable.

Methods
-------

.. index:: pair: function; __init__
.. _doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a788a8a4c49c71a1ca8ecea63043c1c7b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def __init__(
		self self,
		variable_name variable_name,
		variable_type variable_type,
		:ref:`values<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a7955b71fac4270579250247bbdaef199>` values = None
		)

Init :ref:`Variable <doxid-d5/de5/classpygenn_1_1model__preprocessor_1_1Variable>`.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- variable_name

		- string name of the variable

	*
		- variable_type

		- string type of the variable

	*
		- values

		- iterable

.. index:: pair: function; set_values
.. _doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a09e4e8058d9f945990944f7676fed73b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	def set_values(self self, :ref:`values<doxid-d2/d80/classpygenn_1_1model__preprocessor_1_1ExtraGlobalVariable_1a7955b71fac4270579250247bbdaef199>` values)

Set :ref:`Variable <doxid-d5/de5/classpygenn_1_1model__preprocessor_1_1Variable>` 's values.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- values

		- iterable, single value or VarInit instance


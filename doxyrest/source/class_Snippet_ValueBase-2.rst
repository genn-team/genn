.. index:: pair: class; Snippet::ValueBase<0>
.. _doxid-dd/df2/classSnippet_1_1ValueBase_3_010_01_4:

template class Snippet::ValueBase<0>
====================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Template specialisation of :ref:`ValueBase <doxid-da/d76/classSnippet_1_1ValueBase>` to avoid compiler warnings in the case when a model requires no parameters or state variables :ref:`More...<details-dd/df2/classSnippet_1_1ValueBase_3_010_01_4>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <snippet.h>
	
	template <>
	class ValueBase<0>
	{
	public:
		// construction
	
		template <typename... T>
		:target:`ValueBase<doxid-dd/df2/classSnippet_1_1ValueBase_3_010_01_4_1a7fab07343a7f0b9ae1a694559fbcdfbf>`(T&&... vals);

		// methods
	
		std::vector<double> :ref:`getValues<doxid-dd/df2/classSnippet_1_1ValueBase_3_010_01_4_1ae25702eaea986281230a9396a28e03ce>`() const;
	};
.. _details-dd/df2/classSnippet_1_1ValueBase_3_010_01_4:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Template specialisation of :ref:`ValueBase <doxid-da/d76/classSnippet_1_1ValueBase>` to avoid compiler warnings in the case when a model requires no parameters or state variables

Methods
-------

.. index:: pair: function; getValues
.. _doxid-dd/df2/classSnippet_1_1ValueBase_3_010_01_4_1ae25702eaea986281230a9396a28e03ce:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<double> getValues() const

Gets values as a vector of doubles.


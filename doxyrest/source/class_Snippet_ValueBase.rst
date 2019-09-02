.. index:: pair: class; Snippet::ValueBase
.. _doxid-da/d76/classSnippet_1_1ValueBase:

template class Snippet::ValueBase
=================================

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <snippet.h>
	
	template <size_t NumVars>
	class ValueBase
	{
	public:
		// construction
	
		template <typename... T>
		:target:`ValueBase<doxid-da/d76/classSnippet_1_1ValueBase_1a4988c19ce485c6112dd63218abd13a20>`(T&&... vals);

		// methods
	
		const std::vector<double>& :ref:`getValues<doxid-da/d76/classSnippet_1_1ValueBase_1a69f7917cb03d425eadc1593564eaf792>`() const;
		double :target:`operator []<doxid-da/d76/classSnippet_1_1ValueBase_1ab16d489b13647bff35b63dffb357ad24>` (size_t pos) const;
	};
.. _details-da/d76/classSnippet_1_1ValueBase:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Methods
-------

.. index:: pair: function; getValues
.. _doxid-da/d76/classSnippet_1_1ValueBase_1a69f7917cb03d425eadc1593564eaf792:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const std::vector<double>& getValues() const

Gets values as a vector of doubles.


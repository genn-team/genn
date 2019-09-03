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
		// methods
	
		template <typename... T>
		:target:`ValueBase<doxid-da/d76/classSnippet_1_1ValueBase_1aa001b4d15730423c4915ce75e26ca89f>`(T&&... vals);
	
		const std::vector<double>& :ref:`getValues<doxid-da/d76/classSnippet_1_1ValueBase_1ad87c5118a6be03345c415702dd70eedb>`() const;
		double :target:`operator []<doxid-da/d76/classSnippet_1_1ValueBase_1ab7ffe4a19d6b14c9c4a0d41fb14812f0>` (size_t pos) const;
	};
.. _details-da/d76/classSnippet_1_1ValueBase:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Methods
-------

.. index:: pair: function; getValues
.. _doxid-da/d76/classSnippet_1_1ValueBase_1ad87c5118a6be03345c415702dd70eedb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const std::vector<double>& getValues() const

Gets values as a vector of doubles.


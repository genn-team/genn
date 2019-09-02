.. index:: pair: class; Models::VarInitContainerBase<0>
.. _doxid-db/db0/classModels_1_1VarInitContainerBase_3_010_01_4:

template class Models::VarInitContainerBase<0>
==============================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Template specialisation of ValueInitBase to avoid compiler warnings in the case when a model requires no variable initialisers :ref:`More...<details-db/db0/classModels_1_1VarInitContainerBase_3_010_01_4>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <models.h>
	
	template <>
	class VarInitContainerBase<0>
	{
	public:
		// construction
	
		template <typename... T>
		:target:`VarInitContainerBase<doxid-db/db0/classModels_1_1VarInitContainerBase_3_010_01_4_1a244f67720b960375c1e1e08fad8ea8a6>`(T&&... initialisers);
	
		:target:`VarInitContainerBase<doxid-db/db0/classModels_1_1VarInitContainerBase_3_010_01_4_1ac786815e6fb3da4aa592a31b884d4421>`(const :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<0>&);

		// methods
	
		std::vector<:ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`> :ref:`getInitialisers<doxid-db/db0/classModels_1_1VarInitContainerBase_3_010_01_4_1afd8ea93a253ce545aedcff4d5f34533b>`() const;
	};
.. _details-db/db0/classModels_1_1VarInitContainerBase_3_010_01_4:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Template specialisation of ValueInitBase to avoid compiler warnings in the case when a model requires no variable initialisers

Methods
-------

.. index:: pair: function; getInitialisers
.. _doxid-db/db0/classModels_1_1VarInitContainerBase_3_010_01_4_1afd8ea93a253ce545aedcff4d5f34533b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::vector<:ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`> getInitialisers() const

Gets initialisers as a vector of Values.


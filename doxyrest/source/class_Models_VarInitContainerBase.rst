.. index:: pair: class; Models::VarInitContainerBase
.. _doxid-d6/d24/classModels_1_1VarInitContainerBase:

template class Models::VarInitContainerBase
===========================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Wrapper to ensure at compile time that correct number of value initialisers are used when specifying the values of a model's initial state. :ref:`More...<details-d6/d24/classModels_1_1VarInitContainerBase>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <models.h>
	
	template <size_t NumVars>
	class VarInitContainerBase
	{
	public:
		// construction
	
		template <typename... T>
		:target:`VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase_1a354dfe92300e835facb084583df03d9c>`(T&&... initialisers);

		// methods
	
		const std::vector<:ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getInitialisers<doxid-d6/d24/classModels_1_1VarInitContainerBase_1a20b4885664ed4d7b5483d8b57ec4c242>`() const;
		const :ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`& :target:`operator []<doxid-d6/d24/classModels_1_1VarInitContainerBase_1a2de564fbceeb1e57aad2199313e8ead6>` (size_t pos) const;
	};
.. _details-d6/d24/classModels_1_1VarInitContainerBase:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Wrapper to ensure at compile time that correct number of value initialisers are used when specifying the values of a model's initial state.

Methods
-------

.. index:: pair: function; getInitialisers
.. _doxid-d6/d24/classModels_1_1VarInitContainerBase_1a20b4885664ed4d7b5483d8b57ec4c242:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const std::vector<:ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& getInitialisers() const

Gets initialisers as a vector of Values.


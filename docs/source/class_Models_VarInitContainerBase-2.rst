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
		// methods
	
		template <typename... T>
		:target:`VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase_1ab32b684a4402a77ad46018f48ef95b3d>`(T&&... initialisers);
	
		const std::vector<:ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getInitialisers<doxid-d6/d24/classModels_1_1VarInitContainerBase_1ad8c91b4dec4d3263425e75ade891aac1>`() const;
		const :ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`& :target:`operator []<doxid-d6/d24/classModels_1_1VarInitContainerBase_1a27ab7cc7f38e1510db4422259a9658b0>` (size_t pos) const;
	};
.. _details-d6/d24/classModels_1_1VarInitContainerBase:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Wrapper to ensure at compile time that correct number of value initialisers are used when specifying the values of a model's initial state.

Methods
-------

.. index:: pair: function; getInitialisers
.. _doxid-d6/d24/classModels_1_1VarInitContainerBase_1ad8c91b4dec4d3263425e75ade891aac1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const std::vector<:ref:`VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& getInitialisers() const

Gets initialisers as a vector of Values.


.. index:: pair: class; Snippet::Base
.. _doxid-db/d97/classSnippet_1_1Base:

class Snippet::Base
===================

.. toctree::
	:hidden:

	struct_Snippet_Base_DerivedParam.rst
	struct_Snippet_Base_EGP.rst
	struct_Snippet_Base_ParamVal.rst

Overview
~~~~~~~~

:ref:`Base <doxid-db/d97/classSnippet_1_1Base>` class for all code snippets. :ref:`More...<details-db/d97/classSnippet_1_1Base>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <snippet.h>
	
	class Base
	{
	public:
		// typedefs
	
		typedef std::vector<std::string> :target:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>`;
		typedef std::vector<:ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`> :target:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>`;
		typedef std::vector<:ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`> :target:`ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>`;
		typedef std::vector<:ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`> :target:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>`;

		// structs
	
		struct :ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`;
		struct :ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`;
		struct :ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`;

		// methods
	
		virtual :target:`~Base<doxid-db/d97/classSnippet_1_1Base_1a17a9ca158277401f2c190afb1e791d1f>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1a0c8374854fbdc457bf0f75e458748580>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1ab01de002618efa59541c927ffdd463f5>`() const;
	};

	// direct descendants

	class :ref:`Base<doxid-d5/d9f/classInitSparseConnectivitySnippet_1_1Base>`;
	class :ref:`Base<doxid-d3/d9e/classInitVarSnippet_1_1Base>`;
	class :ref:`Base<doxid-d6/d97/classModels_1_1Base>`;
.. _details-db/d97/classSnippet_1_1Base:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Base <doxid-db/d97/classSnippet_1_1Base>` class for all code snippets.

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-db/d97/classSnippet_1_1Base_1a0c8374854fbdc457bf0f75e458748580:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getDerivedParams
.. _doxid-db/d97/classSnippet_1_1Base_1ab01de002618efa59541c927ffdd463f5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` getDerivedParams() const

Gets names of derived model parameters and the function objects to call to Calculate their value from a vector of model parameter values


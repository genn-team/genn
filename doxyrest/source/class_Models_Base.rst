.. index:: pair: class; Models::Base
.. _doxid-d6/d97/classModels_1_1Base:

class Models::Base
==================

.. toctree::
	:hidden:

	struct_Models_Base_Var.rst

Overview
~~~~~~~~

:ref:`Base <doxid-d6/d97/classModels_1_1Base>` class for all models - in addition to the parameters snippets have, models can have state variables. :ref:`More...<details-d6/d97/classModels_1_1Base>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <models.h>
	
	class Base: public :ref:`Snippet::Base<doxid-db/d97/classSnippet_1_1Base>`
	{
	public:
		// typedefs
	
		typedef std::vector<:ref:`Var<doxid-d5/d42/structModels_1_1Base_1_1Var>`> :target:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>`;

		// structs
	
		struct :ref:`Var<doxid-d5/d42/structModels_1_1Base_1_1Var>`;

		// methods
	
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d6/d97/classModels_1_1Base_1a9df8ba9bf6d971a574ed4745f6cf946c>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d6/d97/classModels_1_1Base_1a7fdddb7d19382736b330ade62c441de1>`() const;
		size_t :ref:`getVarIndex<doxid-d6/d97/classModels_1_1Base_1afa0e39df5002efc76448e180f82825e4>`(const std::string& varName) const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d6/d97/classModels_1_1Base_1ae046c19ad56dfb2808c5f4d2cc7475fe>`(const std::string& paramName) const;
	};

	// direct descendants

	class :ref:`Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`;
	class :ref:`Base<doxid-d7/dad/classNeuronModels_1_1Base>`;
	class :ref:`Base<doxid-d1/d3a/classPostsynapticModels_1_1Base>`;
	class :ref:`Base<doxid-d2/d05/classWeightUpdateModels_1_1Base>`;

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// typedefs
	
		typedef std::vector<std::string> :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>`;
		typedef std::vector<:ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`> :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>`;
		typedef std::vector<:ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`> :ref:`ParamValVec<doxid-db/d97/classSnippet_1_1Base_1a0156727ddf8f9c9cbcbc0d3d913b6b48>`;
		typedef std::vector<:ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`> :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>`;

		// structs
	
		struct :ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`;
		struct :ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`;
		struct :ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`;

		// methods
	
		virtual :ref:`~Base<doxid-db/d97/classSnippet_1_1Base_1a17a9ca158277401f2c190afb1e791d1f>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1a0c8374854fbdc457bf0f75e458748580>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1ab01de002618efa59541c927ffdd463f5>`() const;

.. _details-d6/d97/classModels_1_1Base:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Base <doxid-d6/d97/classModels_1_1Base>` class for all models - in addition to the parameters snippets have, models can have state variables.

Methods
-------

.. index:: pair: function; getVars
.. _doxid-d6/d97/classModels_1_1Base_1a9df8ba9bf6d971a574ed4745f6cf946c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

.. index:: pair: function; getExtraGlobalParams
.. _doxid-d6/d97/classModels_1_1Base_1a7fdddb7d19382736b330ade62c441de1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` getExtraGlobalParams() const

Gets names and types (as strings) of additional per-population parameters for the weight update model.

.. index:: pair: function; getVarIndex
.. _doxid-d6/d97/classModels_1_1Base_1afa0e39df5002efc76448e180f82825e4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t getVarIndex(const std::string& varName) const

Find the index of a named variable.

.. index:: pair: function; getExtraGlobalParamIndex
.. _doxid-d6/d97/classModels_1_1Base_1ae046c19ad56dfb2808c5f4d2cc7475fe:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t getExtraGlobalParamIndex(const std::string& paramName) const

Find the index of a named extra global parameter.


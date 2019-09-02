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
	
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d6/d97/classModels_1_1Base_1a5da12b4e51f0b969510dd97d45ad285a>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d6/d97/classModels_1_1Base_1ad6a043bb48b7620c4294854c042e561e>`() const;
		size_t :ref:`getVarIndex<doxid-d6/d97/classModels_1_1Base_1ab54e5508872ef8d1558b7da8aa25bb63>`(const std::string& varName) const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d6/d97/classModels_1_1Base_1a693ad5cfedde6e2db10200501c549c81>`(const std::string& paramName) const;
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
	
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1aad4f3bb00c5f29cb9d0e3585db3f4e20>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1a450c7783570d875e19bcd8a88d10bbf6>`() const;

.. _details-d6/d97/classModels_1_1Base:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Base <doxid-d6/d97/classModels_1_1Base>` class for all models - in addition to the parameters snippets have, models can have state variables.

Methods
-------

.. index:: pair: function; getVars
.. _doxid-d6/d97/classModels_1_1Base_1a5da12b4e51f0b969510dd97d45ad285a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` getVars() const

Gets names and types (as strings) of model variables.

.. index:: pair: function; getExtraGlobalParams
.. _doxid-d6/d97/classModels_1_1Base_1ad6a043bb48b7620c4294854c042e561e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` getExtraGlobalParams() const

Gets names and types (as strings) of additional per-population parameters for the weight update model.

.. index:: pair: function; getVarIndex
.. _doxid-d6/d97/classModels_1_1Base_1ab54e5508872ef8d1558b7da8aa25bb63:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t getVarIndex(const std::string& varName) const

Find the index of a named variable.

.. index:: pair: function; getExtraGlobalParamIndex
.. _doxid-d6/d97/classModels_1_1Base_1a693ad5cfedde6e2db10200501c549c81:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	size_t getExtraGlobalParamIndex(const std::string& paramName) const

Find the index of a named extra global parameter.


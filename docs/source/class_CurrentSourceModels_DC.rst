.. index:: pair: class; CurrentSourceModels::DC
.. _doxid-d7/da1/classCurrentSourceModels_1_1DC:

class CurrentSourceModels::DC
=============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

:ref:`DC <doxid-d7/da1/classCurrentSourceModels_1_1DC>` source. :ref:`More...<details-d7/da1/classCurrentSourceModels_1_1DC>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <currentSourceModels.h>
	
	class DC: public :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<1> :target:`ParamValues<doxid-d7/da1/classCurrentSourceModels_1_1DC_1a3999840df60dc6284d6209e8f21f72b3>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`VarValues<doxid-d7/da1/classCurrentSourceModels_1_1DC_1a5f87cf43404ed2416185be2e85f6bd74>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-d7/da1/classCurrentSourceModels_1_1DC_1a979def8db5dd3d0bd585afe965f93698>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-d7/da1/classCurrentSourceModels_1_1DC_1a961309e28760d0e81122bb660716aa54>`;

		// methods
	
		static const DC* :target:`getInstance<doxid-d7/da1/classCurrentSourceModels_1_1DC_1af70581941c10eb3ebad704b3ea79a405>`();
		:target:`SET_INJECTION_CODE<doxid-d7/da1/classCurrentSourceModels_1_1DC_1ab14a6c9700652ac6e741b372c441cfed>`("$(injectCurrent, $(amp));\n");
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d7/da1/classCurrentSourceModels_1_1DC_1a6a95c17036de93f523b846e3105ff1be>`() const;
	};

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
		typedef std::vector<:ref:`Var<doxid-d5/d42/structModels_1_1Base_1_1Var>`> :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>`;

		// structs
	
		struct :ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`;
		struct :ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`;
		struct :ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`;
		struct :ref:`Var<doxid-d5/d42/structModels_1_1Base_1_1Var>`;

		// methods
	
		virtual :ref:`~Base<doxid-db/d97/classSnippet_1_1Base_1a17a9ca158277401f2c190afb1e791d1f>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1a0c8374854fbdc457bf0f75e458748580>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1ab01de002618efa59541c927ffdd463f5>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d6/d97/classModels_1_1Base_1a9df8ba9bf6d971a574ed4745f6cf946c>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d6/d97/classModels_1_1Base_1a7fdddb7d19382736b330ade62c441de1>`() const;
		size_t :ref:`getVarIndex<doxid-d6/d97/classModels_1_1Base_1afa0e39df5002efc76448e180f82825e4>`(const std::string& varName) const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d6/d97/classModels_1_1Base_1ae046c19ad56dfb2808c5f4d2cc7475fe>`(const std::string& paramName) const;
		virtual std::string :ref:`getInjectionCode<doxid-d0/de0/classCurrentSourceModels_1_1Base_1ae03aa4efaacd4a169409f3a08262a0f3>`() const;

.. _details-d7/da1/classCurrentSourceModels_1_1DC:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`DC <doxid-d7/da1/classCurrentSourceModels_1_1DC>` source.

It has a single parameter:

* ``amp`` - amplitude of the current [nA]

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-d7/da1/classCurrentSourceModels_1_1DC_1a6a95c17036de93f523b846e3105ff1be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.


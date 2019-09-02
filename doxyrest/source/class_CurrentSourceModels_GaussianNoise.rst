.. index:: pair: class; CurrentSourceModels::GaussianNoise
.. _doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise:

class CurrentSourceModels::GaussianNoise
========================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Noisy current source with noise drawn from normal distribution. :ref:`More...<details-d0/d7d/classCurrentSourceModels_1_1GaussianNoise>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <currentSourceModels.h>
	
	class GaussianNoise: public :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<2> :target:`ParamValues<doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise_1a306b35cee4a320067542a8cc3c4ce6a3>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`VarValues<doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise_1a28b334e3b77583f34c5b64330307a856>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise_1a077dd38255bda0c91be7b343ba71b72c>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise_1aa2774064d89602d90e78cb4a503736a0>`;

		// methods
	
		static const GaussianNoise* :target:`getInstance<doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise_1a2d29bad232c05a7eae4e5c5ef6712cc8>`();
		:target:`SET_INJECTION_CODE<doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise_1af80be8c3e3030ca673867c974609e2b2>`("$(injectCurrent, $(mean) + $(gennrand_normal)* $(sd));\);
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise_1ad0c69a87bd606978e13bf6c489b39f20>`() const;
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
	
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1aad4f3bb00c5f29cb9d0e3585db3f4e20>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1a450c7783570d875e19bcd8a88d10bbf6>`() const;
		virtual :ref:`VarVec<doxid-d6/d97/classModels_1_1Base_1a5a6bc95969a38ac1ac68ab4a0ba94c75>` :ref:`getVars<doxid-d6/d97/classModels_1_1Base_1a5da12b4e51f0b969510dd97d45ad285a>`() const;
		virtual :ref:`EGPVec<doxid-db/d97/classSnippet_1_1Base_1a43ece29884e2c6cabffe9abf985807c6>` :ref:`getExtraGlobalParams<doxid-d6/d97/classModels_1_1Base_1ad6a043bb48b7620c4294854c042e561e>`() const;
		size_t :ref:`getVarIndex<doxid-d6/d97/classModels_1_1Base_1ab54e5508872ef8d1558b7da8aa25bb63>`(const std::string& varName) const;
		size_t :ref:`getExtraGlobalParamIndex<doxid-d6/d97/classModels_1_1Base_1a693ad5cfedde6e2db10200501c549c81>`(const std::string& paramName) const;
		virtual std::string :ref:`getInjectionCode<doxid-d0/de0/classCurrentSourceModels_1_1Base_1aa1e8f581137f0415dae669e985df99c2>`() const;

.. _details-d0/d7d/classCurrentSourceModels_1_1GaussianNoise:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Noisy current source with noise drawn from normal distribution.

It has 2 parameters:

* ``mean`` - mean of the normal distribution [nA]

* ``sd`` - standard deviation of the normal distribution [nA]

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-d0/d7d/classCurrentSourceModels_1_1GaussianNoise_1ad0c69a87bd606978e13bf6c489b39f20:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.


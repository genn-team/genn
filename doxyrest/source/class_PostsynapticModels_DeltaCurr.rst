.. index:: pair: class; PostsynapticModels::DeltaCurr
.. _doxid-de/da9/classPostsynapticModels_1_1DeltaCurr:

class PostsynapticModels::DeltaCurr
===================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Simple delta current synapse. :ref:`More...<details-de/da9/classPostsynapticModels_1_1DeltaCurr>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <postsynapticModels.h>
	
	class DeltaCurr: public :ref:`PostsynapticModels::Base<doxid-d1/d3a/classPostsynapticModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<0> :target:`ParamValues<doxid-de/da9/classPostsynapticModels_1_1DeltaCurr_1ac79a926831d8a70e524f3a1c57672bd9>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`VarValues<doxid-de/da9/classPostsynapticModels_1_1DeltaCurr_1ad44a55e6f4aa03b4a6e1986bddf39601>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-de/da9/classPostsynapticModels_1_1DeltaCurr_1a3df3b2a0591f9c0f40871157ef9868f6>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-de/da9/classPostsynapticModels_1_1DeltaCurr_1ae64a7f09da43d6d3d08f964df89626a0>`;

		// methods
	
		static const DeltaCurr* :target:`getInstance<doxid-de/da9/classPostsynapticModels_1_1DeltaCurr_1a4ed81c3ad0a14f46b7a2dad1f9b7992f>`();
		virtual std::string :target:`getApplyInputCode<doxid-de/da9/classPostsynapticModels_1_1DeltaCurr_1a4ed0d483fa452dd7405aced6dddf3a12>`() const;
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
		virtual std::string :ref:`getDecayCode<doxid-d1/d3a/classPostsynapticModels_1_1Base_1acc7bd35a4842517a2a046ac17e2e6ad3>`() const;
		virtual std::string :ref:`getApplyInputCode<doxid-d1/d3a/classPostsynapticModels_1_1Base_1a1f34df49b0acab91a302b57a248b2068>`() const;
		virtual std::string :ref:`getSupportCode<doxid-d1/d3a/classPostsynapticModels_1_1Base_1ab2ffae04b5142df1c166bc1c7cd2aa83>`() const;

.. _details-de/da9/classPostsynapticModels_1_1DeltaCurr:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Simple delta current synapse.

Synaptic input provides a direct inject of instantaneous current


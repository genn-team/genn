.. index:: pair: class; PostsynapticModels::ExpCurr
.. _doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr:

class PostsynapticModels::ExpCurr
=================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Exponential decay with synaptic input treated as a current value. :ref:`More...<details-d5/d1e/classPostsynapticModels_1_1ExpCurr>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <postsynapticModels.h>
	
	class ExpCurr: public :ref:`PostsynapticModels::Base<doxid-d1/d3a/classPostsynapticModels_1_1Base>`
	{
	public:
		// typedefs
	
		typedef :ref:`Snippet::ValueBase<doxid-da/d76/classSnippet_1_1ValueBase>`<1> :target:`ParamValues<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1ad30dc0939458f4b258facbab84a83b3b>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`VarValues<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1aa34e3f94e243295537c684f0c3f6eaad>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PreVarValues<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1ac136b34dfafa22ad131aa8ce522a4a12>`;
		typedef :ref:`Models::VarInitContainerBase<doxid-d6/d24/classModels_1_1VarInitContainerBase>`<0> :target:`PostVarValues<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1a13b9ec21322cb9db92ca490949ddefa6>`;

		// methods
	
		static const ExpCurr* :target:`getInstance<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1a76c6e6548b00a2b630d28c54fa2e6f8d>`();
		virtual std::string :target:`getDecayCode<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1ab8eb9a53898579c5d16b64a07901b9dd>`() const;
		virtual std::string :target:`getApplyInputCode<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1a6e85c8dbf187916807951d162d984a27>`() const;
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1a44f8c01c1a6d89292701c410796c5be9>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1a0382c6f54349cd1e332cd5f4f82ec752>`() const;
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

.. _details-d5/d1e/classPostsynapticModels_1_1ExpCurr:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Exponential decay with synaptic input treated as a current value.

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1a44f8c01c1a6d89292701c410796c5be9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.

.. index:: pair: function; getDerivedParams
.. _doxid-d5/d1e/classPostsynapticModels_1_1ExpCurr_1a0382c6f54349cd1e332cd5f4f82ec752:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` getDerivedParams() const

Gets names of derived model parameters and the function objects to call to Calculate their value from a vector of model parameter values


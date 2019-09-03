.. index:: pair: class; InitVarSnippet::Gamma
.. _doxid-d0/d54/classInitVarSnippet_1_1Gamma:

class InitVarSnippet::Gamma
===========================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Initialises variable by sampling from the exponential distribution. :ref:`More...<details-d0/d54/classInitVarSnippet_1_1Gamma>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <initVarSnippet.h>
	
	class Gamma: public :ref:`InitVarSnippet::Base<doxid-d3/d9e/classInitVarSnippet_1_1Base>`
	{
	public:
		// methods
	
		:target:`DECLARE_SNIPPET<doxid-d0/d54/classInitVarSnippet_1_1Gamma_1aa29667a191e433cc5c2629a7a6ce9d1f>`(
			InitVarSnippet::Gamma,
			2
			);
	
		:target:`SET_CODE<doxid-d0/d54/classInitVarSnippet_1_1Gamma_1a164787e99c9fec1117e1ec097face206>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d0/d54/classInitVarSnippet_1_1Gamma_1a9b49d48ecc76239c734f2d2c9fa2a749>`() const;
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

		// structs
	
		struct :ref:`DerivedParam<doxid-d9/d0c/structSnippet_1_1Base_1_1DerivedParam>`;
		struct :ref:`EGP<doxid-dd/d5d/structSnippet_1_1Base_1_1EGP>`;
		struct :ref:`ParamVal<doxid-d7/dda/structSnippet_1_1Base_1_1ParamVal>`;

		// methods
	
		virtual :ref:`~Base<doxid-db/d97/classSnippet_1_1Base_1a17a9ca158277401f2c190afb1e791d1f>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-db/d97/classSnippet_1_1Base_1a0c8374854fbdc457bf0f75e458748580>`() const;
		virtual :ref:`DerivedParamVec<doxid-db/d97/classSnippet_1_1Base_1ad14217cebf11eddffa751a4d5c4792cb>` :ref:`getDerivedParams<doxid-db/d97/classSnippet_1_1Base_1ab01de002618efa59541c927ffdd463f5>`() const;
		virtual std::string :ref:`getCode<doxid-d3/d9e/classInitVarSnippet_1_1Base_1af6547fd34390034643ed1651f7cf1797>`() const;

.. _details-d0/d54/classInitVarSnippet_1_1Gamma:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Initialises variable by sampling from the exponential distribution.

This snippet takes 1 parameter:

* ``lambda`` - mean event rate (events per unit time/distance)

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-d0/d54/classInitVarSnippet_1_1Gamma_1a9b49d48ecc76239c734f2d2c9fa2a749:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.


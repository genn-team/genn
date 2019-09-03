.. index:: pair: class; InitVarSnippet::Normal
.. _doxid-d5/dc1/classInitVarSnippet_1_1Normal:

class InitVarSnippet::Normal
============================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Initialises variable by sampling from the normal distribution. :ref:`More...<details-d5/dc1/classInitVarSnippet_1_1Normal>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <initVarSnippet.h>
	
	class Normal: public :ref:`InitVarSnippet::Base<doxid-d3/d9e/classInitVarSnippet_1_1Base>`
	{
	public:
		// methods
	
		:target:`DECLARE_SNIPPET<doxid-d5/dc1/classInitVarSnippet_1_1Normal_1a42ca6ff098f4372bb7a559c8444f77e2>`(
			InitVarSnippet::Normal,
			2
			);
	
		:target:`SET_CODE<doxid-d5/dc1/classInitVarSnippet_1_1Normal_1a07ddbd072769b7919448034c47af470d>`();
		virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` :ref:`getParamNames<doxid-d5/dc1/classInitVarSnippet_1_1Normal_1a9191e7e1ca6aa72c753e26520b784854>`() const;
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

.. _details-d5/dc1/classInitVarSnippet_1_1Normal:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Initialises variable by sampling from the normal distribution.

This snippet takes 2 parameters:

* ``mean`` - The mean

* ``sd`` - The standard distribution

Methods
-------

.. index:: pair: function; getParamNames
.. _doxid-d5/dc1/classInitVarSnippet_1_1Normal_1a9191e7e1ca6aa72c753e26520b784854:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual :ref:`StringVec<doxid-db/d97/classSnippet_1_1Base_1a06cd0f6da1424a20163e12b6fec62519>` getParamNames() const

Gets names of of (independent) model parameters.


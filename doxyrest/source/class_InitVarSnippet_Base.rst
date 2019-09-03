.. index:: pair: class; InitVarSnippet::Base
.. _doxid-d3/d9e/classInitVarSnippet_1_1Base:

class InitVarSnippet::Base
==========================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <initVarSnippet.h>
	
	class Base: public :ref:`Snippet::Base<doxid-db/d97/classSnippet_1_1Base>`
	{
	public:
		// methods
	
		virtual std::string :target:`getCode<doxid-d3/d9e/classInitVarSnippet_1_1Base_1af6547fd34390034643ed1651f7cf1797>`() const;
	};

	// direct descendants

	class :ref:`Constant<doxid-dd/dcb/classInitVarSnippet_1_1Constant>`;
	class :ref:`Exponential<doxid-d8/d70/classInitVarSnippet_1_1Exponential>`;
	class :ref:`Gamma<doxid-d0/d54/classInitVarSnippet_1_1Gamma>`;
	class :ref:`Normal<doxid-d5/dc1/classInitVarSnippet_1_1Normal>`;
	class :ref:`Uniform<doxid-dd/da0/classInitVarSnippet_1_1Uniform>`;
	class :ref:`Uninitialised<doxid-d9/db6/classInitVarSnippet_1_1Uninitialised>`;

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


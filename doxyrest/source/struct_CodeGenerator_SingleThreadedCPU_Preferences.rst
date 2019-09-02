.. index:: pair: struct; CodeGenerator::SingleThreadedCPU::Preferences
.. _doxid-d2/d1e/structCodeGenerator_1_1SingleThreadedCPU_1_1Preferences:

struct CodeGenerator::SingleThreadedCPU::Preferences
====================================================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <backend.h>
	
	struct Preferences: public :ref:`CodeGenerator::PreferencesBase<doxid-d1/d7a/structCodeGenerator_1_1PreferencesBase>`
	{
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// fields
	
		bool :ref:`optimizeCode<doxid-d1/d7a/structCodeGenerator_1_1PreferencesBase_1a78a5449e9e05425cebb11d1ffba5dc21>`;
		bool :ref:`debugCode<doxid-d1/d7a/structCodeGenerator_1_1PreferencesBase_1a58b816a5e133a98fa9aa88ec71890a89>`;
		std::string :ref:`userCxxFlagsGNU<doxid-d1/d7a/structCodeGenerator_1_1PreferencesBase_1a86fb454bb8ca003d22eedff8c8c7f4e2>`;
		std::string :ref:`userNvccFlagsGNU<doxid-d1/d7a/structCodeGenerator_1_1PreferencesBase_1acad9c31842a33378f83f524093f6011c>`;
		plog::Severity :ref:`logLevel<doxid-d1/d7a/structCodeGenerator_1_1PreferencesBase_1a901bd4125e2fff733bee452613175063>`;


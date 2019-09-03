.. index:: pair: class; CodeGenerator::Substitutions
.. _doxid-de/d22/classCodeGenerator_1_1Substitutions:

class CodeGenerator::Substitutions
==================================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <substitutions.h>
	
	class Substitutions
	{
	public:
		// methods
	
		:target:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a7fe0822b5faffcadce98ffe3b8e10197>`(const Substitutions* parent = nullptr);
	
		:target:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a6e7c057f59d2af8be75d7bd4f81165b7>`(
			const std::vector<:ref:`FunctionTemplate<doxid-dc/df1/structCodeGenerator_1_1FunctionTemplate>`>& functions,
			const std::string& ftype
			);
	
		void :target:`addVarSubstitution<doxid-de/d22/classCodeGenerator_1_1Substitutions_1ae3ed06630ef19dce975fd16651768c32>`(
			const std::string& source,
			const std::string& destionation,
			bool allowOverride = false
			);
	
		void :target:`addFuncSubstitution<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a61c71d0dc4863950a652ecce2fa1b843>`(
			const std::string& source,
			unsigned int numArguments,
			const std::string& funcTemplate,
			bool allowOverride = false
			);
	
		bool :target:`hasVarSubstitution<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a4fd50f7493be2c4951ce0876d0ad55cc>`(const std::string& source) const;
		const std::string& :target:`getVarSubstitution<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a83802d6384bbf6422b5af56f6cc64ab0>`(const std::string& source) const;
		void :target:`apply<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a6b81121e62bcf604409a0cbaa57de79e>`(std::string& code) const;
		const std::string :target:`operator []<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a524b56ee41ee756c1fff8bae1ddf90d8>` (const std::string& source) const;
	};

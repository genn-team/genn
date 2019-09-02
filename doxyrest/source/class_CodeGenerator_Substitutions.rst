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
		// construction
	
		:target:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a7fe0822b5faffcadce98ffe3b8e10197>`(const Substitutions* parent = nullptr);
	
		:target:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a6e7c057f59d2af8be75d7bd4f81165b7>`(
			const std::vector<:ref:`FunctionTemplate<doxid-dc/df1/structCodeGenerator_1_1FunctionTemplate>`>& functions,
			const std::string& ftype
			);

		// methods
	
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
	
		bool :target:`hasVarSubstitution<doxid-de/d22/classCodeGenerator_1_1Substitutions_1aef723becd5e3a3d5d6dcc72ddce7dddb>`(const std::string& source) const;
		const std::string& :target:`getVarSubstitution<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a2499ee956ad8f25a7a725c3f7e35ca31>`(const std::string& source) const;
		void :target:`apply<doxid-de/d22/classCodeGenerator_1_1Substitutions_1ae2f104d7fb3d30afa560f4ea00eed80a>`(std::string& code) const;
		const std::string :target:`operator []<doxid-de/d22/classCodeGenerator_1_1Substitutions_1a72a45ab0c1741af1611a6f6ab71e7ee9>` (const std::string& source) const;
	};

.. index:: pair: class; CodeGenerator::CodeStream
.. _doxid-d9/df8/classCodeGenerator_1_1CodeStream:

class CodeGenerator::CodeStream
===============================

.. toctree::
	:hidden:

	struct_CodeGenerator_CodeStream_CB.rst
	struct_CodeGenerator_CodeStream_OB.rst
	class_CodeGenerator_CodeStream_IndentBuffer.rst
	class_CodeGenerator_CodeStream_Scope.rst




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <codeStream.h>
	
	class CodeStream: public std::ostream
	{
	public:
		// structs
	
		struct :ref:`CB<doxid-d4/d3d/structCodeGenerator_1_1CodeStream_1_1CB>`;
		struct :ref:`OB<doxid-d4/d6b/structCodeGenerator_1_1CodeStream_1_1OB>`;

		// classes
	
		class :ref:`IndentBuffer<doxid-d7/db2/classCodeGenerator_1_1CodeStream_1_1IndentBuffer>`;
		class :ref:`Scope<doxid-d4/d6e/classCodeGenerator_1_1CodeStream_1_1Scope>`;

		// methods
	
		:target:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream_1a04a5a264774068dd66b7cd84d2e7a816>`();
		:target:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream_1aa16ba20eb1d2da8dc23045ed8ca74d59>`(std::ostream& stream);
		void :target:`setSink<doxid-d9/df8/classCodeGenerator_1_1CodeStream_1abe44259e6c0aa0bedd34b99e641e2d87>`(std::ostream& stream);
	};

.. index:: pair: struct; CodeGenerator::CodeStream::OB
.. _doxid-d4/d6b/structCodeGenerator_1_1CodeStream_1_1OB:

struct CodeGenerator::CodeStream::OB
====================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

An open bracket marker. :ref:`More...<details-d4/d6b/structCodeGenerator_1_1CodeStream_1_1OB>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <codeStream.h>
	
	struct OB
	{
		// fields
	
		const unsigned int :target:`Level<doxid-d4/d6b/structCodeGenerator_1_1CodeStream_1_1OB_1adc50a112eb3588a5526a80810c83a14b>`;

		// methods
	
		:target:`OB<doxid-d4/d6b/structCodeGenerator_1_1CodeStream_1_1OB_1a890a4c3888f1439c4c51166d8518135c>`(unsigned int level);
	};
.. _details-d4/d6b/structCodeGenerator_1_1CodeStream_1_1OB:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

An open bracket marker.

Write to code stream ``os`` using:

.. ref-code-block:: cpp

	os << :ref:`OB <doxid-d4/d6b/structCodeGenerator_1_1CodeStream_1_1OB_1a890a4c3888f1439c4c51166d8518135c>`(16);


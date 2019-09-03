.. index:: pair: struct; CodeGenerator::CodeStream::CB
.. _doxid-d4/d3d/structCodeGenerator_1_1CodeStream_1_1CB:

struct CodeGenerator::CodeStream::CB
====================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

A close bracket marker. :ref:`More...<details-d4/d3d/structCodeGenerator_1_1CodeStream_1_1CB>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <codeStream.h>
	
	struct CB
	{
		// fields
	
		const unsigned int :target:`Level<doxid-d4/d3d/structCodeGenerator_1_1CodeStream_1_1CB_1a34c7eaa2f7502df3ddc7a138ee2e4d1b>`;

		// methods
	
		:target:`CB<doxid-d4/d3d/structCodeGenerator_1_1CodeStream_1_1CB_1a1bdb3de24897472fac38d36a60783347>`(unsigned int level);
	};
.. _details-d4/d3d/structCodeGenerator_1_1CodeStream_1_1CB:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

A close bracket marker.

Write to code stream ``os`` using:

.. ref-code-block:: cpp

	os << :ref:`CB <doxid-d4/d3d/structCodeGenerator_1_1CodeStream_1_1CB_1a1bdb3de24897472fac38d36a60783347>`(16);


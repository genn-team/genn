.. index:: pair: enum; BlockSizeSelect
.. _doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7ab:

enum CodeGenerator::CUDA::BlockSizeSelect
=========================================

Overview
~~~~~~~~

Methods for selecting :ref:`CUDA <doxid-d1/df6/namespaceCodeGenerator_1_1CUDA>` kernel block size. :ref:`More...<details-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7ab>`

.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <backend.h>

	enum BlockSizeSelect
	{
	    :ref:`OCCUPANCY<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7abad835e9b82eae5eafdd8c3cb305a7d7a5>`,
	    :ref:`MANUAL<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7abaa60a6a471c0681e5a49c4f5d00f6bc5a>`,
	};

.. _details-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7ab:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Methods for selecting :ref:`CUDA <doxid-d1/df6/namespaceCodeGenerator_1_1CUDA>` kernel block size.

Enum Values
-----------

.. index:: pair: enumvalue; OCCUPANCY
.. _doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7abad835e9b82eae5eafdd8c3cb305a7d7a5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	OCCUPANCY

Pick optimal blocksize for each kernel based on occupancy.

.. index:: pair: enumvalue; MANUAL
.. _doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7abaa60a6a471c0681e5a49c4f5d00f6bc5a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	MANUAL

Use block sizes specified by user.


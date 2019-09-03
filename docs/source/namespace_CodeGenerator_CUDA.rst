.. index:: pair: namespace; CodeGenerator::CUDA
.. _doxid-d1/df6/namespaceCodeGenerator_1_1CUDA:

namespace CodeGenerator::CUDA
=============================

.. toctree::
	:hidden:

	namespace_CodeGenerator_CUDA_Optimiser.rst
	namespace_CodeGenerator_CUDA_PresynapticUpdateStrategy.rst
	namespace_CodeGenerator_CUDA_Utils.rst
	enum_CodeGenerator_CUDA_BlockSizeSelect.rst
	enum_CodeGenerator_CUDA_DeviceSelect.rst
	enum_CodeGenerator_CUDA_Kernel.rst
	struct_CodeGenerator_CUDA_Preferences.rst
	class_CodeGenerator_CUDA_Backend.rst

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	
	namespace CUDA {

	// namespaces

	namespace :ref:`CodeGenerator::CUDA::Optimiser<doxid-d9/d85/namespaceCodeGenerator_1_1CUDA_1_1Optimiser>`;
	namespace :ref:`CodeGenerator::CUDA::PresynapticUpdateStrategy<doxid-da/d97/namespaceCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy>`;
	namespace :ref:`CodeGenerator::CUDA::Utils<doxid-d0/dd2/namespaceCodeGenerator_1_1CUDA_1_1Utils>`;

	// typedefs

	typedef std::array<size_t, :ref:`KernelMax<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05381dc4178da4eb5cd21384a44dace4a50aff7d81597c0195a06734c9fa4ada8>`> :ref:`KernelBlockSize<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a834e8ff4a9b37453a04e5bfa2743423b>`;

	// enums

	enum :ref:`BlockSizeSelect<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7ab>`;
	enum :ref:`DeviceSelect<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05cdf29b66af2e1899cbb1d9c702f9d0>`;
	enum :ref:`Kernel<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05381dc4178da4eb5cd21384a44dace4>`;

	// structs

	struct :ref:`Preferences<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences>`;

	// classes

	class :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`;

	} // namespace CUDA
.. _details-d1/df6/namespaceCodeGenerator_1_1CUDA:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Typedefs
--------

.. index:: pair: typedef; KernelBlockSize
.. _doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a834e8ff4a9b37453a04e5bfa2743423b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef std::array<size_t, :ref:`KernelMax<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05381dc4178da4eb5cd21384a44dace4a50aff7d81597c0195a06734c9fa4ada8>`> KernelBlockSize

Array of block sizes for each kernel.


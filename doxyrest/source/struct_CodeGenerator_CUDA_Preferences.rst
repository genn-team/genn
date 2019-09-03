.. index:: pair: struct; CodeGenerator::CUDA::Preferences
.. _doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences:

struct CodeGenerator::CUDA::Preferences
=======================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

:ref:`Preferences <doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences>` for :ref:`CUDA <doxid-d1/df6/namespaceCodeGenerator_1_1CUDA>` backend. :ref:`More...<details-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <backend.h>
	
	struct Preferences: public :ref:`CodeGenerator::PreferencesBase<doxid-d1/d7a/structCodeGenerator_1_1PreferencesBase>`
	{
		// fields
	
		bool :ref:`showPtxInfo<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1ad8e937d78148dcd94e8174b3fba45e86>`;
		:ref:`DeviceSelect<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05cdf29b66af2e1899cbb1d9c702f9d0>` :ref:`deviceSelectMethod<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1ab3f7e871cfa06d52cdff493f80a9289e>`;
		unsigned int :ref:`manualDeviceID<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1a3b75fa868ca95ea3c644efcaaff3308d>`;
		:ref:`BlockSizeSelect<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7ab>` :ref:`blockSizeSelectMethod<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1ad9ddce1e46c8707bf1c30116f9a799be>`;
		:ref:`KernelBlockSize<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a834e8ff4a9b37453a04e5bfa2743423b>` :ref:`manualBlockSizes<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1a24e6c8b33837783988259baa53fd4dda>`;
		std::string :ref:`userNvccFlags<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1aa370bc1e7c48d0928ace7bf4baaa7e73>`;

		// methods
	
		:target:`Preferences<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1a81d2f28e0431fcbb375e6c0c547a879d>`();
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

.. _details-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

:ref:`Preferences <doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences>` for :ref:`CUDA <doxid-d1/df6/namespaceCodeGenerator_1_1CUDA>` backend.

Fields
------

.. index:: pair: variable; showPtxInfo
.. _doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1ad8e937d78148dcd94e8174b3fba45e86:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool showPtxInfo

Should PTX assembler information be displayed for each :ref:`CUDA <doxid-d1/df6/namespaceCodeGenerator_1_1CUDA>` kernel during compilation.

.. index:: pair: variable; deviceSelectMethod
.. _doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1ab3f7e871cfa06d52cdff493f80a9289e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`DeviceSelect<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05cdf29b66af2e1899cbb1d9c702f9d0>` deviceSelectMethod

How to select GPU device.

.. index:: pair: variable; manualDeviceID
.. _doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1a3b75fa868ca95ea3c644efcaaff3308d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unsigned int manualDeviceID

If device select method is set to :ref:`DeviceSelect::MANUAL <doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05cdf29b66af2e1899cbb1d9c702f9d0aa60a6a471c0681e5a49c4f5d00f6bc5a>`, id of device to use.

.. index:: pair: variable; blockSizeSelectMethod
.. _doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1ad9ddce1e46c8707bf1c30116f9a799be:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`BlockSizeSelect<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a54abdd5e5351c160ba420cd758edb7ab>` blockSizeSelectMethod

How to select :ref:`CUDA <doxid-d1/df6/namespaceCodeGenerator_1_1CUDA>` blocksize.

.. index:: pair: variable; manualBlockSizes
.. _doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1a24e6c8b33837783988259baa53fd4dda:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`KernelBlockSize<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a834e8ff4a9b37453a04e5bfa2743423b>` manualBlockSizes

If block size select method is set to :ref:`BlockSizeSelect::MANUAL <doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05cdf29b66af2e1899cbb1d9c702f9d0aa60a6a471c0681e5a49c4f5d00f6bc5a>`, block size to use for each kernel.

.. index:: pair: variable; userNvccFlags
.. _doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences_1aa370bc1e7c48d0928ace7bf4baaa7e73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::string userNvccFlags

NVCC compiler options for all GPU code.


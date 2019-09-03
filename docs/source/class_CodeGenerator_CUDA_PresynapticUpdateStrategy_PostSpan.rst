.. index:: pair: class; CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpan
.. _doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan:

class CodeGenerator::CUDA::PresynapticUpdateStrategy::PostSpan
==============================================================

.. toctree::
	:hidden:

Overview
~~~~~~~~

Postsynaptic parallelism. :ref:`More...<details-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <presynapticUpdateStrategy.h>
	
	class PostSpan: public :ref:`CodeGenerator::CUDA::PresynapticUpdateStrategy::Base<doxid-d1/d48/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1Base>`
	{
	public:
		// methods
	
		virtual size_t :ref:`getNumThreads<doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1aa6547dbd6195b74f0c5e156d40e69981>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg) const;
		virtual bool :ref:`isCompatible<doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1a8f48bb3bab41339ac00e4eca4649bd24>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg) const;
		virtual bool :ref:`shouldAccumulateInRegister<doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1ae55cec8c17b68cb38cd5922f37f2df44>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg, const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend) const;
		virtual bool :ref:`shouldAccumulateInSharedMemory<doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1a1daef37e18265b842431dfaba74be20e>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg, const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend) const;
	
		virtual void :ref:`genCode<doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1a418208734dd6b518796cf3ed3ae6cee2>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
			const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& popSubs,
			const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend,
			bool trueSpike,
			:ref:`BackendBase::SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumThreshHandler,
			:ref:`BackendBase::SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumSimHandler
			) const;
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// methods
	
		virtual size_t :ref:`getNumThreads<doxid-d1/d48/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1Base_1a23dcb4398c882c4af6811b13cb9ebe8d>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg) const = 0;
		virtual bool :ref:`isCompatible<doxid-d1/d48/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1Base_1a2613968de8aebfbb8e97972b265e6f72>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg) const = 0;
		virtual bool :ref:`shouldAccumulateInRegister<doxid-d1/d48/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1Base_1a9cc23f259f780552598d5424bf4c51e6>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg, const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend) const = 0;
		virtual bool :ref:`shouldAccumulateInSharedMemory<doxid-d1/d48/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1Base_1a4903a1c176412710ec2ca83641be6da4>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg, const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend) const = 0;
	
		virtual void :ref:`genCode<doxid-d1/d48/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1Base_1a3a665d09ec8064093fba498c558bcbc3>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
			const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& popSubs,
			const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend,
			bool trueSpike,
			:ref:`BackendBase::SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumThreshHandler,
			:ref:`BackendBase::SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumSimHandler
			) const = 0;

.. _details-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Postsynaptic parallelism.

Methods
-------

.. index:: pair: function; getNumThreads
.. _doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1aa6547dbd6195b74f0c5e156d40e69981:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual size_t getNumThreads(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg) const

Get the number of threads that presynaptic updates should be parallelised across.

.. index:: pair: function; isCompatible
.. _doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1a8f48bb3bab41339ac00e4eca4649bd24:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool isCompatible(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg) const

Is this presynaptic update strategy compatible with a given synapse group?

.. index:: pair: function; shouldAccumulateInRegister
.. _doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1ae55cec8c17b68cb38cd5922f37f2df44:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool shouldAccumulateInRegister(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg, const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend) const

Are input currents emitted by this presynaptic update accumulated into a register?

.. index:: pair: function; shouldAccumulateInSharedMemory
.. _doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1a1daef37e18265b842431dfaba74be20e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool shouldAccumulateInSharedMemory(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg, const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend) const

Are input currents emitted by this presynaptic update accumulated into a shared memory array?

.. index:: pair: function; genCode
.. _doxid-d1/d23/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1PostSpan_1a418208734dd6b518796cf3ed3ae6cee2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genCode(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
		const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& popSubs,
		const :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`& backend,
		bool trueSpike,
		:ref:`BackendBase::SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumThreshHandler,
		:ref:`BackendBase::SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumSimHandler
		) const

Generate presynaptic update code.


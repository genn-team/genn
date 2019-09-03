.. index:: pair: class; CodeGenerator::BackendBase
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase:

class CodeGenerator::BackendBase
================================

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <backendBase.h>
	
	class BackendBase
	{
	public:
		// typedefs
	
		typedef std::function<void(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`&, :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`&)> :target:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>`;
		typedef std::function<void(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`&, const T&, :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`&)> :target:`GroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1adba20f0748ab61dd226b26bf116b04c2>`;
		typedef :ref:`GroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1adba20f0748ab61dd226b26bf116b04c2>`<:ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`> :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>`;
		typedef :ref:`GroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1adba20f0748ab61dd226b26bf116b04c2>`<:ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`> :ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>`;
		typedef std::function<void(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`&, const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`&, :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`&, :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>`, :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>`)> :ref:`NeuronGroupSimHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a938379883f4cdf06998dd969a3d74a73>`;

		// methods
	
		:target:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a0ce0c48ea6c5041fb3a6d3d3cd1d24bc>`(
			int localHostID,
			const std::string& scalarType
			);
	
		virtual :target:`~BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ac4ba1c12b2a97dfb6cb4f3cd5df10f50>`();
		virtual void :ref:`genNeuronUpdate<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ae8915df91011c1c65ff2a9b52bc8e05c>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model, :ref:`NeuronGroupSimHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a938379883f4cdf06998dd969a3d74a73>` simHandler, :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>` wuVarUpdateHandler) const = 0;
	
		virtual void :ref:`genSynapseUpdate<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a7a63d712f5cb7e8053c93acb6df77d58>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumThreshHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumSimHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumEventHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` postLearnHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` synapseDynamicsHandler
			) const = 0;
	
		virtual void :target:`genInit<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a879f4a9c41a101ba527729733cdc0a9e>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
			:ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>` localNGHandler,
			:ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>` remoteNGHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` sgDenseInitHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` sgSparseConnectHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` sgSparseInitHandler
			) const = 0;
	
		virtual void :ref:`genDefinitionsPreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1af8d1180742c4bb56a68a244a4348f070>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const = 0;
		virtual void :ref:`genDefinitionsInternalPreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a5d4f7275f37639e6563ca37a9c77be17>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const = 0;
		virtual void :target:`genRunnerPreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ac4a836ef8f5c4b5ada55ff298e7b086b>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const = 0;
		virtual void :ref:`genAllocateMemPreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a044451afab7cb5aeaffe801e9a0f2b73>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0;
		virtual void :ref:`genStepTimeFinalisePreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a6706c769fd857c084ca85e27985b18b7>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0;
	
		virtual void :target:`genVariableDefinition<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a0c979af40143720432811dd1768eaec2>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :target:`genVariableImplementation<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ae9c337953415ab88551891ced09d8aa6>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :target:`genVariableAllocation<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a228feb3756df365e230122b2abf43985>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count
			) const = 0;
	
		virtual void :target:`genVariableFree<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a6be7117419486f6e221aa90601ed66de>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :target:`genExtraGlobalParamDefinition<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a4ccb6a0724f29df2a999bd1daf645585>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :target:`genExtraGlobalParamImplementation<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a5007d624c8dec1ae8cc2a180a8405214>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :target:`genExtraGlobalParamAllocation<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a4d5d7e06e8e7176ece20c8fb08cffca4>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :target:`genExtraGlobalParamPush<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a113a50a43bdc1d52ab5ea67409f66fe1>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :target:`genExtraGlobalParamPull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a48febc23266ae499c211af7d8c4b52d3>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :target:`genPopVariableInit<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a6ae3becde57e41f1151b280aaac90f26>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs,
			:ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler
			) const = 0;
	
		virtual void :target:`genVariableInit<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a37be1d74b9fa1bdd516f1afa5638077a>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count,
			const std::string& indexVarName,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs,
			:ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler
			) const = 0;
	
		virtual void :target:`genSynapseVariableRowInit<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aa99d6c99c1fbf2e12bf15de68518f542>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs,
			:ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler
			) const = 0;
	
		virtual void :target:`genVariablePush<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aa04def4fec77c03d419b79e11abfb035>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			bool autoInitialized,
			size_t count
			) const = 0;
	
		virtual void :target:`genVariablePull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1addcfe5c7751b6dc87dd13f59127fcb3b>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count
			) const = 0;
	
		virtual void :target:`genCurrentTrueSpikePush<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a8ac39bcb2a2b737ec85910009d064deb>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng
			) const = 0;
	
		virtual void :target:`genCurrentTrueSpikePull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a7299c60d97b3cfc6f71c5863ce3dee45>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng
			) const = 0;
	
		virtual void :target:`genCurrentSpikeLikeEventPush<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a852dac4b14d5f25a93f30d58239bb4a2>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng
			) const = 0;
	
		virtual void :target:`genCurrentSpikeLikeEventPull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a8fcf09ea91aaa84d16ff44cb44b6b472>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng
			) const = 0;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :target:`genGlobalRNG<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a9a94866ddd2bd9fe2f63649caab8b7b5>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model
			) const = 0;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :target:`genPopulationRNG<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a45611d965af4bf0c86285add875cdeaa>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			const std::string& name,
			size_t count
			) const = 0;
	
		virtual void :target:`genTimer<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a054418ff1f9aa42e69c89aa7709807ff>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& stepTimeFinalise,
			const std::string& name,
			bool updateInStepTime
			) const = 0;
	
		virtual void :ref:`genMakefilePreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a05760de268731a3432901739b1a69ee5>`(std::ostream& os) const = 0;
		virtual void :ref:`genMakefileLinkRule<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ac28594d0b0b207844932ffb105846ce5>`(std::ostream& os) const = 0;
		virtual void :ref:`genMakefileCompileRule<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a5835ec300bc0b5c6b2b476164d9da13b>`(std::ostream& os) const = 0;
		virtual void :ref:`genMSBuildConfigProperties<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a4e8a1201a6cf7342ede5b7bbb797cdb8>`(std::ostream& os) const = 0;
		virtual void :target:`genMSBuildImportProps<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1af6fd371fd89c03caefed7cd8d40b13f6>`(std::ostream& os) const = 0;
		virtual void :ref:`genMSBuildItemDefinitions<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a955c31de992db529595956bc33ca1525>`(std::ostream& os) const = 0;
	
		virtual void :target:`genMSBuildCompileModule<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aaa0922c18f30dabce3693503a5df3a3d>`(
			const std::string& moduleName,
			std::ostream& os
			) const = 0;
	
		virtual void :target:`genMSBuildImportTarget<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a3530f453f5bc8d354d16958a09da56ed>`(std::ostream& os) const = 0;
		virtual std::string :ref:`getVarPrefix<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a817fca06615b5436d0a00e25e7eb04bb>`() const;
		virtual bool :ref:`isGlobalRNGRequired<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1abd63eb9b142bf73c95921747ad708f14>`(const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0;
		virtual bool :target:`isSynRemapRequired<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a064f208bb1d5c47dad599d737fe9e3bc>`() const = 0;
		virtual bool :target:`isPostsynapticRemapRequired<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a98d6344fec8a12ae7ea7ac8bece8ed50>`() const = 0;
		virtual size_t :ref:`getDeviceMemoryBytes<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a034f408b28b10f750cd501a4dfecceac>`() const = 0;
	
		void :ref:`genVariablePushPull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a88e6465a9217de179050154d3779e957>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& push,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& pull,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			bool autoInitialized,
			size_t count
			) const;
	
		:ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :ref:`genArray<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aec83f512127c6a85a285a1c8b02925a1>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count
			) const;
	
		void :ref:`genScalar<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a3f7a0752cdee6b7667d729de1578bbbf>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		int :ref:`getLocalHostID<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aca884725d8e8efcc6d2fa4e69dadca8a>`() const;
	};

	// direct descendants

	class :ref:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend>`;
	class :ref:`Backend<doxid-d2/dc5/classCodeGenerator_1_1SingleThreadedCPU_1_1Backend>`;
.. _details-d3/d15/classCodeGenerator_1_1BackendBase:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Typedefs
--------

.. index:: pair: typedef; NeuronGroupHandler
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef :ref:`GroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1adba20f0748ab61dd226b26bf116b04c2>`<:ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`> NeuronGroupHandler

Standard callback type which provides a :ref:`CodeStream <doxid-d9/df8/classCodeGenerator_1_1CodeStream>` to write platform-independent code for the specified :ref:`NeuronGroup <doxid-d7/d3b/classNeuronGroup>` to.

.. index:: pair: typedef; SynapseGroupHandler
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef :ref:`GroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1adba20f0748ab61dd226b26bf116b04c2>`<:ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`> SynapseGroupHandler

Standard callback type which provides a :ref:`CodeStream <doxid-d9/df8/classCodeGenerator_1_1CodeStream>` to write platform-independent code for the specified :ref:`SynapseGroup <doxid-dc/dfa/classSynapseGroup>` to.

.. index:: pair: typedef; NeuronGroupSimHandler
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a938379883f4cdf06998dd969a3d74a73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	typedef std::function<void(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`&, const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`&, :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`&, :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>`, :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>`)> NeuronGroupSimHandler

Callback function type for generation neuron group simulation code.

Provides additional callbacks to insert code to emit spikes

Methods
-------

.. index:: pair: function; genNeuronUpdate
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ae8915df91011c1c65ff2a9b52bc8e05c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genNeuronUpdate(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		:ref:`NeuronGroupSimHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a938379883f4cdf06998dd969a3d74a73>` simHandler,
		:ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>` wuVarUpdateHandler
		) const = 0

Generate platform-specific function to update the state of all neurons.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- os

		- :ref:`CodeStream <doxid-d9/df8/classCodeGenerator_1_1CodeStream>` to write function to

	*
		- model

		- model to generate code for

	*
		- simHandler

		- callback to write platform-independent code to update an individual :ref:`NeuronGroup <doxid-d7/d3b/classNeuronGroup>`

	*
		- wuVarUpdateHandler

		- callback to write platform-independent code to update pre and postsynaptic weight update model variables when neuron spikes

.. index:: pair: function; genSynapseUpdate
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a7a63d712f5cb7e8053c93acb6df77d58:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genSynapseUpdate(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumThreshHandler,
		:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumSimHandler,
		:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumEventHandler,
		:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` postLearnHandler,
		:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` synapseDynamicsHandler
		) const = 0

Generate platform-specific function to update the state of all synapses.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- os

		- :ref:`CodeStream <doxid-d9/df8/classCodeGenerator_1_1CodeStream>` to write function to

	*
		- model

		- model to generate code for

	*
		- wumThreshHandler

		- callback to write platform-independent code to update an individual :ref:`NeuronGroup <doxid-d7/d3b/classNeuronGroup>`

	*
		- wumSimHandler

		- callback to write platform-independent code to process presynaptic spikes. "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided to callback via :ref:`Substitutions <doxid-de/d22/classCodeGenerator_1_1Substitutions>`.

	*
		- wumEventHandler

		- callback to write platform-independent code to process presynaptic spike-like events. "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided to callback via :ref:`Substitutions <doxid-de/d22/classCodeGenerator_1_1Substitutions>`.

	*
		- postLearnHandler

		- callback to write platform-independent code to process postsynaptic spikes. "id_pre", "id_post" and "id_syn" variables will be provided to callback via :ref:`Substitutions <doxid-de/d22/classCodeGenerator_1_1Substitutions>`.

	*
		- synapseDynamicsHandler

		- callback to write platform-independent code to update time-driven synapse dynamics. "id_pre", "id_post" and "id_syn" variables; and either "addToInSynDelay" or "addToInSyn" function will be provided to callback via :ref:`Substitutions <doxid-de/d22/classCodeGenerator_1_1Substitutions>`.

.. index:: pair: function; genDefinitionsPreamble
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1af8d1180742c4bb56a68a244a4348f070:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genDefinitionsPreamble(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const = 0

Definitions is the usercode-facing header file for the generated code. This function generates a 'preamble' to this header file.

This will be included from a standard C++ compiler so shouldn't include any platform-specific types or headers

.. index:: pair: function; genDefinitionsInternalPreamble
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a5d4f7275f37639e6563ca37a9c77be17:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genDefinitionsInternalPreamble(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const = 0

Definitions internal is the internal header file for the generated code. This function generates a 'preamble' to this header file.

This will only be included by the platform-specific compiler used to build this backend so can include platform-specific types or headers

.. index:: pair: function; genAllocateMemPreamble
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a044451afab7cb5aeaffe801e9a0f2b73:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genAllocateMemPreamble(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0

Allocate memory is the first function in GeNN generated code called by usercode and it should only ever be called once. Therefore it's a good place for any global initialisation. This function generates a 'preamble' to this function.

.. index:: pair: function; genStepTimeFinalisePreamble
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a6706c769fd857c084ca85e27985b18b7:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genStepTimeFinalisePreamble(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0

After all timestep logic is complete.

.. index:: pair: function; genMakefilePreamble
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a05760de268731a3432901739b1a69ee5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMakefilePreamble(std::ostream& os) const = 0

This function can be used to generate a preamble for the GNU makefile used to build.

.. index:: pair: function; genMakefileLinkRule
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ac28594d0b0b207844932ffb105846ce5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMakefileLinkRule(std::ostream& os) const = 0

The GNU make build system will populate a variable called ```` with a list of objects to link. This function should generate a GNU make rule to build these objects into a shared library.

.. index:: pair: function; genMakefileCompileRule
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a5835ec300bc0b5c6b2b476164d9da13b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMakefileCompileRule(std::ostream& os) const = 0

The GNU make build system uses 'pattern rules' (`https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html <https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html>`__) to build backend modules into objects. This function should generate a GNU make pattern rule capable of building each module (i.e. compiling .cc file $< into .o file $@).

.. index:: pair: function; genMSBuildConfigProperties
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a4e8a1201a6cf7342ede5b7bbb797cdb8:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMSBuildConfigProperties(std::ostream& os) const = 0

In MSBuild, 'properties' are used to configure global project settings e.g. whether the MSBuild project builds a static or dynamic library This function can be used to add additional XML properties to this section.

see `https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-properties <https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-properties>`__ for more information.

.. index:: pair: function; genMSBuildItemDefinitions
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a955c31de992db529595956bc33ca1525:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMSBuildItemDefinitions(std::ostream& os) const = 0

In MSBuild, the 'item definitions' are used to override the default properties of 'items' such as ``<ClCompile>`` or ``<Link>``. This function should generate XML to correctly configure the 'items' required to build the generated code, taking into account ```` etc.

see `https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-items#item-definitions <https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-items#item-definitions>`__ for more information.

.. index:: pair: function; getVarPrefix
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a817fca06615b5436d0a00e25e7eb04bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getVarPrefix() const

When backends require separate 'device' and 'host' versions of variables, they are identified with a prefix. This function returns this prefix so it can be used in otherwise platform-independent code.

.. index:: pair: function; isGlobalRNGRequired
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1abd63eb9b142bf73c95921747ad708f14:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool isGlobalRNGRequired(const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0

Different backends use different RNGs for different things. Does this one require a global RNG for the specified model?

.. index:: pair: function; getDeviceMemoryBytes
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a034f408b28b10f750cd501a4dfecceac:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual size_t getDeviceMemoryBytes() const = 0

How many bytes of memory does 'device' have.

.. index:: pair: function; genVariablePushPull
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a88e6465a9217de179050154d3779e957:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void genVariablePushPull(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& push,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& pull,
		const std::string& type,
		const std::string& name,
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
		bool autoInitialized,
		size_t count
		) const

Helper function to generate matching push and pull functions for a variable.

.. index:: pair: function; genArray
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aec83f512127c6a85a285a1c8b02925a1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` genArray(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
		const std::string& type,
		const std::string& name,
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
		size_t count
		) const

Helper function to generate matching definition, declaration, allocation and free code for an array.

.. index:: pair: function; genScalar
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a3f7a0752cdee6b7667d729de1578bbbf:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void genScalar(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
		const std::string& type,
		const std::string& name,
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
		) const

Helper function to generate matching definition and declaration code for a scalar variable.

.. index:: pair: function; getLocalHostID
.. _doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aca884725d8e8efcc6d2fa4e69dadca8a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	int getLocalHostID() const

Gets ID of local host backend is building code for.


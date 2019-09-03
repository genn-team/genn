.. index:: pair: class; CodeGenerator::CUDA::Backend
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend:

class CodeGenerator::CUDA::Backend
==================================

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <backend.h>
	
	class Backend: public :ref:`CodeGenerator::BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase>`
	{
	public:
		// fields
	
		static const char* :target:`KernelNames<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a9efc38e97bacfddb2925921ce5df91e3>`[KernelMax];

		// methods
	
		:target:`Backend<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a9f01f9b7a15ea7e31510d3725c7147ab>`(
			const :ref:`KernelBlockSize<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a834e8ff4a9b37453a04e5bfa2743423b>`& kernelBlockSizes,
			const :ref:`Preferences<doxid-da/dae/structCodeGenerator_1_1CUDA_1_1Preferences>`& preferences,
			int localHostID,
			const std::string& scalarType,
			int device
			);
	
		virtual void :ref:`genNeuronUpdate<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aac00294b1ef371c52c105bcd85abf790>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model, :ref:`NeuronGroupSimHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a938379883f4cdf06998dd969a3d74a73>` simHandler, :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>` wuVarUpdateHandler) const;
	
		virtual void :ref:`genSynapseUpdate<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aac58a792834d2b4496ec98739626f0a5>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumThreshHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumSimHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` wumEventHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` postLearnHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` synapseDynamicsHandler
			) const;
	
		virtual void :target:`genInit<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1abe93b38aadfe4ed096af8813b24ae0ce>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
			:ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>` localNGHandler,
			:ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>` remoteNGHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` sgDenseInitHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` sgSparseConnectHandler,
			:ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>` sgSparseInitHandler
			) const;
	
		virtual void :ref:`genDefinitionsPreamble<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a109afc9d732bd75c6798e557b81edbf5>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const;
		virtual void :ref:`genDefinitionsInternalPreamble<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a414975d93a2b28aa60025feaf54b3ede>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const;
		virtual void :target:`genRunnerPreamble<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a57e64689756d610f54217c0e25ca2713>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const;
		virtual void :ref:`genAllocateMemPreamble<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a4d1687bdddf44ec71b727ea02400827a>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const;
		virtual void :ref:`genStepTimeFinalisePreamble<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a87840adc59004fb10082c9ca41e23521>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const;
	
		virtual void :target:`genVariableDefinition<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a0a67e211bc1daff47b3cd7c544febac5>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		virtual void :target:`genVariableImplementation<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a3e67d08a18050d67108fba175075f7a4>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :target:`genVariableAllocation<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a478ea6510c24bbaa24cd7ee9858abc75>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count
			) const;
	
		virtual void :target:`genVariableFree<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a4a401d26402bcf5103a099ca2182a645>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		virtual void :target:`genExtraGlobalParamDefinition<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1ac18bff299474a8b6887a7030dd9e82a1>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		virtual void :target:`genExtraGlobalParamImplementation<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a24ee42fcca1d7dc8d8fcb5fdeaa31f54>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		virtual void :target:`genExtraGlobalParamAllocation<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1afeda7210082ace94714fbbf6db957398>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		virtual void :target:`genExtraGlobalParamPush<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a8cc904c2dd8b652b279db9c8e83fd9d2>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		virtual void :target:`genExtraGlobalParamPull<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1acb57d65a381315ec0812bb4525b05f4f>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const;
	
		virtual void :target:`genPopVariableInit<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1acf63cb845f7143a4d88e805e9451a336>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs,
			:ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler
			) const;
	
		virtual void :target:`genVariableInit<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a9e6a42b4b655cad7f5aca8ff7caa1ad4>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count,
			const std::string& indexVarName,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs,
			:ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler
			) const;
	
		virtual void :target:`genSynapseVariableRowInit<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a2dbbe8b4b880a1fb433c6226565df124>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs,
			:ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler
			) const;
	
		virtual void :target:`genVariablePush<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1af0c2251d4ce823c13850bc3201ae49d3>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			bool autoInitialized,
			size_t count
			) const;
	
		virtual void :target:`genVariablePull<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a5cc94a36f46e4e524ee30afb87aafbc9>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count
			) const;
	
		virtual void :target:`genCurrentTrueSpikePush<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aad49bc822127947488ca8969344192bf>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng
			) const;
	
		virtual void :target:`genCurrentTrueSpikePull<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aafe1abd025030c9629a4487aec891e38>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng
			) const;
	
		virtual void :target:`genCurrentSpikeLikeEventPush<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1ab39ba2ca850b9b3f79c986b29434dace>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng
			) const;
	
		virtual void :target:`genCurrentSpikeLikeEventPull<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a074aa6cc89325a4c48f175aeb6d82f29>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng
			) const;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :target:`genGlobalRNG<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a0925d8dd21a15c31209b63b522282bca>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model
			) const;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :target:`genPopulationRNG<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1ad6174620c02abe41cd16bc7003019e85>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			const std::string& name,
			size_t count
			) const;
	
		virtual void :target:`genTimer<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aa0e52eda3f398a5b0de12ee72e48511d>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& stepTimeFinalise,
			const std::string& name,
			bool updateInStepTime
			) const;
	
		virtual void :ref:`genMakefilePreamble<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a7ed64a7cf237943bf788247522ffb755>`(std::ostream& os) const;
		virtual void :ref:`genMakefileLinkRule<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a92603cab842a2cb33adc6978abdb555d>`(std::ostream& os) const;
		virtual void :ref:`genMakefileCompileRule<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a7f35cab83eefa913d2a784bc1aed1226>`(std::ostream& os) const;
		virtual void :ref:`genMSBuildConfigProperties<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a40494ed5e5a10c9d36d487bb911edee5>`(std::ostream& os) const;
		virtual void :target:`genMSBuildImportProps<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a5c50a47bba810aef8be56e11043b4dd2>`(std::ostream& os) const;
		virtual void :ref:`genMSBuildItemDefinitions<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aa66ebe80963428ed54b237e26d488119>`(std::ostream& os) const;
	
		virtual void :target:`genMSBuildCompileModule<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1abc3c700a50c0fdfe4bf7b8fba881b0fb>`(
			const std::string& moduleName,
			std::ostream& os
			) const;
	
		virtual void :target:`genMSBuildImportTarget<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a550e85db3844113ac1edc689469fa5e7>`(std::ostream& os) const;
		virtual std::string :ref:`getVarPrefix<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a4b10e92d9f9f86252136e0b31a769464>`() const;
		virtual bool :ref:`isGlobalRNGRequired<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1acd9a5796a357e5b633eb81bee2c35d65>`(const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const;
		virtual bool :target:`isSynRemapRequired<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1ac393b97a8a98ad75ee65883ac77f6a2b>`() const;
		virtual bool :target:`isPostsynapticRemapRequired<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1af4eb927f939b6416b193848660287f48>`() const;
		virtual size_t :ref:`getDeviceMemoryBytes<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1ae5788df928832109eb10606611e45066>`() const;
		const cudaDeviceProp& :target:`getChosenCUDADevice<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a03814877a752ed9ef0c63dbb1320da70>`() const;
		int :target:`getChosenDeviceID<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a428484ac026d692312c76cee6e67dcc5>`() const;
		int :target:`getRuntimeVersion<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a28686e0f104341e1eaba42648a5edf2c>`() const;
		std::string :target:`getNVCCFlags<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a3cbd03ccb955922efa300de9ff9c90b6>`() const;
		std::string :target:`getFloatAtomicAdd<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a22ae1c31c460cc6b1a38120652d2179d>`(const std::string& ftype) const;
		size_t :target:`getKernelBlockSize<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1afe9f3858faa6dd680a368884091ffd35>`(:ref:`Kernel<doxid-d1/df6/namespaceCodeGenerator_1_1CUDA_1a05381dc4178da4eb5cd21384a44dace4>` kernel) const;
		static size_t :target:`getNumPresynapticUpdateThreads<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a329a55cd79838f58b20a699e92858720>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg);
		static size_t :target:`getNumPostsynapticUpdateThreads<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1ad960a69ff210f6bdea4d3979645dc947>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg);
		static size_t :target:`getNumSynapseDynamicsThreads<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a3fc7808cb0b427e57cbc71a416613155>`(const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg);
		static void :ref:`addPresynapticUpdateStrategy<doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a7b408f27067e5ac5962567925a3c62cb>`(:ref:`PresynapticUpdateStrategy::Base<doxid-d1/d48/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1Base>`* strategy);
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// typedefs
	
		typedef std::function<void(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`&, :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`&)> :ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>`;
		typedef std::function<void(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`&, const T&, :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`&)> :ref:`GroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1adba20f0748ab61dd226b26bf116b04c2>`;
		typedef :ref:`GroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1adba20f0748ab61dd226b26bf116b04c2>`<:ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`> :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>`;
		typedef :ref:`GroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1adba20f0748ab61dd226b26bf116b04c2>`<:ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`> :ref:`SynapseGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a85aea4cafbf9a7e9ceb6b0d12e9d4997>`;
		typedef std::function<void(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`&, const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`&, :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`&, :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>`, :ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>`)> :ref:`NeuronGroupSimHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a938379883f4cdf06998dd969a3d74a73>`;

		// methods
	
		:ref:`BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a0ce0c48ea6c5041fb3a6d3d3cd1d24bc>`(int localHostID, const std::string& scalarType);
		virtual :ref:`~BackendBase<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ac4ba1c12b2a97dfb6cb4f3cd5df10f50>`();
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
	
		virtual void :ref:`genInit<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a879f4a9c41a101ba527729733cdc0a9e>`(
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
		virtual void :ref:`genRunnerPreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ac4a836ef8f5c4b5ada55ff298e7b086b>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const = 0;
		virtual void :ref:`genAllocateMemPreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a044451afab7cb5aeaffe801e9a0f2b73>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0;
		virtual void :ref:`genStepTimeFinalisePreamble<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a6706c769fd857c084ca85e27985b18b7>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0;
	
		virtual void :ref:`genVariableDefinition<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a0c979af40143720432811dd1768eaec2>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :ref:`genVariableImplementation<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ae9c337953415ab88551891ced09d8aa6>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const std::string& type, const std::string& name, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc) const = 0;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :ref:`genVariableAllocation<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a228feb3756df365e230122b2abf43985>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count
			) const = 0;
	
		virtual void :ref:`genVariableFree<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a6be7117419486f6e221aa90601ed66de>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const std::string& name, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc) const = 0;
	
		virtual void :ref:`genExtraGlobalParamDefinition<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a4ccb6a0724f29df2a999bd1daf645585>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc
			) const = 0;
	
		virtual void :ref:`genExtraGlobalParamImplementation<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a5007d624c8dec1ae8cc2a180a8405214>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const std::string& type, const std::string& name, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc) const = 0;
		virtual void :ref:`genExtraGlobalParamAllocation<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a4d5d7e06e8e7176ece20c8fb08cffca4>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const std::string& type, const std::string& name, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc) const = 0;
		virtual void :ref:`genExtraGlobalParamPush<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a113a50a43bdc1d52ab5ea67409f66fe1>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const std::string& type, const std::string& name, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc) const = 0;
		virtual void :ref:`genExtraGlobalParamPull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a48febc23266ae499c211af7d8c4b52d3>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const std::string& type, const std::string& name, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc) const = 0;
		virtual void :ref:`genPopVariableInit<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a6ae3becde57e41f1151b280aaac90f26>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc, const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs, :ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler) const = 0;
	
		virtual void :ref:`genVariableInit<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a37be1d74b9fa1bdd516f1afa5638077a>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count,
			const std::string& indexVarName,
			const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs,
			:ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler
			) const = 0;
	
		virtual void :ref:`genSynapseVariableRowInit<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aa99d6c99c1fbf2e12bf15de68518f542>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc, const :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`& sg, const :ref:`Substitutions<doxid-de/d22/classCodeGenerator_1_1Substitutions>`& kernelSubs, :ref:`Handler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1ab56bdf519ec4ae2c476bf1915c9a3cc5>` handler) const = 0;
	
		virtual void :ref:`genVariablePush<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aa04def4fec77c03d419b79e11abfb035>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			bool autoInitialized,
			size_t count
			) const = 0;
	
		virtual void :ref:`genVariablePull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1addcfe5c7751b6dc87dd13f59127fcb3b>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
			const std::string& type,
			const std::string& name,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc,
			size_t count
			) const = 0;
	
		virtual void :ref:`genCurrentTrueSpikePush<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a8ac39bcb2a2b737ec85910009d064deb>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng) const = 0;
		virtual void :ref:`genCurrentTrueSpikePull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a7299c60d97b3cfc6f71c5863ce3dee45>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng) const = 0;
		virtual void :ref:`genCurrentSpikeLikeEventPush<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a852dac4b14d5f25a93f30d58239bb4a2>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng) const = 0;
		virtual void :ref:`genCurrentSpikeLikeEventPull<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a8fcf09ea91aaa84d16ff44cb44b6b472>`(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`& ng) const = 0;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :ref:`genGlobalRNG<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a9a94866ddd2bd9fe2f63649caab8b7b5>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model
			) const = 0;
	
		virtual :ref:`MemAlloc<doxid-d2/d06/classCodeGenerator_1_1MemAlloc>` :ref:`genPopulationRNG<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a45611d965af4bf0c86285add875cdeaa>`(
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitions,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& definitionsInternal,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& runner,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& allocations,
			:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& free,
			const std::string& name,
			size_t count
			) const = 0;
	
		virtual void :ref:`genTimer<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a054418ff1f9aa42e69c89aa7709807ff>`(
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
		virtual void :ref:`genMSBuildImportProps<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1af6fd371fd89c03caefed7cd8d40b13f6>`(std::ostream& os) const = 0;
		virtual void :ref:`genMSBuildItemDefinitions<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a955c31de992db529595956bc33ca1525>`(std::ostream& os) const = 0;
		virtual void :ref:`genMSBuildCompileModule<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1aaa0922c18f30dabce3693503a5df3a3d>`(const std::string& moduleName, std::ostream& os) const = 0;
		virtual void :ref:`genMSBuildImportTarget<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a3530f453f5bc8d354d16958a09da56ed>`(std::ostream& os) const = 0;
		virtual std::string :ref:`getVarPrefix<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a817fca06615b5436d0a00e25e7eb04bb>`() const;
		virtual bool :ref:`isGlobalRNGRequired<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1abd63eb9b142bf73c95921747ad708f14>`(const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const = 0;
		virtual bool :ref:`isSynRemapRequired<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a064f208bb1d5c47dad599d737fe9e3bc>`() const = 0;
		virtual bool :ref:`isPostsynapticRemapRequired<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a98d6344fec8a12ae7ea7ac8bece8ed50>`() const = 0;
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

.. _details-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Methods
-------

.. index:: pair: function; genNeuronUpdate
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aac00294b1ef371c52c105bcd85abf790:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genNeuronUpdate(
		:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os,
		const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model,
		:ref:`NeuronGroupSimHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a938379883f4cdf06998dd969a3d74a73>` simHandler,
		:ref:`NeuronGroupHandler<doxid-d3/d15/classCodeGenerator_1_1BackendBase_1a55e34429d3f08186318fe811d5bc6531>` wuVarUpdateHandler
		) const

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
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aac58a792834d2b4496ec98739626f0a5:

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
		) const

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
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a109afc9d732bd75c6798e557b81edbf5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genDefinitionsPreamble(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const

Definitions is the usercode-facing header file for the generated code. This function generates a 'preamble' to this header file.

This will be included from a standard C++ compiler so shouldn't include any platform-specific types or headers

.. index:: pair: function; genDefinitionsInternalPreamble
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a414975d93a2b28aa60025feaf54b3ede:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genDefinitionsInternalPreamble(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os) const

Definitions internal is the internal header file for the generated code. This function generates a 'preamble' to this header file.

This will only be included by the platform-specific compiler used to build this backend so can include platform-specific types or headers

.. index:: pair: function; genAllocateMemPreamble
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a4d1687bdddf44ec71b727ea02400827a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genAllocateMemPreamble(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const

Allocate memory is the first function in GeNN generated code called by usercode and it should only ever be called once. Therefore it's a good place for any global initialisation. This function generates a 'preamble' to this function.

.. index:: pair: function; genStepTimeFinalisePreamble
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a87840adc59004fb10082c9ca41e23521:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genStepTimeFinalisePreamble(:ref:`CodeStream<doxid-d9/df8/classCodeGenerator_1_1CodeStream>`& os, const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const

After all timestep logic is complete.

.. index:: pair: function; genMakefilePreamble
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a7ed64a7cf237943bf788247522ffb755:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMakefilePreamble(std::ostream& os) const

This function can be used to generate a preamble for the GNU makefile used to build.

.. index:: pair: function; genMakefileLinkRule
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a92603cab842a2cb33adc6978abdb555d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMakefileLinkRule(std::ostream& os) const

The GNU make build system will populate a variable called ```` with a list of objects to link. This function should generate a GNU make rule to build these objects into a shared library.

.. index:: pair: function; genMakefileCompileRule
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a7f35cab83eefa913d2a784bc1aed1226:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMakefileCompileRule(std::ostream& os) const

The GNU make build system uses 'pattern rules' (`https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html <https://www.gnu.org/software/make/manual/html_node/Pattern-Intro.html>`__) to build backend modules into objects. This function should generate a GNU make pattern rule capable of building each module (i.e. compiling .cc file $< into .o file $@).

.. index:: pair: function; genMSBuildConfigProperties
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a40494ed5e5a10c9d36d487bb911edee5:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMSBuildConfigProperties(std::ostream& os) const

In MSBuild, 'properties' are used to configure global project settings e.g. whether the MSBuild project builds a static or dynamic library This function can be used to add additional XML properties to this section.

see `https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-properties <https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-properties>`__ for more information.

.. index:: pair: function; genMSBuildItemDefinitions
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1aa66ebe80963428ed54b237e26d488119:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual void genMSBuildItemDefinitions(std::ostream& os) const

In MSBuild, the 'item definitions' are used to override the default properties of 'items' such as ``<ClCompile>`` or ``<Link>``. This function should generate XML to correctly configure the 'items' required to build the generated code, taking into account ```` etc.

see `https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-items#item-definitions <https://docs.microsoft.com/en-us/visualstudio/msbuild/msbuild-items#item-definitions>`__ for more information.

.. index:: pair: function; getVarPrefix
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a4b10e92d9f9f86252136e0b31a769464:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual std::string getVarPrefix() const

When backends require separate 'device' and 'host' versions of variables, they are identified with a prefix. This function returns this prefix so it can be used in otherwise platform-independent code.

.. index:: pair: function; isGlobalRNGRequired
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1acd9a5796a357e5b633eb81bee2c35d65:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual bool isGlobalRNGRequired(const :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`& model) const

Different backends use different RNGs for different things. Does this one require a global RNG for the specified model?

.. index:: pair: function; getDeviceMemoryBytes
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1ae5788df928832109eb10606611e45066:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	virtual size_t getDeviceMemoryBytes() const

How many bytes of memory does 'device' have.

.. index:: pair: function; addPresynapticUpdateStrategy
.. _doxid-d6/d3a/classCodeGenerator_1_1CUDA_1_1Backend_1a7b408f27067e5ac5962567925a3c62cb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	static void addPresynapticUpdateStrategy(:ref:`PresynapticUpdateStrategy::Base<doxid-d1/d48/classCodeGenerator_1_1CUDA_1_1PresynapticUpdateStrategy_1_1Base>`* strategy)

Register a new presynaptic update strategy.

This function should be called with strategies in ascending order of preference


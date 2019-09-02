.. index:: pair: class; SynapseGroupInternal
.. _doxid-dd/d48/classSynapseGroupInternal:

class SynapseGroupInternal
==========================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <synapseGroupInternal.h>
	
	class SynapseGroupInternal: public :ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`
	{
	public:
		// construction
	
		:target:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal_1a0ac57bb652676847ab10edc6ba1f5399>`(
			const std::string name,
			:ref:`SynapseMatrixType<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9c>` matrixType,
			unsigned int delaySteps,
			const :ref:`WeightUpdateModels::Base<doxid-d2/d05/classWeightUpdateModels_1_1Base>`* wu,
			const std::vector<double>& wuParams,
			const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& wuVarInitialisers,
			const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& wuPreVarInitialisers,
			const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& wuPostVarInitialisers,
			const :ref:`PostsynapticModels::Base<doxid-d1/d3a/classPostsynapticModels_1_1Base>`* ps,
			const std::vector<double>& psParams,
			const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& psVarInitialisers,
			:ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`* srcNeuronGroup,
			:ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`* trgNeuronGroup,
			const :ref:`InitSparseConnectivitySnippet::Init<doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>`& connectivityInitialiser,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` defaultVarLocation,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` defaultExtraGlobalParamLocation,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` defaultSparseConnectivityLocation
			);
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// enums
	
		enum :ref:`SpanType<doxid-dc/dfa/classSynapseGroup_1a3da23a0e726b05a12e95c3d58645b1a2>`;

		// methods
	
		void :ref:`setWUVarLocation<doxid-dc/dfa/classSynapseGroup_1a36fd4856ed157898059c1aab176c02b8>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setWUPreVarLocation<doxid-dc/dfa/classSynapseGroup_1a2b4a14a357b0f00020f632a440a3c048>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setWUPostVarLocation<doxid-dc/dfa/classSynapseGroup_1abce72af57aaeb5cbeb3b6e1a849b1e1e>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setWUExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1a67c9478a20f8181df57a43e91ecb3ea3>`(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setPSVarLocation<doxid-dc/dfa/classSynapseGroup_1ad394ea032564c35d3228c3e1c1704f54>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setPSExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1a90b0bda40690467d37ce993f12236e39>`(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setSparseConnectivityExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1a3cb510f7c9530a61f4ab7a603ef01ac3>`(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setInSynVarLocation<doxid-dc/dfa/classSynapseGroup_1a871ba5677d4b088443eb43d3c3036114>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setSparseConnectivityLocation<doxid-dc/dfa/classSynapseGroup_1ae30487a9c1dc728cce45130821766fc8>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setDendriticDelayLocation<doxid-dc/dfa/classSynapseGroup_1a74211da769cfc9a1597f5f1c07e26002>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setMaxConnections<doxid-dc/dfa/classSynapseGroup_1aab6b2fb0ad30189bc11ee3dd7d48dbb2>`(unsigned int maxConnections);
		void :ref:`setMaxSourceConnections<doxid-dc/dfa/classSynapseGroup_1a93b12c08d634f1a2300f1b91ef34ea24>`(unsigned int maxPostConnections);
		void :ref:`setMaxDendriticDelayTimesteps<doxid-dc/dfa/classSynapseGroup_1a220307d4043e8bf1bed07552829f2a17>`(unsigned int maxDendriticDelay);
		void :ref:`setSpanType<doxid-dc/dfa/classSynapseGroup_1a97cfec638d856e6e07628bc19490690c>`(:ref:`SpanType<doxid-dc/dfa/classSynapseGroup_1a3da23a0e726b05a12e95c3d58645b1a2>` spanType);
		void :ref:`setNumThreadsPerSpike<doxid-dc/dfa/classSynapseGroup_1a50da6b80e10ac9175f34e901b252803d>`(unsigned int numThreadsPerSpike);
		void :ref:`setBackPropDelaySteps<doxid-dc/dfa/classSynapseGroup_1ac080d0115f8d3aa274e9f95898b1a443>`(unsigned int timesteps);
		const std::string& :ref:`getName<doxid-dc/dfa/classSynapseGroup_1a775e738338993cc2523af0af086527e8>`() const;
		:ref:`SpanType<doxid-dc/dfa/classSynapseGroup_1a3da23a0e726b05a12e95c3d58645b1a2>` :ref:`getSpanType<doxid-dc/dfa/classSynapseGroup_1ad4c8ae8472813f9f77d09e80c6927491>`() const;
		unsigned int :ref:`getNumThreadsPerSpike<doxid-dc/dfa/classSynapseGroup_1ad2f28afff3f0748f9df31e59e2cca004>`() const;
		unsigned int :ref:`getDelaySteps<doxid-dc/dfa/classSynapseGroup_1a46792ae6aadf1948ccc42450175a1c42>`() const;
		unsigned int :ref:`getBackPropDelaySteps<doxid-dc/dfa/classSynapseGroup_1aaddfca073f33dadfc1766689bf0b15f4>`() const;
		unsigned int :ref:`getMaxConnections<doxid-dc/dfa/classSynapseGroup_1a66f15c24d427c6a7e88ff3cf20b6d39e>`() const;
		unsigned int :ref:`getMaxSourceConnections<doxid-dc/dfa/classSynapseGroup_1aa46e2837b0e10c6973ef8b0e2b57e362>`() const;
		unsigned int :ref:`getMaxDendriticDelayTimesteps<doxid-dc/dfa/classSynapseGroup_1aa582cd38678138e44cd0ba9a3a8da7d4>`() const;
		:ref:`SynapseMatrixType<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9c>` :ref:`getMatrixType<doxid-dc/dfa/classSynapseGroup_1a91d96a0c0135910329c2922fdd82169c>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getInSynLocation<doxid-dc/dfa/classSynapseGroup_1aa02124fcc921b0e05785d277f0c9f415>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getSparseConnectivityLocation<doxid-dc/dfa/classSynapseGroup_1aafdea35798050ee57893d0ab6988bd81>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getDendriticDelayLocation<doxid-dc/dfa/classSynapseGroup_1ac91f10810d28560a1b60968834f8e41f>`() const;
		int :ref:`getClusterHostID<doxid-dc/dfa/classSynapseGroup_1a4c645f5e34e80fe452630e177a8d318f>`() const;
		bool :ref:`isTrueSpikeRequired<doxid-dc/dfa/classSynapseGroup_1a17511671ab71f34c65c37837ee6fa7f3>`() const;
		bool :ref:`isSpikeEventRequired<doxid-dc/dfa/classSynapseGroup_1affc58ea4ce0833992eb99cb2debf9459>`() const;
		const :ref:`WeightUpdateModels::Base<doxid-d2/d05/classWeightUpdateModels_1_1Base>`* :ref:`getWUModel<doxid-dc/dfa/classSynapseGroup_1aff9ca57ce84ca974596b02752172601a>`() const;
		const std::vector<double>& :ref:`getWUParams<doxid-dc/dfa/classSynapseGroup_1ab02794caf82b3124b2e7a43819df6df5>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getWUVarInitialisers<doxid-dc/dfa/classSynapseGroup_1ac085e2eaf41431b8307a6f87e1de9031>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getWUPreVarInitialisers<doxid-dc/dfa/classSynapseGroup_1a5f60a80d62699a4b6edfda93e720ca9f>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getWUPostVarInitialisers<doxid-dc/dfa/classSynapseGroup_1a9fbbc52ee9d6e1d3671f22f023b7fde3>`() const;
		const std::vector<double> :ref:`getWUConstInitVals<doxid-dc/dfa/classSynapseGroup_1aff546833d043184f30272a3f38d0865f>`() const;
		const :ref:`PostsynapticModels::Base<doxid-d1/d3a/classPostsynapticModels_1_1Base>`* :ref:`getPSModel<doxid-dc/dfa/classSynapseGroup_1a2cb6ffc942af020a39dea9d9b5d7ca8f>`() const;
		const std::vector<double>& :ref:`getPSParams<doxid-dc/dfa/classSynapseGroup_1a0948f8acc0ef725a79504c743642f4b2>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getPSVarInitialisers<doxid-dc/dfa/classSynapseGroup_1adaf521cec872b314e875e54c676dc99d>`() const;
		const std::vector<double> :ref:`getPSConstInitVals<doxid-dc/dfa/classSynapseGroup_1ad38209d5f3abed8bf3c76be66aa48f5f>`() const;
		const :ref:`InitSparseConnectivitySnippet::Init<doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>`& :ref:`getConnectivityInitialiser<doxid-dc/dfa/classSynapseGroup_1a401ac8c180dec94006d1c4331d0a0e0d>`() const;
		bool :ref:`isZeroCopyEnabled<doxid-dc/dfa/classSynapseGroup_1ae49b55ca7bf25160ae98cc43972fca7b>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getWUVarLocation<doxid-dc/dfa/classSynapseGroup_1a372211c03132fe062dfceb2c946fbdaf>`(const std::string& var) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getWUVarLocation<doxid-dc/dfa/classSynapseGroup_1aeedc94782da1c991bd7fd7974cc69021>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getWUPreVarLocation<doxid-dc/dfa/classSynapseGroup_1ae89e853febe50d5c78fce67a9c8a502c>`(const std::string& var) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getWUPreVarLocation<doxid-dc/dfa/classSynapseGroup_1ad490eebd29a779d3abd26b87b1f97179>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getWUPostVarLocation<doxid-dc/dfa/classSynapseGroup_1a614ee94c1475d536f2b22057140bdccd>`(const std::string& var) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getWUPostVarLocation<doxid-dc/dfa/classSynapseGroup_1a09c5025c7b424c52350abce5661d0721>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getWUExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1acfc97089dd26259510ee2df544c97ce8>`(const std::string& paramName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getWUExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1a821b41019d9ce1f587984c7b55f3d5aa>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getPSVarLocation<doxid-dc/dfa/classSynapseGroup_1a7e20da90e94f95655b0814e24fe5ca7b>`(const std::string& var) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getPSVarLocation<doxid-dc/dfa/classSynapseGroup_1ad2f81f34af78fc2ef7250be19034d50f>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getPSExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1ae94d4f06f927a8e064fde501dc79fba1>`(const std::string& paramName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getPSExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1a1cace5cda6eb25bafe04ed5a786cb0ab>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getSparseConnectivityExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1a66b8a41dab082aae12e7664c673dfabf>`(const std::string& paramName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getSparseConnectivityExtraGlobalParamLocation<doxid-dc/dfa/classSynapseGroup_1a1677a4308ef9b902970033c93d42e9a9>`(size_t index) const;
		bool :ref:`isDendriticDelayRequired<doxid-dc/dfa/classSynapseGroup_1a0ab4cea002580bb64da2a55db37ab189>`() const;
		bool :ref:`isPSInitRNGRequired<doxid-dc/dfa/classSynapseGroup_1a0cb20a9bde00e1ba8132143bfc3d96c9>`() const;
		bool :ref:`isWUInitRNGRequired<doxid-dc/dfa/classSynapseGroup_1aa8191f2f567437c1129e3209fa07273d>`() const;
		bool :ref:`isWUVarInitRequired<doxid-dc/dfa/classSynapseGroup_1a04792c6fbefc32a49c0b936bc6ea5318>`() const;
		bool :ref:`isSparseConnectivityInitRequired<doxid-dc/dfa/classSynapseGroup_1a1c2454ff904e5df412539aa6477e4f0e>`() const;


.. index:: pair: class; ModelSpecInternal
.. _doxid-dc/dfa/classModelSpecInternal:

class ModelSpecInternal
=======================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <modelSpecInternal.h>
	
	class ModelSpecInternal: public :ref:`ModelSpec<doxid-da/dfd/classModelSpec>`
	{
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// typedefs
	
		typedef std::map<std::string, :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`>::value_type :ref:`NeuronGroupValueType<doxid-da/dfd/classModelSpec_1ac724c12166b5ee5fb4492277c1d8deb5>`;
		typedef std::map<std::string, :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`>::value_type :ref:`SynapseGroupValueType<doxid-da/dfd/classModelSpec_1a8af14d4e037788bb97854c4d0da801df>`;

		// methods
	
		:ref:`ModelSpec<doxid-da/dfd/classModelSpec_1a5f71e866bd0b2af2abf2b1e8dde7d6d4>`();
		:ref:`ModelSpec<doxid-da/dfd/classModelSpec_1ab5f3d20c830593a4452be5878e43bba8>`(const :ref:`ModelSpec<doxid-da/dfd/classModelSpec>`&);
		:ref:`ModelSpec<doxid-da/dfd/classModelSpec>`& :ref:`operator =<doxid-da/dfd/classModelSpec_1aaf415f0379159d74d57d18126cf2982e>` (const :ref:`ModelSpec<doxid-da/dfd/classModelSpec>`&);
		:ref:`~ModelSpec<doxid-da/dfd/classModelSpec_1a60ffaa6deb779cff61da6a7ea651613f>`();
		void :ref:`setName<doxid-da/dfd/classModelSpec_1ada1aff7a94eeb36dff721f09d5cf94b4>`(const std::string& name);
		void :ref:`setPrecision<doxid-da/dfd/classModelSpec_1a7548f1bf634884c051e4fbac3cf6212c>`(:ref:`FloatType<doxid-dc/de1/modelSpec_8h_1aa039815ec6b74d0fe4cb016415781c08>` floattype);
		void :ref:`setTimePrecision<doxid-da/dfd/classModelSpec_1a379793c6fcbe1f834ad18cf4c5789537>`(:ref:`TimePrecision<doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8f>` timePrecision);
		void :ref:`setDT<doxid-da/dfd/classModelSpec_1a329236a3b07044b82bfda5b4f741d8e1>`(double dt);
		void :ref:`setTiming<doxid-da/dfd/classModelSpec_1ae1678fdcd6c8381a402c58673064fa6a>`(bool timingEnabled);
		void :ref:`setSeed<doxid-da/dfd/classModelSpec_1a1c6bc48d22a8f7b3fb70b46a4ca87646>`(unsigned int rngSeed);
		void :ref:`setDefaultVarLocation<doxid-da/dfd/classModelSpec_1a55c87917355d34463a3c19fc6887e67a>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setDefaultExtraGlobalParamLocation<doxid-da/dfd/classModelSpec_1aa0d462099083f12bc9f98b9b0fb86d64>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setDefaultSparseConnectivityLocation<doxid-da/dfd/classModelSpec_1a9bc61e7c5dce757de3a9b7479852ca72>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setMergePostsynapticModels<doxid-da/dfd/classModelSpec_1ac40ba51579f94af5ca7fa51c4f67fe8f>`(bool merge);
		const std::string& :ref:`getName<doxid-da/dfd/classModelSpec_1a4a4fc34aa4a09b384bd0ea7134bd49fa>`() const;
		const std::string& :ref:`getPrecision<doxid-da/dfd/classModelSpec_1aafede0edcd474eb000230b63b8c1562d>`() const;
		std::string :ref:`getTimePrecision<doxid-da/dfd/classModelSpec_1aa02041a6c0e06ba384a2bbf5e8d925a6>`() const;
		double :ref:`getDT<doxid-da/dfd/classModelSpec_1a79796e2faf44bc12bb716cb376603fc2>`() const;
		unsigned int :ref:`getSeed<doxid-da/dfd/classModelSpec_1a4f032e3eb72f40ea3dcdafee8e0ad289>`() const;
		bool :ref:`isTimingEnabled<doxid-da/dfd/classModelSpec_1a2ba95653dc8c75539a7e582e66999f2e>`() const;
		unsigned int :ref:`getNumLocalNeurons<doxid-da/dfd/classModelSpec_1a46ab18306ec13a61d4aff75645b8646e>`() const;
		unsigned int :ref:`getNumRemoteNeurons<doxid-da/dfd/classModelSpec_1aaf562353e81f02732b6bb49311f16dda>`() const;
		unsigned int :ref:`getNumNeurons<doxid-da/dfd/classModelSpec_1a453b3a92fb74742103d48cbf81fa47bb>`() const;
		:ref:`NeuronGroup<doxid-d7/d3b/classNeuronGroup>`* :ref:`findNeuronGroup<doxid-da/dfd/classModelSpec_1a7508ff35c5957bf8a4385168fed50e2c>`(const std::string& name);
	
		template <typename NeuronModel>
		:ref:`NeuronGroup<doxid-d7/d3b/classNeuronGroup>`* :ref:`addNeuronPopulation<doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b>`(
			const std::string& name,
			unsigned int size,
			const NeuronModel* model,
			const typename NeuronModel::ParamValues& paramValues,
			const typename NeuronModel::VarValues& varInitialisers,
			int hostID = 0
			);
	
		template <typename NeuronModel>
		:ref:`NeuronGroup<doxid-d7/d3b/classNeuronGroup>`* :ref:`addNeuronPopulation<doxid-da/dfd/classModelSpec_1a5eec26674996c3504f1c85b1e190f82f>`(
			const std::string& name,
			unsigned int size,
			const typename NeuronModel::ParamValues& paramValues,
			const typename NeuronModel::VarValues& varInitialisers,
			int hostID = 0
			);
	
		:ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`* :ref:`findSynapseGroup<doxid-da/dfd/classModelSpec_1ac61c646d1c4e5a56a8e1b5cf81de9088>`(const std::string& name);
	
		template <typename WeightUpdateModel, typename PostsynapticModel>
		:ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`* :ref:`addSynapsePopulation<doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2>`(
			const std::string& name,
			:ref:`SynapseMatrixType<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9c>` mtype,
			unsigned int delaySteps,
			const std::string& src,
			const std::string& trg,
			const WeightUpdateModel* wum,
			const typename WeightUpdateModel::ParamValues& weightParamValues,
			const typename WeightUpdateModel::VarValues& weightVarInitialisers,
			const typename WeightUpdateModel::PreVarValues& weightPreVarInitialisers,
			const typename WeightUpdateModel::PostVarValues& weightPostVarInitialisers,
			const PostsynapticModel* psm,
			const typename PostsynapticModel::ParamValues& postsynapticParamValues,
			const typename PostsynapticModel::VarValues& postsynapticVarInitialisers,
			const :ref:`InitSparseConnectivitySnippet::Init<doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>`& connectivityInitialiser = :ref:`uninitialisedConnectivity<doxid-dc/de1/modelSpec_8h_1a367c112babcc14b58db730731b798073>`()
			);
	
		template <typename WeightUpdateModel, typename PostsynapticModel>
		:ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`* :ref:`addSynapsePopulation<doxid-da/dfd/classModelSpec_1a0bde9e959e2d306b6af799e5b9fb9eaa>`(
			const std::string& name,
			:ref:`SynapseMatrixType<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9c>` mtype,
			unsigned int delaySteps,
			const std::string& src,
			const std::string& trg,
			const typename WeightUpdateModel::ParamValues& weightParamValues,
			const typename WeightUpdateModel::VarValues& weightVarInitialisers,
			const typename PostsynapticModel::ParamValues& postsynapticParamValues,
			const typename PostsynapticModel::VarValues& postsynapticVarInitialisers,
			const :ref:`InitSparseConnectivitySnippet::Init<doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>`& connectivityInitialiser = :ref:`uninitialisedConnectivity<doxid-dc/de1/modelSpec_8h_1a367c112babcc14b58db730731b798073>`()
			);
	
		template <typename WeightUpdateModel, typename PostsynapticModel>
		:ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`* :ref:`addSynapsePopulation<doxid-da/dfd/classModelSpec_1a9c0d07277fbdedf094f06279d13f4d54>`(
			const std::string& name,
			:ref:`SynapseMatrixType<doxid-dd/dd5/synapseMatrixType_8h_1a24a045033b9a7e987843a67ff5ddec9c>` mtype,
			unsigned int delaySteps,
			const std::string& src,
			const std::string& trg,
			const typename WeightUpdateModel::ParamValues& weightParamValues,
			const typename WeightUpdateModel::VarValues& weightVarInitialisers,
			const typename WeightUpdateModel::PreVarValues& weightPreVarInitialisers,
			const typename WeightUpdateModel::PostVarValues& weightPostVarInitialisers,
			const typename PostsynapticModel::ParamValues& postsynapticParamValues,
			const typename PostsynapticModel::VarValues& postsynapticVarInitialisers,
			const :ref:`InitSparseConnectivitySnippet::Init<doxid-d2/d7f/classInitSparseConnectivitySnippet_1_1Init>`& connectivityInitialiser = :ref:`uninitialisedConnectivity<doxid-dc/de1/modelSpec_8h_1a367c112babcc14b58db730731b798073>`()
			);
	
		:ref:`CurrentSource<doxid-d1/d48/classCurrentSource>`* :ref:`findCurrentSource<doxid-da/dfd/classModelSpec_1a1f9d972f4f93c65dd254a27992980600>`(const std::string& name);
	
		template <typename CurrentSourceModel>
		:ref:`CurrentSource<doxid-d1/d48/classCurrentSource>`* :ref:`addCurrentSource<doxid-da/dfd/classModelSpec_1aaf260ae8ffd52473b61a27974867c3e3>`(
			const std::string& currentSourceName,
			const CurrentSourceModel* model,
			const std::string& targetNeuronGroupName,
			const typename CurrentSourceModel::ParamValues& paramValues,
			const typename CurrentSourceModel::VarValues& varInitialisers
			);
	
		template <typename CurrentSourceModel>
		:ref:`CurrentSource<doxid-d1/d48/classCurrentSource>`* :ref:`addCurrentSource<doxid-da/dfd/classModelSpec_1a54bfff6bcd9ae2bf4d3424177a68265c>`(
			const std::string& currentSourceName,
			const std::string& targetNeuronGroupName,
			const typename CurrentSourceModel::ParamValues& paramValues,
			const typename CurrentSourceModel::VarValues& varInitialisers
			);


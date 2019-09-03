.. index:: pair: class; ModelSpec
.. _doxid-da/dfd/classModelSpec:

class ModelSpec
===============

.. toctree::
	:hidden:

Overview
~~~~~~~~

Object used for specifying a neuronal network model. :ref:`More...<details-da/dfd/classModelSpec>`


.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <modelSpec.h>
	
	class ModelSpec
	{
	public:
		// typedefs
	
		typedef std::map<std::string, :ref:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal>`>::value_type :target:`NeuronGroupValueType<doxid-da/dfd/classModelSpec_1ac724c12166b5ee5fb4492277c1d8deb5>`;
		typedef std::map<std::string, :ref:`SynapseGroupInternal<doxid-dd/d48/classSynapseGroupInternal>`>::value_type :target:`SynapseGroupValueType<doxid-da/dfd/classModelSpec_1a8af14d4e037788bb97854c4d0da801df>`;

		// methods
	
		:target:`ModelSpec<doxid-da/dfd/classModelSpec_1a5f71e866bd0b2af2abf2b1e8dde7d6d4>`();
		:target:`ModelSpec<doxid-da/dfd/classModelSpec_1ab5f3d20c830593a4452be5878e43bba8>`(const ModelSpec&);
		ModelSpec& :target:`operator =<doxid-da/dfd/classModelSpec_1aaf415f0379159d74d57d18126cf2982e>` (const ModelSpec&);
		:target:`~ModelSpec<doxid-da/dfd/classModelSpec_1a60ffaa6deb779cff61da6a7ea651613f>`();
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
	};

	// direct descendants

	class :ref:`ModelSpecInternal<doxid-dc/dfa/classModelSpecInternal>`;
.. _details-da/dfd/classModelSpec:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~

Object used for specifying a neuronal network model.

Methods
-------

.. index:: pair: function; setName
.. _doxid-da/dfd/classModelSpec_1ada1aff7a94eeb36dff721f09d5cf94b4:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setName(const std::string& name)

Method to set the neuronal network model name.

.. index:: pair: function; setPrecision
.. _doxid-da/dfd/classModelSpec_1a7548f1bf634884c051e4fbac3cf6212c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setPrecision(:ref:`FloatType<doxid-dc/de1/modelSpec_8h_1aa039815ec6b74d0fe4cb016415781c08>` floattype)

Set numerical precision for floating point.

This function sets the numerical precision of floating type variables. By default, it is GENN_GENN_FLOAT.

.. index:: pair: function; setTimePrecision
.. _doxid-da/dfd/classModelSpec_1a379793c6fcbe1f834ad18cf4c5789537:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setTimePrecision(:ref:`TimePrecision<doxid-dc/de1/modelSpec_8h_1a71ece086a364ee04c7ffc3f626218b8f>` timePrecision)

Set numerical precision for time.

.. index:: pair: function; setDT
.. _doxid-da/dfd/classModelSpec_1a329236a3b07044b82bfda5b4f741d8e1:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setDT(double dt)

Set the integration step size of the model.

.. index:: pair: function; setTiming
.. _doxid-da/dfd/classModelSpec_1ae1678fdcd6c8381a402c58673064fa6a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setTiming(bool timingEnabled)

Set whether timers and timing commands are to be included.

.. index:: pair: function; setSeed
.. _doxid-da/dfd/classModelSpec_1a1c6bc48d22a8f7b3fb70b46a4ca87646:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setSeed(unsigned int rngSeed)

Set the random seed (disables automatic seeding if argument not 0).

.. index:: pair: function; setDefaultVarLocation
.. _doxid-da/dfd/classModelSpec_1a55c87917355d34463a3c19fc6887e67a:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setDefaultVarLocation(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc)

What is the default location for model state variables?

Historically, everything was allocated on both the host AND device

.. index:: pair: function; setDefaultExtraGlobalParamLocation
.. _doxid-da/dfd/classModelSpec_1aa0d462099083f12bc9f98b9b0fb86d64:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setDefaultExtraGlobalParamLocation(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc)

What is the default location for model extra global parameters?

Historically, this was just left up to the user to handle

.. index:: pair: function; setDefaultSparseConnectivityLocation
.. _doxid-da/dfd/classModelSpec_1a9bc61e7c5dce757de3a9b7479852ca72:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setDefaultSparseConnectivityLocation(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc)

What is the default location for sparse synaptic connectivity?

Historically, everything was allocated on both the host AND device

.. index:: pair: function; setMergePostsynapticModels
.. _doxid-da/dfd/classModelSpec_1ac40ba51579f94af5ca7fa51c4f67fe8f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setMergePostsynapticModels(bool merge)

Should compatible postsynaptic models and dendritic delay buffers be merged?

This can significantly reduce the cost of updating neuron population but means that per-synapse group inSyn arrays can not be retrieved

.. index:: pair: function; getName
.. _doxid-da/dfd/classModelSpec_1a4a4fc34aa4a09b384bd0ea7134bd49fa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const std::string& getName() const

Gets the name of the neuronal network model.

.. index:: pair: function; getPrecision
.. _doxid-da/dfd/classModelSpec_1aafede0edcd474eb000230b63b8c1562d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const std::string& getPrecision() const

Gets the floating point numerical precision.

.. index:: pair: function; getTimePrecision
.. _doxid-da/dfd/classModelSpec_1aa02041a6c0e06ba384a2bbf5e8d925a6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	std::string getTimePrecision() const

Gets the floating point numerical precision used to represent time.

.. index:: pair: function; getDT
.. _doxid-da/dfd/classModelSpec_1a79796e2faf44bc12bb716cb376603fc2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	double getDT() const

Gets the model integration step size.

.. index:: pair: function; getSeed
.. _doxid-da/dfd/classModelSpec_1a4f032e3eb72f40ea3dcdafee8e0ad289:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unsigned int getSeed() const

Get the random seed.

.. index:: pair: function; isTimingEnabled
.. _doxid-da/dfd/classModelSpec_1a2ba95653dc8c75539a7e582e66999f2e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	bool isTimingEnabled() const

Are timers and timing commands enabled.

.. index:: pair: function; getNumLocalNeurons
.. _doxid-da/dfd/classModelSpec_1a46ab18306ec13a61d4aff75645b8646e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unsigned int getNumLocalNeurons() const

How many neurons are simulated locally in this model.

.. index:: pair: function; getNumRemoteNeurons
.. _doxid-da/dfd/classModelSpec_1aaf562353e81f02732b6bb49311f16dda:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unsigned int getNumRemoteNeurons() const

How many neurons are simulated remotely in this model.

.. index:: pair: function; getNumNeurons
.. _doxid-da/dfd/classModelSpec_1a453b3a92fb74742103d48cbf81fa47bb:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	unsigned int getNumNeurons() const

How many neurons make up the entire model.

.. index:: pair: function; findNeuronGroup
.. _doxid-da/dfd/classModelSpec_1a7508ff35c5957bf8a4385168fed50e2c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`NeuronGroup<doxid-d7/d3b/classNeuronGroup>`* findNeuronGroup(const std::string& name)

Find a neuron group by name.

.. index:: pair: function; addNeuronPopulation
.. _doxid-da/dfd/classModelSpec_1a0b765be273f3c6cec15092d7dbfdd52b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename NeuronModel>
	:ref:`NeuronGroup<doxid-d7/d3b/classNeuronGroup>`* addNeuronPopulation(
		const std::string& name,
		unsigned int size,
		const NeuronModel* model,
		const typename NeuronModel::ParamValues& paramValues,
		const typename NeuronModel::VarValues& varInitialisers,
		int hostID = 0
		)

Adds a new neuron group to the model using a neuron model managed by the user.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- NeuronModel

		- type of neuron model (derived from :ref:`NeuronModels::Base <doxid-d7/dad/classNeuronModels_1_1Base>`).

	*
		- name

		- string containing unique name of neuron population.

	*
		- size

		- integer specifying how many neurons are in the population.

	*
		- model

		- neuron model to use for neuron group.

	*
		- paramValues

		- parameters for model wrapped in NeuronModel::ParamValues object.

	*
		- varInitialisers

		- state variable initialiser snippets and parameters wrapped in NeuronModel::VarValues object.

	*
		- hostID

		- if using MPI, the ID of the node to simulate this population on.



.. rubric:: Returns:

pointer to newly created :ref:`NeuronGroup <doxid-d7/d3b/classNeuronGroup>`

.. index:: pair: function; addNeuronPopulation
.. _doxid-da/dfd/classModelSpec_1a5eec26674996c3504f1c85b1e190f82f:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename NeuronModel>
	:ref:`NeuronGroup<doxid-d7/d3b/classNeuronGroup>`* addNeuronPopulation(
		const std::string& name,
		unsigned int size,
		const typename NeuronModel::ParamValues& paramValues,
		const typename NeuronModel::VarValues& varInitialisers,
		int hostID = 0
		)

Adds a new neuron group to the model using a singleton neuron model created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- NeuronModel

		- type of neuron model (derived from :ref:`NeuronModels::Base <doxid-d7/dad/classNeuronModels_1_1Base>`).

	*
		- name

		- string containing unique name of neuron population.

	*
		- size

		- integer specifying how many neurons are in the population.

	*
		- paramValues

		- parameters for model wrapped in NeuronModel::ParamValues object.

	*
		- varInitialisers

		- state variable initialiser snippets and parameters wrapped in NeuronModel::VarValues object.

	*
		- hostID

		- if using MPI, the ID of the node to simulate this population on.



.. rubric:: Returns:

pointer to newly created :ref:`NeuronGroup <doxid-d7/d3b/classNeuronGroup>`

.. index:: pair: function; findSynapseGroup
.. _doxid-da/dfd/classModelSpec_1ac61c646d1c4e5a56a8e1b5cf81de9088:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`* findSynapseGroup(const std::string& name)

Find a synapse group by name.

.. index:: pair: function; addSynapsePopulation
.. _doxid-da/dfd/classModelSpec_1abd4e9128a5d4f5f993907134218af0c2:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename WeightUpdateModel, typename PostsynapticModel>
	:ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`* addSynapsePopulation(
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
		)

Adds a synapse population to the model using weight update and postsynaptic models managed by the user.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- WeightUpdateModel

		- type of weight update model (derived from :ref:`WeightUpdateModels::Base <doxid-d2/d05/classWeightUpdateModels_1_1Base>`).

	*
		- PostsynapticModel

		- type of postsynaptic model (derived from :ref:`PostsynapticModels::Base <doxid-d1/d3a/classPostsynapticModels_1_1Base>`).

	*
		- name

		- string containing unique name of neuron population.

	*
		- mtype

		- how the synaptic matrix associated with this synapse population should be represented.

	*
		- delaySteps

		- integer specifying number of timesteps delay this synaptic connection should incur (or NO_DELAY for none)

	*
		- src

		- string specifying name of presynaptic (source) population

	*
		- trg

		- string specifying name of postsynaptic (target) population

	*
		- wum

		- weight update model to use for synapse group.

	*
		- weightParamValues

		- parameters for weight update model wrapped in WeightUpdateModel::ParamValues object.

	*
		- weightVarInitialisers

		- weight update model state variable initialiser snippets and parameters wrapped in WeightUpdateModel::VarValues object.

	*
		- weightPreVarInitialisers

		- weight update model presynaptic state variable initialiser snippets and parameters wrapped in WeightUpdateModel::VarValues object.

	*
		- weightPostVarInitialisers

		- weight update model postsynaptic state variable initialiser snippets and parameters wrapped in WeightUpdateModel::VarValues object.

	*
		- psm

		- postsynaptic model to use for synapse group.

	*
		- postsynapticParamValues

		- parameters for postsynaptic model wrapped in PostsynapticModel::ParamValues object.

	*
		- postsynapticVarInitialisers

		- postsynaptic model state variable initialiser snippets and parameters wrapped in NeuronModel::VarValues object.

	*
		- connectivityInitialiser

		- sparse connectivity initialisation snippet used to initialise connectivity for :ref:`SynapseMatrixConnectivity::SPARSE <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0459833ba9cad7cfd7bbfe10d7bbbe6e>` or :ref:`SynapseMatrixConnectivity::BITMASK <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0287e103671bf22378919a64d4b70699>`. Typically wrapped with it's parameters using ``initConnectivity`` function



.. rubric:: Returns:

pointer to newly created :ref:`SynapseGroup <doxid-dc/dfa/classSynapseGroup>`

.. index:: pair: function; addSynapsePopulation
.. _doxid-da/dfd/classModelSpec_1a0bde9e959e2d306b6af799e5b9fb9eaa:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename WeightUpdateModel, typename PostsynapticModel>
	:ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`* addSynapsePopulation(
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
		)

Adds a synapse population to the model using singleton weight update and postsynaptic models created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- WeightUpdateModel

		- type of weight update model (derived from :ref:`WeightUpdateModels::Base <doxid-d2/d05/classWeightUpdateModels_1_1Base>`).

	*
		- PostsynapticModel

		- type of postsynaptic model (derived from :ref:`PostsynapticModels::Base <doxid-d1/d3a/classPostsynapticModels_1_1Base>`).

	*
		- name

		- string containing unique name of neuron population.

	*
		- mtype

		- how the synaptic matrix associated with this synapse population should be represented.

	*
		- delaySteps

		- integer specifying number of timesteps delay this synaptic connection should incur (or NO_DELAY for none)

	*
		- src

		- string specifying name of presynaptic (source) population

	*
		- trg

		- string specifying name of postsynaptic (target) population

	*
		- weightParamValues

		- parameters for weight update model wrapped in WeightUpdateModel::ParamValues object.

	*
		- weightVarInitialisers

		- weight update model state variable initialiser snippets and parameters wrapped in WeightUpdateModel::VarValues object.

	*
		- postsynapticParamValues

		- parameters for postsynaptic model wrapped in PostsynapticModel::ParamValues object.

	*
		- postsynapticVarInitialisers

		- postsynaptic model state variable initialiser snippets and parameters wrapped in NeuronModel::VarValues object.

	*
		- connectivityInitialiser

		- sparse connectivity initialisation snippet used to initialise connectivity for :ref:`SynapseMatrixConnectivity::SPARSE <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0459833ba9cad7cfd7bbfe10d7bbbe6e>` or :ref:`SynapseMatrixConnectivity::BITMASK <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0287e103671bf22378919a64d4b70699>`. Typically wrapped with it's parameters using ``initConnectivity`` function



.. rubric:: Returns:

pointer to newly created :ref:`SynapseGroup <doxid-dc/dfa/classSynapseGroup>`

.. index:: pair: function; addSynapsePopulation
.. _doxid-da/dfd/classModelSpec_1a9c0d07277fbdedf094f06279d13f4d54:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename WeightUpdateModel, typename PostsynapticModel>
	:ref:`SynapseGroup<doxid-dc/dfa/classSynapseGroup>`* addSynapsePopulation(
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
		)

Adds a synapse population to the model using singleton weight update and postsynaptic models created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- WeightUpdateModel

		- type of weight update model (derived from :ref:`WeightUpdateModels::Base <doxid-d2/d05/classWeightUpdateModels_1_1Base>`).

	*
		- PostsynapticModel

		- type of postsynaptic model (derived from :ref:`PostsynapticModels::Base <doxid-d1/d3a/classPostsynapticModels_1_1Base>`).

	*
		- name

		- string containing unique name of neuron population.

	*
		- mtype

		- how the synaptic matrix associated with this synapse population should be represented.

	*
		- delaySteps

		- integer specifying number of timesteps delay this synaptic connection should incur (or NO_DELAY for none)

	*
		- src

		- string specifying name of presynaptic (source) population

	*
		- trg

		- string specifying name of postsynaptic (target) population

	*
		- weightParamValues

		- parameters for weight update model wrapped in WeightUpdateModel::ParamValues object.

	*
		- weightVarInitialisers

		- weight update model per-synapse state variable initialiser snippets and parameters wrapped in WeightUpdateModel::VarValues object.

	*
		- weightPreVarInitialisers

		- weight update model presynaptic state variable initialiser snippets and parameters wrapped in WeightUpdateModel::VarValues object.

	*
		- weightPostVarInitialisers

		- weight update model postsynaptic state variable initialiser snippets and parameters wrapped in WeightUpdateModel::VarValues object.

	*
		- postsynapticParamValues

		- parameters for postsynaptic model wrapped in PostsynapticModel::ParamValues object.

	*
		- postsynapticVarInitialisers

		- postsynaptic model state variable initialiser snippets and parameters wrapped in NeuronModel::VarValues object.

	*
		- connectivityInitialiser

		- sparse connectivity initialisation snippet used to initialise connectivity for :ref:`SynapseMatrixConnectivity::SPARSE <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0459833ba9cad7cfd7bbfe10d7bbbe6e>` or :ref:`SynapseMatrixConnectivity::BITMASK <doxid-dd/dd5/synapseMatrixType_8h_1aedb0946699027562bc78103a5d2a578da0287e103671bf22378919a64d4b70699>`. Typically wrapped with it's parameters using ``initConnectivity`` function



.. rubric:: Returns:

pointer to newly created :ref:`SynapseGroup <doxid-dc/dfa/classSynapseGroup>`

.. index:: pair: function; findCurrentSource
.. _doxid-da/dfd/classModelSpec_1a1f9d972f4f93c65dd254a27992980600:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`CurrentSource<doxid-d1/d48/classCurrentSource>`* findCurrentSource(const std::string& name)

Find a current source by name.

This function attempts to find an existing current source.

.. index:: pair: function; addCurrentSource
.. _doxid-da/dfd/classModelSpec_1aaf260ae8ffd52473b61a27974867c3e3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename CurrentSourceModel>
	:ref:`CurrentSource<doxid-d1/d48/classCurrentSource>`* addCurrentSource(
		const std::string& currentSourceName,
		const CurrentSourceModel* model,
		const std::string& targetNeuronGroupName,
		const typename CurrentSourceModel::ParamValues& paramValues,
		const typename CurrentSourceModel::VarValues& varInitialisers
		)

Adds a new current source to the model using a current source model managed by the user.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- CurrentSourceModel

		- type of current source model (derived from :ref:`CurrentSourceModels::Base <doxid-d0/de0/classCurrentSourceModels_1_1Base>`).

	*
		- currentSourceName

		- string containing unique name of current source.

	*
		- model

		- current source model to use for current source.

	*
		- targetNeuronGroupName

		- string name of the target neuron group

	*
		- paramValues

		- parameters for model wrapped in CurrentSourceModel::ParamValues object.

	*
		- varInitialisers

		- state variable initialiser snippets and parameters wrapped in CurrentSource::VarValues object.



.. rubric:: Returns:

pointer to newly created :ref:`CurrentSource <doxid-d1/d48/classCurrentSource>`

.. index:: pair: function; addCurrentSource
.. _doxid-da/dfd/classModelSpec_1a54bfff6bcd9ae2bf4d3424177a68265c:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	template <typename CurrentSourceModel>
	:ref:`CurrentSource<doxid-d1/d48/classCurrentSource>`* addCurrentSource(
		const std::string& currentSourceName,
		const std::string& targetNeuronGroupName,
		const typename CurrentSourceModel::ParamValues& paramValues,
		const typename CurrentSourceModel::VarValues& varInitialisers
		)

Adds a new current source to the model using a singleton current source model created using standard DECLARE_MODEL and IMPLEMENT_MODEL macros.



.. rubric:: Parameters:

.. list-table::
	:widths: 20 80

	*
		- CurrentSourceModel

		- type of neuron model (derived from CurrentSourceModel::Base).

	*
		- currentSourceName

		- string containing unique name of current source.

	*
		- targetNeuronGroupName

		- string name of the target neuron group

	*
		- paramValues

		- parameters for model wrapped in CurrentSourceModel::ParamValues object.

	*
		- varInitialisers

		- state variable initialiser snippets and parameters wrapped in CurrentSourceModel::VarValues object.



.. rubric:: Returns:

pointer to newly created :ref:`CurrentSource <doxid-d1/d48/classCurrentSource>`


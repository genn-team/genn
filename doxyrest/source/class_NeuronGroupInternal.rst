.. index:: pair: class; NeuronGroupInternal
.. _doxid-dc/da3/classNeuronGroupInternal:

class NeuronGroupInternal
=========================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <neuronGroupInternal.h>
	
	class NeuronGroupInternal: public :ref:`NeuronGroup<doxid-d7/d3b/classNeuronGroup>`
	{
	public:
		// construction
	
		:target:`NeuronGroupInternal<doxid-dc/da3/classNeuronGroupInternal_1a10d1a8498a9bd91151ff1b4b3c480a95>`(
			const std::string& name,
			int numNeurons,
			const :ref:`NeuronModels::Base<doxid-d7/dad/classNeuronModels_1_1Base>`* neuronModel,
			const std::vector<double>& params,
			const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& varInitialisers,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` defaultVarLocation,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` defaultExtraGlobalParamLocation,
			int hostID
			);
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// methods
	
		void :ref:`setSpikeLocation<doxid-d7/d3b/classNeuronGroup_1a9df1df6d85dde4a46ddef63954828a95>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setSpikeEventLocation<doxid-d7/d3b/classNeuronGroup_1a95f0660e93790ea764119002db68f706>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setSpikeTimeLocation<doxid-d7/d3b/classNeuronGroup_1a63004d6ff9f5b2982ef401e95314d531>`(:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setVarLocation<doxid-d7/d3b/classNeuronGroup_1a75951040bc142c60c4f0b5a8aa84bd57>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setExtraGlobalParamLocation<doxid-d7/d3b/classNeuronGroup_1a9f54ec7c3dbf68196a62c2c953eeccd4>`(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		const std::string& :ref:`getName<doxid-d7/d3b/classNeuronGroup_1a5ca1529217cfe4c8b7858cae99ec3315>`() const;
		unsigned int :ref:`getNumNeurons<doxid-d7/d3b/classNeuronGroup_1aa4b9215c3c33eadc85c233d403034ac8>`() const;
		const :ref:`NeuronModels::Base<doxid-d7/dad/classNeuronModels_1_1Base>`* :ref:`getNeuronModel<doxid-d7/d3b/classNeuronGroup_1a69fb51cd9c6fd422fd0bc9ea19204d7b>`() const;
		const std::vector<double>& :ref:`getParams<doxid-d7/d3b/classNeuronGroup_1a230ada09fa356b73c897d3999a55bd9c>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getVarInitialisers<doxid-d7/d3b/classNeuronGroup_1a2217c2f1119bff7d7a471312d8d51004>`() const;
		int :ref:`getClusterHostID<doxid-d7/d3b/classNeuronGroup_1a578cd96df28b2e7b606aee6c7b51706d>`() const;
		bool :ref:`isSpikeTimeRequired<doxid-d7/d3b/classNeuronGroup_1a46de5e06d867bc56b257f69def0698d5>`() const;
		bool :ref:`isTrueSpikeRequired<doxid-d7/d3b/classNeuronGroup_1a779ffce6e38704ae33ae45affbffdaa2>`() const;
		bool :ref:`isSpikeEventRequired<doxid-d7/d3b/classNeuronGroup_1af3cc12aa18eaf34faa2297a16f257fab>`() const;
		unsigned int :ref:`getNumDelaySlots<doxid-d7/d3b/classNeuronGroup_1af165bbd3f1269def4e667f204d6fbda4>`() const;
		bool :ref:`isDelayRequired<doxid-d7/d3b/classNeuronGroup_1aaee088799cba1183037c1a412dbf6be6>`() const;
		bool :ref:`isZeroCopyEnabled<doxid-d7/d3b/classNeuronGroup_1a808eebf23213351d0972b5765e29775d>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getSpikeLocation<doxid-d7/d3b/classNeuronGroup_1ac9d0acacfc7fcbfb793bc71c9b5df302>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getSpikeEventLocation<doxid-d7/d3b/classNeuronGroup_1ab29e95a81d84dbd352794ed30e6bcd18>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getSpikeTimeLocation<doxid-d7/d3b/classNeuronGroup_1a65601133dc934a9c46462e43e6d096dc>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d7/d3b/classNeuronGroup_1adcbb2b02e02286d4206e94bc3905883b>`(const std::string& varName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d7/d3b/classNeuronGroup_1a5f40746255edeee5e14404ef71921846>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d7/d3b/classNeuronGroup_1a2a98e9145a85b0d54ed7593a9572823c>`(const std::string& paramName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d7/d3b/classNeuronGroup_1a13e3e1fba765cbcf3a5424c7a6ea53a0>`(size_t index) const;
		bool :ref:`isSimRNGRequired<doxid-d7/d3b/classNeuronGroup_1a193c659939f14aa5442295804f203477>`() const;
		bool :ref:`isInitRNGRequired<doxid-d7/d3b/classNeuronGroup_1a3377276c4239b87c72ba7a538fcca825>`() const;
		bool :ref:`hasOutputToHost<doxid-d7/d3b/classNeuronGroup_1a3cb183967be00b695b931bf3f4659076>`(int targetHostID) const;


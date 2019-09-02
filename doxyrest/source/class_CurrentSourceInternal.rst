.. index:: pair: class; CurrentSourceInternal
.. _doxid-d6/de6/classCurrentSourceInternal:

class CurrentSourceInternal
===========================

.. toctree::
	:hidden:




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <currentSourceInternal.h>
	
	class CurrentSourceInternal: public :ref:`CurrentSource<doxid-d1/d48/classCurrentSource>`
	{
	public:
		// construction
	
		:target:`CurrentSourceInternal<doxid-d6/de6/classCurrentSourceInternal_1a64ac197ca7840535908cb73f9799d8fa>`(
			const std::string& name,
			const :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`* currentSourceModel,
			const std::vector<double>& params,
			const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& varInitialisers,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` defaultVarLocation,
			:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` defaultExtraGlobalParamLocation
			);
	};

Inherited Members
-----------------

.. ref-code-block:: cpp
	:class: doxyrest-overview-inherited-code-block

	public:
		// methods
	
		void :ref:`setVarLocation<doxid-d1/d48/classCurrentSource_1aed244c33e6e0830d203f462d71cd949e>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1a43687bec0ce75867db3661b587e2b6b9>`(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		const std::string& :ref:`getName<doxid-d1/d48/classCurrentSource_1a59dd0ff630ed03251751ec2634457753>`() const;
		const :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`* :ref:`getCurrentSourceModel<doxid-d1/d48/classCurrentSource_1a4db1f74f237b66e235981ea290d86ada>`() const;
		const std::vector<double>& :ref:`getParams<doxid-d1/d48/classCurrentSource_1a020b8cd27da6eaadbac238ff5dbc016a>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getVarInitialisers<doxid-d1/d48/classCurrentSource_1af0e876921169fc46d0d801dc9514688a>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d1/d48/classCurrentSource_1a0b5061240ea86ae09f52bf128b158aed>`(const std::string& varName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d1/d48/classCurrentSource_1a4082262b23f60d230840455ffbb8c5f3>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1ac589d53c052cf96ff6eb4f10adca997d>`(const std::string& paramName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1aeb8c8e3d3ce51295c8ccbc6386640a5b>`(size_t index) const;


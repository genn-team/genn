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
		// methods
	
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
	
		:ref:`CurrentSource<doxid-d1/d48/classCurrentSource_1a23699d48b18506030c0d5afdf72ffd20>`(const :ref:`CurrentSource<doxid-d1/d48/classCurrentSource>`&);
		:ref:`CurrentSource<doxid-d1/d48/classCurrentSource_1a837a3749724b52ad1ab76751b2464c21>`();
		void :ref:`setVarLocation<doxid-d1/d48/classCurrentSource_1aed244c33e6e0830d203f462d71cd949e>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1a43687bec0ce75867db3661b587e2b6b9>`(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		const std::string& :ref:`getName<doxid-d1/d48/classCurrentSource_1a4efd934fc92578118b4a7935051071ab>`() const;
		const :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`* :ref:`getCurrentSourceModel<doxid-d1/d48/classCurrentSource_1adede32230bc2e0264ae6562f5cdc7272>`() const;
		const std::vector<double>& :ref:`getParams<doxid-d1/d48/classCurrentSource_1aafd49bcc780ba345747091af04d72885>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :ref:`getVarInitialisers<doxid-d1/d48/classCurrentSource_1aab4ea708233ebab9f90876d4c5def063>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d1/d48/classCurrentSource_1aaa778782b59a8cb3e88b59644c7fd4ba>`(const std::string& varName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d1/d48/classCurrentSource_1a68dd0c0732470342dc8392309a471207>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1a06d4317ea9bf386dbab1d4730f4558f6>`(const std::string& paramName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1ab723e7326963f12bbdd05da760bf579b>`(size_t index) const;


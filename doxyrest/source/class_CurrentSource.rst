.. index:: pair: class; CurrentSource
.. _doxid-d1/d48/classCurrentSource:

class CurrentSource
===================

.. toctree::
	:hidden:

Overview
~~~~~~~~




.. ref-code-block:: cpp
	:class: doxyrest-overview-code-block

	#include <currentSource.h>
	
	class CurrentSource
	{
	public:
		// methods
	
		:target:`CurrentSource<doxid-d1/d48/classCurrentSource_1a23699d48b18506030c0d5afdf72ffd20>`(const CurrentSource&);
		:target:`CurrentSource<doxid-d1/d48/classCurrentSource_1a837a3749724b52ad1ab76751b2464c21>`();
		void :ref:`setVarLocation<doxid-d1/d48/classCurrentSource_1aed244c33e6e0830d203f462d71cd949e>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1a43687bec0ce75867db3661b587e2b6b9>`(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		const std::string& :target:`getName<doxid-d1/d48/classCurrentSource_1a4efd934fc92578118b4a7935051071ab>`() const;
		const :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`* :ref:`getCurrentSourceModel<doxid-d1/d48/classCurrentSource_1adede32230bc2e0264ae6562f5cdc7272>`() const;
		const std::vector<double>& :target:`getParams<doxid-d1/d48/classCurrentSource_1aafd49bcc780ba345747091af04d72885>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :target:`getVarInitialisers<doxid-d1/d48/classCurrentSource_1aab4ea708233ebab9f90876d4c5def063>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d1/d48/classCurrentSource_1aaa778782b59a8cb3e88b59644c7fd4ba>`(const std::string& varName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d1/d48/classCurrentSource_1a68dd0c0732470342dc8392309a471207>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1a06d4317ea9bf386dbab1d4730f4558f6>`(const std::string& paramName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1ab723e7326963f12bbdd05da760bf579b>`(size_t index) const;
	};

	// direct descendants

	class :ref:`CurrentSourceInternal<doxid-d6/de6/classCurrentSourceInternal>`;
.. _details-d1/d48/classCurrentSource:

Detailed Documentation
~~~~~~~~~~~~~~~~~~~~~~



Methods
-------

.. index:: pair: function; setVarLocation
.. _doxid-d1/d48/classCurrentSource_1aed244c33e6e0830d203f462d71cd949e:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setVarLocation(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc)

Set location of current source state variable.

.. index:: pair: function; setExtraGlobalParamLocation
.. _doxid-d1/d48/classCurrentSource_1a43687bec0ce75867db3661b587e2b6b9:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	void setExtraGlobalParamLocation(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc)

Set location of extra global parameter.

This is ignored for simulations on hardware with a single memory space and only applies to extra global parameters which are pointers.

.. index:: pair: function; getCurrentSourceModel
.. _doxid-d1/d48/classCurrentSource_1adede32230bc2e0264ae6562f5cdc7272:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`* getCurrentSourceModel() const

Gets the current source model used by this group.

.. index:: pair: function; getVarLocation
.. _doxid-d1/d48/classCurrentSource_1aaa778782b59a8cb3e88b59644c7fd4ba:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` getVarLocation(const std::string& varName) const

Get variable location for current source model state variable.

.. index:: pair: function; getVarLocation
.. _doxid-d1/d48/classCurrentSource_1a68dd0c0732470342dc8392309a471207:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` getVarLocation(size_t index) const

Get variable location for current source model state variable.

.. index:: pair: function; getExtraGlobalParamLocation
.. _doxid-d1/d48/classCurrentSource_1a06d4317ea9bf386dbab1d4730f4558f6:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` getExtraGlobalParamLocation(const std::string& paramName) const

Get location of neuron model extra global parameter by name.

This is only used by extra global parameters which are pointers

.. index:: pair: function; getExtraGlobalParamLocation
.. _doxid-d1/d48/classCurrentSource_1ab723e7326963f12bbdd05da760bf579b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` getExtraGlobalParamLocation(size_t index) const

Get location of neuron model extra global parameter by omdex.

This is only used by extra global parameters which are pointers


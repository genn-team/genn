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
		// construction
	
		:target:`CurrentSource<doxid-d1/d48/classCurrentSource_1a23699d48b18506030c0d5afdf72ffd20>`(const CurrentSource&);
		:target:`CurrentSource<doxid-d1/d48/classCurrentSource_1a837a3749724b52ad1ab76751b2464c21>`();

		// methods
	
		void :ref:`setVarLocation<doxid-d1/d48/classCurrentSource_1aed244c33e6e0830d203f462d71cd949e>`(const std::string& varName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		void :ref:`setExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1a43687bec0ce75867db3661b587e2b6b9>`(const std::string& paramName, :ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` loc);
		const std::string& :target:`getName<doxid-d1/d48/classCurrentSource_1a59dd0ff630ed03251751ec2634457753>`() const;
		const :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`* :ref:`getCurrentSourceModel<doxid-d1/d48/classCurrentSource_1a4db1f74f237b66e235981ea290d86ada>`() const;
		const std::vector<double>& :target:`getParams<doxid-d1/d48/classCurrentSource_1a020b8cd27da6eaadbac238ff5dbc016a>`() const;
		const std::vector<:ref:`Models::VarInit<doxid-d8/dee/classModels_1_1VarInit>`>& :target:`getVarInitialisers<doxid-d1/d48/classCurrentSource_1af0e876921169fc46d0d801dc9514688a>`() const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d1/d48/classCurrentSource_1a0b5061240ea86ae09f52bf128b158aed>`(const std::string& varName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getVarLocation<doxid-d1/d48/classCurrentSource_1a4082262b23f60d230840455ffbb8c5f3>`(size_t index) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1ac589d53c052cf96ff6eb4f10adca997d>`(const std::string& paramName) const;
		:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` :ref:`getExtraGlobalParamLocation<doxid-d1/d48/classCurrentSource_1aeb8c8e3d3ce51295c8ccbc6386640a5b>`(size_t index) const;
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
.. _doxid-d1/d48/classCurrentSource_1a4db1f74f237b66e235981ea290d86ada:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	const :ref:`CurrentSourceModels::Base<doxid-d0/de0/classCurrentSourceModels_1_1Base>`* getCurrentSourceModel() const

Gets the current source model used by this group.

.. index:: pair: function; getVarLocation
.. _doxid-d1/d48/classCurrentSource_1a0b5061240ea86ae09f52bf128b158aed:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` getVarLocation(const std::string& varName) const

Get variable location for current source model state variable.

.. index:: pair: function; getVarLocation
.. _doxid-d1/d48/classCurrentSource_1a4082262b23f60d230840455ffbb8c5f3:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` getVarLocation(size_t index) const

Get variable location for current source model state variable.

.. index:: pair: function; getExtraGlobalParamLocation
.. _doxid-d1/d48/classCurrentSource_1ac589d53c052cf96ff6eb4f10adca997d:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` getExtraGlobalParamLocation(const std::string& paramName) const

Get location of neuron model extra global parameter by name.

This is only used by extra global parameters which are pointers

.. index:: pair: function; getExtraGlobalParamLocation
.. _doxid-d1/d48/classCurrentSource_1aeb8c8e3d3ce51295c8ccbc6386640a5b:

.. ref-code-block:: cpp
	:class: doxyrest-title-code-block

	:ref:`VarLocation<doxid-d6/d8f/variableMode_8h_1a2807180f6261d89020cf7d7d498fb087>` getExtraGlobalParamLocation(size_t index) const

Get location of neuron model extra global parameter by omdex.

This is only used by extra global parameters which are pointers

